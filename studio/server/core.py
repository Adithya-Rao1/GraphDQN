"""Run management, limits and caching for the studio server (no auth, no database)."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

from studio.schema import RunConfig

log = logging.getLogger("studio.server")


def _env(name: str, default, cast=float):
    try:
        return cast(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


@dataclass
class Settings:
    max_concurrent: int = 2       # engine runs executing at once: the box has 2 vCPUs
    max_queue: int = 4            # waiting runs; beyond this -> 429
    run_timeout: float = 90.0     # wall-clock seconds from execution start
    max_generations: int = 15
    max_population: int = 40
    runs_per_minute: int = 6      # per session
    validates_per_minute: int = 30
    max_body_bytes: int = 200_000
    heartbeat: float = 15.0
    mock_delay: float = 0.4       # pacing of the mock replay (seconds / event)
    replay_delay: float = 0.25    # pacing of cache replays
    cache_size: int = 64
    max_runs_kept: int = 100
    force_mock: bool = False

    @classmethod
    def from_env(cls) -> "Settings":
        d = cls()
        return cls(
            max_concurrent=_env("STUDIO_MAX_CONCURRENT", d.max_concurrent, int),
            max_queue=_env("STUDIO_MAX_QUEUE", d.max_queue, int),
            run_timeout=_env("STUDIO_RUN_TIMEOUT", d.run_timeout),
            max_generations=_env("STUDIO_MAX_GENERATIONS", d.max_generations, int),
            max_population=_env("STUDIO_MAX_POPULATION", d.max_population, int),
            runs_per_minute=_env("STUDIO_RUNS_PER_MIN", d.runs_per_minute, int),
            validates_per_minute=_env("STUDIO_VALIDATES_PER_MIN", d.validates_per_minute, int),
            heartbeat=_env("STUDIO_HEARTBEAT", d.heartbeat),
            mock_delay=_env("STUDIO_MOCK_DELAY", d.mock_delay),
            replay_delay=_env("STUDIO_REPLAY_DELAY", d.replay_delay),
            force_mock=os.environ.get("STUDIO_FORCE_MOCK", "") not in ("", "0", "false"),
        )


class Rejected(Exception):
    def __init__(self, status: int, message: str, retry_after: Optional[int] = None):
        self.status, self.message, self.retry_after = status, message, retry_after


class RateWindow:
    """Sliding one-minute window per key."""

    def __init__(self):
        self._hits: Dict[str, Deque[float]] = {}
        self._lock = threading.Lock()

    def hit(self, key: str, limit: int, window: float = 60.0) -> Optional[int]:
        """Record a hit; return None if allowed else seconds until allowed."""
        now = time.monotonic()
        with self._lock:
            q = self._hits.setdefault(key, deque())
            while q and now - q[0] > window:
                q.popleft()
            if len(q) >= limit:
                return max(1, int(window - (now - q[0])) + 1)
            q.append(now)
            if len(self._hits) > 5000:  # bound memory under abuse
                for k in [k for k, v in self._hits.items() if not v or now - v[-1] > window]:
                    self._hits.pop(k, None)
            return None


@dataclass
class Run:
    run_id: str
    session: str
    config: RunConfig
    source: str                       # live | mock | replay
    cache_key: Optional[str] = None
    pace: float = 0.0                 # delay applied by the SSE writer (replays only)
    events: List[dict] = field(default_factory=list)
    finished: bool = False
    status: str = "queued"            # queued | running | done | error | cancelled
    started: Optional[float] = None
    created: float = field(default_factory=time.monotonic)
    lock: threading.Lock = field(default_factory=threading.Lock)
    timer: Optional[threading.Timer] = None

    def push(self, type_: str, payload: dict) -> bool:
        with self.lock:
            if self.finished:
                return False
            self.events.append({"type": type_, "run_id": self.run_id, "seq": len(self.events), "payload": payload})
            return True

    def finish(self, status: str, error: Optional[str] = None, recoverable: bool = False) -> bool:
        """Terminate the stream exactly once; optionally append a terminal error event."""
        with self.lock:
            if self.finished:
                return False
            if error is not None:
                self.events.append({"type": "error", "run_id": self.run_id, "seq": len(self.events),
                                    "payload": {"message": error, "recoverable": recoverable}})
            self.finished, self.status = True, status
            if self.timer:
                self.timer.cancel()
            return True


def config_key(seed_canonical: str, cfg: RunConfig) -> str:
    body = cfg.model_dump(mode="json", exclude={"preset_id"})
    body["seed_smiles"] = seed_canonical
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


class RunManager:
    def __init__(self, settings: Settings, engine: Any, mock_events: List[dict]):
        self.s, self.engine, self.mock_events = settings, engine, mock_events
        self.pool = ThreadPoolExecutor(max_workers=settings.max_concurrent, thread_name_prefix="run")
        self.runs: "OrderedDict[str, Run]" = OrderedDict()
        self.cache: "OrderedDict[str, List[dict]]" = OrderedDict()
        self.lock = threading.Lock()
        self.rate = RateWindow()

    # ------------------------------------------------------------ admission
    def _active(self) -> List[Run]:
        return [r for r in self.runs.values() if not r.finished and r.source != "replay"]

    def submit(self, session: str, cfg: RunConfig, run_id: str) -> Run:
        wait = self.rate.hit(f"run:{session}", self.s.runs_per_minute)
        if wait:
            raise Rejected(429, "Too many runs from this session, slow down", wait)
        key = config_key(cfg.seed_smiles, cfg)
        with self.lock:
            cached = self.cache.get(key)
            if cached is not None:
                self.cache.move_to_end(key)
                run = Run(run_id, session, cfg, "replay", cache_key=key, pace=self.s.replay_delay)
                for i, ev in enumerate(cached):
                    ev = json.loads(json.dumps(ev))
                    ev.update(run_id=run_id, seq=i)
                    if ev["type"] == "run_start":
                        ev["payload"]["source"] = "replay"
                    run.events.append(ev)
                run.finished, run.status = True, "done"
                self._register(run)
                return run
            active = self._active()
            if any(r.session == session for r in active):
                raise Rejected(429, "You already have a run in progress", self._retry_after(active))
            if len(active) >= self.s.max_concurrent + self.s.max_queue:
                raise Rejected(429, "Server is busy, try again shortly", self._retry_after(active))
            source = "mock" if self.engine is None else "live"
            run = Run(run_id, session, cfg, source, cache_key=key if source == "live" else None)
            self._register(run)
        self.pool.submit(self._execute, run)
        return run

    def _retry_after(self, active: List[Run]) -> int:
        now, best = time.monotonic(), self.s.run_timeout
        for r in active:
            if r.started:
                best = min(best, max(1.0, self.s.run_timeout - (now - r.started)))
        return int(best) + 1

    def _register(self, run: Run) -> None:
        self.runs[run.run_id] = run
        while len(self.runs) > self.s.max_runs_kept:
            oldest = next((k for k, r in self.runs.items() if r.finished), None)
            if oldest is None:
                break
            self.runs.pop(oldest)

    def get(self, run_id: str) -> Optional[Run]:
        return self.runs.get(run_id)

    def cancel(self, run: Run) -> bool:
        return run.finish("cancelled", "Run cancelled", recoverable=True)

    # ------------------------------------------------------------ execution
    def _on_timeout(self, run: Run) -> None:
        if run.finish("error", f"Run exceeded the {self.s.run_timeout:g}s time limit", recoverable=True):
            log.warning("run %s timed out", run.run_id[:6])

    def _execute(self, run: Run) -> None:
        if run.finished:  # cancelled while queued
            return
        run.status, run.started = "running", time.monotonic()
        run.timer = threading.Timer(self.s.run_timeout, self._on_timeout, args=(run,))
        run.timer.daemon = True
        run.timer.start()
        try:
            gen = self._mock_iter(run) if run.source == "mock" else self.engine.run(run.run_id, run.config)
            try:
                for ev in gen:
                    d = ev if isinstance(ev, dict) else ev.model_dump(mode="json")
                    if not run.push(d["type"], d["payload"]):
                        break  # timed out / cancelled
                    if d["type"] in ("run_end", "error"):
                        break
            finally:
                close = getattr(gen, "close", None)
                if close:
                    close()
            with run.lock:
                kinds = [e["type"] for e in run.events]
            if "run_end" in kinds and "error" not in kinds:
                if run.finish("done") and run.cache_key:
                    self._remember(run)
            elif "error" in kinds:
                run.finish("error")
            else:
                run.finish("error", "Engine ended without a result", recoverable=True)
        except Exception as exc:  # engine bugs must not kill the worker
            log.exception("run %s failed", run.run_id[:6])
            run.finish("error", f"Engine failure: {type(exc).__name__}", recoverable=True)
        finally:
            log.info("run %s source=%s status=%s events=%d secs=%.1f", run.run_id[:6], run.source,
                     run.status, len(run.events), time.monotonic() - (run.started or run.created))

    def _mock_iter(self, run: Run):
        for ev in self.mock_events:
            if run.finished:
                return
            time.sleep(self.s.mock_delay)
            ev = json.loads(json.dumps(ev))
            if ev["type"] == "run_start":
                ev["payload"]["source"] = "mock"
            yield ev

    def _remember(self, run: Run) -> None:
        with self.lock:
            self.cache[run.cache_key] = list(run.events)
            while len(self.cache) > self.s.cache_size:
                self.cache.popitem(last=False)
