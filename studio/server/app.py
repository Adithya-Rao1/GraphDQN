"""Metis Lead Studio server: one FastAPI app serving the static front-end and the API.

Real parts (Engine, Scorer, geometry, ...) are injected when importable; otherwise the
mock replay is used and runs are labelled source="mock". STUDIO_FORCE_MOCK=1 forces the mock.
"""
from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import logging
import secrets
import threading
import time
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from studio.schema import (SCHEMA_VERSION, RunConfig, RunCreated, ScoreResult, ValidateRequest,
                           ValidateResponse)
from studio.server import molio
from studio.server.core import Rejected, RunManager, Settings
from studio.wiring import describe

log = logging.getLogger("studio.server")

ROOT = Path(__file__).resolve().parents[1]
MOCK = ROOT / "mock" / "mock_run.json"
PRESETS = ROOT / "mock" / "presets.json"
WEB = ROOT / "web"
COOKIE = "studio_sid"

# Conventional entry points we look for in the other streams (first match wins).
_FACTORIES = {
    "studio.wiring:engine": (("build_engine",), ()),
    "studio.wiring:scorer": (("build_scorer",), ()),
}


def _public_names(mod) -> list:
    return [n for n in dir(mod) if not n.startswith("_") and not inspect.ismodule(getattr(mod, n))]


def _load_component(key: str) -> Optional[Any]:
    """Instantiate a stream's main object, or None if it isn't there yet."""
    modname = key.split(":")[0]
    try:
        mod = importlib.import_module(modname)
        factories, classes = _FACTORIES[key]
        for n in factories:
            if callable(getattr(mod, n, None)):
                return getattr(mod, n)()
        for n in classes:
            c = getattr(mod, n, None)
            if inspect.isclass(c) and not getattr(c, "_is_protocol", False):
                return c()
    except Exception as exc:
        log.info("component %s unavailable: %s", modname, type(exc).__name__)
    return None


def _module_present(modname: str) -> bool:
    try:
        return bool(_public_names(importlib.import_module(modname)))
    except Exception:
        return False


def create_app(engine: Any = None, scorer: Any = None, settings: Optional[Settings] = None) -> FastAPI:
    s = settings or Settings.from_env()
    if s.force_mock:
        engine, scorer = None, None
    else:
        engine = engine or _load_component("studio.wiring:engine")
        scorer = scorer or _load_component("studio.wiring:scorer")
    mock_events = json.loads(MOCK.read_text()) if MOCK.exists() else []
    mgr = RunManager(s, engine, mock_events)
    details = describe(engine)
    # Legacy boolean keys the front-end reads; derived from what the engine actually holds.
    components = {
        "engine": engine is not None,
        "scorer": scorer is not None,
        "geometry": engine is not None and _module_present("studio.engine.geometry"),
        "registry": details["registry_ops"] > 0,
        "explain": details["explainer"] is not None,
        "embed": details["embedder"] is not None,
        "policy": details["policy"] is not None,
        "mock_run": bool(mock_events),
    }

    app = FastAPI(title="Metis Lead Studio")
    app.state.manager, app.state.settings = mgr, s

    # ------------------------------------------------------------ middleware
    @app.middleware("http")
    async def guard(request: Request, call_next):
        if request.method in ("POST", "PUT", "PATCH"):
            cl = request.headers.get("content-length")
            if cl is None:
                return JSONResponse({"detail": "Content-Length required"}, status_code=411)
            if not cl.isdigit() or int(cl) > s.max_body_bytes:
                return JSONResponse({"detail": f"Body too large (max {s.max_body_bytes // 1000} KB)"},
                                    status_code=413)
        sid = request.cookies.get(COOKIE)
        fresh = not sid or len(sid) > 64
        if fresh:
            sid = secrets.token_urlsafe(16)
        request.state.sid = sid
        resp = await call_next(request)
        if fresh and request.url.path.startswith("/api/"):
            resp.set_cookie(COOKIE, sid, httponly=True, samesite="lax", max_age=86400)
        return resp

    @app.exception_handler(Rejected)
    async def _rejected(_: Request, exc: Rejected):
        headers = {"Retry-After": str(exc.retry_after)} if exc.retry_after else {}
        return JSONResponse({"detail": exc.message}, status_code=exc.status, headers=headers)

    # ---------------------------------------------------------------- routes
    @app.get("/api/health")
    def health():
        return {
            "ok": True,
            "schema_version": SCHEMA_VERSION,
            "engine_source": "live" if engine is not None else "mock",
            "force_mock": s.force_mock,
            "components": components,
            "details": details,
            "limits": {
                "max_concurrent_runs": s.max_concurrent, "max_queue": s.max_queue,
                "run_timeout_s": s.run_timeout, "max_generations": s.max_generations,
                "max_population": s.max_population, "runs_per_minute_per_session": s.runs_per_minute,
                "max_heavy_atoms": molio.MAX_HEAVY_ATOMS, "max_body_bytes": s.max_body_bytes,
            },
        }

    @app.get("/api/presets")
    def presets():
        try:
            return json.loads(PRESETS.read_text()) if PRESETS.exists() else []
        except (OSError, ValueError):
            log.warning("presets.json unreadable")
            return []

    @app.post("/api/molecules/validate", response_model=ValidateResponse)
    def validate(req: ValidateRequest, request: Request):
        wait = mgr.rate.hit(f"val:{request.state.sid}", s.validates_per_minute)
        if wait:
            raise Rejected(429, "Too many requests", wait)
        try:
            mol = molio.parse_molecule(req.smiles, req.molfile)
        except molio.InvalidMolecule as exc:
            return ValidateResponse(ok=False, error=str(exc))
        smi = molio.canonical(mol)
        block, provider = molio.molblock_3d(smi)
        if block is None:
            return ValidateResponse(ok=False, canonical_smiles=smi, heavy_atoms=mol.GetNumHeavyAtoms(),
                                    error="Could not generate a 3D conformer for this molecule")
        score: Optional[ScoreResult] = None
        if scorer is not None:
            try:
                score = scorer.score(smi, RunConfig(seed_smiles=smi))
            except Exception as exc:
                log.warning("seed scoring failed: %s", type(exc).__name__)
        return ValidateResponse(ok=True, canonical_smiles=smi, heavy_atoms=mol.GetNumHeavyAtoms(),
                                molblock=block, score=score)

    @app.post("/api/runs", response_model=RunCreated)
    def create_run(cfg: RunConfig, request: Request):
        try:
            mol = molio.parse_molecule(cfg.seed_smiles)
        except molio.InvalidMolecule as exc:
            raise HTTPException(422, str(exc))
        n = mol.GetNumHeavyAtoms()
        if any(i < 0 or i >= n for i in cfg.locked_atoms):
            raise HTTPException(422, "locked_atoms index out of range")
        cfg = cfg.model_copy(update={
            "seed_smiles": molio.canonical(mol),
            "generations": max(1, min(cfg.generations, s.max_generations)),
            "population_size": max(2, min(cfg.population_size, s.max_population)),
            "policy_guidance": max(0.0, min(cfg.policy_guidance, 1.0)),
            "locked_atoms": sorted(set(cfg.locked_atoms)),
            "use_reactions": cfg.use_reactions if engine is not None else False,
        })
        run = mgr.submit(request.state.sid, cfg, secrets.token_urlsafe(9))
        log.info("run %s created source=%s heavy_atoms=%d gens=%d pop=%d", run.run_id[:6], run.source, n,
                 cfg.generations, cfg.population_size)
        return RunCreated(run_id=run.run_id)

    @app.delete("/api/runs/{run_id}")
    def cancel_run(run_id: str, request: Request):
        run = mgr.get(run_id)
        if run is None or run.session != request.state.sid:
            raise HTTPException(404, "unknown run")
        return {"cancelled": mgr.cancel(run)}

    @app.get("/api/runs/{run_id}/events")
    async def run_events(run_id: str, request: Request):
        run = mgr.get(run_id)
        if run is None:
            raise HTTPException(404, "unknown run")
        last = request.headers.get("last-event-id")
        start = int(last) + 1 if last and last.isdigit() else 0

        async def stream():
            i, idle = start, time.monotonic()
            yield "retry: 3000\n\n"
            while True:
                if i < len(run.events):
                    ev = run.events[i]
                    if run.pace and i > start:
                        await asyncio.sleep(run.pace)
                    yield f"id: {ev['seq']}\nevent: {ev['type']}\ndata: {json.dumps(ev, separators=(',', ':'))}\n\n"
                    i, idle = i + 1, time.monotonic()
                    continue
                if run.finished:
                    return
                if await request.is_disconnected():
                    return
                if time.monotonic() - idle >= s.heartbeat:
                    yield ": heartbeat\n\n"
                    idle = time.monotonic()
                await asyncio.sleep(0.05)

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    if WEB.exists():
        app.mount("/", StaticFiles(directory=WEB, html=True), name="web")
    return app


class _LazyApp:
    """ASGI shim so `uvicorn studio.server.app:app` imports instantly; the real engine is
    built by create_app() on the first ASGI scope (lifespan startup under uvicorn)."""

    def __init__(self):
        self._app: Optional[FastAPI] = None
        self._lock = threading.Lock()

    def _get(self) -> FastAPI:
        with self._lock:
            if self._app is None:
                self._app = create_app()
            return self._app

    async def __call__(self, scope, receive, send):
        await self._get()(scope, receive, send)

    def __getattr__(self, name):
        return getattr(self._get(), name)


app = _LazyApp()
