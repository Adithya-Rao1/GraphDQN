"""Stage plumbing for the public API: targets, structure files, stage-aware runs, auth.

`Services` resolves the optional parts (target loader, binding stage, registry, pose client) lazily and lets
tests inject fakes. Nothing here fabricates data: a missing part yields a clear 503/failed status.
"""
from __future__ import annotations

import hmac
import importlib
import inspect
import json
import logging
import os
import re
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

from studio.schema import RunConfig, TargetInfo

log = logging.getLogger("studio.server")

STUDIO = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).with_name("fixtures")
ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,40}$")
DATA_EXT = {".pdb": "text/plain", ".sdf": "text/plain", ".json": "application/json"}
MAX_MW_CAP = 700.0


def structures_dir() -> Path:
    return Path(os.environ.get("STUDIO_STRUCTURES_DIR") or STUDIO / "data" / "structures")


def _cap(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


class Services:
    """Late-bound collaborators. Every attribute may be injected (tests) or resolved on first use."""

    def __init__(self, binding_factory: Optional[Callable[..., Any]] = None, pose_client: Any = None,
                 binding_scorer: Any = None, registry: Any = None, targets: Any = None,
                 explainer: Any = None):
        self._binding_factory, self.pose_client = binding_factory, pose_client
        self._binding_scorer, self._registry, self._targets = binding_scorer, registry, targets
        self._explainer = explainer
        self._engines: Dict[Optional[str], Any] = {}
        self._lock = threading.Lock()

    # ----------------------------------------------------------- targets
    def _loader(self):
        if self._targets is not None:
            return self._targets
        try:
            return importlib.import_module("studio.data.loader")
        except Exception:
            return None

    def target_source(self) -> str:
        return "loader" if self._loader() is not None else "fixture"

    @staticmethod
    def _norm(obj: Any) -> TargetInfo:
        return obj if isinstance(obj, TargetInfo) else TargetInfo.model_validate(obj)

    def _fixture_targets(self) -> List[TargetInfo]:
        return [TargetInfo.model_validate(json.loads(p.read_text())) for p in sorted(FIXTURES.glob("*_target.json"))]

    def list_targets(self) -> List[TargetInfo]:
        ld = self._loader()
        if ld is None:
            return self._fixture_targets()
        out = []
        for item in ld.list_targets():
            if isinstance(item, str):
                tid = item
            elif isinstance(item, dict):
                tid = item["id"]
            else:
                tid = item.id
            t = self.get_target(tid)
            if t is not None:
                out.append(t)
        return out

    def get_target(self, tid: Optional[str]) -> Optional[TargetInfo]:
        if not tid or not ID_RE.match(tid):
            return None
        ld = self._loader()
        if ld is None:
            return next((t for t in self._fixture_targets() if t.id == tid), None)
        try:
            return self._norm(ld.get_target(tid))
        except (KeyError, FileNotFoundError, ValueError):
            return None

    def structure_path(self, tid: str) -> Optional[Path]:
        t = self.get_target(tid)
        if t is None:
            return None
        p = resolve_data_path(t.structure_url)
        return p

    # ----------------------------------------------------------- binding
    def binding_available(self) -> bool:
        if self._binding_factory is not None:
            return True
        try:
            self._binding_factory = importlib.import_module("studio.binding.stage").build_binding_engine
            return True
        except Exception as exc:
            log.info("binding stage unavailable: %s", type(exc).__name__)
            return False

    def binding_engine(self, config: RunConfig):
        key = config.target_id
        with self._lock:
            if key not in self._engines:
                fn = self._binding_factory
                params = inspect.signature(fn).parameters
                avail = {"target_id": config.target_id, "config": config}
                kw = {k: v for k, v in avail.items() if k in params}
                self._engines[key] = fn(**kw)
            return self._engines[key]

    def binding_scorer(self):
        if self._binding_scorer is None:
            try:
                self._binding_scorer = importlib.import_module("studio.binding.scorer").BindingScorer()
            except Exception as exc:
                log.info("binding scorer unavailable: %s", type(exc).__name__)
                return None
        return self._binding_scorer

    # ----------------------------------------------------------- registry
    def registry(self):
        if self._registry is None:
            self._registry = importlib.import_module("studio.registry").get_registry()
        return self._registry

    def explainer(self):
        if self._explainer is None:
            self._explainer = importlib.import_module("studio.explain.explainer").RDKitExplainer()
        return self._explainer

    def health(self) -> dict:
        return {"targets": self.target_source(), "binding_stage": self.binding_available(),
                "dynamicbind_url": os.environ.get("DYNAMICBIND_URL", "http://127.0.0.1:8100")}


class StageEngine:
    """Engine facade: routes `binding` runs to the binding engine, everything else to the base engine."""

    def __init__(self, base: Any, services: Services):
        self.base, self.svc = base, services

    def run(self, run_id: str, config: RunConfig):
        if config.stage == "binding":
            return self.svc.binding_engine(config).run(run_id, config)
        return self.base.run(run_id, config)


def resolve_data_path(rel: str) -> Optional[Path]:
    """Map a URL path under /api/data to a whitelisted file inside the structures dir, or None."""
    if not rel or "\x00" in rel or "\\" in rel or rel.startswith("/"):
        return None
    parts = rel.split("/")
    if parts[0] == "structures":
        parts = parts[1:]
    if not parts or any(p in ("", ".", "..") or p.startswith(".") for p in parts):
        return None
    root = structures_dir().resolve()
    p = (root.joinpath(*parts)).resolve()
    if p.suffix.lower() not in DATA_EXT or not p.is_relative_to(root) or not p.is_file():
        return None
    return p


def prepare_config(cfg: RunConfig, svc: Services, engine_live: bool) -> RunConfig:
    """Stage validation and caps for POST /api/runs. Raises HTTPException."""
    if cfg.stage == "manual":
        raise HTTPException(422, "Manual mode does not run the GA; use /api/manual/*")
    if cfg.target_id is not None and svc.get_target(cfg.target_id) is None:
        raise HTTPException(422, f"unknown target_id {cfg.target_id!r}")
    if cfg.stage != "binding":
        return cfg
    if cfg.target_id is None:
        raise HTTPException(422, "stage 'binding' requires target_id")
    if not engine_live:
        raise HTTPException(503, "Binding stage needs the live engine, but the server is running in mock mode")
    if not svc.binding_available():
        raise HTTPException(503, "Binding stage is not available on this server (studio.binding.stage not importable)")
    upd: Dict[str, Any] = {
        "policy_guidance": 0.0,  # stage 1 is a pure GA; the RL policy is untrained
        "use_reactions": False,
        "generations": min(cfg.generations, _cap("STUDIO_BINDING_MAX_GENERATIONS", 10)),
        "population_size": min(cfg.population_size, _cap("STUDIO_BINDING_MAX_POPULATION", 30)),
        "min_similarity_to_seed": max(0.0, min(cfg.min_similarity_to_seed, 1.0)),
    }
    if cfg.max_mw is not None:
        upd["max_mw"] = max(100.0, min(cfg.max_mw, MAX_MW_CAP))
    return cfg.model_copy(update=upd)


def install_auth(app: FastAPI, key: Optional[str]) -> None:
    """If `key` is set, every /api/* route except /api/health needs X-Studio-Key == key."""
    if not key:
        return
    kb = key.encode()

    @app.middleware("http")
    async def auth(request: Request, call_next):
        p = request.url.path
        if p.startswith("/api") and p != "/api/health":
            given = request.headers.get("x-studio-key", "").encode()
            if not hmac.compare_digest(given, kb):
                return JSONResponse({"detail": "missing or invalid X-Studio-Key"}, status_code=401)
        return await call_next(request)


def install(app: FastAPI, svc: Services, mgr: Any, scorer: Any, settings: Any) -> None:
    from studio.server import manual, poses

    @app.get("/api/targets")
    def targets():
        return [t.model_dump(mode="json") for t in svc.list_targets()]

    @app.get("/api/targets/{target_id}")
    def target(target_id: str):
        t = svc.get_target(target_id)
        if t is None:
            raise HTTPException(404, "unknown target")
        return t.model_dump(mode="json")

    @app.get("/api/data/{path:path}")
    def data(path: str):
        p = resolve_data_path(path)
        if p is None:
            raise HTTPException(404, "not found")
        return FileResponse(p, media_type=DATA_EXT[p.suffix.lower()])

    poses.register(app, svc, mgr)
    manual.register(app, svc, scorer, mgr)
