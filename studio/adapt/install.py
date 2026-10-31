"""`install(app)`: mounts /api/adapt/* on the studio FastAPI app.

Wiring (integrator): in studio/server/app.py after the app is built:
    from studio.adapt.install import install as install_adapt; install_adapt(app)
and when building a run's policy, if `config.adapter_id` / the request carries one:
    from studio.adapt.policy_factory import load_adapted_policy
    policy = load_adapted_policy(adapter_id, owner_of(request))   # .adapter_info says if it fell back
"""
from __future__ import annotations

import json
import re
from typing import Any, List, Optional

from fastapi import APIRouter, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from studio.adapt import store
from studio.adapt.dataset import MAX_TRANSITIONS, MIN_VALID_STEPS
from studio.adapt.finetune import TooFewSteps
from studio.adapt.jobs import JobManager, QueueFull

SID_RE = re.compile(r"^[A-Za-z0-9_-]{6,64}$")
MAX_LOG_BYTES = 2_000_000


def owner_of(request: Request) -> str:
    """Owner of adapters/jobs. Swap for the accounts login when it lands (must be stable per user)."""
    sid = getattr(request.state, "sid", None) or request.cookies.get("studio_sid")
    if not sid:
        raise HTTPException(400, "No session cookie (studio_sid); reload the page and retry.")
    return f"sid:{sid}"


class JobRequest(BaseModel):
    transitions: Optional[List[dict]] = Field(default=None, max_length=MAX_TRANSITIONS)
    manual_session_id: Optional[str] = Field(default=None, max_length=64)


def _manual_log(session_id: str) -> List[dict]:
    from pathlib import Path
    import os
    if not SID_RE.match(session_id):
        raise HTTPException(422, "Bad manual_session_id")
    p = Path(os.environ.get("STUDIO_CACHE_DIR") or ".cache/studio") / "manual" / f"{session_id}.jsonl"
    try:
        if p.stat().st_size > MAX_LOG_BYTES:
            raise HTTPException(413, "Manual log too large")
        lines = p.read_text().splitlines()
    except FileNotFoundError:
        raise HTTPException(404, "No logged steps for that manual session")
    out = []
    for ln in lines[:MAX_TRANSITIONS]:
        try:
            out.append(json.loads(ln))
        except ValueError:
            continue
    return out


def make_runner(registry=None):
    def runner(transitions: Any, owner: str, progress=None) -> dict:
        from studio.adapt import pipeline
        from studio.registry import get_registry
        return pipeline.run(transitions, owner, registry=registry or get_registry(), progress=progress)
    return runner


def install(app: FastAPI, manager: Optional[JobManager] = None, owner_fn=owner_of) -> JobManager:
    mgr = manager or JobManager(make_runner())
    r = APIRouter(prefix="/api/adapt", tags=["adapt"])

    @r.post("/jobs")
    def create_job(req: JobRequest, request: Request):
        owner = owner_fn(request)
        if (req.transitions is None) == (req.manual_session_id is None):
            raise HTTPException(422, "Send exactly one of transitions or manual_session_id")
        trans = req.transitions if req.transitions is not None else _manual_log(req.manual_session_id)
        if len(trans) < MIN_VALID_STEPS:
            raise HTTPException(422, f"Need at least {MIN_VALID_STEPS} logged edits to adapt; got {len(trans)}.")
        try:
            return {"job_id": mgr.submit(owner, trans)}
        except QueueFull as exc:
            raise HTTPException(429, str(exc), headers={"Retry-After": "30"})

    @r.get("/jobs/{job_id}")
    def job(job_id: str, request: Request):
        j = mgr.get(job_id, owner_fn(request))
        if j is None:
            raise HTTPException(404, "Unknown job")
        return j

    @r.get("/adapters")
    def adapters(request: Request):
        return {"adapters": [{k: v for k, v in m.items() if k != "owner"} for m in store.list_adapters(owner_fn(request))]}

    @r.delete("/adapters/{adapter_id}")
    def delete_adapter(adapter_id: str, request: Request):
        if not store.delete(owner_fn(request), adapter_id):
            raise HTTPException(404, "Unknown adapter")
        return {"deleted": adapter_id}

    app.include_router(r)
    return mgr
