"""Manual mode: preview a single hand-picked edit, and log every accepted step as a training transition.

Transitions are written in the policy format {parent_smiles, edit_id, site, child_smiles, reward} (plus
metadata) with chosen_by="user". `reward` is the change in RDKit composite, not binding.
"""
from __future__ import annotations

import json
import logging
import os
import secrets
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import Response
from pydantic import BaseModel
from rdkit import Chem

from studio.schema import (EditRecord, ManualPreview, ManualPreviewRequest, ManualSession, ManualStep,
                           RunConfig)
from studio.server import molio
from studio.server.core import Rejected

log = logging.getLogger("studio.server")

MAX_SESSIONS = 500
MAX_STEPS = 500
MAX_APPLICABLE = 500


class SessionCreate(BaseModel):
    seed_smiles: str
    target_id: Optional[str] = None


def cache_dir() -> Path:
    return Path(os.environ.get("STUDIO_CACHE_DIR") or ".cache/studio") / "manual"


def transition(sess: ManualSession, st: ManualStep) -> dict:
    return {"parent_smiles": st.parent_smiles, "edit_id": st.edit_id, "site": list(st.site),
            "child_smiles": st.child_smiles, "reward": st.reward_delta, "chosen_by": "user",
            "t": round(st.t, 3), "session_id": sess.session_id, "seed_smiles": sess.seed_smiles,
            "target_id": sess.target_id}


class Sessions:
    def __init__(self):
        self.items: "OrderedDict[str, ManualSession]" = OrderedDict()
        self.started: Dict[str, float] = {}
        self.lock = threading.Lock()

    def create(self, seed: str, target_id: Optional[str]) -> ManualSession:
        with self.lock:
            s = ManualSession(session_id=secrets.token_urlsafe(9), seed_smiles=seed, target_id=target_id)
            self.items[s.session_id] = s
            self.started[s.session_id] = time.monotonic()
            while len(self.items) > MAX_SESSIONS:
                k, _ = self.items.popitem(last=False)
                self.started.pop(k, None)
            return s

    def get(self, sid: str) -> Optional[ManualSession]:
        return self.items.get(sid)

    def append(self, sid: str, st: ManualStep) -> Optional[ManualSession]:
        with self.lock:
            s = self.items.get(sid)
            if s is None:
                return None
            if len(s.steps) >= MAX_STEPS:
                raise Rejected(429, f"Session is limited to {MAX_STEPS} steps")
            st = st.model_copy(update={"t": time.monotonic() - self.started[sid]})
            s.steps.append(st)
            try:  # append-only log; a disk problem must not lose the in-memory session
                d = cache_dir()
                d.mkdir(parents=True, exist_ok=True)
                with open(d / f"{sid}.jsonl", "a") as f:
                    f.write(json.dumps(transition(s, st)) + "\n")
            except OSError as exc:
                log.warning("manual log write failed: %s", type(exc).__name__)
            return s.model_copy(deep=True)


def register(app: FastAPI, svc: Any, scorer: Any, mgr: Any) -> None:
    sessions = Sessions()
    app.state.manual_sessions = sessions
    state: Dict[str, Any] = {"scorer": scorer}

    def rdkit_scorer():
        if state["scorer"] is None:
            from studio.scoring import RDKitScorer
            state["scorer"] = RDKitScorer()
        return state["scorer"]

    def registry():
        try:
            return svc.registry()
        except Exception as exc:
            raise HTTPException(503, f"edit registry unavailable ({type(exc).__name__})")

    def sanity(parent: str, child: str):
        try:
            from studio.sanity import check_child
        except Exception:
            return True, None  # no sanity module on this build
        ok, why = check_child(parent, child)
        return bool(ok), None if ok else str(why)

    def throttle(request: Request, kind: str, limit: int) -> None:
        wait = mgr.rate.hit(f"{kind}:{request.state.sid}", limit)
        if wait:
            raise Rejected(429, "Too many requests", wait)

    def check_smiles(smi: str) -> None:
        try:
            molio.parse_molecule(smi)
        except molio.InvalidMolecule as exc:
            raise HTTPException(422, str(exc))

    @app.get("/api/manual/applicable")
    def applicable(request: Request, smiles: str = Query(..., max_length=2000), locked: str = Query("", max_length=500)):
        throttle(request, "app", 120)
        check_smiles(smiles)
        try:
            lock = [int(x) for x in locked.split(",") if x.strip()]
        except ValueError:
            raise HTTPException(422, "locked must be comma-separated atom indices")
        out: List[dict] = []
        for op, site in registry().applicable(smiles, lock):
            out.append({"edit_id": op.id, "name": op.name, "category": op.category, "site": list(site),
                        "rationale": op.rationale})
            if len(out) >= MAX_APPLICABLE:
                break
        return out

    @app.post("/api/manual/preview", response_model=ManualPreview)
    def preview(req: ManualPreviewRequest, request: Request):
        throttle(request, "prev", 120)
        check_smiles(req.smiles)
        if req.target_id is not None and svc.get_target(req.target_id) is None:
            raise HTTPException(422, f"unknown target_id {req.target_id!r}")
        reg = registry()
        try:
            op = reg.get(req.edit_id)
        except KeyError:
            return ManualPreview(ok=False, reason=f"unknown edit {req.edit_id!r}")
        n = Chem.MolFromSmiles(req.smiles).GetNumAtoms()
        if not req.site or any(i < 0 or i >= n for i in req.site):
            return ManualPreview(ok=False, reason="site index out of range")
        products = op.apply(req.smiles, list(req.site))
        if not products:
            return ManualPreview(ok=False, reason="edit does not apply at this site")
        child = products[0]
        sane, why = sanity(req.smiles, child)

        cfg = RunConfig(seed_smiles=req.smiles)
        sc = rdkit_scorer()
        p_score, c_score = sc.score(req.smiles, cfg), sc.score(child, cfg)
        if req.target_id is not None:
            bs = svc.binding_scorer()
            if bs is not None:
                try:
                    bcfg = RunConfig(seed_smiles=req.smiles, stage="binding", target_id=req.target_id)
                    c_score = c_score.model_copy(update={"extras": {**c_score.extras, **bs.score(child, bcfg).extras}})
                    p_score = p_score.model_copy(update={"extras": {**p_score.extras, **bs.score(req.smiles, bcfg).extras}})
                except Exception as exc:  # binding is optional decoration on the preview
                    log.warning("binding extras failed: %s", type(exc).__name__)
        rec = EditRecord(edit_id=op.id, name=op.name, category=op.category, site=list(req.site),
                         chosen_by="user", parents=[req.smiles])
        expl = svc.explainer().explain(req.smiles, child, rec, p_score, c_score, op)

        from studio.engine import geometry
        try:
            _, pmol = geometry.embed_3d(req.smiles)
            block, cmol = geometry.embed_3d(child, pmol)
            amap = geometry.atom_map(pmol, cmol)
        except ValueError:
            return ManualPreview(ok=False, child_smiles=child, score=c_score, explanation=expl, sane=sane,
                                 reason="could not build a 3D structure for the product")
        return ManualPreview(ok=True, child_smiles=child, molblock=block, atom_map=amap, score=c_score,
                             explanation=expl, sane=sane, reason=why)

    @app.post("/api/manual/sessions", response_model=ManualSession)
    def create_session(req: SessionCreate, request: Request):
        throttle(request, "msess", 20)
        try:
            seed = molio.canonical(molio.parse_molecule(req.seed_smiles))
        except molio.InvalidMolecule as exc:
            raise HTTPException(422, str(exc))
        if req.target_id is not None and svc.get_target(req.target_id) is None:
            raise HTTPException(422, f"unknown target_id {req.target_id!r}")
        return sessions.create(seed, req.target_id)

    @app.post("/api/manual/sessions/{session_id}/steps", response_model=ManualSession)
    def add_step(session_id: str, step: ManualStep, request: Request):
        throttle(request, "mstep", 240)
        if sessions.get(session_id) is None:
            raise HTTPException(404, "unknown session")
        check_smiles(step.parent_smiles)
        check_smiles(step.child_smiles)
        if step.reward_delta is None:  # server-side RDKit composite delta
            sc, cfg = rdkit_scorer(), RunConfig(seed_smiles=step.parent_smiles)
            step = step.model_copy(update={
                "reward_delta": sc.score(step.child_smiles, cfg).composite - sc.score(step.parent_smiles, cfg).composite})
        s = sessions.append(session_id, step)
        if s is None:
            raise HTTPException(404, "unknown session")
        return s

    @app.get("/api/manual/sessions/{session_id}/export")
    def export(session_id: str):
        s = sessions.get(session_id)
        if s is None:
            raise HTTPException(404, "unknown session")
        body = "".join(json.dumps(transition(s, st)) + "\n" for st in s.steps)
        return Response(body, media_type="application/x-ndjson",
                        headers={"Content-Disposition": f'attachment; filename="manual_{session_id}.jsonl"'})
