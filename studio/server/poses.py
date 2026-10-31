"""Pose jobs: precomputed lookup first, else a DynamicBind service (contract in POSES.md).

No pose is ever synthesised: if the service cannot be reached the job is `failed` with a note.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException, Request

from studio.schema import PoseRequest, PoseResult
from studio.server import molio
from studio.server.core import Rejected

log = logging.getLogger("studio.server")

STUDIO = Path(__file__).resolve().parents[1]
MAX_ACTIVE = 8


def smiles_hash(canonical_smiles: str) -> str:
    """Filename stem of a precomputed pose: sha1 of the RDKit-canonical SMILES, first 16 hex chars."""
    return hashlib.sha1(canonical_smiles.encode()).hexdigest()[:16]


def precomputed_dir() -> Path:
    return Path(os.environ.get("STUDIO_POSES_DIR") or STUDIO / "data" / "precomputed_poses")


class PoseServiceError(RuntimeError):
    pass


class HttpDynamicBind:
    """Client for the tiny DynamicBind service contract (POSES.md). stdlib only."""

    def __init__(self, url: Optional[str] = None, timeout: float = 10.0):
        self.url = (url or os.environ.get("DYNAMICBIND_URL") or "http://127.0.0.1:8100").rstrip("/")
        self.timeout = timeout

    def _call(self, method: str, path: str, body: Optional[dict] = None) -> dict:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.url + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as exc:
            raise PoseServiceError(f"DynamicBind service returned HTTP {exc.code}") from None
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise PoseServiceError(f"DynamicBind service unreachable at {self.url} ({type(exc).__name__})") from None

    def dock(self, target_id: str, smiles: str) -> str:
        d = self._call("POST", "/dock", {"target_id": target_id, "smiles": smiles})
        if not isinstance(d, dict) or "job_id" not in d:
            raise PoseServiceError("DynamicBind service gave no job_id")
        return str(d["job_id"])

    def job(self, remote_id: str) -> dict:
        return self._call("GET", f"/jobs/{remote_id}")


class PoseManager:
    def __init__(self, svc, client=None, ttl: float = 3600.0, poll: float = 2.0, timeout: float = 900.0):
        self.svc, self.ttl, self.poll, self.timeout = svc, ttl, poll, timeout
        self.client = client or svc.pose_client or HttpDynamicBind()
        self.jobs: Dict[str, PoseResult] = {}
        self.stamp: Dict[str, float] = {}
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pose")  # global concurrency 1

    def _purge(self) -> None:
        now = time.monotonic()
        for k in [k for k, t in self.stamp.items() if now - t > self.ttl]:
            self.jobs.pop(k, None)
            self.stamp.pop(k, None)

    def get(self, job_id: str) -> Optional[PoseResult]:
        with self.lock:
            self._purge()
            j = self.jobs.get(job_id)
            return j.model_copy() if j else None

    def _set(self, job_id: str, **upd) -> None:
        with self.lock:
            if job_id in self.jobs:
                self.jobs[job_id] = self.jobs[job_id].model_copy(update=upd)

    def _precomputed(self, target_id: str, smi: str) -> Optional[PoseResult]:
        p = precomputed_dir() / target_id / f"{smiles_hash(smi)}.json"
        try:
            if p.is_file():
                return PoseResult.model_validate_json(p.read_text())
        except (OSError, ValueError) as exc:
            log.warning("precomputed pose %s unreadable: %s", p.name, type(exc).__name__)
        return None

    def submit(self, target_id: str, smi: str) -> PoseResult:
        with self.lock:
            self._purge()
            for j in self.jobs.values():  # one job per (target, molecule) while it is alive and not failed
                if j.target_id == target_id and j.smiles == smi and j.status != "failed":
                    return j.model_copy()
            job_id = secrets.token_urlsafe(9)
            pre = self._precomputed(target_id, smi)
            if pre is not None:
                job = pre.model_copy(update={"job_id": job_id, "status": "done", "progress": 1.0,
                                             "target_id": target_id, "smiles": smi, "source": "precomputed"})
            else:
                if sum(j.status in ("queued", "running") for j in self.jobs.values()) >= MAX_ACTIVE:
                    raise Rejected(429, "Pose queue is full, try again shortly", 30)
                job = PoseResult(job_id=job_id, status="queued", target_id=target_id, smiles=smi,
                                 source="dynamicbind", note="queued for DynamicBind")
            self.jobs[job_id], self.stamp[job_id] = job, time.monotonic()
        if pre is None:
            self.pool.submit(self._execute, job_id, target_id, smi)
        return job.model_copy()

    def _apo(self, target_id: str) -> Optional[str]:
        try:
            p = self.svc.structure_path(target_id)
            return p.read_text() if p else None
        except OSError:
            return None

    def _execute(self, job_id: str, target_id: str, smi: str) -> None:
        self._set(job_id, status="running", note="submitted to DynamicBind")
        try:
            remote = self.client.dock(target_id, smi)
            deadline = time.monotonic() + self.timeout
            while True:
                d = self.client.job(remote)
                st = d.get("status")
                if st == "done":
                    fields = {k: d.get(k) for k in ("ligand_sdf", "protein_pdb", "apo_pdb", "pose_confidence",
                                                    "affinity") if d.get(k) is not None}
                    fields.setdefault("apo_pdb", self._apo(target_id))
                    self._set(job_id, status="done", progress=1.0, source="dynamicbind",
                              note=d.get("note") or "DynamicBind prediction", **fields)
                    return
                if st == "failed":
                    self._set(job_id, status="failed", note=f"DynamicBind failed: {d.get('note') or 'no detail'}")
                    return
                if time.monotonic() > deadline:
                    self._set(job_id, status="failed", note=f"DynamicBind did not finish within {self.timeout:g}s")
                    return
                self._set(job_id, status="running", progress=float(d.get("progress") or 0.0),
                          note=d.get("note") or "DynamicBind running")
                time.sleep(self.poll)
        except PoseServiceError as exc:
            self._set(job_id, status="failed", note=str(exc))
        except Exception as exc:  # never kill the worker
            log.exception("pose job failed")
            self._set(job_id, status="failed", note=f"pose job error: {type(exc).__name__}")


def register(app: FastAPI, svc, mgr) -> None:
    def _f(name, d):
        try:
            return float(os.environ.get(name, d))
        except ValueError:
            return d

    pm = PoseManager(svc, poll=_f("DYNAMICBIND_POLL", 2.0), timeout=_f("DYNAMICBIND_TIMEOUT", 900.0))
    app.state.poses = pm

    @app.post("/api/poses", response_model=PoseResult)
    def create_pose(req: PoseRequest, request: Request):
        wait = mgr.rate.hit(f"pose:{request.state.sid}", 10)
        if wait:
            raise Rejected(429, "Too many pose requests", wait)
        if svc.get_target(req.target_id) is None:
            raise HTTPException(422, f"unknown target_id {req.target_id!r}")
        try:
            smi = molio.canonical(molio.parse_molecule(req.smiles))
        except molio.InvalidMolecule as exc:
            raise HTTPException(422, str(exc))
        return pm.submit(req.target_id, smi)

    @app.get("/api/poses/{job_id}", response_model=PoseResult)
    def get_pose(job_id: str):
        j = pm.get(job_id)
        if j is None:
            raise HTTPException(404, "unknown or expired pose job")
        return j
