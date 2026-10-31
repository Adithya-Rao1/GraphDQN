"""DynamicBind job service (runs on the Lambda GPU box, port 8100, concurrency 1).

    POST /dock      {target_id, smiles, force?: bool, samples?: int}  -> {job_id}
    GET  /jobs/{id}                                                    -> studio.schema.PoseResult
    GET  /health

Shells out to the DynamicBind conda envs (micromamba, see ops_log.md). Everything returned is a *predicted*
pose / affinity from DynamicBind; `source` says whether it was computed now ("dynamicbind") or read from the
precomputed cache ("precomputed"). Two disclosed deviations from the upstream README pipeline:

* ESM-2 650M embeddings come from the Hugging Face copy of the same model (hf_esm_extract.py), because the
  upstream weight host is not reachable from the box.
* `--samples_per_complex` defaults to 3 (upstream 10) to keep a live dock at tens of seconds, not minutes.

Run:  ~/studio-venv/bin/uvicorn dynamicbind_service:app --host 127.0.0.1 --port 8100   (cwd = this directory)
"""
from __future__ import annotations

import csv
import hashlib
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Callable, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rdkit import Chem

HOME = Path(os.environ.get("HOME", "/home/ubuntu"))
STUDIO_APP = Path(os.environ.get("STUDIO_APP", HOME / "studio-app"))
DB_HOME = Path(os.environ.get("DB_HOME", HOME / "DynamicBind"))
MAMBA_PREFIX = Path(os.environ.get("MAMBA_ROOT_PREFIX", HOME / "micromamba"))
DB_ENV = Path(os.environ.get("DB_ENV", MAMBA_PREFIX / "envs" / "dynamicbind"))
RELAX_ENV = Path(os.environ.get("DB_RELAX_ENV", MAMBA_PREFIX / "envs" / "relax"))
STORAGE = Path(os.environ.get("DB_STORAGE", "/home/ubuntu/metis-v1-storage/dynamicbind"))
CKPT_DIR = Path(os.environ.get("DB_CKPT_DIR", STORAGE / "workdir" / "big_score_model_sanyueqi_with_time"))
CKPT = os.environ.get("DB_CKPT", "pro_ema_inference_epoch138_model.pt")
RUNS = Path(os.environ.get("DB_RUNS", STORAGE / "runs"))
HF_EXTRACT = Path(os.environ.get("DB_HF_EXTRACT", HOME / "hf_esm_extract.py"))
SAMPLES = int(os.environ.get("DB_SAMPLES", "3"))
STEPS = int(os.environ.get("DB_STEPS", "20"))
RELAX_WORKERS = int(os.environ.get("DB_RELAX_WORKERS", "8"))
DATA = STUDIO_APP / "studio" / "data"
PRECOMPUTED = Path(os.environ.get("DB_PRECOMPUTED", DATA / "precomputed_poses"))

# progress milestones
P_PREP, P_INFER_END, P_RELAX_END = 0.10, 0.80, 0.95

if str(STUDIO_APP) not in sys.path:
    sys.path.insert(0, str(STUDIO_APP))
from studio.schema import PoseResult  # noqa: E402  (repo contract; PoseResult is the wire format)


# --------------------------------------------------------------------------- helpers
def canonical_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"unparseable SMILES: {smiles!r}")
    return Chem.MolToSmiles(mol)


def smiles_key(smiles: str) -> str:
    """sha1 of the RDKit canonical SMILES: file stem of precomputed pose JSONs."""
    return hashlib.sha1(canonical_smiles(smiles).encode()).hexdigest()


def precomputed_path(target_id: str, smiles: str) -> Path:
    return PRECOMPUTED / target_id / f"{smiles_key(smiles)}.json"


def _env() -> Dict[str, str]:
    env = dict(os.environ)
    env.update(
        LD_LIBRARY_PATH=f"{DB_ENV}/lib",
        PATH=f"{RELAX_ENV}/bin:{HOME}/bin:" + env.get("PATH", ""),
        MAMBA_ROOT_PREFIX=str(MAMBA_PREFIX),
        CONDA_PREFIX=str(RELAX_ENV),  # openff-toolkit asks `conda list` for the AmberTools version
        CUDA_VISIBLE_DEVICES=env.get("CUDA_VISIBLE_DEVICES", "0"),
        PYTHONNOUSERSITE="1",
    )
    env.setdefault("HF_HOME", str(STORAGE.parent / "hf"))
    return env


_TQDM = re.compile(rb"(\d+)/(\d+) \[")


def _run(cmd: List[str], cwd: Path, log, env: Dict[str, str], on_tqdm: Optional[Callable[[float], None]] = None,
         relax_env: bool = False) -> int:
    e = dict(env)
    if relax_env:
        e["LD_LIBRARY_PATH"] = f"{RELAX_ENV}/lib"
    log.write(f"\n$ {' '.join(cmd)}\n".encode())
    log.flush()
    p = subprocess.Popen(cmd, cwd=cwd, env=e, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    buf = b""
    assert p.stdout is not None
    while True:
        chunk = p.stdout.read1(4096)  # type: ignore[attr-defined]
        if not chunk:
            break
        log.write(chunk)
        buf += chunk
        parts = re.split(rb"[\r\n]", buf)
        buf = parts.pop()
        if on_tqdm:
            for part in parts:
                m = _TQDM.search(part)
                if m and int(m.group(2)) > 0:
                    on_tqdm(int(m.group(1)) / int(m.group(2)))
    log.flush()
    return p.wait()


_POSE = re.compile(r"^rank(\d+)_ligand_lddt([\d.]+)_affinity(-?[\d.]+?)(_relaxed)?\.sdf$")


def _best_pose(complex_dir: Path) -> Optional[dict]:
    """rank1 (best DynamicBind pose score) ligand SDF + matching receptor PDB, relaxed if available."""
    best = None
    for f in complex_dir.glob("rank*_ligand_lddt*.sdf"):
        m = _POSE.match(f.name)
        if not m:
            continue
        rank, lddt, aff, relaxed = int(m.group(1)), float(m.group(2)), float(m.group(3)), bool(m.group(4))
        if rank != 1:
            continue
        rec = complex_dir / f.name.replace("_ligand_", "_receptor_").replace(".sdf", ".pdb")
        if not rec.exists():
            continue
        cand = dict(ligand=f, receptor=rec, lddt=lddt, affinity=aff, relaxed=relaxed)
        if best is None or (cand["relaxed"] and not best["relaxed"]):
            best = cand
    return best


# --------------------------------------------------------------------------- pipeline
def dock_batch(target_id: str, smiles_list: List[str], job_id: str, progress: Callable[[float, str], None],
               samples: int = SAMPLES, steps: int = STEPS) -> List[PoseResult]:
    """Dock several ligands against one target in a single DynamicBind run. Returns one PoseResult per input
    (status 'done' or 'failed'). Model + protein prep load once for the whole batch."""
    struct = DATA / "structures" / f"{target_id}.pdb"
    if not struct.is_file():
        raise FileNotFoundError(f"no structure for target {target_id}: {struct}")
    job_dir = RUNS / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "data").mkdir(exist_ok=True)
    env = _env()
    py, rpy = str(DB_ENV / "bin" / "python"), str(RELAX_ENV / "bin" / "python")
    log = open(job_dir / "pipeline.log", "ab")
    t0 = time.time()
    cache = RUNS / "_target_cache" / target_id
    cleaned = job_dir / "data" / "cleaned_input_proteinFile.pdb"

    progress(0.01, "preparing protein")
    if (cache / "ok").exists() and (cache / "struct_sha1").read_text() == hashlib.sha1(struct.read_bytes()).hexdigest():
        shutil.copy(cache / "cleaned_input_proteinFile.pdb", cleaned)
        shutil.copytree(cache / "esm2_output", job_dir / "data" / "esm2_output", dirs_exist_ok=True)
        need_esm = False
    else:
        shutil.copy(struct, job_dir / "data" / "input.pdb")
        if _run([rpy, str(DB_HOME / "clean_pdb.py"), "data/input.pdb", "data/cleaned_input_proteinFile.pdb"], job_dir, log, env, relax_env=True):
            raise RuntimeError("clean_pdb failed")
        need_esm = True

    with open(job_dir / "data" / "ligandFile_with_protein_path.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ligand", "protein_path"])
        for smi in smiles_list:
            w.writerow([smi, "data/cleaned_input_proteinFile.pdb"])
    if need_esm:
        if _run([py, str(DB_HOME / "datasets" / "esm_embedding_preparation.py"), "--protein_ligand_csv",
                 "data/ligandFile_with_protein_path.csv", "--out_file", "data/prepared_for_esm_job.fasta"], job_dir, log, env):
            raise RuntimeError("esm_embedding_preparation failed")
        progress(0.04, "ESM-2 embedding (HF copy of esm2_t33_650M)")
        if _run([py, str(HF_EXTRACT), "data/prepared_for_esm_job.fasta", "data/esm2_output"], job_dir, log, env):
            raise RuntimeError("ESM embedding failed")
        cache.mkdir(parents=True, exist_ok=True)
        shutil.copy(cleaned, cache / "cleaned_input_proteinFile.pdb")
        shutil.copytree(job_dir / "data" / "esm2_output", cache / "esm2_output", dirs_exist_ok=True)
        (cache / "struct_sha1").write_text(hashlib.sha1(struct.read_bytes()).hexdigest())
        (cache / "ok").write_text("1")

    progress(P_PREP, "DynamicBind pose sampling")
    infer_cmd = [py, str(DB_HOME / "inference.py"), "--seed", "42", "--ckpt", CKPT, "--protein_dynamic",
                 "--save_visualisation", "--model_dir", str(CKPT_DIR),
                 "--protein_ligand_csv", "data/ligandFile_with_protein_path.csv",
                 "--esm_embeddings_path", "data/esm2_output", "--out_dir", "results/job",
                 "--inference_steps", str(steps), "--samples_per_complex", str(samples),
                 "--savings_per_complex", "3", "--batch_size", "5", "--actual_steps", str(steps),
                 "--no_final_step_noise"]
    seen = {"max": 0.0}

    def on_tqdm(frac: float) -> None:
        seen["max"] = max(seen["max"], frac)
        progress(P_PREP + (P_INFER_END - P_PREP) * seen["max"], "DynamicBind pose sampling")

    if _run(infer_cmd, job_dir, log, env, on_tqdm=on_tqdm):
        raise RuntimeError("DynamicBind inference failed (see pipeline.log)")

    progress(P_INFER_END, "relaxing final poses (OpenMM)")
    relax_ok = _run([rpy, str(DB_HOME / "relax_final.py"), "--results_path", "results/job",
                     "--samples_per_complex", str(samples), "--num_workers", str(RELAX_WORKERS)],
                    job_dir, log, env, relax_env=True) == 0
    progress(P_RELAX_END, "collecting results")

    aff_csv = job_dir / "results" / "job" / "affinity_prediction.csv"
    csv_aff: Dict[str, float] = {}
    if aff_csv.exists():
        with open(aff_csv, newline="") as fh:
            csv_aff = {r["name"]: float(r["affinity"]) for r in csv.DictReader(fh)}
    apo = cleaned.read_text()
    out: List[PoseResult] = []
    wall = time.time() - t0
    for i, smi in enumerate(smiles_list):
        cdir = job_dir / "results" / "job" / f"index{i}_idx_{i}"
        best = _best_pose(cdir) if cdir.is_dir() else None
        if best is None:
            out.append(PoseResult(job_id=job_id, status="failed", target_id=target_id, smiles=smi, progress=1.0,
                                  source="dynamicbind", note="DynamicBind produced no pose for this ligand"))
            continue
        note = (f"DynamicBind v2 checkpoint (pro_ema_inference_epoch138); {samples} sampled poses, {steps} steps; "
                f"rank-1 by predicted pose score; {'OpenMM-relaxed' if best['relaxed'] else 'NOT relaxed (relax step failed)'}; "
                f"affinity is DynamicBind's predicted pAffinity (-log10 M), pose affinity {best['affinity']:.2f}; "
                f"apo_pdb is the cleaned input structure; ESM-2 embeddings from the HF copy of the same model; "
                f"batch of {len(smiles_list)} ligand(s) took {wall:.0f}s.")
        out.append(PoseResult(
            job_id=job_id, status="done", target_id=target_id, smiles=smi, progress=1.0,
            ligand_sdf=best["ligand"].read_text(), protein_pdb=best["receptor"].read_text(), apo_pdb=apo,
            pose_confidence=best["lddt"], affinity=csv_aff.get(f"idx_{i}", best["affinity"]),
            source="dynamicbind", note=note))
    log.close()
    return out


# --------------------------------------------------------------------------- service
class DockRequest(BaseModel):
    target_id: str
    smiles: str
    force: bool = False  # ignore the precomputed cache
    samples: Optional[int] = None


app = FastAPI(title="Metis DynamicBind service")
_jobs: Dict[str, PoseResult] = {}
_lock = threading.Lock()
_q: "queue.Queue[tuple]" = queue.Queue()


def _set(job_id: str, **kw) -> None:
    with _lock:
        _jobs[job_id] = _jobs[job_id].model_copy(update=kw)


def _worker() -> None:
    while True:
        job_id, req = _q.get()
        _set(job_id, status="running", progress=0.01)
        try:
            def prog(p: float, note: str) -> None:
                _set(job_id, progress=round(min(p, 0.99), 3), note=note)

            res = dock_batch(req.target_id, [req.smiles], job_id, prog, samples=req.samples or SAMPLES)[0]
            with _lock:
                _jobs[job_id] = res
            (RUNS / job_id / "result.json").write_text(res.model_dump_json())
        except Exception as exc:  # noqa: BLE001  report failure honestly to the client
            _set(job_id, status="failed", progress=1.0, note=f"{type(exc).__name__}: {exc}"[:500])


threading.Thread(target=_worker, daemon=True).start()


def _targets() -> List[str]:
    d = DATA / "targets"
    return sorted(p.stem for p in d.glob("*.json") if p.stem != "index") if d.is_dir() else []


@app.get("/health")
def health() -> dict:
    gpu = None
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,utilization.gpu", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    return {"ok": True, "queue": _q.qsize(), "targets": _targets(), "gpu": gpu, "samples": SAMPLES, "steps": STEPS,
            "checkpoint": CKPT, "precomputed_dir": str(PRECOMPUTED)}


@app.post("/dock")
def dock(req: DockRequest) -> dict:
    if req.target_id not in _targets() or not (DATA / "structures" / f"{req.target_id}.pdb").is_file():
        raise HTTPException(404, f"unknown target {req.target_id!r}")
    try:
        canonical_smiles(req.smiles)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    job_id = uuid.uuid4().hex[:12]
    pre = None if req.force else precomputed_path(req.target_id, req.smiles)
    if pre is not None and pre.is_file():
        res = PoseResult.model_validate_json(pre.read_text()).model_copy(
            update=dict(job_id=job_id, source="precomputed", status="done", progress=1.0))
        with _lock:
            _jobs[job_id] = res
        return {"job_id": job_id}
    with _lock:
        _jobs[job_id] = PoseResult(job_id=job_id, status="queued", target_id=req.target_id, smiles=req.smiles,
                                   note=f"queued ({_q.qsize()} ahead)")
    _q.put((job_id, req))
    return {"job_id": job_id}


@app.get("/jobs/{job_id}", response_model=PoseResult)
def get_job(job_id: str) -> PoseResult:
    with _lock:
        if job_id in _jobs:
            return _jobs[job_id]
    f = RUNS / job_id / "result.json"
    if re.fullmatch(r"[0-9a-f]{12}", job_id) and f.is_file():
        return PoseResult.model_validate_json(f.read_text())
    raise HTTPException(404, "unknown job")
