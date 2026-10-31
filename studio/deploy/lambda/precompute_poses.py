#!/usr/bin/env python
"""Dock every known drug of every target with DynamicBind and write PoseResult JSONs the app serves instantly.

    studio/data/precomputed_poses/<target_id>/<sha1(canonical smiles)>.json   (+ _index.json per target)

Run on the box with the studio venv, while the dynamicbind service is NOT running a job (GPU is shared):
    cd ~/studio-app/studio/deploy/lambda && ~/studio-venv/bin/python precompute_poses.py [--targets egfr abl1] [--samples 3] [--steps 20]

All drugs of one target go through a single DynamicBind run (model/protein prep loaded once). Resumable:
existing files are skipped unless --force. source is set to "precomputed" so the UI can label it.
"""
from __future__ import annotations

import argparse
import json
import time
import uuid

import dynamicbind_service as svc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="*")
    ap.add_argument("--samples", type=int, default=svc.SAMPLES)
    ap.add_argument("--steps", type=int, default=svc.STEPS)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", default=str(svc.PRECOMPUTED))
    a = ap.parse_args()
    out_root = svc.Path(a.out)
    summary = []
    for tid in a.targets or svc._targets():
        info = json.loads((svc.DATA / "targets" / f"{tid}.json").read_text())
        todo = []
        for d in info["drugs"]:
            f = out_root / tid / f"{svc.smiles_key(d['smiles'])}.json"
            if f.exists() and not a.force:
                print(f"[{tid}] {d['name']}: exists, skipping")
                continue
            todo.append((d, f))
        if not todo:
            continue
        t0 = time.time()
        job_id = uuid.uuid4().hex[:12]
        print(f"[{tid}] docking {[d['name'] for d, _ in todo]} (job {job_id})", flush=True)
        try:
            results = svc.dock_batch(tid, [d["smiles"] for d, _ in todo], job_id,
                                     lambda p, n: print(f"   {p:.2f} {n}", flush=True) if n.startswith("relax") else None,
                                     samples=a.samples, steps=a.steps)
        except Exception as exc:  # noqa: BLE001
            print(f"[{tid}] FAILED: {exc}")
            summary.append({"target": tid, "error": str(exc)})
            continue
        wall = time.time() - t0
        (out_root / tid).mkdir(parents=True, exist_ok=True)
        for (d, f), r in zip(todo, results):
            row = {"target": tid, "drug": d["name"], "status": r.status, "pose_confidence": r.pose_confidence,
                   "affinity": r.affinity, "wall_s_batch": round(wall, 1)}
            summary.append(row)
            print("  ", row, flush=True)
            if r.status == "done":
                r = r.model_copy(update=dict(source="precomputed", job_id=f.stem))
                f.write_text(r.model_dump_json())
        (out_root / tid / "_index.json").write_text(json.dumps(
            {"target": tid, "drugs": [s for s in summary if s.get("target") == tid]}, indent=1))
    (out_root / "_summary.json").parent.mkdir(parents=True, exist_ok=True)
    (out_root / "_summary.json").write_text(json.dumps(summary, indent=1))
    print("done", len(summary), "entries; wall s per batch in summary")


if __name__ == "__main__":
    main()
