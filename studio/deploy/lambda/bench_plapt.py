#!/usr/bin/env python
"""PLAPT throughput on the box. usage: bench_plapt.py [target_id]   (run from ~/studio-app)
cold = embedding cache cleared (protein + molecules encoded); mol-cold = protein cached, molecules not;
warm = everything cached (only the ONNX head runs). SMILES are randomised-atom-order strings of the target's drugs
(distinct strings => distinct cache keys; throughput only, not a quality test)."""
import json, random, sys, tempfile, time
from pathlib import Path
from rdkit import Chem

sys.path.insert(0, str(Path.cwd()))
from binding_module.binding_affinity.plapt import Plapt

tid = sys.argv[1] if len(sys.argv) > 1 else "egfr"
info = json.loads(Path(f"studio/data/targets/{tid}.json").read_text())
seq = info["sequence"]
random.seed(0)
pool = set()
bases = [Chem.MolFromSmiles(d["smiles"]) for d in info["drugs"]]
while len(pool) < 256:
    pool.add(Chem.MolToSmiles(random.choice(bases), doRandom=True))
smiles = sorted(pool)

t = time.perf_counter()
cache_dir = tempfile.mkdtemp(prefix="plapt_cache_", dir="/home/ubuntu/metis-v1-storage")
p = Plapt(device="cuda", cache_dir=cache_dir)
print(f"model load: {time.perf_counter()-t:.1f}s  device={p.device}  ort_providers={p.prediction_module.session.get_providers()}")

def run(n, label):
    t = time.perf_counter()
    out = p.score_candidates(seq, smiles[:n])
    dt = time.perf_counter() - t
    print(f"{label:9s} n={n:4d}  {dt:7.3f}s  {n/dt:8.1f} mol/s   first={out[0]['neg_log10_affinity_M']:.3f}")
    return out

for n in (1, 32, 256):
    p.cache.clear(); run(n, "cold")
    p.cache.clear(); p.encode_proteins([seq], 1); run(n, "mol-cold")
    run(n, "warm")
a = p.score_candidates(seq, smiles[:32]); b = p.score_candidates(seq, smiles[:32])
print("deterministic:", all(x["neg_log10_affinity_M"] == y["neg_log10_affinity_M"] for x, y in zip(a, b)))
