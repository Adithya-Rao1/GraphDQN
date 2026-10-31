"""Is PLAPT a usable fitness for this target, and can the GA game it?

    python -m studio.binding.gaming_check --target <id> [--backend plapt|fake] [--target-json file.json]

(a) known drugs vs ~200 synthetic drug-like decoys: AUROC of predicted pKd, and of heavy-atom count alone
    (a size-only baseline), plus a size-matched AUROC.
(b) a 10-generation pure-GA run from each known drug: best child's pKd, Tanimoto to the seed, MW, alerts, and
    the correlation of predicted pKd with heavy-atom count across everything the run scored.
All numbers are PREDICTED by the chosen backend. `--backend fake` only exercises the plumbing.
Decoys are random registry-edit walks from studio/policy/data/start_pool.smi, size-matched (heavy atoms) to the
known drugs but otherwise NOT curated or property-matched.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from rdkit import Chem
from scipy.stats import spearmanr

from studio.binding.backends import canon, make_backend
from studio.binding.scorer import BindingScorer, tanimoto
from studio.schema import RunConfig, ScoreResult

POOL = Path(__file__).resolve().parents[1] / "policy" / "data" / "start_pool.smi"


def load_target(target_id: str, target_json: Optional[str]):
    if target_json:
        d = json.loads(Path(target_json).read_text())
        d = d[target_id] if target_id in d else d
        return type("T", (), {"sequence": d["sequence"], "drugs": [type("D", (), x) for x in d["drugs"]],
                              "id": target_id})()
    from studio.data.loader import get_target
    return get_target(target_id)


def make_decoys(n: int, exclude: Sequence[str], seed: int = 0, ha_range=(8, 60)) -> List[str]:
    """BRICS recombinations of fragments of the start pool: valid, drug-like-ish, size-controlled, NOT curated."""
    from rdkit.Chem import BRICS
    rng = random.Random(seed)
    pool = [l.split()[0] for l in POOL.read_text().splitlines() if l.strip()]
    frags = sorted({f for s in pool for f in BRICS.BRICSDecompose(Chem.MolFromSmiles(s))})
    seen, out = {canon(s) for s in exclude}, []
    for _ in range(60):  # rounds of random fragment subsets
        sub = [Chem.MolFromSmiles(f) for f in rng.sample(frags, min(len(frags), 14))]
        sub = [m for m in sub if m is not None]
        gen = BRICS.BRICSBuild(sub, onlyCompleteMols=True, scrambleReagents=True, maxDepth=4)
        for k, m in enumerate(gen):
            if k >= 400 or len(out) >= n:
                break
            try:
                Chem.SanitizeMol(m)
                s = Chem.MolToSmiles(m)
            except Exception:
                continue
            if s in seen or not ha_range[0] <= m.GetNumHeavyAtoms() <= ha_range[1] or "." in s:
                continue
            seen.add(s)
            out.append(s)
        if len(out) >= n:
            break
    return out


def auroc(pos: Sequence[float], neg: Sequence[float]) -> float:
    pos, neg = np.asarray(pos), np.asarray(neg)
    if not len(pos) or not len(neg):
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(gt / (len(pos) * len(neg)))


class _Recording(BindingScorer):
    """BindingScorer that remembers everything it scored during a run."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.log: Dict[str, ScoreResult] = {}

    def score_many(self, smiles, config):
        res = super().score_many(smiles, config)
        for s, r in zip(smiles, res):
            self.log[canon(s)] = r
        return res


def part_a(backend, target, n_decoys: int) -> dict:
    seeds = [canon(d.smiles) for d in target.drugs]
    h_seed = [Chem.MolFromSmiles(s).GetNumHeavyAtoms() for s in seeds]
    # decoys are SIZE-MATCHED to the known drugs so a high AUROC cannot just be a size effect
    decoys = make_decoys(n_decoys, seeds, ha_range=(min(h_seed) - 3, max(h_seed) + 3))
    t0 = time.time()
    p_known = backend.predict(target.sequence, seeds)
    p_decoy = backend.predict(target.sequence, decoys)
    ha = lambda ss: [Chem.MolFromSmiles(s).GetNumHeavyAtoms() for s in ss]
    h_known, h_decoy = ha(seeds), ha(decoys)
    lo, hi = min(h_known) - 5, max(h_known) + 5
    keep = [i for i, h in enumerate(h_decoy) if lo <= h <= hi]
    mean_rank = float(np.mean([(np.asarray(p_decoy) > p).sum() + 1 for p in p_known]))
    return {
        "n_known": len(seeds), "n_decoys": len(decoys), "seconds": round(time.time() - t0, 1),
        "known_pkd": {d.name: round(p, 2) for d, p in zip(target.drugs, p_known)},
        "decoy_pkd_mean_std": [round(float(np.mean(p_decoy)), 2), round(float(np.std(p_decoy)), 2)],
        "auroc_pkd": round(auroc(p_known, p_decoy), 3),
        "auroc_heavy_atoms_only": round(auroc(h_known, h_decoy), 3),
        "n_size_matched_decoys": len(keep),
        "auroc_pkd_size_matched": round(auroc(p_known, [p_decoy[i] for i in keep]), 3),
        "mean_rank_of_known_among_decoys(1=best)": round(mean_rank, 1),
        "spearman_pkd_vs_heavy_atoms_decoys": round(float(spearmanr(p_decoy, h_decoy)[0]), 3),
    }


def part_b(backend, target, drug, generations: int, pop: int) -> dict:
    from studio.wiring import build_engine
    scorer = _Recording(backend, lambda _t: target)
    engine = build_engine(scorer)
    seed = canon(drug.smiles)
    cfg = RunConfig(seed_smiles=seed, target_id=getattr(target, "id", "x"), generations=generations,
                    population_size=pop, policy_guidance=0.0, stage="binding", use_crossover=True)
    t0 = time.time()
    events = list(engine.run("gaming", cfg))
    errs = [e for e in events if e.type == "error"]
    if errs:
        return {"drug": drug.name, "error": str(errs[0].payload)}
    seed_r = scorer.log[seed]
    ok = {s: r for s, r in scorer.log.items() if s != seed and "plapt_pkd" in r.extras}
    guarded = sum(1 for s, r in scorer.log.items() if r.extras.get("guard_failed"))
    out = {"drug": drug.name, "seconds": round(time.time() - t0, 1), "scored": len(scorer.log),
           "guard_rejected": guarded, "seed_pkd": round(seed_r.extras["plapt_pkd"], 2)}
    if not ok:
        return {**out, "note": "no child passed the guards"}
    best_s = max(ok, key=lambda s: ok[s].extras["plapt_pkd"])
    b = ok[best_s]
    ha = [Chem.MolFromSmiles(s).GetNumHeavyAtoms() for s in ok]
    pk = [ok[s].extras["plapt_pkd"] for s in ok]
    rho = float(spearmanr(pk, ha)[0]) if len(ok) > 2 and len(set(ha)) > 1 else float("nan")
    d_ha = Chem.MolFromSmiles(best_s).GetNumHeavyAtoms() - Chem.MolFromSmiles(seed).GetNumHeavyAtoms()
    out.update({
        "best_child": best_s, "best_pkd": round(b.extras["plapt_pkd"], 2),
        "delta_pkd": round(b.extras["plapt_pkd"] - out["seed_pkd"], 2),
        "tanimoto_to_seed": round(tanimoto(seed, best_s), 2), "mw": round(b.descriptors.mw, 1),
        "seed_mw": round(seed_r.descriptors.mw, 1), "alerts": [a.name for a in b.alerts],
        "delta_heavy_atoms": d_ha, "n_children_passing": len(ok),
        "spearman_pkd_vs_heavy_atoms_in_run": round(rho, 3),
        "size_effect_suspected": bool(rho > 0.5 and d_ha > 0),
    })
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--backend", choices=["plapt", "fake"], default="plapt")
    ap.add_argument("--target-json", help="dev fallback: {sequence, drugs:[{name,smiles}]} when studio.data is absent")
    ap.add_argument("--decoys", type=int, default=200)
    ap.add_argument("--generations", type=int, default=10)
    ap.add_argument("--population", type=int, default=20)
    ap.add_argument("--max-drugs-ga", type=int, default=2)
    a = ap.parse_args(argv)
    target = load_target(a.target, a.target_json)
    backend = make_backend(a.backend)
    print(f"backend={backend.name} unit={backend.unit} (PREDICTED values)")
    res = {"backend": backend.name, "target": a.target, "a_known_vs_decoys": part_a(backend, target, a.decoys),
           "b_ga_runs": [part_b(backend, target, d, a.generations, a.population)
                         for d in target.drugs[: a.max_drugs_ga]]}
    print(json.dumps(res, indent=2, default=str))
    return res


if __name__ == "__main__":
    main()
