"""Regenerate the UI-manual dev fixtures from the real studio components (RDKit only, no network).

    .venv/bin/python studio/web/fixtures/build_manual_fixtures.py

Writes (next to this file):
  run_rdkit_local.json   a real GA run recorded locally (UniformPolicy, Morgan map), events stamped source="live"
  run_binding_dev.json   the same run with SYNTHETIC `extras.plapt_affinity` values, stamped source="mock" (UI test data only)
  manual_fixture.json    real ManualPreview responses for a small tree of edits from the seed (dev fetch shim)
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from rdkit import Chem  # noqa: E402

from studio.embed import get_embedder  # noqa: E402
from studio.engine import Engine, geometry  # noqa: E402
from studio.explain.explainer import RDKitExplainer  # noqa: E402
from studio.policy.policy import UniformPolicy  # noqa: E402
from studio.registry import get_registry  # noqa: E402
from studio.schema import EditRecord, ManualPreview, RunConfig  # noqa: E402
from studio.scoring import RDKitScorer  # noqa: E402

HERE = Path(__file__).parent
SEED = "CCOC(=O)c1ccc(cc1)[N+](=O)[O-]"
reg, scorer, expl = get_registry(), RDKitScorer(), RDKitExplainer()


def record_run():
    cfg = RunConfig(seed_smiles=SEED, population_size=10, generations=6, rng_seed=7, policy_guidance=0.0, use_crossover=True)
    eng = Engine(reg, scorer, expl, get_embedder("morgan"), UniformPolicy())
    evs = [e.model_dump(mode="json") for e in eng.run("fixture-run", cfg)]
    (HERE / "run_rdkit_local.json").write_text(json.dumps(evs))
    print("run events", len(evs), {t: sum(1 for e in evs if e["type"] == t) for t in {e["type"] for e in evs}})
    # synthetic binding values for UI tests: a smooth function of QED so the charts have a trend. NOT a prediction.
    syn = json.loads(json.dumps(evs))
    for e in syn:
        e["payload"].pop("source", None)
        if e["type"] == "run_start":
            e["payload"]["source"] = "mock"
        sc = []
        if e["type"] == "run_start":
            sc = [e["payload"]["seed_score"]]
        elif e["type"] == "candidate":
            sc = [e["payload"]["score"]]
        elif e["type"] == "run_end":
            sc = [e["payload"]["seed_score"], e["payload"]["final_score"]]
        for s in sc:
            s["extras"] = {"plapt_affinity": round(5.2 + 2.4 * s["descriptors"]["qed"] + 0.15 * s["descriptors"]["logp"], 3)}
    (HERE / "run_binding_dev.json").write_text(json.dumps(syn))
    return evs


def preview(parent_smiles, parent_mol, op, site):
    prods = op.apply(parent_smiles, site)
    if not prods:
        return None
    child = prods[0]
    block, cmol = geometry.embed_3d(child, parent_mol)
    amap = geometry.atom_map(parent_mol, cmol)
    ps, cs = scorer.score(parent_smiles, RunConfig(seed_smiles=SEED)), scorer.score(child, RunConfig(seed_smiles=SEED))
    ex = expl.explain(parent_smiles, child, EditRecord(edit_id=op.id, name=op.name, category=op.category, site=list(site), chosen_by="user", parents=[parent_smiles]), ps, cs, op)
    pv = ManualPreview(ok=True, child_smiles=child, molblock=block, atom_map=amap, score=cs, explanation=ex, sane=True)
    return pv.model_dump(mode="json"), cmol


def applicable_list(smiles):
    out = []
    for op, site in reg.applicable(smiles):
        out.append({"edit_id": op.id, "name": op.name, "category": op.category, "site": list(site), "rationale": op.rationale, "tags": list(op.tags)})
    return out


def build_manual(depth=2, branch=4, per_mol=60):
    seed_block, seed_mol = geometry.embed_3d(SEED)
    fx = {"seed": {"smiles": SEED, "molblock": seed_block, "score": scorer.score(SEED, RunConfig(seed_smiles=SEED)).model_dump(mode="json")},
          "applicable": {}, "previews": {}}
    frontier = [(SEED, seed_mol)]
    for d in range(depth):
        nxt = []
        for smi, mol in frontier:
            lst, made = [], 0
            for a in applicable_list(smi):
                if len(lst) >= per_mol:
                    break
                r = preview(smi, mol, reg.get(a["edit_id"]), a["site"])
                if r is None:  # the registry lists the site but the edit yields no valid product: leave it out of the fixture
                    continue
                lst.append(a)
                pv, cmol = r
                fx["previews"][f'{smi}|{a["edit_id"]}|{",".join(map(str, a["site"]))}'] = pv
                if d + 1 < depth and made < branch and pv["child_smiles"] not in {s for s, _ in nxt}:
                    nxt.append((pv["child_smiles"], cmol))
                    made += 1
            fx["applicable"][smi] = lst
        frontier = nxt
    (HERE / "manual_fixture.json").write_text(json.dumps(fx))
    print("manual: molecules", len(fx["applicable"]), "previews", len(fx["previews"]), "bytes", (HERE / "manual_fixture.json").stat().st_size)


if __name__ == "__main__":
    record_run()
    build_manual()
