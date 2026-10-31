"""Build the UI fixtures (served with ?fixtures=1). EVERYTHING here is synthetic and labelled so:

* fixture1.pdb: a made-up 6-helix bundle (backbone + a few side-chain atoms), NOT a real protein
* pose_result.json: a PoseResult-shaped file; coordinates are a made-up placement, numbers are placeholders
* binding_run.json: a real GA run of studio.engine (real edits, real RDKit descriptors) with a TOY
  `extras.plapt_affinity` (a function of RDKit descriptors; NOT a PLAPT prediction), Tanimoto computed by RDKit.

Run:  python studio/web/fixtures/build_fixtures.py   (needs the repo venv: rdkit, torch, deap)
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))  # repo root

AA = "LEAKVGSDTIRNPFQHYMWC"
THREE = {"L": "LEU", "E": "GLU", "A": "ALA", "K": "LYS", "V": "VAL", "G": "GLY", "S": "SER", "D": "ASP", "T": "THR", "I": "ILE",
         "R": "ARG", "N": "ASN", "P": "PRO", "F": "PHE", "Q": "GLN", "H": "HIS", "Y": "TYR", "M": "MET", "W": "TRP", "C": "CYS"}
NRES = 22
NHEL = 6
RING_R = 10.5
SEED_SMILES = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"  # ibuprofen, used only as a fixture drug


def helix_atoms(h: int):
    """Idealised alpha helix along z, on a ring of radius RING_R; helices alternate direction."""
    ang0 = 2 * math.pi * h / NHEL
    cx, cy = RING_R * math.cos(ang0), RING_R * math.sin(ang0)
    up = h % 2 == 0
    out = []
    for i in range(NRES):
        t = math.radians(100 * i) + ang0 * 0.5
        z = (i * 1.5) * (1 if up else -1) + (0 if up else NRES * 1.5)
        def pt(r, dt, dz):
            return np.array([cx + r * math.cos(t + dt), cy + r * math.sin(t + dt), z + dz * (1 if up else -1)])
        ca = pt(2.3, 0, 0)
        out.append({"CA": ca, "N": pt(2.0, math.radians(-28), -0.9), "C": pt(2.0, math.radians(26), 0.7), "O": pt(2.9, math.radians(30), 1.0),
                    "CB": pt(3.7, math.radians(5), 0.4), "centre": np.array([cx, cy, z])})
    return out


def build_pdb(displace: bool) -> tuple[str, list[int], str, np.ndarray]:
    lines = ["REMARK  FIXTURE: synthetic 6-helix bundle for UI development. NOT a real protein, not a structure of anything."]
    serial, resi, seq, pocket = 1, 1, "", []
    centre = np.array([0.0, 0.0, NRES * 0.75])
    for h in range(NHEL):
        for i, res in enumerate(helix_atoms(h)):
            aa = AA[(h * 7 + i * 3) % len(AA)]
            seq += aa
            z_mid = abs(res["CA"][2] - centre[2]) < 9
            in_pocket = z_mid and (i % 2 == 0)
            if in_pocket:
                pocket.append(resi)
            names = ["N", "CA", "C", "O", "CB"]
            pos = {k: res[k].copy() for k in names}
            if in_pocket:  # a short side chain pointing at the pocket centre
                d = centre - pos["CB"]; d[2] *= 0.2; d /= np.linalg.norm(d)
                pos["CG"] = pos["CB"] + 1.5 * d
                pos["CD"] = pos["CB"] + 2.9 * d
                names += ["CG", "CD"]
                if displace:  # fixture "induced fit": pocket residues swing away from the centre, smoothly
                    for k in pos:
                        r = pos[k] - centre; r[2] = 0
                        r /= max(np.linalg.norm(r), 1e-6)
                        pos[k] = pos[k] + r * (1.6 + 0.5 * math.sin(resi))
            elif displace and z_mid:
                for k in pos:
                    r = pos[k] - centre; r[2] = 0
                    r /= max(np.linalg.norm(r), 1e-6)
                    pos[k] = pos[k] + r * 0.5
            for n in names:
                x, y, z = pos[n]
                lines.append(f"ATOM  {serial:5d} {n:^4s} {THREE[aa]} A{resi:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           {n[0]:>2s}")
                serial += 1
            resi += 1
        lines.append("TER")
    lines.append("END")
    return "\n".join(lines) + "\n", pocket, seq, centre


def main():
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs, Descriptors as D, rdMolTransforms

    apo, pocket, seq, centre = build_pdb(False)
    pred, _, _, _ = build_pdb(True)
    (HERE / "fixture1.pdb").write_text(apo)
    drugs = [
        {"name": "Ibuprofen (fixture drug)", "smiles": SEED_SMILES, "chembl_id": None, "source": "FIXTURE"},
        {"name": "Aspirin (fixture drug)", "smiles": "CC(=O)Oc1ccccc1C(=O)O", "chembl_id": None, "source": "FIXTURE"},
    ]
    targets = [{
        "id": "fixture1", "name": "FIXTURE target (synthetic helix bundle)", "uniprot": None, "pdb_id": "FAKE", "chain": "A",
        "sequence": seq, "drugs": drugs, "structure_url": "fixtures/fixture1.pdb", "pocket_residues": pocket,
        "note": "FIXTURE: made-up protein for UI development. Not a real target, not a real binding site.",
    }]
    (HERE / "targets.json").write_text(json.dumps(targets, indent=1))

    # ---- pose fixture: ligand placed at the pocket centre in a made-up orientation
    def placed_sdf(smiles: str, seedn: int, shift=(0, 0, 0)) -> str:
        m = Chem.AddHs(Chem.MolFromSmiles(smiles))
        AllChem.EmbedMolecule(m, randomSeed=7)
        AllChem.MMFFOptimizeMolecule(m)
        m = Chem.RemoveHs(m)
        conf = m.GetConformer()
        pts = np.array([list(conf.GetAtomPosition(i)) for i in range(m.GetNumAtoms())])
        pts -= pts.mean(0)
        a = math.radians(40 + 25 * seedn)
        R = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
        pts = pts @ R.T + centre + np.array(shift)
        for i, p in enumerate(pts):
            conf.SetAtomPosition(i, p.tolist())
        return Chem.MolToMolBlock(m)

    pose = {"job_id": "fixture-pose", "status": "done", "target_id": "fixture1", "smiles": SEED_SMILES, "progress": 1.0,
            "ligand_sdf": placed_sdf(SEED_SMILES, 1), "protein_pdb": pred, "apo_pdb": apo, "pose_confidence": 0.71, "affinity": 6.3,
            "source": "fallback",
            "note": "FIXTURE: made-up placement and placeholder numbers. A real DynamicBind result replaces this."}
    (HERE / "pose_result.json").write_text(json.dumps(pose))

    # ---- binding-run fixture: real GA, toy affinity
    from studio.schema import RunConfig
    from studio.wiring import build_engine
    eng = build_engine()
    cfg = RunConfig(seed_smiles=SEED_SMILES, population_size=10, generations=7, rng_seed=3, policy_guidance=0.0, use_crossover=True,
                    stage="binding", target_id="fixture1", min_similarity_to_seed=0.0)
    seed_fp = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(SEED_SMILES), 2, 2048)

    def toy_affinity(smiles: str) -> float:
        m = Chem.MolFromSmiles(smiles)
        return round(4.5 + 0.9 * D.MolLogP(m) ** 0.5 * (1 if D.MolLogP(m) > 0 else 0) + 0.012 * D.MolWt(m) + 0.15 * Chem.rdMolDescriptors.CalcNumAromaticRings(m), 2)

    def decorate(score: dict, smiles: str):
        m = Chem.MolFromSmiles(smiles)
        fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048)
        score["extras"] = {"plapt_affinity": toy_affinity(smiles), "tanimoto_to_seed": round(DataStructs.TanimotoSimilarity(seed_fp, fp), 3), "fixture": 1.0}

    events = []
    for ev in eng.run("fixture-binding", cfg):
        d = json.loads(ev.model_dump_json())
        p = d["payload"]
        if d["type"] == "run_start":
            p["source"] = "mock"
            decorate(p["seed_score"], p["config"]["seed_smiles"])
        elif d["type"] == "candidate":
            decorate(p["score"], p["smiles"])
        elif d["type"] == "run_end":
            decorate(p["seed_score"], p["seed_smiles"]); decorate(p["final_score"], p["final_smiles"])
        events.append(d)
    (HERE / "binding_run.json").write_text(json.dumps(events))
    print("events", len(events), "pocket residues", len(pocket), "seq len", len(seq))


if __name__ == "__main__":
    main()
