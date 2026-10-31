"""SYNTHETIC scientist, for tests and the demo only. Not user data and not evidence about real chemists.

The synthetic scientist always picks, among the applicable edits, the one whose product has the highest TPSA,
then continues from that product. This gives a learnable, op-driven preference to check the machinery with.
"""
from __future__ import annotations

from typing import List

from rdkit import Chem
from rdkit.Chem import Descriptors

SEEDS = [
    "CC(=O)Nc1ccc(O)cc1", "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "c1ccc2c(c1)cc[nH]2", "CCN(CC)CCOC(=O)c1ccc(N)cc1",
    "COc1ccc2[nH]c(nc2c1)S(=O)Cc1ncc(C)c(OC)c1C", "Cc1ccccc1NC(=O)c1ccccc1", "O=C(c1ccccc1)c1ccc(Cl)cc1",
    "CC1=CC=C(C=C1)S(=O)(=O)N", "c1ccc(cc1)C(=O)Nc1ccncc1", "CCCCc1ccc(cc1)OC", "Clc1ccc(cc1)C(c1ccccc1)N1CCNCC1",
    "CN1CCC(CC1)c1c[nH]c2ccccc12", "O=C(O)c1ccccc1OC(C)=O", "CC(C)NCC(O)COc1cccc2ccccc12",
]


def _tpsa(smiles: str) -> float:
    m = Chem.MolFromSmiles(smiles)
    return Descriptors.TPSA(m) if m else float("-inf")


def make_transitions(registry, n: int, chain: int = 4) -> List[dict]:
    """n logged steps (dicts in the manual-export shape) from the always-raise-TPSA scientist."""
    out: List[dict] = []
    for seed in SEEDS:
        cur = seed
        for _ in range(chain):
            best = None
            for op, site in registry.applicable(cur):
                for child in op.apply(cur, site)[:1]:
                    t = _tpsa(child)
                    if best is None or t > best[0]:
                        best = (t, op.id, list(site), child)
            if best is None:
                break
            tp, eid, site, child = best
            out.append({"parent_smiles": cur, "edit_id": eid, "site": site, "child_smiles": child,
                        "reward": round(tp - _tpsa(cur), 3), "chosen_by": "user", "synthetic": True})
            cur = child
            if len(out) >= n:
                return out
    return out
