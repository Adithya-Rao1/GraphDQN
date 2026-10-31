"""Chemical-plausibility filter for generated children (independent of the descriptor scorer).

``check_child(parent_smiles, child_smiles) -> (ok, reason)``. Only RDKit + the SMARTS in patterns.yaml; the
binding stage reuses it, so it must not import ``studio.scoring``. ``parent_smiles`` may be None/"" (then charge
and pattern baselines are "neutral, no motifs"). Reasons are short machine strings; pattern rejections read
``bad_pattern:<name>``. Typical cost < 2 ms.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

PATTERNS_PATH = Path(__file__).with_name("patterns.yaml")

_BAD: List[Tuple[str, Chem.Mol]] = []
for _e in yaml.safe_load(PATTERNS_PATH.read_text())["patterns"]:
    _q = Chem.MolFromSmarts(_e["smarts"])
    if _q is None:
        raise ValueError(f"bad SMARTS in patterns.yaml: {_e['name']}")
    _BAD.append((_e["name"], _q))

# neutral zwitterion groups whose charged atoms are accepted (indices of the charged atoms are collected)
_OK_CHARGED = [Chem.MolFromSmarts(s) for s in (
    "[N+](=O)[O-]",              # nitro
    "[#7+][O-]",                 # N-oxide (incl. aromatic)
    "[S+][O-]",                  # sulfoxide-type
    "[N-]=[N+]=[N,N-]",          # azide (either resonance form)
    "[N+]=[N-]",                 # diazo-like azide fragment
    "[P+][O-]",                  # phosphine oxide / phosphonate drawn charge-separated
)]
_AMMONIUM = Chem.MolFromSmarts("[NX4+;!$([N+]~[O-])]")
_CARBOXYLATE = Chem.MolFromSmarts("[O-][CX3]=O")


@lru_cache(maxsize=4096)
def _parse(smiles: str) -> Optional[Chem.Mol]:
    return Chem.MolFromSmiles(smiles) if smiles else None


def _pattern_counts(mol: Chem.Mol) -> Dict[str, int]:
    out = {}
    for name, q in _BAD:
        n = len(mol.GetSubstructMatches(q))
        if n:
            out[name] = n
    return out


def _unrecognised_charged(mol: Chem.Mol) -> int:
    charged = {a.GetIdx() for a in mol.GetAtoms() if a.GetFormalCharge()}
    if not charged:
        return 0
    for q in _OK_CHARGED:
        for m in mol.GetSubstructMatches(q):
            charged.difference_update(m)
    amm = [m[0] for m in mol.GetSubstructMatches(_AMMONIUM) if m[0] in charged]
    car = [m[0] for m in mol.GetSubstructMatches(_CARBOXYLATE) if m[0] in charged]
    for idx in amm[: len(car)] + car[: len(amm)]:  # ammonium/carboxylate zwitterion pairs
        charged.discard(idx)
    return len(charged)


@lru_cache(maxsize=2048)
def _parent_profile(parent: str) -> Optional[Tuple[int, int, Dict[str, int]]]:
    mol = _parse(parent)
    if mol is None:
        return None
    return Chem.GetFormalCharge(mol), _unrecognised_charged(mol), _pattern_counts(mol)


def check_child(parent_smiles: Optional[str], child_smiles: str) -> Tuple[bool, str]:
    """(True, "ok") if the child is chemically plausible relative to its parent, else (False, reason)."""
    if not child_smiles:
        return False, "sanitize_failed"
    if "." in child_smiles:
        return False, "fragments"
    mol = _parse(child_smiles)
    if mol is None:
        return False, "sanitize_failed"
    if len(Chem.GetMolFrags(mol)) > 1:
        return False, "fragments"
    if any(a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
        return False, "radical"
    prof = _parent_profile(parent_smiles) if parent_smiles else None
    p_charge, p_unrec, p_pat = prof if prof else (0, 0, {})
    if Chem.GetFormalCharge(mol) != p_charge:
        return False, "net_charge_changed"
    if _unrecognised_charged(mol) > p_unrec:
        return False, "unrecognised_charge"
    for name, n in _pattern_counts(mol).items():
        if n > p_pat.get(name, 0):
            return False, f"bad_pattern:{name}"
    return True, "ok"
