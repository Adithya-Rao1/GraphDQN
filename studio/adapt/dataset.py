"""Policy transitions from a scientist's manual edits, validated against RDKit and the registry.

Transition format (see studio/policy/README.md): (parent_smiles, edit_id, site, child_smiles, reward).
Input is a list of dicts (the manual-mode export, extra keys ignored) or 5-tuples.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

MIN_VALID_STEPS = 5
MAX_TRANSITIONS = 2000
MAX_SMILES_LEN = 2000


@dataclass(frozen=True)
class Transition:
    parent_smiles: str
    edit_id: str
    site: Tuple[int, ...]
    child_smiles: str
    reward: float


@dataclass
class Example:
    """A validated step: the parent's full applicable candidate list and the index the user chose."""
    transition: Transition
    candidates: List[tuple]  # [(EditOp, site)]
    target: int


@dataclass
class CleanReport:
    received: int
    valid: int
    dropped: int
    reasons: Dict[str, int]

    def as_dict(self) -> dict:
        return {"received": self.received, "valid": self.valid, "dropped": self.dropped, "reasons": dict(self.reasons)}


def _canon(smiles: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) and 0 < len(smiles) <= MAX_SMILES_LEN else None
    return Chem.MolToSmiles(mol) if mol is not None and mol.GetNumAtoms() else None


def _parse(raw: Any) -> Optional[Transition]:
    try:
        if isinstance(raw, dict):
            parent, edit_id, site, child, reward = (raw["parent_smiles"], raw["edit_id"], raw["site"],
                                                    raw["child_smiles"], raw["reward"])
        else:
            parent, edit_id, site, child, reward = raw
        if not (isinstance(parent, str) and isinstance(edit_id, str) and isinstance(child, str)):
            return None
        if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(reward):
            return None
        if isinstance(site, (str, bytes)) or any(isinstance(i, bool) or not isinstance(i, int) for i in site):
            return None
        return Transition(parent, edit_id, tuple(site), child, float(reward))
    except (KeyError, TypeError, ValueError):
        return None


def clean(raw: Sequence[Any], registry) -> Tuple[List[Example], CleanReport]:
    """Validate each step; drop what cannot be used as an imitation target and count why.

    A step is kept only if: parent and child parse in RDKit, the edit id is in the registry, the
    (edit, site) is applicable to the parent, applying it yields the logged child, the parent offers
    >= 2 candidates (otherwise there is nothing to rank) and it is not an exact duplicate.
    """
    reasons: Counter = Counter()
    out: List[Example] = []
    seen = set()
    cand_cache: Dict[str, List[tuple]] = {}
    for item in raw:
        t = _parse(item)
        if t is None:
            reasons["malformed"] += 1
            continue
        if _canon(t.parent_smiles) is None:
            reasons["invalid_parent"] += 1
            continue
        child_canon = _canon(t.child_smiles)
        if child_canon is None:
            reasons["invalid_child"] += 1
            continue
        try:
            op = registry.get(t.edit_id)
        except KeyError:
            reasons["unknown_edit"] += 1
            continue
        key = (t.parent_smiles, t.edit_id, tuple(sorted(t.site)))
        if key in seen:
            reasons["duplicate"] += 1
            continue
        cands = cand_cache.get(t.parent_smiles)
        if cands is None:
            cands = cand_cache[t.parent_smiles] = list(registry.applicable(t.parent_smiles))
        target = next((i for i, (o, s) in enumerate(cands) if o.id == t.edit_id and tuple(sorted(s)) == key[2]), None)
        if target is None:
            reasons["not_applicable"] += 1
            continue
        if child_canon not in {_canon(c) for c in op.apply(t.parent_smiles, cands[target][1])}:
            reasons["child_mismatch"] += 1
            continue
        if len(cands) < 2:
            reasons["single_candidate"] += 1
            continue
        seen.add(key)
        out.append(Example(t, cands, target))
    return out, CleanReport(len(raw), len(out), len(raw) - len(out), dict(reasons))
