"""Cached RDKit-descriptor scorer implementing contracts.Scorer."""
from __future__ import annotations

import hashlib
import math
from typing import Dict, List, Optional, Sequence, Tuple

from rdkit import Chem, RDLogger

from studio.schema import AlertHit, Descriptors, ObjectiveSpec, RunConfig, ScoreResult
from studio.scoring.descriptors import LRU, compute_descriptors

RDLogger.DisableLog("rdApp.*")

# Drug-like defaults used when config.objectives is empty.
DEFAULT_OBJECTIVES: Dict[str, ObjectiveSpec] = {
    "qed": ObjectiveSpec(weight=2.0, direction="maximize", low=0.3, high=0.9),
    "logp": ObjectiveSpec(weight=1.0, direction="range", low=1.0, high=3.0),
    "tpsa": ObjectiveSpec(weight=1.0, direction="range", low=40.0, high=100.0),
    "mw": ObjectiveSpec(weight=1.0, direction="range", low=200.0, high=450.0),
    "alerts": ObjectiveSpec(weight=1.5, direction="minimize", low=0.0, high=2.0),
    "lipinski_violations": ObjectiveSpec(weight=1.0, direction="minimize", low=0.0, high=2.0),
}

# Soft widths (falloff scale) per descriptor for "range" objectives without room to infer one.
_SCALE = {"qed": 0.2, "sa": 1.0, "logp": 1.0, "tpsa": 20.0, "mw": 75.0, "hbd": 1.0, "hba": 2.0,
          "rotb": 2.0, "fsp3": 0.15, "heavy_atoms": 5.0, "lipinski_violations": 1.0, "alerts": 1.0}


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, z))))


def desirability(name: str, value: float, spec: ObjectiveSpec) -> float:
    """Desirability in [0, 1].

    range: 1 inside [low, high], smooth (Gaussian) falloff outside with a per-descriptor scale.
    maximize: sigmoid rising through the midpoint of [low, high] (0.5 there; ~0.88 at high).
    minimize: mirror image (higher value -> lower desirability). `low`/`high` bound the transition.
    Missing bounds are filled with sensible per-descriptor widths.
    """
    scale = _SCALE.get(name, 1.0)
    low, high = spec.low, spec.high
    if spec.direction == "range":
        if low is not None and value < low:
            return math.exp(-0.5 * ((low - value) / scale) ** 2)
        if high is not None and value > high:
            return math.exp(-0.5 * ((value - high) / scale) ** 2)
        return 1.0
    if low is None and high is None:
        low, high = 0.0, scale
    elif low is None:
        low = high - scale
    elif high is None:
        high = low + scale
    if high <= low:
        high = low + scale
    mid = 0.5 * (low + high)
    k = 4.0 / (high - low)  # sigmoid(2) ~ 0.88 at the far bound
    z = k * (value - mid)
    return _sigmoid(z) if spec.direction == "maximize" else _sigmoid(-z)


def config_hash(objectives: Dict[str, ObjectiveSpec]) -> str:
    items = sorted((k, v.model_dump()) for k, v in objectives.items())
    return hashlib.sha1(repr(items).encode()).hexdigest()[:16]


class RDKitScorer:
    def __init__(self, cache_size: int = 20000):
        self._desc_cache = LRU(cache_size)    # canonical SMILES -> (Descriptors, alerts)
        self._canon_cache = LRU(cache_size)   # input SMILES -> canonical SMILES
        self._score_cache = LRU(cache_size)   # (canonical SMILES, config hash) -> ScoreResult

    # -- descriptors
    def _canonical(self, smiles: str) -> Tuple[str, Optional[Chem.Mol]]:
        c = self._canon_cache.get(smiles)
        if c is not None:
            return c, None
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles!r}")
        c = Chem.MolToSmiles(mol)
        self._canon_cache.put(smiles, c)
        return c, mol

    def descriptors(self, smiles: str) -> Tuple[Descriptors, List[AlertHit]]:
        canon, mol = self._canonical(smiles)
        hit = self._desc_cache.get(canon)
        if hit is not None:
            return hit
        if mol is None:
            mol = Chem.MolFromSmiles(canon)
        res = compute_descriptors(mol)
        self._desc_cache.put(canon, res)
        return res

    # -- scoring
    def score(self, smiles: str, config: RunConfig) -> ScoreResult:
        objectives = config.objectives or DEFAULT_OBJECTIVES
        canon, _ = self._canonical(smiles)
        key = (canon, config_hash(objectives))
        cached = self._score_cache.get(key)
        if cached is not None:
            return cached
        desc, alerts = self.descriptors(canon)
        comps: Dict[str, float] = {}
        num = den = 0.0
        for name, spec in objectives.items():
            value = getattr(desc, name, None)
            if value is None:  # unknown field, or SA unavailable: skip rather than invent a value
                continue
            d = desirability(name, float(value), spec)
            comps[name] = d
            num += spec.weight * d
            den += spec.weight
        composite = num / den if den > 0 else 0.0
        res = ScoreResult(descriptors=desc, components=comps, composite=composite, alerts=alerts)
        self._score_cache.put(key, res)
        return res

    def score_many(self, smiles: Sequence[str], config: RunConfig) -> List[ScoreResult]:
        return [self.score(s, config) for s in smiles]
