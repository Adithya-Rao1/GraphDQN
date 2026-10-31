"""BindingScorer: PLAPT-predicted binding as the stage-1 fitness, implementing contracts.Scorer.

composite = sigmoid((pKd - seed_pKd) / BINDING_SCALE) * guard factors, so the seed drug sits at 0.5 and
improvements show up. Hard guards (composite 0, PLAPT not even called): Tanimoto(Morgan r2) to seed < floor,
MW > cap, failed sanity. Soft guard: 0.5 ** (number of NEW structural alerts vs the seed).
Everything is PREDICTED; `extras` carries the raw model outputs.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

from studio.binding.backends import BindingBackend, canon
from studio.schema import RunConfig, ScoreResult
from studio.scoring import RDKitScorer

RDLogger.DisableLog("rdApp.*")

BINDING_SCALE = 0.75   # pKd units over which desirability goes 0.5 -> 0.73 (one scale above the seed)
DEFAULT_MW_HEADROOM = 120.0
NEW_ALERT_FACTOR = 0.5

try:  # sanity module lives on another branch; fall back to a small local check, labelled as such
    from studio.sanity.checks import check_child as _check_child
    SANITY_SOURCE = "studio.sanity.checks"
except Exception:  # pragma: no cover - depends on branch
    _check_child = None
    SANITY_SOURCE = "local_fallback"


def _local_sanity(parent: str, child: str) -> Tuple[bool, str]:
    """LOCAL FALLBACK (minimal): single fragment, no radicals, no new charged atoms."""
    mol = Chem.MolFromSmiles(child)
    if mol is None:
        return False, "invalid"
    if "." in child:
        return False, "multiple_fragments"
    if any(a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
        return False, "radical"
    pm = Chem.MolFromSmiles(parent) if parent else None
    pc = sum(1 for a in pm.GetAtoms() if a.GetFormalCharge()) if pm else 0
    if sum(1 for a in mol.GetAtoms() if a.GetFormalCharge()) > pc + 2:
        return False, "extra_charges"
    return True, "ok"


def sanity(parent: str, child: str) -> Tuple[bool, str]:
    return _check_child(parent, child) if _check_child else _local_sanity(parent, child)


def morgan_fp(smiles: str):
    return AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), 2, 2048)


def tanimoto(a: str, b: str) -> float:
    return float(DataStructs.TanimotoSimilarity(morgan_fp(a), morgan_fp(b)))


def binding_desirability(pkd: float, seed_pkd: float, scale: float = BINDING_SCALE) -> float:
    """Monotone increasing in pkd, 0.5 at the seed's own predicted pKd."""
    z = max(-60.0, min(60.0, (pkd - seed_pkd) / scale))
    return 1.0 / (1.0 + math.exp(-z))


def _alert_counts(alerts) -> Counter:
    return Counter(a.name for a in alerts)


class BindingScorer:
    """Scorer whose fitness is PLAPT-predicted binding to `config.target_id` (guarded)."""

    def __init__(self, backend: BindingBackend, target_provider: Optional[Callable[[str], object]] = None,
                 base: Optional[RDKitScorer] = None):
        self.backend = backend
        self.base = base or RDKitScorer()
        self._provider = target_provider
        self._seed_cache: Dict[Tuple[str, str], float] = {}

    # -- target
    def _sequence(self, config: RunConfig) -> str:
        if not config.target_id:
            raise ValueError("binding stage needs config.target_id")
        provider = self._provider
        if provider is None:
            from studio.data.loader import get_target  # owned by the targets stream
            provider = get_target
        return provider(config.target_id).sequence

    def _seed_pkd(self, seq: str, seed: str, config: RunConfig) -> float:
        key = (config.target_id or "", seed)
        if key not in self._seed_cache:
            self._seed_cache[key] = self.backend.predict(seq, [seed])[0]
        return self._seed_cache[key]

    # -- scoring
    def score(self, smiles: str, config: RunConfig) -> ScoreResult:
        return self.score_many([smiles], config)[0]

    def score_many(self, smiles: Sequence[str], config: RunConfig) -> List[ScoreResult]:
        seq = self._sequence(config)
        seed = canon(config.seed_smiles)
        seed_res = self.base.score(seed, config)
        seed_pkd = self._seed_pkd(seq, seed, config)
        cap = config.max_mw if config.max_mw is not None else seed_res.descriptors.mw + DEFAULT_MW_HEADROOM
        seed_alerts = _alert_counts(seed_res.alerts)

        rows = []  # (base ScoreResult, extras, hard_fail, canonical)
        for s in smiles:
            c = canon(s)
            res = self.base.score(c, config)
            sim = tanimoto(seed, c)
            ok, reason = (True, "seed") if c == seed else sanity(seed, c)
            n_new = sum((_alert_counts(res.alerts) - seed_alerts).values())
            hard = sim < config.min_similarity_to_seed or res.descriptors.mw > cap or not ok
            ex = {"sim_to_seed": sim, "mw_cap": float(cap), "new_alerts": float(n_new),
                  "sanity_ok": float(ok), "sanity_is_fallback": float(SANITY_SOURCE == "local_fallback"),
                  "guard_failed": float(hard), "plapt_seed_pkd": seed_pkd}
            rows.append((res, ex, hard, c))

        todo = list(dict.fromkeys(c for _, _, hard, c in rows if not hard and c != seed))
        pred: Dict[str, float] = dict(zip(todo, self.backend.predict(seq, todo))) if todo else {}
        pred[seed] = seed_pkd

        out = []
        for res, ex, hard, c in rows:
            comps = dict(res.components)
            if hard:
                comps["binding"] = 0.0
                out.append(res.model_copy(update={"composite": 0.0, "components": comps, "extras": ex}))
                continue
            pkd = pred[c]
            d = binding_desirability(pkd, seed_pkd)
            comps["binding"] = d
            ex = {**ex, "plapt_pkd": pkd, "plapt_affinity": pkd,  # plapt_affinity: alias, same unit (-log10 Kd/M)
                  "plapt_affinity_uM": 1e6 * 10 ** (-pkd), "plapt_delta_vs_seed": pkd - seed_pkd}
            comp = d * NEW_ALERT_FACTOR ** ex["new_alerts"]
            out.append(res.model_copy(update={"composite": comp, "components": comps, "extras": ex}))
        return out
