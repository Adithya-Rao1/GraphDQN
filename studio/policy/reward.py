"""Pure-RDKit composite reward in [0, 1] (no ADMET-AI, no PLAPT).

Mirrors the objectives agent's defaults in spirit (QED maximise, logP and TPSA
target ranges, structural alerts, Lipinski). This is the policy's *own* copy so
training never depends on studio/scoring; the weights below are the policy
defaults, not read from studio/scoring.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, Optional

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Lipinski, QED, Crippen, rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

RDLogger.DisableLog("rdApp.*")

# name -> (weight, low, high); desirability is 1 inside [low, high], decays linearly outside.
DEFAULT_WEIGHTS: Dict[str, float] = {"qed": 2.0, "logp": 1.0, "tpsa": 1.0, "alerts": 1.0, "lipinski": 1.0}
LOGP_RANGE = (1.0, 3.0)
TPSA_RANGE = (40.0, 90.0)

_catalog: Optional[FilterCatalog] = None


def _alert_catalog() -> FilterCatalog:
    global _catalog
    if _catalog is None:
        p = FilterCatalogParams()
        p.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
        p.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
        _catalog = FilterCatalog(p)
    return _catalog


def _range_desirability(x: float, low: float, high: float, width: float) -> float:
    if low <= x <= high:
        return 1.0
    d = low - x if x < low else x - high
    return max(0.0, 1.0 - d / width)


def components(mol: Chem.Mol) -> Dict[str, float]:
    logp = Crippen.MolLogP(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    viol = sum([
        Descriptors.MolWt(mol) > 500,
        logp > 5,
        Lipinski.NumHDonors(mol) > 5,
        Lipinski.NumHAcceptors(mol) > 10,
    ])
    alerts = len(_alert_catalog().GetMatches(mol))
    return {
        "qed": float(QED.qed(mol)),
        "logp": _range_desirability(logp, *LOGP_RANGE, width=3.0),
        "tpsa": _range_desirability(tpsa, *TPSA_RANGE, width=50.0),
        "alerts": 1.0 / (1.0 + alerts),
        "lipinski": 1.0 - viol / 4.0,
    }


@lru_cache(maxsize=200_000)
def composite(smiles: str) -> float:
    """Weighted mean of per-component desirabilities; 0.0 for unparsable SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumHeavyAtoms() == 0:
        return 0.0
    c = components(mol)
    tot = sum(DEFAULT_WEIGHTS.values())
    return sum(DEFAULT_WEIGHTS[k] * c[k] for k in DEFAULT_WEIGHTS) / tot
