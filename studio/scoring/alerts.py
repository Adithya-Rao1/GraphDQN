"""Structural alerts via RDKit FilterCatalog (PAINS, BRENK, NIH)."""
from __future__ import annotations

from typing import List

from rdkit import Chem
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

from studio.schema import AlertHit

_FAMILIES = (
    ("PAINS", FilterCatalogParams.FilterCatalogs.PAINS),
    ("BRENK", FilterCatalogParams.FilterCatalogs.BRENK),
    ("NIH", FilterCatalogParams.FilterCatalogs.NIH),
)
_CATALOGS = []
for _name, _cat in _FAMILIES:
    _p = FilterCatalogParams()
    _p.AddCatalog(_cat)
    _CATALOGS.append((_name, FilterCatalog(_p)))


def find_alerts(mol: Chem.Mol) -> List[AlertHit]:
    """All alert matches, one AlertHit per (entry, match); atom indices are sorted mol atom indices.

    Per-family catalogs are used so `family` is exact (PAINS_A/B/C are reported as PAINS).
    """
    hits: List[AlertHit] = []
    for family, cat in _CATALOGS:
        for entry in cat.GetMatches(mol):
            for fm in entry.GetFilterMatches(mol):
                # fm.atomPairs is (query_idx, mol_idx); keep the molecule index
                atoms = sorted({int(mol_idx) for _, mol_idx in fm.atomPairs})
                hits.append(AlertHit(name=entry.GetDescription(), family=family, atom_indices=atoms))
    return hits
