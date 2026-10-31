"""Per-atom RDKit decompositions of one molecule (exact, deterministic)."""
from __future__ import annotations

import math
from typing import List

from rdkit import Chem
from rdkit.Chem import rdPartialCharges, rdMolDescriptors

from studio.schema import AtomContributions


def gasteiger_charges(mol: Chem.Mol) -> List[float]:
    """Gasteiger partial charges, one per atom. Works on a copy so `mol` is not mutated.
    Atoms RDKit cannot parametrize (NaN/inf) report 0.0."""
    m = Chem.Mol(mol)
    rdPartialCharges.ComputeGasteigerCharges(m)
    out = []
    for a in m.GetAtoms():
        v = a.GetDoubleProp("_GasteigerCharge") if a.HasProp("_GasteigerCharge") else 0.0
        out.append(v if math.isfinite(v) else 0.0)
    return out


def _crippen_with_hydrogens(mol: Chem.Mol) -> List[float]:
    """Per-heavy-atom Crippen logP with each hydrogen's contribution folded into its
    neighbour, so the list sums to Crippen.MolLogP exactly."""
    mh = Chem.AddHs(mol)
    contribs = rdMolDescriptors._CalcCrippenContribs(mh)
    n = mol.GetNumAtoms()
    out = [float(contribs[i][0]) for i in range(n)]
    for a in mh.GetAtoms():
        if a.GetIdx() >= n:
            out[a.GetNeighbors()[0].GetIdx()] += float(contribs[a.GetIdx()][0])
    return out


def atom_contributions(mol: Chem.Mol) -> AtomContributions:
    """Crippen logP, TPSA and Gasteiger contributions, index-aligned with `mol`'s atoms
    (heavy atoms only, as hydrogens are implicit)."""
    return AtomContributions(
        crippen_logp=_crippen_with_hydrogens(mol),
        tpsa=[float(c) for c in rdMolDescriptors._CalcTPSAContribs(mol)],
        gasteiger=gasteiger_charges(mol),
    )
