"""3D geometry helpers: ETKDG + MMFF embedding, MCS alignment, atom maps.

Atom indices always refer to ``Chem.MolFromSmiles(smiles)`` of the SMILES passed in
(heavy atoms only; hydrogens are removed after embedding, which preserves heavy-atom order).
"""
from __future__ import annotations

from typing import List, Optional, Tuple, Union

from rdkit import Chem
from rdkit.Chem import AllChem, rdFMCS

EMBED_SEED = 7
MCS_TIMEOUT_S = 2

MolLike = Union[str, Chem.Mol]


def _as_mol(m: MolLike) -> Optional[Chem.Mol]:
    return Chem.MolFromSmiles(m) if isinstance(m, str) else m


def embed_raw(smiles: str, seed: int = EMBED_SEED) -> Optional[Chem.Mol]:
    """ETKDG + MMFF (UFF fallback) embedding. Returns a heavy-atom mol with one 3D conformer, or None."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mh = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.maxIterations = 200
    if AllChem.EmbedMolecule(mh, params) != 0:
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(mh, params) != 0:
            return None
    try:
        if AllChem.MMFFHasAllMoleculeParams(mh):
            AllChem.MMFFOptimizeMolecule(mh, maxIters=200)
        else:
            AllChem.UFFOptimizeMolecule(mh, maxIters=200)
    except Exception:  # optimization is a refinement; ETKDG coordinates are already usable
        pass
    return Chem.RemoveHs(mh)


def mcs_match(parent: MolLike, child: MolLike) -> List[Tuple[int, int]]:
    """Partial (parent_idx, child_idx) map over the maximum common substructure."""
    p, c = _as_mol(parent), _as_mol(child)
    if p is None or c is None:
        return []
    res = rdFMCS.FindMCS(
        [p, c], timeout=MCS_TIMEOUT_S, ringMatchesRingOnly=True,
        atomCompare=rdFMCS.AtomCompare.CompareElements, bondCompare=rdFMCS.BondCompare.CompareOrder,
    )
    if res.numAtoms == 0:
        return []
    patt = Chem.MolFromSmarts(res.smartsString)
    a, b = p.GetSubstructMatch(patt), c.GetSubstructMatch(patt)
    return list(zip(a, b)) if a and b else []


def atom_map(parent: MolLike, child: MolLike) -> List[Tuple[int, int]]:
    """Schema ``Candidate.atom_map``: (parent idx, child idx) pairs over the MCS."""
    return mcs_match(parent, child)


def align_to(child: Chem.Mol, ref: Chem.Mol, amap: List[Tuple[int, int]]) -> None:
    """Rigidly align ``child`` onto ``ref`` in place using the (parent, child) atom map."""
    if len(amap) >= 3:
        AllChem.AlignMol(child, ref, atomMap=[(c, p) for p, c in amap])


def embed_3d(smiles: str, ref_mol: Optional[Chem.Mol] = None) -> Tuple[str, Chem.Mol]:
    """(molblock, mol): 3D heavy-atom embedding, rigid-aligned on the MCS to ``ref_mol`` when given.

    Raises ValueError if the molecule cannot be embedded.
    """
    mol = embed_raw(smiles)
    if mol is None:
        raise ValueError(f"3D embedding failed for {smiles}")
    if ref_mol is not None:
        align_to(mol, ref_mol, mcs_match(ref_mol, mol))
    return Chem.MolToMolBlock(mol), mol
