"""Graph featurisation for the policy network.

Same 45-dim atom / 10-dim bond layout as dqn/utils.py (get_atom_features /
get_bond_features), re-implemented here in plain numpy because importing
`dqn.utils` pulls in ADMET, PLAPT and torch_geometric at module level, which the
studio runtime must not need. Layout is kept identical so the feature semantics
match the original DQN.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from rdkit import Chem

ATOM_DIM = 45
BOND_DIM = 10
_PT = Chem.GetPeriodicTable()
_ATOMS = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "OutOfRange"]
_BONDS = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC]


def _oh(x, items) -> list:
    if x not in items:
        x = items[-1]
    return [1.0 if i == x else 0.0 for i in items]


def atom_features(a: Chem.Atom) -> list:
    f = _oh(a.GetSymbol(), _ATOMS)
    f += _oh(int(a.GetDegree()), [0, 1, 2, 3, 4, "OutOfRange"])
    f += _oh(a.GetFormalCharge(), [-3, -2, -1, 0, 1, 2, 3, "OutOfRange"])
    f += _oh(str(a.GetHybridization()), ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OutOfRange"])
    f += [float(a.IsInRing()), float(a.GetIsAromatic())]
    f += [(_PT.GetRvdw(a.GetAtomicNum()) - 1.5) / 0.6, (_PT.GetRcovalent(a.GetAtomicNum()) - 0.64) / 0.76]
    f += _oh(str(a.GetChiralTag()), ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"])
    f += _oh(int(a.GetTotalNumHs()), [0, 1, 2, 3, 4, "OutOfRange"])
    return f


def bond_features(b: Chem.Bond) -> list:
    f = _oh(b.GetBondType(), _BONDS) + [float(b.GetIsConjugated()), float(b.IsInRing())]
    f += _oh(str(b.GetStereo()), ["STEREOZ", "STEREOE", "STEREOANY", "STEREONONE"])
    return f


def mol_graph(smiles: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(node_x [N,45], edge_index [2,E] both directions, edge_attr [E,10])."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles!r}")
    x = np.array([atom_features(a) for a in mol.GetAtoms()], dtype=np.float32).reshape(-1, ATOM_DIM)
    src, dst, ea = [], [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bf = bond_features(b)
        src += [i, j]
        dst += [j, i]
        ea += [bf, bf]
    ei = np.array([src, dst], dtype=np.int64).reshape(2, -1)
    return x, ei, np.array(ea, dtype=np.float32).reshape(-1, BOND_DIM)
