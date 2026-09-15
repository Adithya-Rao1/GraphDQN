import random
from rdkit import Chem
from rdkit.Chem import rdchem
from rdkit.Chem import Descriptors
from rdkit.Chem.rdchem import HybridizationType
from typing import List, Union, Optional, Tuple, Dict
from molecular_modifications.chemistry_constants import ELECTRONEGATIVITY, ATOMIC_NUMBERS, VALENCE_ELECTRON_COUNTS, VAN_DER_WAALS_RADII, COVALENT_RADII


def reclaim_h_for_new_bond(atom, bond_order: int = 1) -> None:
    explicit_hs = atom.GetNumExplicitHs()
    if explicit_hs > 0:
        atom.SetNumExplicitHs(max(0, explicit_hs - bond_order))
