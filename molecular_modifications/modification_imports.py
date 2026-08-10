import random
from rdkit import Chem
from rdkit.Chem import rdchem
from rdkit.Chem.rdchem import HybridizationType
from typing import List, Union, Optional, Tuple, Dict
from molecular_modifications.chemistry_constants import ELECTRONEGATIVITY, ATOMIC_NUMBERS, VALENCE_ELECTRON_COUNTS, VAN_DER_WAALS_RADII, COVALENT_RADII
