import collections
import copy
import itertools

import torch
import torch.nn as nn
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit import RDLogger

# The Modify* classes below try many candidate edits per step and discard the
# ones that fail sanitization/valence checks -- that's expected, already
# handled via try/except, and produces a flood of RDKit C++-level warnings
# ("Explicit valence...", "non-ring atom marked aromatic") that aren't
# actionable. Silence them; real issues are still surfaced via each class's
# own Python logger when log=True.
RDLogger.DisableLog('rdApp.*')

from molecular_modifications.bioisosteres_optimization import ModifyBioisosteres
from molecular_modifications.atom_optimization import ModifyAtom
from molecular_modifications.bond_optimization import ModifyBond
from molecular_modifications.functional_group_optimization import ModifyFunctionalGroup
from molecular_modifications.logger import setup_molecule_logger

modify_atom = ModifyAtom(setup_molecule_logger())
modify_bond = ModifyBond(setup_molecule_logger())
modify_bio = ModifyBioisosteres(setup_molecule_logger())
modify_fg = ModifyFunctionalGroup(setup_molecule_logger())

# Functional-group edits to try each step. Removal groups are single-attachment
# substituents ModifyFunctionalGroup can find and remove; modify pairs are
# limited to fragments FG_FRAGMENT_SMILES actually defines (both ends);
# add groups use whatever attachment site functional_group_add_sites finds
# first, mirroring how modify_atom/modify_bond pick one site per call rather
# than enumerating every possibility.
_FG_REMOVE_GROUPS = [
    'methyl', 'hydroxyl', 'amino', 'carboxyl', 'carbonyl', 'aldehyde',
    'ketone', 'ether', 'ester', 'amide', 'nitro', 'cyano', 'thiol',
    'halogen', 'azide', 'sulfonamide',
]
_FG_MODIFY_PAIRS = [
    ('hydroxyl', 'amino'), ('hydroxyl', 'thiol'), ('hydroxyl', 'methyl'), ('hydroxyl', 'halogen'),
    ('amino', 'hydroxyl'), ('amino', 'amide'),
    ('carboxyl', 'ester'), ('carboxyl', 'amide'),
    ('aldehyde', 'ketone'), ('aldehyde', 'carboxyl'),
    ('thiol', 'hydroxyl'), ('cyano', 'carboxyl'), ('nitro', 'amino'),
    ('ester', 'carboxyl'), ('amide', 'carboxyl'),
]
_FG_ADD_GROUPS = ['methyl', 'hydroxyl', 'amino', 'halogen', 'cyano', 'carboxyl']

class Result(collections.namedtuple("Result", ["state", "reward", "terminated"])):
    "Named tuple to store the result of an environment step."

def get_all_actions(state):
    if not state:
        return copy.deepcopy(set(['C', 'O']))
    
    mol = Chem.MolFromSmiles(state)
    if mol is None:
        raise ValueError(f"Invalid state: {state}")

    actions = set()

    # Atom actions
    actions.add(
        modify_atom.add_atom(mol, 5)
    )
    actions.add(
        modify_atom.remove_atom(mol, 5)
    )
    actions.add(
        modify_atom.modify_atom(mol, 5)
    )

    # Bond actions
    actions.add(
        modify_bond.optimize_bond(mol, 0)
    )
    actions.add(
        modify_bond.optimize_bond(mol, 1)
    )
    actions.add(
        modify_bond.optimize_bond(mol, 2)
    )

    # Bioisosteric groups actions
    actions.add(
        modify_bio.apply_modification(mol, 'carboxylic_acid', 'tetrazole')
    )
    actions.add(
        modify_bio.apply_modification(mol, 'carboxylic_acid', 'phosphonic_acid')
    )
    actions.add(
        modify_bio.apply_modification(mol, 'amine', None)
    )
    actions.add(
        modify_bio.apply_modification(mol, 'amide', 'sulfonamide')
    )
    actions.add(
        modify_bio.apply_modification(mol, 'amide', 'retroamide')
    )
    actions.add(
        modify_bio.apply_modification(mol, 'phenyl', 'pyridyl')
    )
    actions.add(
        modify_bio.apply_modification(mol, 'phenyl', 'thiophene')
    )
    actions.add(
        modify_bio.apply_modification(mol, 'phenyl', 'furan')
    )
    actions.add(
        modify_bio.apply_modification(mol, 'phenyl', 'pyrrole')
    )

    # Functional group actions
    for fg in _FG_REMOVE_GROUPS:
        actions.add(modify_fg.remove_functional_group(mol, fg))

    for source_fg, target_fg in _FG_MODIFY_PAIRS:
        actions.add(modify_fg.modify_functional_group(mol, source_fg, target_fg))

    for fg in _FG_ADD_GROUPS:
        add_sites = modify_fg.functional_group_add_sites(mol, fg)
        if add_sites:
            actions.add(modify_fg.add_functional_group(mol, fg, add_sites[0]))

    candidates = {smiles for smiles in actions if smiles and Chem.MolFromSmiles(smiles) is not None}

    if not candidates:
        return {state}

    return candidates

def goal_by_similarity(smile, target_smile):
    smile_struct = AllChem.GetMorganFingerprint(Chem.MolFromSmiles(smile), radius=2)
    target_smile_struct = AllChem.GetMorganFingerprint(Chem.MolFromSmiles(target_smile), radius=2)
    t_sim = DataStructs.TanimotoSimilarity(target_smile_struct, smile_struct)

    return t_sim

class MoleculeEnv(object):
    def __init__(self, 
                 init_mol, 
                 max_steps, 
                 target_fn=None,
                 target_seq=None,
                 off_target_seq=None):
        super(MoleculeEnv, self).__init__()

        if isinstance(init_mol, (Chem.Mol, Chem.RWMol)):
            self.init_mol = Chem.MolToSmiles(init_mol)
        
        self.init_mol = init_mol
        self.target_seq = target_seq
        self.off_target_seq = off_target_seq
        self.max_steps = max_steps
        self._counter = self.max_steps
        self._state = None
        self._valid_actions = []
        self._target_fn = target_fn
        self._path = []
    
    @property
    def state(self):
        return self._state

    @property
    def num_steps_taken(self):
        return self._counter
    
    def get_path(self):
        return self._path
    
    def initialize(self):
        self._state = self.init_mol
        self._path = [self._state]
        self._valid_actions = get_all_actions(None)
        self._counter = 0
    
    def get_valid_actions(self):
        if not self._state:
            return copy.deepcopy(self._valid_actions)
        if isinstance(self._state, (Chem.Mol, Chem.RWMol)):
            state = Chem.MolToSmiles(self._state)
        state = self._state
        self._valid_actions = list(get_all_actions(state))
        
        return copy.deepcopy(self._valid_actions)

    def _reward(self):
        if self._state == '':
            return 0.0
        return 0.0

    def _goal_reached(self):
        if self._target_fn is None:
            return False
        else:
            return self._target_fn(self._state)
    
    def step(self, action):
        if self._counter >= self.max_steps or self._goal_reached():
            raise ValueError("This episode has been terminated.")
        if action not in self._valid_actions:
            raise ValueError(f"Invalid action: {action}")
        
        self._state = action
        self._path.append(self._state)
        self._counter += 1
        result = Result(
            state=self._state,
            reward=self._reward(),
            terminated=(self._counter >= self.max_steps) or self._goal_reached(),
        )

        return result