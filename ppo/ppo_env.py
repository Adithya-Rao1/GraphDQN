import random

import torch
import gymnasium as gym
from gymnasium import spaces
from rdkit import Chem

from molecular_modifications.atom_optimization import ModifyAtom
from molecular_modifications.bond_optimization import ModifyBond
from molecular_modifications.bioisosteres_optimization import ModifyBioisosteres
from molecular_modifications.functional_group_optimization import ModifyFunctionalGroup
from molecular_modifications.logger import setup_molecule_logger
from ppo.mol_graph import GraphDataset
import ppo.ppo_hyperparams as hp
import dqn.dqn_hyperparams as reward_hp
from ADMET.model import ADMETModel
from binding_module.binding_affinity.plapt import Plapt
from synthetic_accessibility.sa_score import SyntheticAccessibility
from reward.multi_objective import compute_reward


class MoleculeEnv(gym.Env):
    def __init__(self, target_seq, off_target_seq=None, device=None, base_mols=None):
        super(MoleculeEnv, self).__init__()
        self.mol_logger = setup_molecule_logger()
        self.graph_dataset = GraphDataset()
        self.target_seq = target_seq
        self.off_target_seq = off_target_seq
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.base_mols = list(base_mols) if base_mols is not None else list(hp.base_mols)
        self.admet_model = ADMETModel(self.device)
        self.binding_model = Plapt(device=str(self.device))
        self.sa_model = SyntheticAccessibility()

        self.action_space = None
        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(self.graph_dataset.node_feature_dim,))

        self.current_mol = None
        self.current_actions = []
        self.total_steps = 0
        self.step_count = 0
        self.current_reward = 0
        self.done = False
        self.episode = 0
        self.episode_reward = 0
        self.episode_lengths = []

    def reset(self, *, seed=None, options=None):
        self.current_mol = self.sample_initial_molecule()
        self.current_actions = self.get_actions(self.current_mol)
        self.step_count = 0
        self.current_reward = 0
        self.episode_reward = 0
        self.done = False

        return self.graph_dataset.mol_to_state_vector(self.current_mol), {}

    def sample_initial_molecule(self):
        return Chem.MolFromSmiles(random.choice(self.base_mols))

    def step(self, action: int):
        self.current_mol = self.current_actions[action]
        self.current_actions = self.get_actions(self.current_mol)

        reward = self.reward(self.current_mol)
        self.current_reward = reward
        self.step_count += 1

        terminated = self.step_count >= hp.steps_per_episode
        self.done = terminated

        info = {'smiles': Chem.MolToSmiles(self.current_mol)}
        obs = self.graph_dataset.mol_to_state_vector(self.current_mol)

        return obs, reward, terminated, False, info

    def get_actions(self, state=None):
        modify_atom = ModifyAtom(self.mol_logger)
        modify_bio = ModifyBioisosteres(self.mol_logger)
        modify_bond = ModifyBond(self.mol_logger)
        modify_func = ModifyFunctionalGroup(self.mol_logger)

        if state is None:
            return self._pad_actions([Chem.MolFromSmiles(smi) for smi in self.base_mols])

        mol = Chem.MolFromSmiles(state) if isinstance(state, str) else state

        actions = set()

        # Atom actions
        actions.add(modify_atom.add_atom(mol, 5))
        actions.add(modify_atom.remove_atom(mol, 5))
        actions.add(modify_atom.modify_atom(mol, 5))

        # Bond actions
        for i in range(3):
            actions.add(modify_bond.optimize_bond(mol, i))

        # Bioisosteric groups actions
        bio_mappings = [
            ('carboxylic_acid', 'tetrazole'),
            ('carboxylic_acid', 'phosphonic_acid'),
            ('amine', None),
            ('amide', 'sulfonamide'),
            ('amide', 'retroamide'),
            ('phenyl', 'pyridyl'),
            ('phenyl', 'thiophene'),
            ('phenyl', 'furan'),
            ('phenyl', 'pyrrole'),
        ]
        for bio1, bio2 in bio_mappings:
            actions.add(modify_bio.apply_modification(mol, bio1, bio2))

        # Functional group actions
        fg_groups = ['methyl', 'hyroxyl', 'amino', 'carboxyl', 'carbonyl', 'aldehyde',
                     'ketone', 'ether', 'ester', 'amide', 'nitro', 'cyano', 'thiol',
                     'halogen', 'azide', 'sulfonamide']
        for fg in fg_groups:
            actions.add(modify_func.remove_functional_group(mol, fg))

        fg_mappings = [
            ('hydroxyl', 'amino'),
            ('hydroxyl', 'thiol'),
            ('hydroxyl', 'methyl'),
            ('hydroxyl', 'halogen'),
            ('amino', 'hydroxyl'),
            ('amino', 'amide'),
            ('carboxyl', 'ester'),
            ('carboxyl', 'amide'),
            ('aldehyde', 'ketone'),
            ('aldehyde', 'carboxyl'),
            ('ketone', 'alcohol'),
            ('thiol', 'hydroxyl'),
            ('cyano', 'carboxyl'),
            ('nitro', 'amino'),
            ('ester', 'carboxyl'),
            ('amide', 'carboxyl'),
        ]
        for fg1, fg2 in fg_mappings:
            actions.add(modify_func.modify_functional_group(mol, fg1, fg2))

        candidates = [Chem.MolFromSmiles(smi) for smi in actions if smi]
        candidates = [mol for mol in candidates if mol is not None]

        return self._pad_actions(candidates if candidates else [mol])

    @staticmethod
    def _pad_actions(candidates):
        if len(candidates) >= hp.max_actions:
            return candidates[:hp.max_actions]
        return [candidates[i % len(candidates)] for i in range(hp.max_actions)]

    def reward(self, mol):
        result = compute_reward(
            smiles=Chem.MolToSmiles(mol),
            target_seq=self.target_seq,
            device=self.device,
            off_target_seq=self.off_target_seq,
            admet_weight=reward_hp.admet_weight,
            binding_weight=reward_hp.binding_weight,
            synthetic_weight=reward_hp.synthetic_weight,
            selectivity_weight=reward_hp.selectivity_weight,
            admet_model=self.admet_model,
            binding_model=self.binding_model,
            sa_model=self.sa_model,
        )
        return result["reward"]
