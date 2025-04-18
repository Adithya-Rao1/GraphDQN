import gymnasium as gym
from gymnasium import spaces
from molecular_modifications.atom_optimization import ModifyAtom
from molecular_modifications.bond_optimization import ModifyBond
from molecular_modifications.bioisosteres_optimization import ModifyBioisosteres
from molecular_modifications.functional_group_optimization import ModifyFunctionalGroup
from molecular_modifications.logger import setup_molecule_logger
from mol_graph import GraphDataset
import ppo_hyperparams as hp
import rdkit
from rdkit import Chem

# OBSERVATIONS = [PYG OBJECT]

class MoleculeEnv:
    def __init__(self):
        super(MoleculeEnv, self).__init__()
        self.mol_logger = setup_molecule_logger()
        self.action_space = spaces.Discrete(len(self.actions))
        self.current_mol = None

    def reset(self):
        self.current_mol = self.sample_initial_molecule()
        return GraphDataset.graph_to_pyg(GraphDataset.mol_to_graph(self.current_mol))

    def get_actions(self, state=None):
        modify_atom = ModifyAtom(self.mol_logger)
        modify_bio = ModifyBioisosteres(self.mol_logger)
        modify_bond = ModifyBond(self.mol_logger)
        modify_func = ModifyFunctionalGroup(self.mol_logger)

        if state is None:
            return hp.base_mols
        
        mol = Chem.MolFromSmiles(state)
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
        for i in range(3):
            actions.add(
                modify_bond.optimize_bond(mol, i)
            )

        # Bioisosteric groups actions
        modify_mappings = [
            ('carboxylic_acid', 'tetrazole'),
            ('carboxylic_acid', 'phosphonic_acid'),
            ('amine', None),
            ('amide', 'sulfonamide'),
            ('amide', 'retroamide'),
            ('phenyl', 'pyridyl'),
            ('phenyl', 'thiophene'),
            ('phenyl', 'furan'),
            ('phenyl', 'pyrrole')
        ]

        for bio1, bio2 in modify_mappings:
            actions.add(
                modify_bio.apply_modification(mol, bio1, bio2)
            )

        # Functional group actions
        fg_groups = ['methyl',
                     'hyroxyl',
                     'amino',
                     'carboxyl',
                     'carbonyl',
                     'aldehyde',
                     'ketone',
                     'ether',
                     'ester',
                     'amide',
                     'nitro',
                     'cyano',
                     'thiol',
                     'halogen',
                     'azide',
                     'sulfonamide']
        
        for fg in fg_groups:
            actions.add(
                modify_func.remove_functional_group(mol, fg)
            )
        
        modify_mappings = [
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

        for fg1, fg2 in modify_mappings:
            actions.add(
                modify_func.modify_functional_group(mol, fg1, fg2)
            )

        return {GraphDataset.graph_to_pyg(GraphDataset.mol_to_graph(smiles)) for smiles in actions if smiles}

    def apply_action(self, mol, action_idx):
        pass


"""
class MoleculeEnv:
    def __init__(self, admet_model, binding_model):
        self.admet_model = admet_model  # nn.Module
        self.binding_model = binding_model  # nn.Module
        self.current_mol = None
        ...

    def reset(self):
        self.current_mol = sample_initial_molecule()
        return featurize(self.current_mol)

    def step(self, action):
        new_mol = apply_action(self.current_mol, action)
        reward = self.compute_reward(new_mol)
        self.current_mol = new_mol
        return featurize(new_mol), reward, done, {}

    def compute_reward(self, mol):
        admet_score = self.admet_model(mol)
        binding_score = self.binding_model(mol)
        return weighted_sum(admet_score, binding_score)
"""

