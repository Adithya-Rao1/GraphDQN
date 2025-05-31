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
from ADMET.model import ADMETModel
from binding_module.binding_affinity.plapt import Plapt, run_predictions
from binding_module.selectivity.compare import compare_affinities
from synthetic_accessibility.sa_score import SyntheticAccessibility
import random

# OBSERVATIONS = [PYG OBJECT]

class MoleculeEnv:
    def __init__(self, target_seq, off_target_seq=None):
        super(MoleculeEnv, self).__init__()
        self.mol_logger = setup_molecule_logger()
        self.action_space = spaces.Discrete(len(self.actions))
        self.current_mol = None
        self.target_seq = target_seq
        self.off_target_seq = off_target_seq
        self.admet_model = ADMETModel()
        self.binding_aff_model = Plapt()
        self.sa_model = SyntheticAccessibility()
        self.step_count = 0

    def reset(self):
        self.current_mol = self.sample_initial_molecule()
        self.step_count = 0
        return self.current_mol

    def sample_initial_molecule(self):
        sample_mol = Chem.MolFromSmiles(hp.base_mols[random.randint(0, len(hp.base_mols) - 1)])
        return sample_mol
    
    def step(self, action: int):
        """
        Step for the agent in the environment

        Args:
            action (int): action to take in the environment

        Returns:
            observation, reward, done, info
        """
        # Get observation
        actions = self.get_actions(self.current_mol)
        obs = actions[action]
        self.current_mol = Chem.MolFromSmiles(GraphDataset.graph_to_mol(obs))

        # Compute reward
        reward = self.reward(self.current_mol)

        # Check if done
        done = self.step_count >= hp.steps_per_episode

        # Info --> Add for logging/metrics
        info = {
            'smiles': Chem.MolToSmiles(obs),
        }

        return obs, reward, done, info

    def get_actions(self, state=None):
        modify_atom = ModifyAtom(self.mol_logger)
        modify_bio = ModifyBioisosteres(self.mol_logger)
        modify_bond = ModifyBond(self.mol_logger)
        modify_func = ModifyFunctionalGroup(self.mol_logger)

        if state is None:
            return hp.base_mols
        
        if isinstance(state, str):
            mol = Chem.MolFromSmiles(state)
        else:
            mol = state

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

        for fg_pair in modify_mappings:
            actions.add(
                modify_func.modify_functional_group(mol, fg_pair[0], fg_pair[1])
            )

        return {Chem.MolFromSmiles(smile) for smile in actions if smile}

    def reward(self, mol, weights=None):
        smiles = [Chem.MolToSmiles(m) for m in [mol]]

        admet_properties = ['QED',
                            'Lipinski',
                            'Bioavailability_Ma',
                            'BBB_Martins',
                            'DILI',
                            'Clearance_Hepatocyte_AZ',
                            'Clearance_Microsome_AZ',
                            'Half_Life_Obach',
                            'hERG',
                            'ClinTox',
                            'LD50_Zhu']
        admet_optim_directions = [1, 1, 1, 1, -1, 1, 1, 1, -1, -1, -1]

        # ADMET Predictions
        admet_preds = self.admet_model.predict(smiles)
        admet_rewards = []
        all_admet_values = []

        for i in range(len(admet_preds)):
            admet_values = [admet_preds.iloc[i][prop] for prop in admet_properties]
            all_admet_values.append(admet_values)  # Store all ADMET values for obj_vector

            admet_reward = sum(
                admet_values[j] if admet_optim_directions[j] == 1 else (1 / admet_values[j])
                for j in range(len(admet_properties))
            )
            admet_rewards.append(admet_reward)

        # Binding Affinity Predictions
        binding_aff_preds = run_predictions(self.binding_aff_model, self.target_seq, smiles)
        
        if self.off_target_seq:
            binding_off_target_preds = run_predictions(self.binding_aff_model, self.off_target_seq, smiles)    
            selectivity_values = compare_affinities(binding_aff_preds, binding_off_target_preds)

        # Synthetic Accessibility Scores
        sa_preds = self.sa_model.processSMILES(smiles)

        if self.off_target_seq:
            if weights:
                final_rewards = [
                    weights[0]*admet + weights[1]*(1 / binding_aff) + weights[2]*selectivity + weights[3]*(1 / sa)
                    for admet, binding_aff, selectivity, sa in zip(admet_rewards, binding_aff_preds, selectivity_values, sa_preds)
                ]
            else:
                final_rewards = [
                    admet + (1 / binding_aff) + selectivity + (1 / sa)
                    for admet, binding_aff, selectivity, sa in zip(admet_rewards, binding_aff_preds, selectivity_values, sa_preds)
                ]
        else:
            if weights:
                final_rewards = [
                    weights[0]*admet + weights[1]*(1 / binding_aff) + weights[2]*(1 / sa)
                    for admet, binding_aff, sa in zip(admet_rewards, binding_aff_preds, sa_preds)
                ]
            else:
                final_rewards = [
                admet + (1 / binding_aff) + (1 / sa)
                for admet, binding_aff, sa in zip(admet_rewards, binding_aff_preds, sa_preds)
                ]

        return final_rewards

