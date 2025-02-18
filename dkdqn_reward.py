from __future__ import absolute_import

from ADMET.model import ADMETModel
from binding_module.binding_affinity.plapt import Plapt, run_predictions
from binding_module.selectivity.compare import compare_affinities
from synthetic_accessibility.sa_score import SyntheticAccessibility

class Reward(object):
    def __init__(self, init_reward):
        super(Reward, self).__init__()

        self.init_reward = init_reward
    
    def calculate_reward(self, smiles, target_seq, off_target_seq):
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

        admet_model = ADMETModel()
        binding_aff_model = Plapt()
        sa_model = SyntheticAccessibility()

        # Ensure smiles is a list
        if isinstance(smiles, str):
            smiles = [smiles]

        # ADMET Predictions
        admet_preds = admet_model.predict(smiles)
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
        binding_aff_preds = run_predictions(binding_aff_model, target_seq, smiles)
        
        if off_target_seq:
            binding_off_target_preds = run_predictions(binding_aff_model, off_target_seq, smiles)    
            selectivity_values = compare_affinities(binding_aff_preds, binding_off_target_preds)

        # Synthetic Accessibility Scores
        sa_preds = sa_model.processSMILES(smiles)

        if off_target_seq:
            final_rewards = [
                admet + (1 / binding_aff) + selectivity + (1 / sa)
                for admet, binding_aff, selectivity, sa in zip(admet_rewards, binding_aff_preds, selectivity_values, sa_preds)
            ]
            
        else:
            final_rewards = [
                admet + (1 / binding_aff) + (1 / sa)
                for admet, binding_aff, sa in zip(admet_rewards, binding_aff_preds, sa_preds)
            ]

        return final_rewards