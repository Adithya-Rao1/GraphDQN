from dqn.dqn_env import MoleculeEnv
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from dqn.utils import penalized_logp

import dqn.dqn_hyperparams as hyp

from ADMET.model import ADMETModel
from binding_module.binding_affinity.plapt import Plapt, run_predictions
from binding_module.selectivity.compare import compare_affinities
from synthetic_accessibility.sa_score import SyntheticAccessibility

class QEDEnv(MoleculeEnv):
    def __init__(self, discount_factor, **kwargs):
        super(QEDEnv, self).__init__(**kwargs)
        self.discount_factor = discount_factor
    
    def _reward(self):
        if self._state is None:
            return 0.0
        
        mol = Chem.MolFromSmiles(self._state)

        if mol is None:
            return 0.0
        
        reward = Chem.QED.qed(mol)
        
        return reward * self.discount_factor ** (self.max_steps - self._counter)

class LogPEnv(MoleculeEnv):
    def __init__(self, discount_factor, **kwargs):
        super(LogPEnv, self).__init__(**kwargs)
        self.discount_factor = discount_factor
    
    def _reward(self):
        if self._state is None:
            return 0.0
        
        mol = Chem.MolFromSmiles(self._state)
        if mol is None:
            return 0.0

        reward = penalized_logp(mol)

        return reward * self.discount_factor ** (self.max_steps - self._counter)

class SAScoreEnv(MoleculeEnv):
    def __init__(self, discount_factor, **kwargs):
        super(SAScoreEnv, self).__init__(**kwargs)
        self.discount_factor = discount_factor
    
    def _reward(self):
        if self._state is None:
            return 0.0
        
        mol = Chem.MolFromSmiles(self._state)
        if mol is None:
            return 0.0

        reward = SyntheticAccessibility().calculateScore(mol)

        return (1/reward) * self.discount_factor ** (self.max_steps - self._counter)

class BindingEnv(MoleculeEnv):
    def __init__(self, discount_factor, **kwargs):
        super(BindingEnv, self).__init__(**kwargs)
        self.discount_factor = discount_factor
    
    def _reward(self):
        if self._state is None:
            return 0.0
        
        reward = run_predictions(Plapt(), self.target_seq, [self._state])[0]

        return (1/reward) * self.discount_factor ** (self.max_steps - self._counter)

class ADMETEnv(MoleculeEnv):
    def __init__(self, discount_factor, **kwargs):
        super(ADMETEnv, self).__init__(**kwargs)
        self.discount_factor = discount_factor
    
    def _reward(self):
        if self._state is None:
            return 0.0
        
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
        admet_preds = ADMETModel().predict(self._state)
        admet_values = [admet_preds[prop] for prop in admet_properties]

        reward = sum(
                admet_values[j] if admet_optim_directions[j] == 1 else (1 / admet_values[j])
                for j in range(len(admet_properties))
            )
        
        return reward * self.discount_factor ** (self.max_steps - self._counter)

class MultiObjectiveRewardEnv(MoleculeEnv):
    def __init__(self, discount_factor, device, **kwargs):
        super(MultiObjectiveRewardEnv, self).__init__(**kwargs)
        self.discount_factor = discount_factor
        self.device = device
    
    def _reward(self):
        if self._state is None:
            return 0.0
        
        mol = Chem.MolFromSmiles(self._state)
        if mol is None:
            return 0.0
        
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
        admet_preds = ADMETModel(self.device).predict(self._state)
        admet_values = [admet_preds[prop] for prop in admet_properties]

        admet_reward = sum(
                admet_values[j] if admet_optim_directions[j] == 1 else (1 / admet_values[j])
                for j in range(len(admet_properties))
            )
        binding_reward = run_predictions(Plapt(), self.target_seq, [self._state])[0]
        sa_reward = SyntheticAccessibility().calculateScore(mol)
        if self.off_target_seq:
            off_target_pred = run_predictions(Plapt(), self.off_target_seq, [self._state])[0]
            selectivity_reward = compare_affinities(binding_reward, off_target_pred)
            scale_weights = hyp.selectivity_weight/3

            reward = ((hyp.admet_weight - scale_weights)*admet_reward + 
            (hyp.binding_weight - scale_weights)*(1/binding_reward) + 
            (hyp.selectivity_weight - scale_weights)*(1/sa_reward) + 
            hyp.selectivity_weight*selectivity_reward)
        
        reward = (hyp.admet_weight*admet_reward + 
                  hyp.binding_weight*(1/binding_reward) + 
                  hyp.synthetic_weight*(1/sa_reward))

        return reward * self.discount_factor ** (self.max_steps - self._counter)

class LogPConstrainedEnv(MoleculeEnv):
    def __init__(self, target_molecule, discount_factor, **kwargs):
        super(LogPConstrainedEnv, self).__init__(**kwargs)
        self.discount_factor = discount_factor
        self.target_molecule = target_molecule
        self.target_struct = self.return_fingerprint(Chem.MolFromSmiles(self.target_molecule))
    
    def return_fingerprint(self, molecule):
        if isinstance(molecule, str):
            molecule = Chem.MolFromSmiles

        return AllChem.GetMorganFingerprint(molecule, radius=2)
    
    def similarity(self, molecule):
        molecule_struct = self.return_fingerprint(molecule)

        return DataStructs.TanimotoSimilarity(self.target_struct, molecule_struct)
    
    def _reward(self):
        if self._state is None:
            return 0.0
        
        mol = Chem.MolFromSmiles(self._state)
        if mol is None:
            return 0.0
        
        sim = self.similarity(mol)

        if sim <= hyp.similarity_threshold:
            reward = penalized_logp(mol) + 100 * (sim - hyp.similarity_threshold)
        else:
            reward = penalized_logp(mol)

        return reward * self.discount_factor**(self.max_steps - self._counter)

class QEDConstrainedEnv(MoleculeEnv):
    def __init__(self, target_molecule, discount_factor, **kwargs):
        super(QEDConstrainedEnv, self).__init__(**kwargs)
        self.discount_factor = discount_factor
        self.target_molecule = target_molecule
        self.target_struct = self.return_fingerprint(Chem.MolFromSmiles(self.target_molecule))
    
    def return_fingerprint(self, molecule):
        if isinstance(molecule, str):
            molecule = Chem.MolFromSmiles

        return AllChem.GetMorganFingerprint(molecule, radius=2)
    
    def similarity(self, molecule):
        molecule_struct = self.return_fingerprint(molecule)

        return DataStructs.TanimotoSimilarity(self.target_struct, molecule_struct)
    
    def _reward(self):
        if self._state is None:
            return 0.0
        
        mol = Chem.MolFromSmiles(self._state)
        if mol is None:
            return 0.0
        
        sim = self.similarity(mol)

        if sim <= hyp.similarity_threshold:
            reward = Chem.QED.qed(mol) + 100 * (sim - hyp.similarity_threshold)
        else:
            reward = Chem.QED.qed(mol)

        return reward * self.discount_factor**(self.max_steps - self._counter)



    

        