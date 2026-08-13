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
from reward.multi_objective import ADMET_PROPERTIES, ADMET_OPTIM_DIRECTIONS, compute_admet_reward, compute_reward

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
    def __init__(self, discount_factor, device, **kwargs):
        super(ADMETEnv, self).__init__(**kwargs)
        self.discount_factor = discount_factor
        self.device = device
        self.admet_model = ADMETModel(self.device)

    def _reward(self):
        if self._state is None:
            return 0.0

        admet_preds = self.admet_model.predict(self._state)
        reward = compute_admet_reward(admet_preds)

        return reward * self.discount_factor ** (self.max_steps - self._counter)

class MultiObjectiveRewardEnv(MoleculeEnv):
    def __init__(self, discount_factor, device, admet_model=None, binding_model=None, sa_model=None, **kwargs):
        super(MultiObjectiveRewardEnv, self).__init__(**kwargs)
        self.discount_factor = discount_factor
        self.device = device

        self.admet_model = admet_model if admet_model is not None else ADMETModel(self.device)
        self.binding_model = binding_model if binding_model is not None else Plapt(device=str(self.device))
        self.sa_model = sa_model if sa_model is not None else SyntheticAccessibility()

    def _reward(self):
        if self._state is None:
            return 0.0

        mol = Chem.MolFromSmiles(self._state)
        if mol is None:
            return 0.0

        result = compute_reward(
            smiles=self._state,
            target_seq=self.target_seq,
            device=self.device,
            off_target_seq=self.off_target_seq,
            admet_weight=hyp.admet_weight,
            binding_weight=hyp.binding_weight,
            synthetic_weight=hyp.synthetic_weight,
            selectivity_weight=hyp.selectivity_weight,
            admet_model=self.admet_model,
            binding_model=self.binding_model,
            sa_model=self.sa_model,
        )

        return result["reward"] * self.discount_factor ** (self.max_steps - self._counter)

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



    

        