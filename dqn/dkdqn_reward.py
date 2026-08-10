from __future__ import absolute_import

import torch

from ADMET.model import ADMETModel
from binding_module.binding_affinity.plapt import Plapt
from synthetic_accessibility.sa_score import SyntheticAccessibility
from reward.multi_objective import compute_reward

class Reward(object):
    def __init__(self, init_reward, device=None):
        super(Reward, self).__init__()

        self.init_reward = init_reward
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.admet_model = ADMETModel(self.device)
        self.binding_model = Plapt(device=str(self.device))
        self.sa_model = SyntheticAccessibility()

    def calculate_reward(self, smiles, target_seq, off_target_seq=None):
        if isinstance(smiles, str):
            smiles = [smiles]

        return [
            compute_reward(
                smiles=smile,
                target_seq=target_seq,
                device=self.device,
                off_target_seq=off_target_seq,
                admet_model=self.admet_model,
                binding_model=self.binding_model,
                sa_model=self.sa_model,
            )["reward"]
            for smile in smiles
        ]