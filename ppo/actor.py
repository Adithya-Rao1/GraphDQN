import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch_geometric as pyg
from mol_graph import GraphDataset
from rdkit import Chem

class Actor(nn.Module):
    """
    Actor Module.

    Init Args:
        - state_dim: dimension of state vector
        - action_dim: dimension of action vector
        - max_action: max action value

    Forward Args:
        - obs: pytorch_geometric object that represents the observed mol at a given timestep

    Returns:
        - probability distribution over actions
    """
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        
        self.max_action = max_action

    def forward(self, state):
        if isinstance(state, Chem.Mol):
            state = GraphDataset().mol_to_pyg(state)

        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = torch.tanh(self.l3(a)) * self.max_action
        
        return a