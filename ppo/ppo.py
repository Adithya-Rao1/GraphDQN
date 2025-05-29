from actor import Actor
from critic import Critic
from mol_graph import GraphDataset
import torch
import torch.nn as nn

class PPO:
    def __init__(self):
        super(PPO, self).__init__()
        self.actor = Actor
        self.critic = Critic

    def calculate_returns(self, rewards:torch.tensor, gamma:float) -> torch.tensor:
        """
        Args:
            - rewards: batch of rewards for each time step in trajectory (1D vector --> sequence of rewards over episode)
            - gamma: discount factor (scalar)

        Returns:
            - returns: standardized batch of returns for reward at each time step in trajectory (vector)
        
        Formula:
        G_t = r_t + gamma*G_(t+1)
        """

        returns = []
        return_ = 0

        for reward in torch.flip(rewards, (1,)):
            return_ = reward + gamma*return_
            returns.insert(0, return_)

        return returns
    
    def calculate_advantage_estimate(self):
        pass


