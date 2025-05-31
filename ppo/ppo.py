from actor import Actor
from critic import Critic
from ppo_env import MoleculeEnv
from mol_graph import GraphDataset
import torch
import torch.nn as nn
from torch.distributions import Categorical

class PPO:
    def __init__(self, state_dim, action_dim, max_action):
        super(PPO, self).__init__()
        # Need state_dim, action_dim, max_action
        self.actor = Actor(state_dim, action_dim, max_action)
        self.critic = Critic(state_dim)
        self.env = MoleculeEnv("placeholder_target_seq")
        self.memory = PPOMemory()

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
    
    def calculate_advantage_estimate(self, returns:torch.tensor, value:torch.tensor):
        """
        Args:
            - returns: batch of returns for each time step in trajectory (1D vector --> sequence of returns over episode)
            - value: batch of values for each time step in trajectory (1D vector --> sequence of values over episode)

        Returns:
            - advantages: batch of advantages for each time step in trajectory (1D vector --> sequence of advantages over episode)
        """

        advantages = returns - value

        return advantages
    
    def get_action(self, obs):
        """
        Queries an action from the actor network, should be called from rollout.

        Parameters:
            obs - the observation at the current timestep

        Return:
            action - the action to take, as a numpy array
            log_prob - the log probability of the selected action in the distribution
        """
        mean = self.actor(obs)

        dist = Categorical(logits=mean)

        action = dist.sample()

        log_prob = dist.log_prob(action)

        return action.detach().numpy(), log_prob.detach()
    
    def rollot(self):
        """
        Init environment
        1. Get an observation, get an action, get a value, get a log prob
        2. Take a step to get the next observation, reward, and done value
        3. Store the observation, action, reward, value, logprob, done
        """
        init_obs = self.env.reset()
        init_action, init_logprob = self.get_action(init_obs)
        init_value = self.critic(init_obs)

        obs, reward, done, info = self.env.step(init_action)  

        self.memory.add_memory(init_action, init_obs, init_logprob, reward, init_value, done) 

    def learn(self):
        pass

class PPOMemory:
    def __init__(self):
        super(PPOMemory, self).__init__()

        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.values[:]
        del self.dones[:]

        torch.cuda.empty_cache()

    def get_memory(self):
        return self.actions, self.states, self.logprobs, self.rewards, self.values, self.dones
    
    def add_memory(self, action, state, logprob, reward, value, done):
        self.actions.append(action)
        self.states.append(state)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

