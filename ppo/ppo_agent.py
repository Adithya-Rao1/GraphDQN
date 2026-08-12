import os

import torch
import torch.nn as nn
from torch.distributions import Categorical

from ppo.actor import Actor
from ppo.critic import Critic
from ppo.ppo_env import MoleculeEnv
import ppo.ppo_hyperparams as hp


class PPO:
    def __init__(self, state_dim, action_dim, max_action, target_seq, off_target_seq=None,
                 device=None, base_mols=None):
        super(PPO, self).__init__()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.actor = Actor(state_dim, action_dim, max_action, device=self.device)
        self.critic = Critic(state_dim, device=self.device)
        self.env = MoleculeEnv(target_seq, off_target_seq=off_target_seq, device=self.device, base_mols=base_mols)

        self._current_obs = None

    def calculate_returns(self, rewards: torch.tensor, gamma: float) -> torch.tensor:
        returns = []
        return_ = 0

        for reward in reversed(rewards):
            return_ = reward + gamma * return_
            returns.insert(0, return_)

        return torch.tensor(returns, dtype=torch.float, device=self.device)

    def calculate_advantage_estimate(self, returns: torch.tensor, value: torch.tensor):
        return returns - value

    def calculate_gae(self, rewards, gamma: float, values, dones, lambda_: float):
        advantages = []
        advantage_ = 0.0

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * float(values[t + 1]) * (1 - dones[t]) - float(values[t])
            advantage_ = delta + gamma * lambda_ * (1 - dones[t]) * advantage_
            advantages.insert(0, advantage_)

        return torch.tensor(advantages, dtype=torch.float, device=self.device)

    def get_returns(self, advantages: torch.tensor, values: torch.tensor):
        return advantages + values

    def get_action(self, obs):
        mean = self.actor(obs)

        dist = Categorical(logits=mean)

        action = dist.sample()

        log_prob = dist.log_prob(action)

        return action, log_prob.detach()

    def evaluate(self, obs, actions):
        values = self.critic(obs).squeeze(-1)

        mean = self.actor(obs)
        dist = Categorical(logits=mean)
        log_probs = dist.log_prob(actions)

        return values, log_probs

    def get_action_dist(self, obs):
        mean = self.actor(obs)

        return Categorical(logits=mean)

    def rollout(self):
        if self._current_obs is None:
            self._current_obs, _ = self.env.reset()

        obs = self._current_obs
        action, logprob = self.get_action(obs)
        value = self.critic(obs).squeeze(-1)

        next_obs, reward, terminated, truncated, _ = self.env.step(action.item())
        done = terminated or truncated

        self._current_obs = None if done else next_obs

        return action, obs, logprob, value.detach(), reward, done

    def bootstrap_value(self):
        if self._current_obs is None:
            self._current_obs, _ = self.env.reset()
        with torch.no_grad():
            return self.critic(self._current_obs).squeeze(-1).detach()

    def save_checkpoint(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])


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

        if torch.cuda.is_available():
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
