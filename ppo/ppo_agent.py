import torch
from torch.distributions import MultivariateNormal
import torch.nn as nn
from actor import Actor
from critic import Critic
from torch.optim import Adam
import numpy as np

class PPOAgent:
    def __init__(self, env):
        super(PPOAgent, self).__init__()

        self._init_hyperparameters()

        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        self.max_actions = env.action_space.high

        self.actor = Actor(self.obs_dim, self.act_dim, self.max_actions)
        self.critic = Critic(self.obs_dim, self.act_dim, self.max_actions)
        
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)
    
    def learn(self, total_timesteps):
        t = 0
        while t < total_timesteps:
            batch_obs, batch_acts, batch_logprobs, batch_rtgs, batch_lens = self.rollout()

            V, _ = self.evaluate(batch_obs)

            A_k = batch_rtgs - V.detach()
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            for _ in range(self.updates_per_iter):
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)

                ratios = torch.exp(curr_log_probs - batch_logprobs)

                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                actor_loss = (-torch.min(surr1, surr2)).mean()
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                critic_loss = nn.MSELoss()(V, batch_rtgs)
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()
                
            t += np.sum(batch_lens)

    def _init_hyperparameters(self):
        self.timesteps_per_batch = 4800
        self.max_timesteps_per_episode = 1600
        self.gamma = 0.95
        self.updates_per_iter = 5
        self.clip = 0.2
        self.lr = 0.005

    def get_actions(self, obs):
        mean = self.actor(obs)

        dist = MultivariateNormal(mean, self.cov_mat)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.detach().numpy(), log_prob.detach()
    
    def compute_rtgs(self, batch_rews):
        batch_rtgs = []

        for ep_rews in reversed(batch_rews):
            discounted_reward = 0

            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)
            
        batch_rtgs = torch.tensor(batch_rtgs, torch.float)

        return batch_rtgs
    
    def evaluate(self, batch_obs, batch_acts):
        V = self.critic(batch_obs).squeeze()

        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)

        return V, log_probs

    def rollout(self):
        batch_obs = []
        batch_acts = []
        batch_logprobs = []
        batch_rews = []
        batch_rtgs = []
        batch_lens = []

        obs = self.env.reset()
        done = False

        for ep_t in range(self.max_timesteps_per_episode):
            action = self.env.action_space.sample()
            obs, rew, done, _ = self

            if done:
                break

        t = 0
        while t < self.timesteps_per_batch:
            ep_rews = []

            obs = self.env.reset()
            done = False

            for ep_t in range(self.max_timesteps_per_episode):
                t += 1

                batch_obs.append(obs)

                action, log_prob = self.get_action(obs)
                obs, rew, done, _ = self.env.step(action)

                ep_rews.append(rew)
                batch_acts.append(action)
                batch_logprobs.append(log_prob)

                if done:
                    break
            
            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)
        
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_logprobs = torch.tensor(batch_logprobs, dtype=torch.float)

        batch_rtgs = self.compute_rtgs(batch_rews)

        return batch_obs, batch_acts, batch_logprobs, batch_rtgs, batch_lens

