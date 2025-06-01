import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from ppo_main import PPO, PPOMemory
from ppo_env import MoleculeEnv
from arguments import parse_args

"""
TODO:
- Create different runs with different parameters
- Change model architectures
"""

def train_ppo(num_iterations,
              num_epochs, 
              rollout_length,
              batch_size,
              gamma,
              lambda_,
              epsilon,
              c1,
              c2,
              env,
              model,
              memory,
              actor_optimizer,
              critic_optimizer):
    
    for iteration in range(num_iterations):
        env.reset()

        while env.step_count < rollout_length:
            actions = []
            states = []
            logprobs = []
            rewards = []
            values = []
            dones = []

            for step in range(rollout_length):
                action, state, logprob, value, reward, done = model.rollout()
                
                env.current_reward = reward
                env.step_count += 1

                if done:
                    env.episode += 1
                    env.reset()

                actions.append(action)
                states.append(state)
                logprobs.append(logprob)
                rewards.append(reward)
                values.append(value)
                dones.append(done)

            memory.add_memory(action, state, logprob, reward, value, done)

            advantages = (model.calculate_gae(rewards, gamma, values, dones, lambda_)).detach()
            returns = (model.get_returns(advantages, values)).detach()

            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-12)

            data = memory.get_memory()
            trainloader = DataLoader(data, batch_size=batch_size, shuffle=True)

            for epoch in range(num_epochs):
                for data_batch in trainloader:
                    states, actions, logprobs, returns, advantages = data_batch

                    V, curr_log_probs = model.evaluate(states, actions)
                    ratios = torch.exp(curr_log_probs - logprobs.detach())

                    surr1 = ratios * advantages
                    surr2 = torch.clamp(ratios, 1 - epsilon, 1 + epsilon) * advantages

                    policy_loss = -torch.min(surr1, surr2)
                    critic_loss = nn.MSELoss()(V, returns)

                    # entropy = -torch.sum(curr_log_probs.exp() * curr_log_probs)
                    dist = model.get_action_dist(states)
                    entropy = dist.entropy().mean()

                    actor_loss = policy_loss + c1 * critic_loss - c2 * entropy

                    actor_optimizer.zero_grad()
                    critic_optimizer.zero_grad()
                    actor_loss.backward(retain_graph=True)
                    critic_loss.backward()
                    actor_optimizer.step()
                    critic_optimizer.step()
            
            memory.clear_memory()

if __name__ == "__main__":
    args = parse_args()

    env = MoleculeEnv()
    model = PPO()
    memory = PPOMemory()
    actor_optimizer = optim.Adam(model.actor.parameters(), lr=1e-4)
    critic_optimizer = optim.Adam(model.critic.parameters(), lr=1e-3)

    trainloader = DataLoader(memory.get_memory(), batch_size=64, shuffle=True)

    train_ppo(args.num_iterations,
              args.num_epochs, 
              args.rollout_length,
              args.batch_size,
              args.gamma,
              args.lambda_,
              args.epsilon,
              args.c1,
              args.c2,
              env,
              model,
              memory,
              actor_optimizer,
              critic_optimizer)

