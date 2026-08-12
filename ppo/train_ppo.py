import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ppo.ppo_agent import PPO, PPOMemory
from ppo.arguments import parse_args

"""
TODO:
- Ablations
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
              model,
              memory,
              actor_optimizer,
              critic_optimizer):

    for iteration in range(num_iterations):
        for _ in range(rollout_length):
            action, state, logprob, value, reward, done = model.rollout()
            memory.add_memory(action, state, logprob, reward, value, done)

        actions, states, logprobs, rewards, values, dones = memory.get_memory()
        values_for_gae = values + [model.bootstrap_value()]

        advantages = model.calculate_gae(rewards, gamma, values_for_gae, dones, lambda_).detach()
        returns = model.get_returns(advantages, torch.stack(values)).detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-12)

        dataset = list(zip(torch.stack(states), torch.stack(actions), torch.stack(logprobs), returns, advantages))
        trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(num_epochs):
            for batch_states, batch_actions, batch_logprobs, batch_returns, batch_advantages in trainloader:
                V, curr_log_probs = model.evaluate(batch_states, batch_actions)
                ratios = torch.exp(curr_log_probs - batch_logprobs.detach())

                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - epsilon, 1 + epsilon) * batch_advantages

                policy_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.MSELoss()(V, batch_returns)

                dist = model.get_action_dist(batch_states)
                entropy = dist.entropy().mean()

                actor_loss = policy_loss + c1 * critic_loss - c2 * entropy

                actor_optimizer.zero_grad()
                critic_optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
                critic_loss.backward()
                actor_optimizer.step()
                critic_optimizer.step()

        memory.clear_memory()
        print(f"Iteration {iteration + 1}/{num_iterations} -- mean rollout reward: {sum(rewards) / len(rewards):.4f}")

if __name__ == "__main__":
    args = parse_args()

    model = PPO(
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        max_action=args.max_action,
        target_seq=args.target_seq,
        off_target_seq=args.off_target_seq,
    )
    memory = PPOMemory()
    actor_optimizer = optim.Adam(model.actor.parameters(), lr=1e-4)
    critic_optimizer = optim.Adam(model.critic.parameters(), lr=1e-3)

    train_ppo(args.num_iterations,
              args.num_epochs,
              args.rollout_length,
              args.batch_size,
              args.gamma,
              args.lambda_,
              args.epsilon,
              args.c1,
              args.c2,
              model,
              memory,
              actor_optimizer,
              critic_optimizer)
