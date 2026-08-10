import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from ray import tune
from ray.tune.search.optuna import OptunaSearch
import wandb

from ppo.ppo_agent import PPO, PPOMemory

def init_wandb(config):
    return wandb.init(
        entity=os.environ.get("WANDB_ENTITY"),
        project=os.environ.get("WANDB_PROJECT", "graphdqn"),
        config=config,
    )

def objective(config):
    run = init_wandb(config)

    model = PPO(
        state_dim=config['state_dim'],
        action_dim=config['action_dim'],
        max_action=config['max_action'],
        target_seq=config['target_seq'],
        off_target_seq=config.get('off_target_seq'),
    )
    memory = PPOMemory()

    actor_optimizer = optim.Adam(model.actor.parameters(), lr=config['actor_lr'])
    critic_optimizer = optim.Adam(model.critic.parameters(), lr=config['critic_lr'])

    for iteration in range(config['num_iterations']):
        episode_rewards = []
        episode_reward = 0.0

        for _ in range(config['rollout_length']):
            action, state, logprob, value, reward, done = model.rollout()
            memory.add_memory(action, state, logprob, reward, value, done)

            episode_reward += reward
            if done:
                episode_rewards.append(episode_reward)
                episode_reward = 0.0

        actions, states, logprobs, rewards, values, dones = memory.get_memory()
        values_for_gae = values + [model.bootstrap_value()]

        advantages = model.calculate_gae(rewards, config['gamma'], values_for_gae, dones, config['lambda_']).detach()
        returns = model.get_returns(advantages, torch.stack(values)).detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-12)

        dataset = list(zip(torch.stack(states), torch.stack(actions), torch.stack(logprobs), returns, advantages))
        trainloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

        for epoch in range(config['num_epochs']):
            for batch_states, batch_actions, batch_logprobs, batch_returns, batch_advantages in trainloader:
                V, curr_log_probs = model.evaluate(batch_states, batch_actions)

                approx_kl = (batch_logprobs.detach() - curr_log_probs).mean().item()
                ratios = torch.exp(curr_log_probs - batch_logprobs.detach())
                clip_fraction = ((ratios > (1 + config['epsilon'])) | (ratios < (1 - config['epsilon']))).float().mean().item()

                tune.report({'approx_kl': approx_kl, 'clip_fraction': clip_fraction})
                run.log({'approx_kl': approx_kl, 'clip_fraction': clip_fraction})

                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - config['epsilon'], 1 + config['epsilon']) * batch_advantages

                policy_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.MSELoss()(V, batch_returns)

                dist = model.get_action_dist(batch_states)
                entropy = dist.entropy().mean()

                actor_loss = policy_loss + config['c1'] * critic_loss - config['c2'] * entropy

                actor_optimizer.zero_grad()
                critic_optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
                critic_loss.backward()
                actor_optimizer.step()
                critic_optimizer.step()

        memory.clear_memory()

        mean_episode_reward = sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0.0
        tune.report({'episode_reward': mean_episode_reward})
        run.log({'episode_reward': mean_episode_reward})

    run.finish()

def build_search_space(target_seq, off_target_seq=None, state_dim=None, action_dim=None, max_action=1.0):
    if state_dim is None:
        from ppo.mol_graph import GraphDataset
        state_dim = GraphDataset().node_feature_dim
    if action_dim is None:
        import ppo.ppo_hyperparams as hp
        action_dim = hp.max_actions

    return {
        'target_seq': target_seq,
        'off_target_seq': off_target_seq,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'max_action': max_action,
        'num_iterations': tune.grid_search([10, 25, 50]),
        'rollout_length': tune.grid_search([2500, 5000, 10000]),
        'batch_size': tune.grid_search([8, 16, 32, 64]),
        'num_epochs': tune.grid_search([10, 25, 50]),
        'gamma': tune.uniform(0.9, 0.999),
        'lambda_': tune.uniform(0.9, 0.999),
        'epsilon': tune.uniform(0.01, 0.5),
        'c1': tune.uniform(0.01, 0.5),
        'c2': tune.uniform(0.01, 0.5),
        'actor_lr': tune.loguniform(1e-5, 1e-3),
        'critic_lr': tune.loguniform(1e-5, 1e-3),
    }


def stop_fn(trial_id: str, result: dict) -> bool:
    """Stop a trial once its approx KL divergence exceeds the target -- signals the
    policy update stepped too far for the current clip range."""
    target_kl = 0.01
    return result.get('approx_kl', 0.0) > target_kl


def build_tuner(target_seq, off_target_seq=None):
    search_space = build_search_space(target_seq, off_target_seq=off_target_seq)
    alg = OptunaSearch(mode='max', metric='episode_reward')

    return tune.Tuner(
        objective,
        param_space=search_space,
        tune_config=tune.TuneConfig(search_alg=alg),
        run_config=tune.RunConfig(
            name="PPO_Reward_Based_Tuning",
            stop=stop_fn,
        ),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--target-seq', type=str, required=True)
    parser.add_argument('--off-target-seq', type=str, default=None)
    args = parser.parse_args()

    tuner = build_tuner(args.target_seq, off_target_seq=args.off_target_seq)
    results = tuner.fit()
    best_result = results.get_best_result(metric='episode_reward', mode='max')
    print(f"Best config: {best_result.config}")
