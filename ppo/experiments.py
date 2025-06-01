import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from ppo_main import PPO, PPOMemory
from ppo_env import MoleculeEnv
from arguments import parse_args
from ray import tune
from ray.tune.search.optuna import OptunaSearch
import wandb

model = PPO()
memory = PPOMemory()
env = MoleculeEnv()

def init_wanbd(config):
    run = wandb.init(
        entity = "adithya_rao-private",
        project = "MolGRL",
        config=config
    )

    return run

def objective(config):
    run = init_wanbd(config)
    actor_optimizer = optim.Adam(model.actor.parameters(), lr=config['actor_lr'])
    critic_optimizer = optim.Adam(model.critic.parameters(), lr=config['critic_lr'])

    for iteration in range(config['num_iterations']):
        env.reset()

        while env.total_steps < config['rollout_length']:
            reward_tracker = 0

            for step in range(config['rollout_length']):
                action, state, logprob, value, reward, done = model.rollout()

                reward_tracker += reward
                env.current_reward = reward
                env.total_steps += 1
                env.step_count += 1

                memory.add_memory(action, state, logprob, reward, value, done)

                if done:
                    env.episode += 1
                    env.episode_reward = reward_tracker/env.step_count
                    env.episode_lengths.append(env.step_count)
                    tune.report({'episode_reward': env.episode_reward})
                    run.log({'episode_reward': env.episode_reward, 'episode_length': env.step_count})
                    env.reset()
            
            data = [memory.get_memory()]

            advantages = model.calculate_gae(rewards=data[3], gamma=config['gamma'], values=data[4], dones=data[5], lambda_=config['lambda_'])
            returns = model.get_returns(advantages, values=data[4])

            advantages = (advantages - advantages.mean())/(advantages.std() + 1e-12)

            trainloader = DataLoader(data, batch_size=config['batch_size'], shuffle=True)

            for epoch in range(config['num_epochs']):
                for data_batch in trainloader:
                    states, actions, logprobs, returns, advantages = data_batch

                    V, curr_log_probs = model.evaluate(states, actions)

                    approx_kl = (logprobs.detach() - curr_log_probs).mean().item()
                    ratios = torch.exp(curr_log_probs - logprobs.detach())
                    clip_fraction = (ratios > (1 + config['epsilon'])).float() | (ratios < (1 - config['epsilon'])).float()
                    clip_fraction = clip_fraction.float().mean().item()

                    # if approx_kl > 1.5 * config['target_kl']:
                    #     break
                    
                    tune.report({'approx_kl': approx_kl, 'clip_fraction': clip_fraction})
                    run.log({'approx_kl': approx_kl, 'clip_fraction': clip_fraction})

                    surr1 = ratios * advantages
                    surr2 = torch.clamp(ratios, 1 - config['epsilon'], 1 + config['epsilon']) * advantages

                    policy_loss = -torch.min(surr1, surr2)
                    critic_loss = nn.MSELoss()(V, returns)

                    # entropy = -torch.sum(curr_log_probs.exp() * curr_log_probs)
                    dist = model.get_action_dist(states)
                    entropy = dist.entropy().mean()

                    actor_loss = policy_loss + config['c1'] * critic_loss - config['c2'] * entropy

                    actor_optimizer.zero_grad()
                    critic_optimizer.zero_grad()
                    actor_loss.backward(retain_graph=True)
                    critic_loss.backward()
                    actor_optimizer.step()
                    critic_optimizer.step()

            memory.clear_memory()

"""
Log uniform: Geometric series of values (0.001, 0.01, 0.1)
Uniform: Arithmetic series of values (0.1, 0.2, 0.3)
"""
search_space = {
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
    'critic_lr': tune.loguniform(1e-5, 1e-3)
}
alg = OptunaSearch(
    mode='max',
    metric='episode_reward'
)

def stop_fn(trial_id: str, result: dict) -> bool:
    """
    Reasons to stop the experiment:
        1. KL > target_kl
    """

    target_kl = 0.01
    return result['approx_kl'] > target_kl
    


tuner = tune.Tuner(
    objective,
    tune_config=tune.TuneConfig(
        search_alg=alg
    ),
    run_config=tune.RunConfig(
        name = "PPO_Reward_Based_Tuning",
        stop = stop_fn
    )
)



    