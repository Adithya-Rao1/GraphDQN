import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import ppo.ppo_hyperparams as hp
from ppo.ppo_agent import PPO, PPOMemory
from ppo.mol_graph import GraphDataset
from experiments.data.targets import TARGETS, DEFAULT_TARGET
from experiments.data.starting_molecules import sample_pilot_molecules


def run(target_name=DEFAULT_TARGET, seed=0, num_molecules=30, num_iterations=20,
        rollout_length=64, num_epochs=4, batch_size=16, checkpoint_interval=5,
        run_id=None, use_wandb=False, checkpoint_root='./checkpoints/ppo',
        results_root='./experiments/results'):
    run_id = run_id or f"ppo_{target_name}_seed{seed}_{time.strftime('%Y%m%d-%H%M%S')}"
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    target_seq = TARGETS[target_name]
    base_mols = sample_pilot_molecules(n=num_molecules, seed=seed)
    state_dim = GraphDataset().node_feature_dim

    model = PPO(state_dim=state_dim, action_dim=hp.max_actions, max_action=1.0,
                target_seq=target_seq, device=device, base_mols=base_mols)
    memory = PPOMemory()
    actor_optimizer = optim.Adam(model.actor.parameters(), lr=1e-4)
    critic_optimizer = optim.Adam(model.critic.parameters(), lr=1e-3)

    wandb_run = None
    if use_wandb:
        import wandb
        wandb_run = wandb.init(
            entity=os.environ.get("WANDB_ENTITY"),
            project=os.environ.get("WANDB_PROJECT", "graphdqn"),
            name=run_id,
            config={"algorithm": "ppo", "target": target_name, "seed": seed,
                    "num_molecules": num_molecules, "num_iterations": num_iterations,
                    "rollout_length": rollout_length},
            tags=["ppo", target_name, f"seed{seed}"],
        )

    checkpoint_dir = os.path.join(checkpoint_root, run_id)
    start_time = time.time()
    iteration_rewards = []

    for iteration in range(num_iterations):
        for _ in range(rollout_length):
            action, state, logprob, value, reward, done = model.rollout()
            memory.add_memory(action, state, logprob, reward, value, done)

        actions, states, logprobs, rewards, values, dones = memory.get_memory()
        values_for_gae = values + [model.bootstrap_value()]
        advantages = model.calculate_gae(rewards, hp.gamma, values_for_gae, dones, hp.lambda_).detach()
        returns = model.get_returns(advantages, torch.stack(values)).detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-12)

        dataset = list(zip(torch.stack(states), torch.stack(actions), torch.stack(logprobs), returns, advantages))
        trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        last_actor_loss, last_critic_loss = None, None
        for epoch in range(num_epochs):
            for batch_states, batch_actions, batch_logprobs, batch_returns, batch_advantages in trainloader:
                V, curr_log_probs = model.evaluate(batch_states, batch_actions)
                ratios = torch.exp(curr_log_probs - batch_logprobs.detach())
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - hp.epsilon, 1 + hp.epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.MSELoss()(V, batch_returns)
                dist = model.get_action_dist(batch_states)
                entropy = dist.entropy().mean()
                actor_loss = policy_loss + hp.c1 * critic_loss - hp.c2 * entropy

                actor_optimizer.zero_grad()
                critic_optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
                critic_loss.backward()
                actor_optimizer.step()
                critic_optimizer.step()
                last_actor_loss, last_critic_loss = actor_loss.item(), critic_loss.item()

        mean_reward = sum(rewards) / len(rewards)
        iteration_rewards.append(mean_reward)
        memory.clear_memory()

        if wandb_run:
            wandb_run.log({"iteration": iteration, "mean_reward": mean_reward,
                            "actor_loss": last_actor_loss, "critic_loss": last_critic_loss})

        if checkpoint_interval and iteration % checkpoint_interval == 0 and iteration > 0:
            model.save_checkpoint(os.path.join(checkpoint_dir, f"iteration_{iteration}.pt"))

        print(f"[{run_id}] iteration {iteration} mean_reward={mean_reward:.4f}")

    model.save_checkpoint(os.path.join(checkpoint_dir, "final.pt"))

    summary = {
        "run_id": run_id,
        "algorithm": "ppo",
        "target": target_name,
        "seed": seed,
        "num_molecules": num_molecules,
        "num_iterations": num_iterations,
        "final_reward": iteration_rewards[-1] if iteration_rewards else None,
        "mean_reward_last_5": (sum(iteration_rewards[-5:]) / len(iteration_rewards[-5:])) if iteration_rewards else None,
        "wall_clock_seconds": time.time() - start_time,
        "checkpoint_dir": checkpoint_dir,
    }

    os.makedirs(results_root, exist_ok=True)
    with open(os.path.join(results_root, f"{run_id}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    if wandb_run:
        wandb_run.log(summary)
        wandb_run.finish()

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-name", default=DEFAULT_TARGET, choices=list(TARGETS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-molecules", type=int, default=30)
    parser.add_argument("--num-iterations", type=int, default=20)
    parser.add_argument("--rollout-length", type=int, default=64)
    parser.add_argument("--num-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--checkpoint-interval", type=int, default=5)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    result_summary = run(
        target_name=args.target_name,
        seed=args.seed,
        num_molecules=args.num_molecules,
        num_iterations=args.num_iterations,
        rollout_length=args.rollout_length,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        checkpoint_interval=args.checkpoint_interval,
        run_id=args.run_id,
        use_wandb=args.wandb,
    )
    print(json.dumps(result_summary, indent=2))
