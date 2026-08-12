import argparse
import json
import os
import time

import torch

import dqn.dqn_hyperparams as hyp
from dqn.all_envs import MultiObjectiveRewardEnv
from dqn.dqn_network import DKDQNAgent
from dqn.utils import create_graph, setup_dqn_logger
from experiments.data.targets import TARGETS, DEFAULT_TARGET
from experiments.data.starting_molecules import sample_pilot_molecules


def run(target_name=DEFAULT_TARGET, seed=0, num_molecules=30, num_episodes=200,
        checkpoint_interval=50, run_id=None, use_wandb=False,
        checkpoint_root='./checkpoints/dqn', results_root='./experiments/results'):
    run_id = run_id or f"dqn_{target_name}_seed{seed}_{time.strftime('%Y%m%d-%H%M%S')}"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = setup_dqn_logger()

    target_seq = TARGETS[target_name]
    start_mols = sample_pilot_molecules(n=num_molecules, seed=seed)

    agent = DKDQNAgent(output_dim=15, device=device)

    wandb_run = None
    if use_wandb:
        import wandb
        wandb_run = wandb.init(
            entity=os.environ.get("WANDB_ENTITY"),
            project=os.environ.get("WANDB_PROJECT", "graphdqn"),
            name=run_id,
            config={"algorithm": "dqn", "target": target_name, "seed": seed,
                    "num_molecules": num_molecules, "num_episodes": num_episodes},
            tags=["dqn", target_name, f"seed{seed}"],
        )

    checkpoint_dir = os.path.join(checkpoint_root, run_id)
    batch_losses = []
    episode_rewards = []
    eps_threshold = hyp.eps_threshold
    start_time = time.time()

    for episode in range(num_episodes):
        start_mol = start_mols[episode % len(start_mols)]
        environment = MultiObjectiveRewardEnv(
            discount_factor=hyp.discount_factor,
            device=device,
            init_mol=start_mol,
            max_steps=hyp.max_steps,
            target_seq=target_seq,
        )
        environment.initialize()

        final_reward = 0.0
        for step in range(hyp.max_steps):
            all_actions = list(environment.get_valid_actions())
            obs = create_graph(all_actions)
            chosen_act = agent.get_action(obs, eps_threshold)
            action_obs = all_actions[chosen_act]
            result = environment.step(action_obs)
            _, reward, done = result
            all_action_obs = list(environment.get_valid_actions())

            agent.replay_buffer.add(
                obs_t=action_obs, action=0, reward=reward,
                obs_tp1=all_action_obs, done=float(result.terminated),
            )
            final_reward = reward
            if done:
                break

        episode_rewards.append(final_reward)
        eps_threshold = max(0.1, eps_threshold * hyp.eps_decay_factor)

        if agent.replay_buffer.__len__() >= hyp.batch_size and episode % hyp.update_interval == 0:
            update_target = episode % hyp.target_update_interval == 0
            loss = agent.update_params(hyp.batch_size, hyp.gamma, hyp.polyak, update_target=update_target)
            batch_losses.append(loss.item())

        if wandb_run:
            wandb_run.log({"episode": episode, "reward": final_reward,
                            "loss": batch_losses[-1] if batch_losses else None})

        if checkpoint_interval and episode % checkpoint_interval == 0 and episode > 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(agent.qn.state_dict(), os.path.join(checkpoint_dir, f"episode_{episode}.pt"))

        if episode % 10 == 0:
            logger.info(f"[{run_id}] episode {episode} reward={final_reward:.4f}")

    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(agent.qn.state_dict(), os.path.join(checkpoint_dir, "final.pt"))

    summary = {
        "run_id": run_id,
        "algorithm": "dqn",
        "target": target_name,
        "seed": seed,
        "num_molecules": num_molecules,
        "num_episodes": num_episodes,
        "final_reward": episode_rewards[-1] if episode_rewards else None,
        "mean_reward_last_10": (sum(episode_rewards[-10:]) / len(episode_rewards[-10:])) if episode_rewards else None,
        "mean_loss": (sum(batch_losses) / len(batch_losses)) if batch_losses else None,
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
    parser.add_argument("--num-episodes", type=int, default=200)
    parser.add_argument("--checkpoint-interval", type=int, default=50)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    result_summary = run(
        target_name=args.target_name,
        seed=args.seed,
        num_molecules=args.num_molecules,
        num_episodes=args.num_episodes,
        checkpoint_interval=args.checkpoint_interval,
        run_id=args.run_id,
        use_wandb=args.wandb,
    )
    print(json.dumps(result_summary, indent=2))
