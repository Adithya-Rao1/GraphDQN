import os
import threading
import time
from typing import Callable, Optional

import torch

import dqn.dqn_hyperparams as hyp
from dqn.all_envs import MultiObjectiveRewardEnv
from dqn.dqn_network import DKDQNAgent
from dqn.utils import create_graph
from ADMET.model import ADMETModel
from binding_module.binding_affinity.plapt import Plapt
from synthetic_accessibility.sa_score import SyntheticAccessibility
from reward.multi_objective import RewardConfig


class TrainingCancelled(Exception):
    pass


def train_dqn(
    reward_config: RewardConfig,
    target_seq: str,
    off_target_seq: Optional[str],
    init_mol: str,
    seed: int,
    num_episodes: int,
    max_steps: int,
    checkpoint_interval: int,
    run_id: str,
    checkpoint_root: str,
    progress_cb: Optional[Callable[[int, int, float], None]] = None,
    cancel_event: Optional[threading.Event] = None,
    resume_from_checkpoint: Optional[str] = None,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    agent = DKDQNAgent(output_dim=1, device=device)
    if resume_from_checkpoint:
        state_dict = torch.load(resume_from_checkpoint, map_location=device)
        agent.qn.load_state_dict(state_dict)
        agent.target_qn.load_state_dict(state_dict)

    admet_model = ADMETModel(device)
    binding_model = Plapt(device=str(device))
    sa_model = SyntheticAccessibility()

    checkpoint_dir = os.path.join(checkpoint_root, run_id)
    batch_losses = []
    episode_rewards = []
    eps_start = hyp.eps_threshold
    eps_end = 0.1
    tau_start = 2.0
    tau_end = 0.1
    start_time = time.time()

    for episode in range(num_episodes):
        if cancel_event is not None and cancel_event.is_set():
            raise TrainingCancelled(f"Training cancelled at episode {episode}/{num_episodes}")

        progress = episode / num_episodes
        eps_threshold = max(eps_end, eps_start - progress * (eps_start - eps_end))
        tau = max(tau_end, tau_start - progress * (tau_start - tau_end))

        environment = MultiObjectiveRewardEnv(
            discount_factor=hyp.discount_factor,
            device=device,
            init_mol=init_mol,
            max_steps=max_steps,
            target_seq=target_seq,
            off_target_seq=off_target_seq,
            admet_model=admet_model,
            binding_model=binding_model,
            sa_model=sa_model,
            reward_config=reward_config,
        )
        environment.initialize()

        final_reward = 0.0
        for _step in range(max_steps):
            all_actions = list(environment.get_valid_actions())
            obs = create_graph(all_actions)
            chosen_act = agent.get_action(obs, eps_threshold, tau)
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

        if len(agent.replay_buffer) >= hyp.batch_size and episode % hyp.update_interval == 0:
            update_target = episode % hyp.target_update_interval == 0
            loss = agent.update_params(hyp.batch_size, hyp.gamma, hyp.polyak, update_target=update_target)
            batch_losses.append(loss.item())

        if checkpoint_interval and episode % checkpoint_interval == 0 and episode > 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(agent.qn.state_dict(), os.path.join(checkpoint_dir, f"episode_{episode}.pt"))

        if progress_cb is not None:
            progress_cb(episode + 1, num_episodes, final_reward)

    os.makedirs(checkpoint_dir, exist_ok=True)
    final_checkpoint_path = os.path.join(checkpoint_dir, "final.pt")
    torch.save(agent.qn.state_dict(), final_checkpoint_path)

    return {
        "run_id": run_id,
        "algorithm": "dqn",
        "seed": seed,
        "num_episodes": num_episodes,
        "final_reward": episode_rewards[-1] if episode_rewards else None,
        "mean_reward_last_10": (
            sum(episode_rewards[-10:]) / len(episode_rewards[-10:]) if episode_rewards else None
        ),
        "mean_loss": (sum(batch_losses) / len(batch_losses)) if batch_losses else None,
        "wall_clock_seconds": time.time() - start_time,
        "checkpoint_dir": checkpoint_dir,
        "final_checkpoint_path": final_checkpoint_path,
    }
