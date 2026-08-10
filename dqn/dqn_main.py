import os

import torch
from dqn.dqn_network import DKDQNAgent
from dqn.dqn_env import MoleculeEnv, get_all_actions
from dqn.all_envs import QEDEnv
import dqn.dqn_hyperparams as dqn_hyperparams
import math
import numpy as np
from dqn.utils import create_graph, setup_dqn_logger
import rdkit
from rdkit import Chem
from experiments.data.targets import TARGETS, DEFAULT_TARGET

def run_dqn(log=False, target_name=DEFAULT_TARGET, checkpoint_dir='./checkpoints/dqn', checkpoint_interval=100):
    num_episodes = 1000
    update_interval = 20
    batch_size = 20
    num_updates_per_iter = 1
    logger = setup_dqn_logger()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    target_seq = TARGETS[target_name]
    off_target_seq = None

    environment = QEDEnv(
        discount_factor=dqn_hyperparams.discount_factor,
        init_mol = dqn_hyperparams.start_molecule,
        max_steps = dqn_hyperparams.max_steps,
        target_seq = target_seq,
        off_target_seq = off_target_seq
    )

    agent = DKDQNAgent(
        output_dim = 15,
        device = device
    )

    environment.initialize()
    eps_threshold = 0.8
    batch_losses = []

    for episode in range(num_episodes):
        all_actions = list(environment.get_valid_actions())
        obs = create_graph(all_actions) 
        chosen_act = agent.get_action(obs, eps_threshold)
        action_obs = all_actions[chosen_act]
        result = environment.step(action_obs)
        _, reward, done = result

        all_action_obs = list(environment.get_valid_actions())
        agent.replay_buffer.add(
            obs_t=action_obs,
            action=0,
            reward=reward,
            obs_tp1=all_action_obs,
            done=float(result.terminated)
        )

        if done:
            final_reward = reward

            if episode != 0 and len(batch_losses) != 0:
                logger.info(f"Episode {episode+1} Reward: {final_reward}")
                logger.info(f"Episode {episode+1} Loss: {np.array(batch_losses).mean()}")
            if episode != 0 and episode % 2 == 0 and len(batch_losses) != 0:
                print(f"Reward of final molecule at episode {episode+1}: {final_reward}")
                print(f"Average loss in episode {episode+1}: {np.array(batch_losses).mean()}")

            eps_threshold *= 0.99907
            environment.initialize()
        
        if episode % update_interval == 0 and agent.replay_buffer.__len__() >= batch_size:
            for _ in range(num_updates_per_iter):
                loss = agent.update_params(batch_size, dqn_hyperparams.gamma, dqn_hyperparams.polyak)
                loss = loss.item()
                batch_losses.append(loss)

        if checkpoint_dir and episode % checkpoint_interval == 0 and episode > 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(agent.qn.state_dict(), os.path.join(checkpoint_dir, f"episode_{episode}.pt"))

    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(agent.qn.state_dict(), os.path.join(checkpoint_dir, "final.pt"))

    return agent


if __name__ == "__main__":
    run_dqn()