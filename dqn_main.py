import torch
from dqn_network import DKDQNAgent
from dqn_env import MoleculeEnv, get_all_actions
from all_envs import QEDEnv
import dqn_hyperparams as dqn_hyperparams
import math
import numpy as np
from utils import create_graph, setup_dqn_logger
import rdkit
from rdkit import Chem

def run_dqn(log=False):
    num_episodes = 1000
    update_interval = 20
    batch_size = 20
    num_updates_per_iter = 1
    logger = setup_dqn_logger()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    target_seq = 'MDVFMKGLSKAKEGVVAAAEKTKQGVAEAAGKTKEGVLYVGSKTKEGVVHGVATVAEKTKEQVTNVGGAVVTGVTAVAQKTVEGAGSIAAATGFVKKDQLGKNEEGAPQEGILEDMPVDPDNEAYEMPSEEGYQDYEPEA'
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
    #print('environment valid actions: ', environment._valid_actions)

    eps_threshold = 0.8
    batch_losses = []

    for episode in range(num_episodes):
        all_actions = list(environment.get_valid_actions())
        print("#"*100)
        print('')
        print()
        print('all actions initial ', all_actions)
        print('environment state before step ', environment._state)
        
        obs = create_graph(all_actions) 

        # print('observations in graphs: ', obs)

        chosen_act = agent.get_action(obs, eps_threshold)

        action_obs = all_actions[chosen_act]
        print('action_obs', action_obs)
        result = environment.step(action_obs)

        _, reward, done = result
        print('environment state after step ', environment._state)
        all_action_obs = list(environment.get_valid_actions())
        print('all actions after step: ', all_action_obs)
        print('reward ', reward)

        if episode == 1:
            break

        agent.replay_buffer.add(
            obs_t=action_obs,
            action=0,
            reward=reward,
            obs_tp1=all_action_obs,
            done=float(result.terminated)
        )

        print('Agent replay buffer length: ', agent.replay_buffer.__len__())

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


if __name__ == "__main__":
    run_dqn()