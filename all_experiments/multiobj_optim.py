import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import torch
import random
import numpy as np
from all_envs import MultiObjectiveRewardEnv
from dqn_network import DKDQNAgent
from utils import create_graph, setup_dqn_logger
import dqn_hyperparams as hyp
from utils import track, calc_multi_obj_properties
import rdkit
from rdkit import Chem

device = 'cuda:3'
states = []
rewards = []

def run_dqn(
        start_mols,
        target_seq,
        off_target_seq=None,
        device=device,
        num_mols=5,
        logger=setup_dqn_logger(),
        log=False
        ):

    agent = DKDQNAgent(
        output_dim = 15,
        device = device
    )

    batch_losses = []

    for episode in range(hyp.num_episodes):
        for i in range(num_mols):
            for step in range(hyp.max_steps):
                start_mol = start_mols[i]
                environment = MultiObjectiveRewardEnv(
                    device = device,
                    discount_factor = hyp.discount_factor,
                    init_mol = start_mol,
                    max_steps = hyp.max_steps,
                    target_seq = target_seq,
                    off_target_seq = off_target_seq                    
                )

                environment.initialize()

                all_actions = list(environment.get_valid_actions())
                
                obs = create_graph(all_actions) 

                # print('observations in graphs: ', obs)

                chosen_act = agent.get_action(obs, hyp.eps_threshold)

                action_obs = all_actions[chosen_act]
                result = environment.step(action_obs)

                _, reward, done = result
                all_action_obs = list(environment.get_valid_actions())


                # if episode == 150:
                #     break

                agent.replay_buffer.add(
                    obs_t=action_obs,
                    action=0,
                    reward=reward,
                    obs_tp1=all_action_obs,
                    done=float(result.terminated)
                )

                if agent.replay_buffer.__len__() % hyp.batch_size == 0:
                    for obs in agent.replay_buffer._storage:
                        states.append([obs[0], calc_multi_obj_properties(obs[0], target_seq, device=agent.device)])
                        rewards.append(obs[2])
                    if agent.replay_buffer.__len__() == hyp.batch_size:
                        track(True, states, 'multiobj_states', output_dir='./multiobj_results')
                        track(True, rewards, 'multiobj_rewards', output_dir='./multiobj_results')
                    else:
                        track(False, states, 'multiobj_states', output_dir='./multiobj_results')
                        track(False, rewards, 'multiobj_rewards', output_dir='./multiobj_results')


                if done:
                    final_reward = reward

                    if episode != 0 and len(batch_losses) != 0:
                        logger.info(f"Episode {episode+1} Reward: {final_reward}")
                        logger.info(f"Episode {episode+1} Loss: {np.array(batch_losses).mean()}")
                    if episode != 0 and episode % 2 == 0 and len(batch_losses) != 0:
                        print(f"Reward of final molecule at episode {episode+1}: {final_reward}")
                        print(f"Average loss in episode {episode+1}: {np.array(batch_losses).mean()}")

                    environment.initialize()
                
                if (episode+1) % hyp.update_interval == 0:
                    hyp.eps_threshold = max(0.1, 1.0 - (episode / hyp.num_episodes) * (0.9))

                    if (episode+1) % hyp.target_update_interval == 0:
                        update_target = True
                    else:
                        update_target = False
                    for _ in range(hyp.num_updates_per_iter):
                        loss = agent.update_params(hyp.batch_size, hyp.gamma, hyp.polyak, update_target=update_target)
                        loss = loss.item()
                        batch_losses.append(loss)
    
    return agent, states, rewards

if __name__ == "__main__":
    smiles_path = '/home/ubuntu/metis-data-storage1/FactorVAE_Data/all.txt'
    with open(smiles_path, 'r') as f:
        smiles_list = f.read().splitlines()

    start_mols = random.sample(smiles_list, 1000)
    
    target_seq = 'MDVFMKGLSKAKEGVVAAAEKTKQGVAEAAGKTKEGVLYVGSKTKEGVVHGVATVAEKTKEQVTNVGGAVVTGVTAVAQKTVEGAGSIAAATGFVKKDQLGKNEEGAPQEGILEDMPVDPDNEAYEMPSEEGYQDYEPEA'
    agent, states, rewards = run_dqn(start_mols, target_seq)

    track(states, 'multiobj_states')
    track(rewards, 'multiobj_rewards')