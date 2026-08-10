import torch
import random
import numpy as np
import rdkit
from rdkit import Chem
from dqn.all_envs import MultiObjectiveRewardEnv
from dqn.dqn_network import DKDQNAgent
from dqn.utils import create_graph, setup_dqn_logger, track, penalized_logp
import dqn.dqn_hyperparams as hyp
from all_experiments import smiles_800

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_dqn1(
        device=device,
        logger=setup_dqn_logger(),
        log=False
        ):
    agent = DKDQNAgent(
        output_dim = 15,
        device = device
    )

    batch_losses = []
    environment = MultiObjectiveRewardEnv(
            discount_factor = hyp.discount_factor,
            device = device,
            init_mol = hyp.start_molecule,
            max_steps = hyp.max_steps
        )

    environment.initialize()

    for episode in range(hyp.num_episodes):
        for step in range(hyp.max_steps):
            all_actions = list(environment.get_valid_actions())
            obs = create_graph(all_actions) 
            chosen_act = agent.get_action(obs, hyp.eps_threshold)
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
    
    return agent

def run_dqn2(
        start_mols,
        device=device,
        num_mols=800,
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
                start_mol = start_mol[i]
                environment = MultiObjectiveRewardEnv(
                    discount_factor = hyp.discount_factor,
                    init_mol = start_mol,
                    target_molecule = start_mol,
                    max_steps = hyp.max_steps,
                )

                environment.initialize()

                all_actions = list(environment.get_valid_actions())
                obs = create_graph(all_actions) 
                chosen_act = agent.get_action(obs, hyp.eps_threshold)
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

                    environment.initialize()
            
            if episode % hyp.update_interval == 0:
                hyp.eps_threshold = max(0.1, 1.0 - (episode / hyp.num_episodes) * (0.9))
                for _ in range(hyp.num_updates_per_iter):
                    loss = agent.update_params(hyp.batch_size, hyp.gamma, hyp.polyak)
                    loss = loss.item()
                    batch_losses.append(loss)

        print(f"[Episode {episode+1} complete.]")
    
    return agent

if __name__ == "__main__":
    all_obs1 = run_dqn1()
    states1 = []
    rewards1 = []
    
    for obs in all_obs1.replay_buffer._storage:
        states1.append([obs[0], Chem.QED.qed(Chem.MolFromSmiles(obs[0]))])
        rewards1.append(obs[2])

    track(states1, 'qed_states')
    track(rewards1, 'qed_rewards')

    all_obs2 = run_dqn2(smiles_800.start_mols)
    states2 = []
    rewards2 = []

    for obs in all_obs2:
        states1.append([obs[0], penalized_logp(Chem.MolFromSmiles(obs[0]))])
        rewards2.append(obs[2])

    track(states2, 'qedconstrained_states', sort_by_values="low")
    track(rewards2, 'qedconstrained_rewards')
    