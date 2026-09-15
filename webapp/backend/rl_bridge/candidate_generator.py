import threading
from typing import List, Optional

import torch
from rdkit import Chem

from dqn.all_envs import MultiObjectiveRewardEnv
from dqn.dqn_network import DKDQNAgent
from dqn.utils import create_graph
import dqn.dqn_hyperparams as hyp
from ADMET.model import ADMETModel
from binding_module.binding_affinity.plapt import Plapt
from synthetic_accessibility.sa_score import SyntheticAccessibility
from reward.multi_objective import RewardConfig, compute_reward
from experiments.visualize_agent import mol_image_base64, new_atoms_since


def generate_dqn_candidates(
    checkpoint_path: str,
    start_smiles: str,
    target_seq: str,
    off_target_seq: Optional[str],
    reward_config: RewardConfig,
    num_candidates: int,
    max_steps: int,
    sampling: str = "boltzmann",
    temperature: float = 1.0,
    device=None,
    cancel_event: Optional[threading.Event] = None,
) -> List[dict]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = DKDQNAgent(output_dim=1, device=device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    agent.qn.load_state_dict(state_dict)
    agent.target_qn.load_state_dict(state_dict)  # get_action reads target_qn

    admet_model = ADMETModel(device)
    binding_model = Plapt(device=str(device))
    sa_model = SyntheticAccessibility()

    epsilon_threshold = 1.0 if sampling == "boltzmann" else 0.0

    candidates = []
    for _ in range(num_candidates):
        if cancel_event is not None and cancel_event.is_set():
            break

        env = MultiObjectiveRewardEnv(
            discount_factor=hyp.discount_factor,
            device=device,
            init_mol=start_smiles,
            max_steps=max_steps,
            target_seq=target_seq,
            off_target_seq=off_target_seq,
            admet_model=admet_model,
            binding_model=binding_model,
            sa_model=sa_model,
            reward_config=reward_config,
        )
        env.initialize()

        final_smiles = start_smiles
        for _step in range(max_steps):
            all_actions = list(env.get_valid_actions())
            obs = create_graph(all_actions)
            chosen = agent.get_action(obs, epsilon_threshold=epsilon_threshold, tau=temperature)
            action_smiles = all_actions[chosen]
            result = env.step(action_smiles)
            final_smiles = action_smiles
            if result.terminated:
                break

        canonical = Chem.MolToSmiles(Chem.MolFromSmiles(final_smiles)) if Chem.MolFromSmiles(final_smiles) else final_smiles
        score = compute_reward(
            smiles=canonical, target_seq=target_seq, device=device, off_target_seq=off_target_seq,
            admet_model=admet_model, binding_model=binding_model, sa_model=sa_model,
            **reward_config.to_compute_reward_kwargs(),
        )
        highlight = new_atoms_since(start_smiles, canonical)
        candidates.append({
            "smiles": canonical,
            "image_b64": mol_image_base64(canonical, highlight_atoms=highlight),
            "reward": score["reward"],
            "admet_score": score["admet"],
            "binding_uM": score["binding_uM"],
            "sa_score": score["sa_score"],
            "selectivity": score["selectivity"],
        })

    return candidates
