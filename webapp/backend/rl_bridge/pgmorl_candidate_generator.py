import threading
from typing import List, Optional, Sequence, Tuple

import torch
from rdkit import Chem

from ADMET.model import ADMETModel
from binding_module.binding_affinity.plapt import Plapt
from dqn.dqn_env import MoleculeEnv
from molecular_modifications.macro_actions import build_edit_catalog
from ppo.gnn_llm_actor_critic import build_shared_llm_backbone, default_qwen2_lora_config
from ppo.pgmorl_agent import MacroStepMemory, PGMORLAgent
from reward.multi_objective import RewardConfig, compute_reward_batch
from synthetic_accessibility.sa_score import SyntheticAccessibility


def _canonical(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else smiles


def _applied_edits(entry: MacroStepMemory, catalog_by_id: dict) -> List[dict]:
    out = []
    for i, edit_id in enumerate(entry.edit_ids):
        resulting_smiles = (
            entry.pre_edit_states[i + 1] if i + 1 < len(entry.pre_edit_states) else entry.final_smiles
        )
        edit_op = catalog_by_id[edit_id]
        out.append({
            "edit_id": edit_id,
            "category": edit_op.category,
            "description": edit_op.description,
            "resulting_smiles": resulting_smiles,
        })
    return out


def generate_pgmorl_candidates(
    checkpoint_path: str,
    reward_config: RewardConfig,
    target_seq: str,
    off_target_seq: Optional[str],
    init_mol: str,
    max_steps: int,
    edit_count_mode: str,
    fixed_edit_count: int,
    edit_count_range: Optional[Sequence[int]],
    k_max: Optional[int],
    use_llm: bool,
    llm_model_name: Optional[str],
    num_candidates: int,
    sampling: str = "boltzmann",
    temperature: float = 1.0,
    device=None,
    cancel_event: Optional[threading.Event] = None,
) -> Tuple[List[dict], List[List[dict]]]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    catalog = build_edit_catalog()
    catalog_by_id = {op.id: op for op in catalog}

    admet_model = ADMETModel(device)
    binding_model = Plapt(device=str(device))
    sa_model = SyntheticAccessibility()

    shared_llm_backbone = None
    if use_llm:
        shared_llm_backbone = build_shared_llm_backbone(
            llm_model_name, {"actor": default_qwen2_lora_config(), "critic": default_qwen2_lora_config()},
            torch_dtype=torch.bfloat16,
        )

    agent = PGMORLAgent(
        catalog=catalog, reward_config=reward_config, target_seq=target_seq, off_target_seq=off_target_seq,
        device=device, admet_model=admet_model, binding_model=binding_model, sa_model=sa_model,
        use_llm=use_llm, shared_llm_backbone=shared_llm_backbone,
        edit_count_mode=edit_count_mode, fixed_edit_count=fixed_edit_count,
        edit_count_range=tuple(edit_count_range) if edit_count_range else None, k_max=k_max,
        temperature=temperature, greedy=(sampling == "greedy"),
    )
    agent.load_checkpoint(checkpoint_path, map_location=device)

    final_candidates = []
    trajectories = []
    for _ in range(num_candidates):
        if cancel_event is not None and cancel_event.is_set():
            break

        env = MoleculeEnv(init_mol=init_mol, max_steps=max_steps)
        env.initialize()

        trajectory_smiles = [_canonical(init_mol)]
        macro_entries: List[Optional[MacroStepMemory]] = [None]
        for _step in range(max_steps):
            entry = agent.act(env)
            trajectory_smiles.append(_canonical(entry.final_smiles))
            macro_entries.append(entry)
            if entry.done:
                break

        results = compute_reward_batch(
            trajectory_smiles, target_seq=target_seq, device=device, off_target_seq=off_target_seq,
            admet_model=admet_model, binding_model=binding_model, sa_model=sa_model,
            **reward_config.to_compute_reward_kwargs(),
        )

        trajectory_scored = []
        for i, (smiles, result, macro_entry) in enumerate(zip(trajectory_smiles, results, macro_entries)):
            step = {
                "step_index": i,
                "smiles": smiles,
                "reward": result["reward"],
                "admet_score": result["admet"],
                "binding_uM": result["binding_uM"],
                "sa_score": result["sa_score"],
                "selectivity": result["selectivity"],
            }
            if macro_entry is not None:
                step["k_edits_used"] = macro_entry.k_used
                step["edit_count_mode"] = edit_count_mode
                step["applied_edits"] = _applied_edits(macro_entry, catalog_by_id)
            trajectory_scored.append(step)

        trajectories.append(trajectory_scored)
        final_candidates.append(trajectory_scored[-1])

        if use_llm and torch.cuda.is_available():
            torch.cuda.empty_cache()

    return final_candidates, trajectories
