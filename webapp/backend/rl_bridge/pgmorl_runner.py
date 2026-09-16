import os
import threading
import time
from contextlib import nullcontext
from typing import Callable, List, Optional, Tuple

import torch

from ADMET.model import ADMETModel
from binding_module.binding_affinity.plapt import Plapt
from dqn.dqn_env import MoleculeEnv
from molecular_modifications.macro_actions import build_edit_catalog
from ppo.gnn_llm_actor_critic import build_shared_llm_backbone, default_qwen2_lora_config
from ppo.llm_finetune import EditOutcome, FineTuneCycleResult, run_llm_finetune_cycle
from ppo.pgmorl_agent import PGMORLAgent
from ppo.pgmorl_concurrent import run_pareto_sweep_concurrent
from ppo.pgmorl_population import run_pareto_sweep_sequential
from ppo.pgmorl_predictor import hypervolume, non_dominated_indices
from reward.multi_objective import RewardConfig
from synthetic_accessibility.sa_score import SyntheticAccessibility


def run_pareto_sweep(
    reward_config: RewardConfig,
    target_seq: str,
    off_target_seq: Optional[str],
    init_mol: str,
    seed: int,
    population_size: int,
    concentration_alpha: float,
    num_rounds: int,
    episodes_per_round: int,
    eval_episodes_per_round: int,
    max_steps: int,
    edit_count_mode: str,
    fixed_edit_count: int,
    edit_count_range: Optional[tuple],
    k_max: Optional[int],
    use_llm: bool,
    llm_model_name: Optional[str],
    use_predictor: bool,
    concurrent: bool,
    max_concurrent_members: Optional[int],
    run_id: str,
    checkpoint_root: str,
    on_member_created: Callable[[int, List[float]], None],
    progress_cb: Optional[Callable[[int, int, list, list, int], Optional[List[Tuple[int, EditOutcome]]]]] = None,
    cancel_event: Optional[threading.Event] = None,
    on_step_scored: Optional[Callable[[str, object], None]] = None,
    on_finetune_result: Optional[Callable[[FineTuneCycleResult, List[int], str], None]] = None,
    adapter_checkpoint_root: Optional[str] = None,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    catalog = build_edit_catalog()

    admet_model = ADMETModel(device)
    binding_model = Plapt(device=str(device))
    sa_model = SyntheticAccessibility()

    shared_llm_backbone = None
    rw_lock = None
    if use_llm:
        shared_llm_backbone = build_shared_llm_backbone(
            llm_model_name, {"actor": default_qwen2_lora_config(), "critic": default_qwen2_lora_config()},
            torch_dtype=torch.bfloat16,
        )
        from readerwriterlock import rwlock
        rw_lock = rwlock.RWLockFair()

    def finetune_descriptor_fn(smiles: str) -> str:
        return f"Molecule {smiles} against target sequence (pooled fine-tune batch across population)."

    def env_factory():
        env = MoleculeEnv(init_mol=init_mol, max_steps=max_steps)
        env.initialize()
        return env

    member_counter = {"n": 0}

    def build_agent_fn(member_reward_config: RewardConfig) -> PGMORLAgent:
        member_index = member_counter["n"]
        member_counter["n"] += 1
        weight_vector = [
            member_reward_config.admet_weight, member_reward_config.binding_weight,
            member_reward_config.synthetic_weight, member_reward_config.selectivity_weight,
        ]
        on_member_created(member_index, weight_vector)

        return PGMORLAgent(
            catalog=catalog, reward_config=member_reward_config, target_seq=target_seq,
            off_target_seq=off_target_seq, device=device,
            admet_model=admet_model, binding_model=binding_model, sa_model=sa_model,
            use_llm=use_llm, shared_llm_backbone=shared_llm_backbone,
            reward_batch_size=8, ppo_minibatch_size=4,
            edit_count_mode=edit_count_mode, fixed_edit_count=fixed_edit_count,
            edit_count_range=edit_count_range, k_max=k_max,
        )

    records_seen = {"n": 0}

    def on_round_complete(round_idx, members, records):
        if progress_cb is None:
            return
        snapshot = [
            {
                "member_id": m.member_id,
                "weight_vector": m.weight_vector,
                "objective_vector": m.objective_vector,
                "total_training_steps": m.total_training_steps,
            }
            for m in members
        ]
        new_records = records[records_seen["n"]:]
        records_seen["n"] = len(records)
        new_records_out = [
            {
                "member_id": r.member_id,
                "objective_vector_before": r.objective_vector_before,
                "weight_vector_used": r.weight_vector_used,
                "training_steps_this_round": r.training_steps_this_round,
                "objective_vector_after": r.objective_vector_after,
            }
            for r in new_records
        ]
        round_steps = sum(r["training_steps_this_round"] for r in new_records_out)
        finetune_batch = progress_cb(round_idx + 1, num_rounds, snapshot, new_records_out, round_steps)

        if finetune_batch and use_llm and shared_llm_backbone is not None:
            representative_actor = members[0].agent.actor
            row_ids = [row_id for row_id, _ in finetune_batch]
            outcomes = [outcome for _, outcome in finetune_batch]
            result = run_llm_finetune_cycle(
                actor=representative_actor, adapter_name="actor", outcomes=outcomes,
                descriptor_fn=finetune_descriptor_fn, device=device, rw_lock=rw_lock,
            )

            adapter_path = None
            if adapter_checkpoint_root is not None:
                adapter_dir = os.path.join(adapter_checkpoint_root, f"{run_id}_round{round_idx}_{int(time.time())}")
                os.makedirs(adapter_dir, exist_ok=True)
                write_ctx = rw_lock.gen_wlock() if rw_lock is not None else nullcontext()
                with write_ctx:
                    representative_actor.llm.save_pretrained(adapter_dir, selected_adapters=["actor"])
                adapter_path = adapter_dir

            if on_finetune_result is not None:
                on_finetune_result(result, row_ids, adapter_path)

    start_time = time.time()
    common_kwargs = dict(
        base_reward_config=reward_config, build_agent_fn=build_agent_fn, env_factory=env_factory,
        population_size=population_size, concentration_alpha=concentration_alpha, num_rounds=num_rounds,
        episodes_per_round=episodes_per_round, eval_episodes_per_round=eval_episodes_per_round,
        max_steps=max_steps, seed=seed, use_predictor=use_predictor,
        cancel_event=cancel_event, on_round_complete=on_round_complete,
        on_step_scored=on_step_scored,
    )
    if concurrent:
        result = run_pareto_sweep_concurrent(
            **common_kwargs, max_concurrent_members=max_concurrent_members or 3,
            rw_lock=rw_lock, use_cuda_streams=use_llm,
        )
    else:
        result = run_pareto_sweep_sequential(**common_kwargs)
    wall_clock = time.time() - start_time

    checkpoint_dir = os.path.join(checkpoint_root, run_id)
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_paths = {}
    for member in result.members:
        path = os.path.join(checkpoint_dir, f"{member.member_id}.pt")
        member.agent.save_checkpoint(path)
        checkpoint_paths[member.member_id] = path

    import numpy as np
    final_objective_vectors = np.array([m.objective_vector for m in result.members])
    nd_idx = set(non_dominated_indices(final_objective_vectors).tolist())
    active_dims = [
        d for d in range(final_objective_vectors.shape[1])
        if not np.allclose(final_objective_vectors[:, d], 0.0)
    ] if len(final_objective_vectors) else []
    front_hv = (
        hypervolume(final_objective_vectors[sorted(nd_idx)][:, active_dims], reference_point=[0.0] * len(active_dims))
        if nd_idx and active_dims else 0.0
    )

    members_out = [
        {
            "member_id": m.member_id,
            "weight_vector": m.weight_vector,
            "objective_vector": m.objective_vector,
            "total_training_steps": m.total_training_steps,
            "checkpoint_path": checkpoint_paths[m.member_id],
            "non_dominated": i in nd_idx,
        }
        for i, m in enumerate(result.members)
    ]

    return {
        "run_id": run_id,
        "concurrent": concurrent,
        "use_predictor": use_predictor,
        "members": members_out,
        "num_non_dominated": len(nd_idx),
        "final_front_hypervolume": front_hv,
        "hypervolume_active_dims": active_dims,
        "num_performance_records": len(result.records),
        "wall_clock_seconds": wall_clock,
        "checkpoint_dir": checkpoint_dir,
    }
