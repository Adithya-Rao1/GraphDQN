import argparse
import itertools
import json
import os
import time

import torch

import dqn.dqn_hyperparams as dqn_hyp
from ADMET.model import ADMETModel
from binding_module.binding_affinity.plapt import Plapt
from dqn.dqn_env import MoleculeEnv
from experiments.data.targets import DEFAULT_TARGET, TARGETS
from experiments.data.starting_molecules import sample_pilot_molecules
from molecular_modifications.macro_actions import build_edit_catalog
from ppo.gnn_llm_actor_critic import build_shared_llm_backbone, default_qwen2_lora_config
from ppo.pgmorl_agent import DEFAULT_LLM_MODEL_NAME, PGMORLAgent
from ppo.pgmorl_predictor import hypervolume, non_dominated_indices
from ppo.pgmorl_population import run_pareto_sweep_sequential
from ppo.pgmorl_concurrent import run_pareto_sweep_concurrent
from reward.multi_objective import RewardConfig
from synthetic_accessibility.sa_score import SyntheticAccessibility


def run(target_name=DEFAULT_TARGET, seed=0, num_molecules=30,
        population_size=6, concentration_alpha=20.0, num_rounds=5,
        episodes_per_round=10, eval_episodes_per_round=3, max_steps=dqn_hyp.max_steps,
        ppo_epochs=4, fixed_edit_count=1, beta=1.0,
        admet_weight=dqn_hyp.admet_weight, binding_weight=dqn_hyp.binding_weight,
        synthetic_weight=dqn_hyp.synthetic_weight, selectivity_weight=0.0,
        use_llm=False, llm_model_name=DEFAULT_LLM_MODEL_NAME, llm_torch_dtype=torch.bfloat16,
        reward_batch_size=8, predictors_device=None, ppo_minibatch_size=4,
        use_predictor=True, concurrent=False, max_concurrent_members=3,
        run_id=None, results_root='./experiments/results'):
    run_id = run_id or (
        f"pareto_sweep_{'naive' if not use_predictor else 'gp'}_{target_name}_seed{seed}_"
        f"{time.strftime('%Y%m%d-%H%M%S')}"
    )
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    predictors_device = torch.device(predictors_device) if predictors_device else device

    target_seq = TARGETS[target_name]
    start_mols = sample_pilot_molecules(n=num_molecules, seed=seed)
    mol_cycle = itertools.cycle(start_mols)

    def env_factory():
        env = MoleculeEnv(init_mol=next(mol_cycle), max_steps=max_steps)
        env.initialize()
        return env

    catalog = build_edit_catalog()
    base_config = RewardConfig(
        admet_weight=admet_weight, binding_weight=binding_weight,
        synthetic_weight=synthetic_weight, selectivity_weight=selectivity_weight,
    )

    admet_model = ADMETModel(predictors_device)
    binding_model = Plapt(device=str(predictors_device))
    sa_model = SyntheticAccessibility()

    shared_llm_backbone = None
    if use_llm:
        shared_llm_backbone = build_shared_llm_backbone(
            llm_model_name, {"actor": default_qwen2_lora_config(), "critic": default_qwen2_lora_config()},
            torch_dtype=llm_torch_dtype,
        )

    def build_agent_fn(member_reward_config):
        return PGMORLAgent(
            catalog=catalog, reward_config=member_reward_config, target_seq=target_seq, device=device,
            admet_model=admet_model, binding_model=binding_model, sa_model=sa_model,
            use_llm=use_llm, shared_llm_backbone=shared_llm_backbone,
            llm_torch_dtype=llm_torch_dtype, reward_batch_size=reward_batch_size,
            ppo_minibatch_size=ppo_minibatch_size, edit_count_mode="fixed",
            fixed_edit_count=fixed_edit_count,
        )

    start_time = time.time()
    if concurrent:
        from readerwriterlock import rwlock
        rw_lock = rwlock.RWLockFair() if use_llm else None
        result = run_pareto_sweep_concurrent(
            base_reward_config=base_config, build_agent_fn=build_agent_fn, env_factory=env_factory,
            population_size=population_size, concentration_alpha=concentration_alpha, num_rounds=num_rounds,
            episodes_per_round=episodes_per_round, eval_episodes_per_round=eval_episodes_per_round,
            max_steps=max_steps, ppo_epochs=ppo_epochs, beta=beta, seed=seed,
            max_concurrent_members=max_concurrent_members, rw_lock=rw_lock, use_cuda_streams=use_llm,
            use_predictor=use_predictor,
        )
    else:
        result = run_pareto_sweep_sequential(
            base_reward_config=base_config, build_agent_fn=build_agent_fn, env_factory=env_factory,
            population_size=population_size, concentration_alpha=concentration_alpha, num_rounds=num_rounds,
            episodes_per_round=episodes_per_round, eval_episodes_per_round=eval_episodes_per_round,
            max_steps=max_steps, ppo_epochs=ppo_epochs, beta=beta, seed=seed, use_predictor=use_predictor,
        )
    wall_clock = time.time() - start_time

    import numpy as np
    final_objective_vectors = np.array([m.objective_vector for m in result.members])
    nd_idx = non_dominated_indices(final_objective_vectors)
    active_dims = [
        d for d in range(final_objective_vectors.shape[1])
        if not np.allclose(final_objective_vectors[:, d], 0.0)
    ] if len(final_objective_vectors) else []
    front_hv = (
        hypervolume(final_objective_vectors[np.ix_(nd_idx, active_dims)], reference_point=[0.0] * len(active_dims))
        if len(nd_idx) and active_dims else 0.0
    )

    summary = {
        "run_id": run_id,
        "algorithm": "pareto_sweep",
        "use_predictor": use_predictor,
        "concurrent": concurrent,
        "max_concurrent_members": max_concurrent_members if concurrent else None,
        "target": target_name,
        "seed": seed,
        "population_size": population_size,
        "concentration_alpha": concentration_alpha,
        "num_rounds": num_rounds,
        "episodes_per_round": episodes_per_round,
        "use_llm": use_llm,
        "llm_model_name": llm_model_name if use_llm else None,
        "hypervolume_active_dims": active_dims,  
        "members": [
            {
                "member_id": m.member_id,
                "weight_vector": m.weight_vector,
                "objective_vector": m.objective_vector,
                "total_training_steps": m.total_training_steps,
            }
            for m in result.members
        ],
        "num_non_dominated": int(len(nd_idx)),
        "final_front_hypervolume": front_hv,
        "num_performance_records": len(result.records),
        "wall_clock_seconds": wall_clock,
    }

    os.makedirs(results_root, exist_ok=True)
    with open(os.path.join(results_root, f"{run_id}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[{run_id}] {len(nd_idx)}/{population_size} members non-dominated, "
          f"front hypervolume={front_hv:.4f}, wall_clock={wall_clock:.1f}s")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-name", default=DEFAULT_TARGET, choices=list(TARGETS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-molecules", type=int, default=30)
    parser.add_argument("--population-size", type=int, default=6)
    parser.add_argument("--concentration-alpha", type=float, default=20.0)
    parser.add_argument("--num-rounds", type=int, default=5)
    parser.add_argument("--episodes-per-round", type=int, default=10)
    parser.add_argument("--eval-episodes-per-round", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=dqn_hyp.max_steps)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--fixed-edit-count", type=int, default=1)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--llm-model-name", default=DEFAULT_LLM_MODEL_NAME)
    parser.add_argument("--llm-dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--reward-batch-size", type=int, default=8)
    parser.add_argument("--predictors-device", default=None, choices=[None, "cpu", "cuda"])
    parser.add_argument("--ppo-minibatch-size", type=int, default=4)
    parser.add_argument("--naive-baseline", action="store_true",)
    parser.add_argument("--concurrent", action="store_true",)
    parser.add_argument("--max-concurrent-members", type=int, default=3,)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    result_summary = run(
        target_name=args.target_name,
        seed=args.seed,
        num_molecules=args.num_molecules,
        population_size=args.population_size,
        concentration_alpha=args.concentration_alpha,
        num_rounds=args.num_rounds,
        episodes_per_round=args.episodes_per_round,
        eval_episodes_per_round=args.eval_episodes_per_round,
        max_steps=args.max_steps,
        ppo_epochs=args.ppo_epochs,
        fixed_edit_count=args.fixed_edit_count,
        beta=args.beta,
        use_llm=args.use_llm,
        llm_model_name=args.llm_model_name,
        llm_torch_dtype=getattr(torch, args.llm_dtype),
        reward_batch_size=args.reward_batch_size or None,
        predictors_device=args.predictors_device,
        ppo_minibatch_size=args.ppo_minibatch_size or None,
        use_predictor=not args.naive_baseline,
        concurrent=args.concurrent,
        max_concurrent_members=args.max_concurrent_members,
        run_id=args.run_id,
    )
    print(json.dumps(result_summary, indent=2))
