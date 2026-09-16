import argparse
import json
import os
import time

import dqn.dqn_hyperparams as dqn_hyp
from experiments.data.targets import TARGETS, DEFAULT_TARGET
from experiments import run_dqn, run_pgmorl_ppo


def run(target_name=DEFAULT_TARGET, seed=0, num_molecules=30, num_episodes=200,
        checkpoint_interval=50, fixed_edit_count=1, results_root='./experiments/results'):
    ablation_id = f"ablation_ppo_vs_dqn_{target_name}_seed{seed}_{time.strftime('%Y%m%d-%H%M%S')}"

    print(f"[{ablation_id}] running DQN baseline...")
    dqn_summary = run_dqn.run(
        target_name=target_name, seed=seed, num_molecules=num_molecules,
        num_episodes=num_episodes, checkpoint_interval=checkpoint_interval,
        run_id=f"{ablation_id}_dqn", results_root=results_root,
    )

    print(f"[{ablation_id}] running PGMORL-PPO (Phase 2: single member, no LLM, fixed edit count)...")
    ppo_summary = run_pgmorl_ppo.run(
        target_name=target_name, seed=seed, num_molecules=num_molecules,
        num_episodes=num_episodes, checkpoint_interval=checkpoint_interval,
        fixed_edit_count=fixed_edit_count,
        admet_weight=dqn_hyp.admet_weight, binding_weight=dqn_hyp.binding_weight,
        synthetic_weight=dqn_hyp.synthetic_weight, selectivity_weight=0.0,
        run_id=f"{ablation_id}_pgmorl_ppo", results_root=results_root,
    )

    dqn_mean = dqn_summary.get("mean_reward_last_10")
    ppo_mean = ppo_summary.get("mean_reward_last_10")
    delta = (ppo_mean - dqn_mean) if (dqn_mean is not None and ppo_mean is not None) else None

    comparison = {
        "ablation_id": ablation_id,
        "target": target_name,
        "seed": seed,
        "num_molecules": num_molecules,
        "num_episodes": num_episodes,
        "dqn": dqn_summary,
        "pgmorl_ppo": ppo_summary,
        "mean_reward_last_10_delta_ppo_minus_dqn": delta,
        "ppo_is_regression": (delta is not None and delta < 0),
    }

    os.makedirs(results_root, exist_ok=True)
    with open(os.path.join(results_root, f"{ablation_id}.json"), "w") as f:
        json.dump(comparison, f, indent=2)

    print(f"\n[{ablation_id}] RESULT")
    print(f"  DQN        mean_reward_last_10={dqn_mean!r}  wall_clock_s={dqn_summary.get('wall_clock_seconds'):.1f}")
    print(f"  PGMORL-PPO mean_reward_last_10={ppo_mean!r}  wall_clock_s={ppo_summary.get('wall_clock_seconds'):.1f}")
    if delta is not None:
        verdict = "REGRESSION vs DQN" if delta < 0 else "no regression vs DQN"
        print(f"  delta (ppo - dqn) = {delta:.4f}  ->  {verdict}")

    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-name", default=DEFAULT_TARGET, choices=list(TARGETS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-molecules", type=int, default=30)
    parser.add_argument("--num-episodes", type=int, default=200)
    parser.add_argument("--checkpoint-interval", type=int, default=50)
    parser.add_argument("--fixed-edit-count", type=int, default=1)
    parser.add_argument("--results-root", default="./experiments/results")
    args = parser.parse_args()

    result = run(
        target_name=args.target_name,
        seed=args.seed,
        num_molecules=args.num_molecules,
        num_episodes=args.num_episodes,
        checkpoint_interval=args.checkpoint_interval,
        fixed_edit_count=args.fixed_edit_count,
        results_root=args.results_root,
    )
