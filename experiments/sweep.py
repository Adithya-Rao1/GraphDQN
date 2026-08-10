import argparse
import itertools
import json
import time

from experiments.data.targets import TARGETS
from experiments.run_dqn import run as run_dqn
from experiments.run_ppo import run as run_ppo

ALGORITHMS = {"dqn": run_dqn, "ppo": run_ppo}


def sweep(algorithms, targets, seeds, num_molecules, use_wandb=False):
    results = []
    for algo, target_name, seed in itertools.product(algorithms, targets, seeds):
        run_id = f"{algo}_{target_name}_seed{seed}_{time.strftime('%Y%m%d-%H%M%S')}"
        print(f"=== Starting {run_id} ===")
        summary = ALGORITHMS[algo](
            target_name=target_name,
            seed=seed,
            num_molecules=num_molecules,
            run_id=run_id,
            use_wandb=use_wandb,
        )
        results.append(summary)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithms", nargs="+", default=["dqn", "ppo"], choices=list(ALGORITHMS))
    parser.add_argument("--targets", nargs="+", default=list(TARGETS), choices=list(TARGETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--num-molecules", type=int, default=30)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    sweep_results = sweep(args.algorithms, args.targets, args.seeds, args.num_molecules, args.wandb)
    print(json.dumps(sweep_results, indent=2))
