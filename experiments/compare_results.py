import argparse
import glob
import json
import os


def load_results(results_dir='./experiments/results'):
    summaries = []
    for path in sorted(glob.glob(os.path.join(results_dir, '*.json'))):
        with open(path) as f:
            summaries.append(json.load(f))
    return summaries


def print_comparison_table(summaries):
    header = f"{'run_id':<45} {'algo':<6} {'target':<16} {'seed':>5} {'final_reward':>14} {'wall_clock_s':>14}"
    print(header)
    print("-" * len(header))
    for s in sorted(summaries, key=lambda s: (s.get('algorithm', ''), s.get('target', ''), s.get('seed', 0))):
        final_reward = s.get('final_reward')
        wall_clock = s.get('wall_clock_seconds')
        print(
            f"{s.get('run_id', ''):<45} {s.get('algorithm', ''):<6} {s.get('target', ''):<16} "
            f"{s.get('seed', ''):>5} "
            f"{final_reward if final_reward is None else round(final_reward, 4):>14} "
            f"{wall_clock if wall_clock is None else round(wall_clock, 1):>14}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="./experiments/results")
    args = parser.parse_args()

    results = load_results(args.results_dir)
    if not results:
        print(f"No result files found in {args.results_dir}")
    else:
        print_comparison_table(results)
