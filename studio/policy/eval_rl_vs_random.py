"""Policy-guided vs uniform mutation, paired over seeds, same evaluation budget.

python -m studio.policy.eval_rl_vs_random --seeds 20 --budget 120 --gain 0.08

Per seed i: start molecule = held-out pool[train_count + i % n_test], target =
start composite + `gain`, identical GA budget and rng seed for both arms. Metric:
number of scored children (evaluations) until the best composite reaches the target;
runs that never reach it are censored at the budget. The policy is the frozen
trained checkpoint (no online learning during eval). Prints mean +/- 95% CI of the
paired difference (uniform - policy; positive = policy faster), paired t-test and
Wilcoxon signed-rank, and reach rates. Reports whatever it measures.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics as st

import numpy as np
from scipy import stats

from studio.policy.ops import load_ops
from studio.policy.policy import DEFAULT_CKPT, DQNPolicy, UniformPolicy
from studio.policy.reward import composite
from studio.policy.search import run_ga
from studio.policy.train import DEFAULT_POOL, read_pool


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--budget", type=int, default=120)
    ap.add_argument("--gain", type=float, default=0.08)
    ap.add_argument("--guidance", type=float, default=0.5)
    ap.add_argument("--pop", type=int, default=8)
    ap.add_argument("--train-count", type=int, default=44)
    ap.add_argument("--start-pool", default=DEFAULT_POOL)
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--out", default=None, help="optional JSON results path")
    a = ap.parse_args(argv)

    ops = load_ops()
    pool = read_pool(a.start_pool)[a.train_count:]
    pol = DQNPolicy([o.id for o in ops], checkpoint=a.checkpoint)
    trained = pol.checkpoint_loaded is not None
    print(f"policy checkpoint: {pol.checkpoint_loaded or 'NONE (untrained random-init net)'}")
    uni = UniformPolicy()

    rows = []
    for i in range(a.seeds):
        start = pool[i % len(pool)]
        target = min(composite(start) + a.gain, 0.99)
        kw = dict(seed=1000 + i, budget=a.budget, pop_size=a.pop, target=target, ops=ops)
        r_u = run_ga(start, uni, 0.0, **kw)
        r_p = run_ga(start, pol, a.guidance, **kw)
        rows.append({"seed": i, "start": start, "target": round(target, 3), "uniform": r_u["evals"], "policy": r_p["evals"],
                     "uniform_reached": r_u["reached"], "policy_reached": r_p["reached"]})
    u = np.array([r["uniform"] for r in rows], float)
    p = np.array([r["policy"] for r in rows], float)
    d = u - p
    n = len(d)
    mean, sd = d.mean(), d.std(ddof=1)
    half = stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n)
    tt = stats.ttest_rel(u, p)
    try:
        wx = stats.wilcoxon(u, p)
        wp = float(wx.pvalue)
    except ValueError:  # all differences zero
        wp = float("nan")
    print(f"n seeds={n} budget={a.budget} gain={a.gain} guidance={a.guidance} trained_checkpoint={trained}")
    print(f"evals-to-target  uniform: {u.mean():.1f} +/- {stats.t.ppf(0.975,n-1)*u.std(ddof=1)/math.sqrt(n):.1f}  "
          f"policy: {p.mean():.1f} +/- {stats.t.ppf(0.975,n-1)*p.std(ddof=1)/math.sqrt(n):.1f}  (mean +/- 95% CI, censored at budget)")
    print(f"paired diff (uniform - policy; >0 means policy faster): {mean:+.2f} evals, 95% CI [{mean-half:+.2f}, {mean+half:+.2f}]")
    print(f"paired t-test p={tt.pvalue:.4f}  Wilcoxon p={wp:.4f}")
    print(f"reached target: uniform {sum(r['uniform_reached'] for r in rows)}/{n}, policy {sum(r['policy_reached'] for r in rows)}/{n}")
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump({"args": vars(a), "rows": rows, "mean_diff": mean, "ci95": [mean - half, mean + half],
                   "t_p": float(tt.pvalue), "wilcoxon_p": wp, "trained_checkpoint": trained,
                   "op_set": "studio.policy.ops stand-in"}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
