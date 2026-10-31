"""Held-out top-1 / top-3 hit rate: adapted vs base vs uniform random, with bootstrap 95% intervals.

The numbers are whatever they are: nothing here is tuned to make them look good. Small n is flagged in `note`.
Ties (e.g. UniformPolicy scores everything 0) count as the expected hit under random tie-breaking.
"""
from __future__ import annotations

import time
from typing import Callable, List, Optional, Sequence

import numpy as np
import torch

from studio.adapt.dataset import Example
from studio.adapt.finetune import BaseInfo, Prepared, adapt, batch_scores
from studio.policy.policy import DQNPolicy

SEED = 0
SMALL_N = 30
LOO_STEPS = 60  # per-fold optimiser steps in leave-one-out (keeps n folds inside the job budget)
N_BOOT = 2000


def scores_for(policy, ex: Example) -> List[float]:
    if isinstance(policy, DQNPolicy):
        with torch.no_grad():
            return batch_scores(policy.net, Prepared(policy, ex)).tolist()
    return list(policy.rank(ex.transition.parent_smiles, ex.candidates))


def hit_prob(scores: Sequence[float], target: int, k: int) -> float:
    t = scores[target]
    greater = sum(1 for s in scores if s > t)
    tied = sum(1 for s in scores if s == t)  # includes the target
    return float(min(1.0, max(0.0, (k - greater) / tied)))


def random_prob(n_candidates: int, k: int) -> float:
    return min(1.0, k / n_candidates)


def _boot(a: np.ndarray, rng, idx: np.ndarray) -> List[float]:
    m = a[idx].mean(axis=1)
    return [round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)]


def summarise(rows: dict, n_total: int, method: str, n_folds_done: int) -> dict:
    n = len(rows["base1"])
    arr = {k: np.array(v, dtype=float) for k, v in rows.items()}
    idx = np.random.default_rng(SEED).integers(0, n, size=(N_BOOT, n))
    out = {"method": method, "n": n, "n_total": n_total}
    for name, key in (("base", "base"), ("adapted", "adapted"), ("random", "random")):
        out[name] = {}
        for k in (1, 3):
            a = arr[f"{key}{k}"]
            out[name][f"top{k}"] = round(float(a.mean()), 4)
            out[name][f"top{k}_ci95"] = _boot(a, None, idx)
    out["delta_top1"] = {"value": round(float((arr["adapted1"] - arr["base1"]).mean()), 4),
                         "ci95": _boot(arr["adapted1"] - arr["base1"], None, idx)}
    out["delta_top3"] = {"value": round(float((arr["adapted3"] - arr["base3"]).mean()), 4),
                         "ci95": _boot(arr["adapted3"] - arr["base3"], None, idx)}
    out["mean_candidates"] = round(float(np.mean(rows["ncand"])), 1)
    return out


def note_for(res: dict, base: BaseInfo) -> str:
    parts = [f"Evaluated on n={res['n']} held-out edits ({res['method']}); intervals are bootstrap 95% over those edits."]
    if res["n"] < SMALL_N:
        parts.append("Small sample: the intervals are wide and the result is indicative, not conclusive.")
    if res["n_total"] != res["n"]:
        parts.append(f"Only {res['n']} of {res['n_total']} evaluations finished inside the time budget.")
    if base.kind == "uniform_fallback":
        parts.append("No trained base checkpoint is loaded, so 'base' is the uniform fallback (no preference).")
    parts.append("Hit rates measure agreement with this user's past choices on this user's data; they are not a claim "
                 "that the edits are better chemistry.")
    return " ".join(parts)


def evaluate(examples: Sequence[Example], base: BaseInfo, *, steps: int, deadline: Optional[float] = None,
             progress: Optional[Callable[[float], None]] = None) -> dict:
    n = len(examples)
    rows = {k: [] for k in ("base1", "base3", "adapted1", "adapted3", "random1", "random3", "ncand")}

    def record(ex: Example, adapted):
        b, a = scores_for(base.eval_policy, ex), scores_for(adapted, ex)
        for k in (1, 3):
            rows[f"base{k}"].append(hit_prob(b, ex.target, k))
            rows[f"adapted{k}"].append(hit_prob(a, ex.target, k))
            rows[f"random{k}"].append(random_prob(len(ex.candidates), k))
        rows["ncand"].append(len(ex.candidates))

    if n < SMALL_N:
        method, total = "leave-one-out", n
        for i in range(n):
            if deadline is not None and time.monotonic() >= deadline and rows["base1"]:
                break
            tr = [e for j, e in enumerate(examples) if j != i]
            r = adapt(tr, base, steps=min(steps, LOO_STEPS), seed=SEED, deadline=deadline, min_steps=1)
            if r.stopped_early and deadline is not None and time.monotonic() >= deadline:
                break  # fold cut off by the deadline: not counted as an evaluation
            record(examples[i], r.policy)
            if progress:
                progress((i + 1) / n)
    else:
        method = "80/20 holdout"
        perm = np.random.RandomState(SEED).permutation(n)
        n_test = max(1, int(round(0.2 * n)))
        test = [examples[i] for i in perm[:n_test]]
        train = [examples[i] for i in perm[n_test:]]
        total = n_test
        r = adapt(train, base, steps=steps, seed=SEED, deadline=deadline)
        for ex in test:
            record(ex, r.policy)
        if progress:
            progress(1.0)
    res = summarise(rows, total, method, len(rows["base1"]))
    res["note"] = note_for(res, base)
    res["base_kind"] = base.kind
    return res
