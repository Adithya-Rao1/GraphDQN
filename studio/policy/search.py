"""Minimal GA used ONLY for training/eval of the policy (the real GA is studio/engine)."""
from __future__ import annotations

import math
import random
from typing import List, Optional, Sequence, Tuple

from studio.policy.ops import applicable
from studio.policy.reward import composite

TAU = 0.05  # softmax temperature over Q (Q ~ one-step delta composite, typically |q| < 0.2)


def choose(policy, rng: random.Random, smiles: str, cands: Sequence[tuple], guidance: float) -> int:
    """Index of the chosen (op, site): policy softmax with prob `guidance`, else uniform."""
    if policy is not None and guidance > 0 and rng.random() < guidance:
        q = policy.rank(smiles, cands)
        m = max(q)
        w = [math.exp((v - m) / TAU) for v in q]
        r = rng.random() * sum(w)
        acc = 0.0
        for i, wi in enumerate(w):
            acc += wi
            if r <= acc:
                return i
        return len(w) - 1
    return rng.randrange(len(cands))


def run_ga(start: str, policy, guidance: float, seed: int, budget: int = 120, pop_size: int = 8,
           target: float = 0.8, ops: Optional[Sequence] = None, learn: bool = False) -> dict:
    """Evaluation-budgeted GA. Returns evals-to-target (censored at `budget`)."""
    rng = random.Random(seed)
    pop: List[Tuple[float, str]] = [(composite(start), start)]
    seen = {start}
    evals, reached, best = 0, None, pop[0][0]
    if best >= target:
        reached = 0
    attempts = 0
    while reached is None and evals < budget and attempts < budget * 20:
        attempts += 1
        # tournament-of-2 parent selection
        a, b = rng.choice(pop), rng.choice(pop)
        parent = max(a, b)[1]
        cands = applicable(parent, ops)
        if not cands:
            continue
        i = choose(policy, rng, parent, cands, guidance)
        op, site = cands[i]
        kids = op.apply(parent, site)
        if not kids or kids[0] in seen:
            continue
        child = kids[0]
        seen.add(child)
        evals += 1
        sc = composite(child)
        if learn and policy is not None:
            policy.update([(parent, op.id, site, child, sc - composite(parent))])
        pop.append((sc, child))
        pop = sorted(pop, reverse=True)[:pop_size]
        best = max(best, sc)
        if best >= target:
            reached = evals
    return {"evals": reached if reached is not None else budget, "reached": reached is not None, "best": best}
