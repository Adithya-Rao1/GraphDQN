"""Train the DQN edit prior on the RDKit composite reward.

python -m studio.policy.train --steps 3000 --start-pool studio/policy/data/start_pool.smi

Trains on the first `--train-count` molecules of the pool; the rest are held out for
eval_rl_vs_random. Writes studio/policy/checkpoints/policy.pt and a jsonl log.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

from rdkit import Chem

from studio.policy.ops import applicable, load_ops
from studio.policy.policy import CKPT_DIR, DQNPolicy
from studio.policy.reward import composite
from studio.policy.search import choose

DEFAULT_POOL = os.path.join(os.path.dirname(__file__), "data", "start_pool.smi")


def read_pool(path: str):
    out = []
    for line in open(path):
        s = line.split()[0] if line.strip() else ""
        m = Chem.MolFromSmiles(s) if s else None
        if m is not None:
            out.append(Chem.MolToSmiles(m))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--start-pool", default=DEFAULT_POOL)
    ap.add_argument("--train-count", type=int, default=44)
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--eps-start", type=float, default=1.0)
    ap.add_argument("--eps-end", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(CKPT_DIR, "policy.pt"))
    ap.add_argument("--log", default=os.path.join(CKPT_DIR, "train_log.jsonl"))
    a = ap.parse_args(argv)

    pool = read_pool(a.start_pool)[: a.train_count]
    ops = load_ops()
    pol = DQNPolicy([o.id for o in ops], checkpoint=None, seed=a.seed)
    rng = random.Random(a.seed)
    os.makedirs(os.path.dirname(a.log), exist_ok=True)
    log = open(a.log, "w")
    t0, losses, rewards, step = time.time(), [], [], 0
    while step < a.steps:
        state = rng.choice(pool)
        for _ in range(a.horizon):
            if step >= a.steps:
                break
            cands = applicable(state, ops)
            if not cands:
                break
            eps = a.eps_start + (a.eps_end - a.eps_start) * step / max(1, a.steps)
            i = choose(pol, rng, state, cands, guidance=1.0 - eps)
            op, site = cands[i]
            kids = op.apply(state, site)
            if not kids:
                continue
            r = composite(kids[0]) - composite(state)
            loss = pol.update([(state, op.id, site, kids[0], r)])
            state = kids[0]
            step += 1
            rewards.append(r)
            if loss is not None:
                losses.append(loss)
            if step % 100 == 0:
                rec = {"step": step, "eps": round(eps, 3), "mean_reward": sum(rewards[-100:]) / 100,
                       "mean_loss": sum(losses[-100:]) / max(1, len(losses[-100:])), "elapsed_s": round(time.time() - t0, 1)}
                log.write(json.dumps(rec) + "\n"); log.flush()
                print(rec)
    pol.save(a.out, meta={"steps": a.steps, "train_mols": len(pool), "ops": "studio.policy.ops stand-in", "seed": a.seed})
    print(f"saved {a.out} after {step} steps in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
