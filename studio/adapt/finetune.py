"""Imitation fine-tune of the DQN edit policy on a COPY of the network.

Objective: for each logged step, softmax cross-entropy over the scores of ALL applicable (op, site)
candidates of the parent, target = the edit the scientist chose. Not RL: no reward is optimised.
"""
from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

from studio.adapt.dataset import MIN_VALID_STEPS, Example
from studio.policy.policy import DQNPolicy, UniformPolicy, load_policy

MAX_OPT_STEPS = 300
MAX_SECONDS = 60.0
DEFAULT_STEPS = 200
DEFAULT_LR = 5e-4
BATCH = 8


class TooFewSteps(ValueError):
    pass


@dataclass
class BaseInfo:
    eval_policy: object      # what runs without an adapter (may be UniformPolicy)
    start: DQNPolicy         # DQN to copy weights from (fresh init when no checkpoint exists)
    kind: str                # "checkpoint" | "uniform_fallback"
    checkpoint_name: str


def resolve_base(op_ids: Sequence[str]) -> BaseInfo:
    pol = load_policy(list(op_ids))
    if isinstance(pol, DQNPolicy):
        return BaseInfo(pol, pol, "checkpoint", pol.checkpoint_loaded and pol.checkpoint_loaded.split("/")[-1] or "policy.pt")
    torch.manual_seed(0)
    return BaseInfo(UniformPolicy(), DQNPolicy(list(op_ids), checkpoint=None), "uniform_fallback", "none (untrained network)")


class Prepared:
    """Per-parent tensors so all candidates are scored in one batched pass (mirrors QNet.q exactly)."""
    __slots__ = ("graph", "M", "op_idx")

    def __init__(self, policy: DQNPolicy, ex: Example):
        g = policy._graph(ex.transition.parent_smiles)
        n = g[0].shape[0]
        M = torch.zeros(len(ex.candidates), n)
        for r, (_, site) in enumerate(ex.candidates):
            idx = [i for i in site if 0 <= i < n] or list(range(n))
            for i in idx:
                M[r, i] += 1.0
            M[r] /= len(idx)
        self.graph = g
        self.M = M
        self.op_idx = torch.tensor([policy._op_index(op.id) for op, _ in ex.candidates])


def batch_scores(net, p: Prepared) -> torch.Tensor:
    h = net.encode(*p.graph)
    feat = torch.cat([h.mean(0).expand(p.M.shape[0], -1), p.M @ h, net.op_emb(p.op_idx)], dim=1)
    return net.head(feat).squeeze(-1)


@dataclass
class AdaptResult:
    policy: DQNPolicy
    steps_done: int
    final_loss: Optional[float]
    seconds: float
    stopped_early: bool


def adapt(examples: Sequence[Example], base: BaseInfo, *, steps: int = DEFAULT_STEPS, lr: float = DEFAULT_LR,
          seed: int = 0, max_seconds: float = MAX_SECONDS, deadline: Optional[float] = None,
          min_steps: int = MIN_VALID_STEPS, progress: Optional[Callable[[float], None]] = None) -> AdaptResult:
    """Fine-tune a copy of base.start. `deadline` is an absolute time.monotonic() bound shared with the caller."""
    if len(examples) < min_steps:
        raise TooFewSteps(f"Need at least {min_steps} valid edits to adapt; got {len(examples)}.")
    steps = max(1, min(int(steps), MAX_OPT_STEPS))
    t0 = time.monotonic()
    stop_at = t0 + min(max_seconds, MAX_SECONDS)
    if deadline is not None:
        stop_at = min(stop_at, deadline)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    pol = DQNPolicy(base.start.net.op_ids, checkpoint=None, lr=lr, seed=seed)
    pol.net.load_state_dict(copy.deepcopy(base.start.net.state_dict()))
    pol.opt = torch.optim.Adam(pol.net.parameters(), lr=lr)
    prep: Dict[int, Prepared] = {}
    order: List[int] = []
    loss_v, done = None, 0
    pol.net.train()
    for done in range(1, steps + 1):
        if time.monotonic() >= stop_at:
            done -= 1
            break
        if len(order) < min(BATCH, len(examples)):
            order = list(range(len(examples)))
            rng.shuffle(order)
        batch = [order.pop() for _ in range(min(BATCH, len(order)))]
        losses = []
        for i in batch:
            p = prep.get(i) or prep.setdefault(i, Prepared(pol, examples[i]))
            losses.append(F.cross_entropy(batch_scores(pol.net, p)[None], torch.tensor([examples[i].target])))
        loss = torch.stack(losses).mean()
        pol.opt.zero_grad()
        loss.backward()
        pol.opt.step()
        loss_v = float(loss.detach())
        if progress and done % 10 == 0:
            progress(done / steps)
    pol.net.eval()
    return AdaptResult(pol, done, loss_v, time.monotonic() - t0, done < steps)
