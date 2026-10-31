"""DQN-style edit-proposal prior implementing `studio.contracts.Policy`.

Design choice (documented, see README): the original DQN scores *whole child
molecules* (one GNN forward per child, needs the child's graph) with a
15-way output. Here candidates are (EditOp, site) pairs and children are not
materialised at rank time, so instead of mapping ops onto the DQN's fixed
action space we learn an **op-id embedding head**:

    Q(s, op, site) = MLP([ pool(h(s)) ; h(s)[site].mean ; emb(op_id) ])

where h is a small GINE-style message-passing encoder over the same 45/10-dim
graph features as dqn/ (features.py). Q is regressed onto the observed one-step
reward (delta of composite score): a contextual-bandit (gamma = 0) special case
of DQN, with a replay buffer and Huber loss. No target network is needed
because there is no bootstrapping.

Transition tuple accepted by update():
    (parent_smiles, edit_id, site, child_smiles, reward)   or
    (parent_smiles, edit_id, site, reward)
"""
from __future__ import annotations

import os
import random
from collections import deque
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from studio.policy.features import ATOM_DIM, BOND_DIM, mol_graph

CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
DEFAULT_CKPT = os.path.join(CKPT_DIR, "policy.pt")


class QNet(nn.Module):
    def __init__(self, op_ids: Sequence[str], hidden: int = 64, op_dim: int = 16, layers: int = 3):
        super().__init__()
        self.op_ids = list(op_ids)
        self.hidden = hidden
        self.inp = nn.Linear(ATOM_DIM, hidden)
        self.msg = nn.ModuleList([nn.Sequential(nn.Linear(hidden + BOND_DIM, hidden), nn.ReLU()) for _ in range(layers)])
        self.upd = nn.ModuleList([nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU()) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.op_emb = nn.Embedding(len(self.op_ids) + 1, op_dim)  # last row = unknown op
        self.head = nn.Sequential(nn.Linear(2 * hidden + op_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def encode(self, x, ei, ea) -> torch.Tensor:
        h = torch.relu(self.inp(x))
        for msg, upd, norm in zip(self.msg, self.upd, self.norms):
            if ei.shape[1]:
                m = msg(torch.cat([h[ei[0]], ea], dim=1))
                agg = torch.zeros_like(h).index_add_(0, ei[1], m)
            else:
                agg = torch.zeros_like(h)
            h = norm(h + upd(torch.cat([h, agg], dim=1)))
        return h

    def q(self, h: torch.Tensor, site: Sequence[int], op_idx: int) -> torch.Tensor:
        idx = [i for i in site if 0 <= i < h.shape[0]] or list(range(h.shape[0]))
        feat = torch.cat([h.mean(0), h[idx].mean(0), self.op_emb(torch.tensor(op_idx))])
        return self.head(feat).squeeze(-1)


class UniformPolicy:
    """Always-available fallback: every candidate equally preferred."""

    def rank(self, smiles: str, candidates: Sequence[tuple]) -> List[float]:
        return [0.0] * len(candidates)

    def update(self, transitions: Sequence[tuple]) -> Optional[float]:
        return None


class DQNPolicy:
    def __init__(self, op_ids: Sequence[str], checkpoint: Optional[str] = "auto", lr: float = 1e-3,
                 buffer_size: int = 20_000, batch_size: int = 32, seed: int = 0):
        torch.manual_seed(seed)
        self._rng = random.Random(seed)
        self.net = QNet(op_ids)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.buffer: deque = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.checkpoint_loaded: Optional[str] = None
        self._graphs: Dict[str, tuple] = {}
        path = DEFAULT_CKPT if checkpoint == "auto" else checkpoint
        if path and os.path.exists(path):
            self.load(path)

    # -- helpers
    def _op_index(self, op_id: str) -> int:
        try:
            return self.net.op_ids.index(op_id)
        except ValueError:
            return len(self.net.op_ids)

    def _graph(self, smiles: str):
        g = self._graphs.get(smiles)
        if g is None:
            if len(self._graphs) > 5000:
                self._graphs.clear()
            x, ei, ea = mol_graph(smiles)
            g = (torch.from_numpy(x), torch.from_numpy(ei), torch.from_numpy(ea))
            self._graphs[smiles] = g
        return g

    # -- Policy protocol
    def rank(self, smiles: str, candidates: Sequence[tuple]) -> List[float]:
        if not candidates:
            return []
        try:
            g = self._graph(smiles)
        except ValueError:
            return [0.0] * len(candidates)
        with torch.no_grad():
            h = self.net.encode(*g)
            return [float(self.net.q(h, site, self._op_index(op.id))) for op, site in candidates]

    def update(self, transitions: Sequence[tuple]) -> Optional[float]:
        for t in transitions:
            parent, edit_id, site, reward = (t[0], t[1], t[2], t[-1])
            self.buffer.append((parent, edit_id, list(site), float(reward)))
        if len(self.buffer) < min(self.batch_size, 8):
            return None
        batch = self._rng.sample(list(self.buffer), min(self.batch_size, len(self.buffer)))
        preds, targets = [], []
        enc: Dict[str, torch.Tensor] = {}
        for parent, edit_id, site, r in batch:
            if parent not in enc:
                enc[parent] = self.net.encode(*self._graph(parent))
            preds.append(self.net.q(enc[parent], site, self._op_index(edit_id)))
            targets.append(r)
        loss = nn.functional.smooth_l1_loss(torch.stack(preds), torch.tensor(targets, dtype=torch.float32))
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return float(loss)

    # -- persistence
    def save(self, path: str = DEFAULT_CKPT, meta: Optional[dict] = None) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"op_ids": self.net.op_ids, "state": self.net.state_dict(), "meta": meta or {}}, path)

    def load(self, path: str) -> None:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if ck["op_ids"] != self.net.op_ids:
            return  # incompatible op set: stay untrained rather than mis-map ops
        self.net.load_state_dict(ck["state"])
        self.checkpoint_loaded = path


def load_policy(op_ids: Sequence[str], checkpoint: str = "auto") -> object:
    """DQNPolicy if torch + a checkpoint are usable, else UniformPolicy."""
    try:
        p = DQNPolicy(op_ids, checkpoint=checkpoint)
        return p if p.checkpoint_loaded else UniformPolicy()
    except Exception:
        return UniformPolicy()
