import os
import random

from rdkit import Chem

from studio.contracts import Policy
from studio.policy.features import ATOM_DIM, BOND_DIM, mol_graph
from studio.policy.ops import applicable, load_ops
from studio.policy.policy import DQNPolicy, UniformPolicy, load_policy
from studio.policy.reward import composite
from studio.policy.search import run_ga
from studio.policy.train import DEFAULT_POOL, read_pool

OPS = load_ops()


def test_reward_range_and_invalid():
    assert composite("not a smiles") == 0.0
    for s in read_pool(DEFAULT_POOL)[:10]:
        assert 0.0 <= composite(s) <= 1.0


def test_pool_valid():
    assert len(read_pool(DEFAULT_POOL)) == 64


def test_features_shape():
    x, ei, ea = mol_graph("CCO")
    assert x.shape == (3, ATOM_DIM) and ei.shape == (2, 4) and ea.shape == (4, BOND_DIM)


def test_ops_products_valid():
    n = 0
    for smi in read_pool(DEFAULT_POOL)[:15]:
        for op, site in applicable(smi, OPS):
            for p in op.apply(smi, site):
                assert Chem.MolFromSmiles(p) is not None and Chem.MolToSmiles(Chem.MolFromSmiles(p)) == p
                n += 1
    assert n > 50


def test_protocol_and_interchangeable():
    smi = "CCOC(=O)c1ccc(cc1)[N+](=O)[O-]"
    cands = applicable(smi, OPS)
    for pol in (UniformPolicy(), DQNPolicy([o.id for o in OPS], checkpoint=None)):
        assert isinstance(pol, Policy)
        assert len(pol.rank(smi, cands)) == len(cands)
        r = run_ga(smi, pol, 1.0, seed=1, budget=10, ops=OPS)
        assert r["evals"] <= 10
    assert UniformPolicy().rank(smi, []) == []


def test_update_and_checkpoint(tmp_path):
    p = DQNPolicy([o.id for o in OPS], checkpoint=None)
    smi = "CCOC(=O)c1ccc(cc1)[N+](=O)[O-]"
    op, site = applicable(smi, OPS)[0]
    loss = None
    for _ in range(10):
        loss = p.update([(smi, op.id, site, "CC", 0.5)])
    assert loss is not None and loss >= 0
    path = str(tmp_path / "p.pt")
    p.save(path)
    q = DQNPolicy([o.id for o in OPS], checkpoint=path)
    assert q.checkpoint_loaded == path
    c = applicable(smi, OPS)
    assert q.rank(smi, c) == p.rank(smi, c)


def test_load_policy_fallback(tmp_path):
    assert isinstance(load_policy([o.id for o in OPS], checkpoint=str(tmp_path / "missing.pt")), UniformPolicy)
