"""Engine tests, run against the fakes in studio/engine/fakes.py.  python -m pytest studio/engine"""
from __future__ import annotations

import random
import time

import pytest
from rdkit import Chem

from studio.engine import geometry
from studio.engine.fakes import (
    FakeEdit, FakeEmbedder, FakeExplainer, FakePolicy, FakeRegistry, FakeScorer, PreferencePolicy,
    make_fake_engine,
)
from studio.engine.engine import Engine
from studio.engine.ga import Individual, Locks, Niching, Operators
from studio.schema import Event, RunConfig

SEED = "CCOC(=O)c1ccc(cc1)[N+](=O)[O-]"


def cfg(**kw) -> RunConfig:
    base = dict(seed_smiles=SEED, population_size=8, generations=4, rng_seed=3)
    base.update(kw)
    return RunConfig(**base)


def by_type(events, t):
    return [e.typed_payload() for e in events if e.type == t]


class BadEdit(FakeEdit):
    """Registers on every aromatic CH but returns garbage / invalid / disconnected / dummy-atom products."""

    def apply(self, smiles, site):
        return ["not_a_smiles", "CC.O", "*C", "C(C)(C)(C)(C)C", ""]


class IgnoresLocksRegistry(FakeRegistry):
    """Registry that violates its contract by ignoring locked_atoms: the engine must still protect them."""

    def applicable(self, smiles, locked_atoms=()):
        return super().applicable(smiles, ())


class OnlyBadRegistry:
    def __init__(self):
        self._e = BadEdit("bad", "bad", "atom", "[cH]", lambda rw, i: None)

    def all(self):
        return [self._e]

    def get(self, edit_id):
        return self._e

    def applicable(self, smiles, locked_atoms=()):
        return [(self._e, s) for s in self._e.sites(smiles)]


def engine_with(registry=None, policy=None, **kw) -> Engine:
    return Engine(registry or FakeRegistry(), FakeScorer(), FakeExplainer(), FakeEmbedder(),
                  policy or FakePolicy(), **kw)


# ------------------------------------------------------------------ events / contract
def test_events_validate_and_are_ordered():
    events = list(make_fake_engine().run("t1", cfg()))
    assert [e.seq for e in events] == list(range(len(events)))
    assert events[0].type == "run_start" and events[-1].type == "run_end"
    assert sum(e.type == "run_start" for e in events) == 1
    for e in events:
        Event.model_validate(e.model_dump())
        e.typed_payload()
    gens = [e for e in events if e.type == "generation"]
    assert [e.payload["generation"] for e in gens] == [1, 2, 3, 4]
    # candidates of a generation precede its generation event; metrics follow it
    for i, e in enumerate(events):
        if e.type == "generation":
            assert events[i + 1].type == "policy_metrics"
    start = by_type(events, "run_start")[0]
    assert start.source == "live"
    gen = by_type(events, "generation")[0]
    assert gen.map_xy is not None and len(gen.map_xy) == len(gen.population)


def test_same_seed_is_identical_and_different_seed_differs():
    a = [e.model_dump() for e in make_fake_engine().run("d", cfg(rng_seed=11))]
    b = [e.model_dump() for e in make_fake_engine().run("d", cfg(rng_seed=11))]
    c = [e.model_dump() for e in make_fake_engine().run("d", cfg(rng_seed=12))]
    assert a == b
    assert a != c


def test_story_card_edit_path_and_headline():
    events = list(make_fake_engine().run("s", cfg(generations=6)))
    cands = {c.id: c for c in by_type(events, "candidate")}
    story = by_type(events, "run_end")[0]
    best = by_type(events, "generation")[-1].best
    assert story.final_smiles == best.smiles
    assert story.final_score.composite == pytest.approx(best.composite)
    if story.edit_path:
        assert cands[story.edit_path[-1]].smiles == story.final_smiles
        # path is a connected chain starting at the seed
        assert cands[story.edit_path[0]].parent_smiles == story.seed_smiles
        for prev, nxt in zip(story.edit_path, story.edit_path[1:]):
            assert cands[nxt].parent_smiles == cands[prev].smiles
        assert f"in {len(story.edit_path)} edit" in story.headline
    s, f = story.descriptor_table["qed"]
    assert f"QED {s:.2f} -> {f:.2f}" in story.headline


# ------------------------------------------------------------------ validity
def test_invalid_children_are_rejected():
    events = list(engine_with(OnlyBadRegistry()).run("bad", cfg(policy_guidance=0.0, use_crossover=False)))
    assert events[-1].type == "run_end"
    assert by_type(events, "candidate") == []
    gens = by_type(events, "generation")
    assert all(len(g.population) == 1 for g in gens)  # only the seed survives
    assert "no valid new children" in (by_type(events, "policy_metrics")[0].note or "")


def test_every_emitted_smiles_is_valid_canonical_connected_3d():
    events = list(make_fake_engine().run("v", cfg(generations=5)))
    cands = by_type(events, "candidate")
    assert cands
    for c in cands:
        m = Chem.MolFromSmiles(c.smiles)
        assert m is not None and Chem.MolToSmiles(m) == c.smiles and "." not in c.smiles
        blk = Chem.MolFromMolBlock(c.molblock)
        assert blk is not None and blk.GetNumAtoms() == m.GetNumAtoms()
        assert blk.GetConformer().Is3D()
        assert all(a.GetAtomicNum() > 1 for a in blk.GetAtoms())  # heavy atoms only
        for p, ch in c.atom_map:
            assert 0 <= p < Chem.MolFromSmiles(c.parent_smiles).GetNumAtoms() and 0 <= ch < m.GetNumAtoms()


# ------------------------------------------------------------------ locked atoms
CORE_SEED = "Cc1ccc(F)cc1O"


def _ring_idx(smiles):
    m = Chem.MolFromSmiles(smiles)
    return list(m.GetSubstructMatch(Chem.MolFromSmarts("c1ccccc1")))


@pytest.mark.parametrize("registry", [FakeRegistry(), IgnoresLocksRegistry()])
def test_locked_atoms_survive(registry):
    locked = _ring_idx(CORE_SEED)
    assert len(locked) == 6
    config = cfg(seed_smiles=CORE_SEED, locked_atoms=locked, generations=5, population_size=10)
    events = list(engine_with(registry).run("lock", config))
    assert events[-1].type == "run_end"
    cands = by_type(events, "candidate")
    assert cands
    seed_core = Chem.MolFromSmarts("c1ccccc1")
    for c in cands:
        assert Chem.MolFromSmiles(c.smiles).HasSubstructMatch(seed_core), c.smiles


def test_lock_violation_is_rejected_when_registry_ignores_locks():
    """The ring-CH->N edit on a locked ring atom must never reach the population."""
    locked = _ring_idx(CORE_SEED)
    reg = IgnoresLocksRegistry()
    only_n = [(e, s) for e, s in reg.applicable(CORE_SEED) if e.id == "fake_ch_to_n"]
    assert only_n
    ops = Operators(reg, FakePolicy(), cfg(seed_smiles=CORE_SEED, policy_guidance=0.0, use_crossover=False),
                    random.Random(0), Locks(CORE_SEED, locked, geometry.atom_map))
    reg.applicable = lambda smi, locked_atoms=(): only_n  # type: ignore[method-assign]
    ind = Individual(smiles=Chem.MolToSmiles(Chem.MolFromSmiles(CORE_SEED)), locked=tuple(locked))
    # canonical indices differ from the raw SMILES; rebuild the lock set on the canonical molecule
    canon = ind.smiles
    ind.locked = tuple(_ring_idx(canon))
    ops.locks = Locks(canon, ind.locked, geometry.atom_map)
    ops._cands.clear()
    reg.applicable = lambda smi, locked_atoms=(): [(e, [s[0]]) for e, s in only_n if True][:0] or \
        [(e, s) for e, s in FakeRegistry().applicable(canon) if e.id == "fake_ch_to_n"]
    assert all(ops.mutate(ind) is None for _ in range(20))


def test_locked_indices_follow_noncanonical_seed_order():
    # indices refer to MolFromSmiles(seed_smiles) as given; the engine renumbers them to the canonical seed
    seed = "OC1=CC=C(F)C=C1C"
    m = Chem.MolFromSmiles(seed)
    locked = list(m.GetSubstructMatch(Chem.MolFromSmarts("c1ccccc1")))
    events = list(make_fake_engine().run("nc", cfg(seed_smiles=seed, locked_atoms=locked, generations=3)))
    assert events[-1].type == "run_end"
    for c in by_type(events, "candidate"):
        assert Chem.MolFromSmiles(c.smiles).HasSubstructMatch(Chem.MolFromSmarts("c1ccccc1"))


def test_bad_seed_and_bad_locks_yield_error_event():
    ev = list(make_fake_engine().run("e1", cfg(seed_smiles="not a smiles")))
    assert [e.type for e in ev] == ["error"]
    ev = list(make_fake_engine().run("e2", cfg(locked_atoms=[999])))
    assert [e.type for e in ev] == ["error"] and "out of range" in ev[0].payload["message"]


# ------------------------------------------------------------------ provenance
def test_provenance_present_on_every_candidate():
    events = list(make_fake_engine().run("p", cfg(generations=5)))
    cands = by_type(events, "candidate")
    assert cands
    for c in cands:
        assert c.edit is not None
        assert c.edit.edit_id and c.edit.name and c.edit.category
        assert c.edit.chosen_by in ("policy", "random", "crossover")
        assert c.edit.parents and c.edit.parents[0] == c.parent_smiles
        if c.edit.category == "crossover":
            assert len(c.edit.parents) == 2
        else:
            assert c.edit.site, "mutations record the parent atoms they touched"
        assert c.explanation is not None and c.explanation.rationale
        assert c.parent_molblock
        assert c.explanation.descriptor_deltas["qed"] == pytest.approx(
            c.score.descriptors.qed - engine_parent_qed(events, c))


def engine_parent_qed(events, cand):
    if cand.parent_smiles == by_type(events, "run_end")[0].seed_smiles:
        return by_type(events, "run_start")[0].seed_score.descriptors.qed
    return next(x.score.descriptors.qed for x in by_type(events, "candidate") if x.smiles == cand.parent_smiles)


# ------------------------------------------------------------------ crossover
def test_crossover_produces_valid_molecules():
    a = Individual(smiles=Chem.CanonSmiles("CCOC(=O)c1ccc(cc1)C#N"))
    b = Individual(smiles=Chem.CanonSmiles("CCNC(=O)c1ccc(F)cc1OC"))
    ops = Operators(FakeRegistry(), FakePolicy(), cfg(), random.Random(0), Locks(a.smiles, [], geometry.atom_map))
    made = set()
    for i in range(30):
        ops.rng = random.Random(i)
        p = ops.crossover(a, b)
        if p is not None:
            m = Chem.MolFromSmiles(p.smiles)
            assert m is not None and "." not in p.smiles and p.smiles not in (a.smiles, b.smiles)
            assert p.edit.category == "crossover" and p.edit.chosen_by == "crossover"
            assert p.edit.parents == [a.smiles, b.smiles]
            made.add(p.smiles)
    assert made, "BRICS crossover should produce at least one child for these parents"


def test_engine_emits_crossover_candidates():
    events = list(engine_with(crossover_rate=1.0).run("x", cfg(generations=6, population_size=10)))
    cx = [c for c in by_type(events, "candidate") if c.edit.category == "crossover"]
    assert cx, "no crossover children were produced"
    for c in cx:
        assert Chem.MolFromSmiles(c.smiles) is not None and len(c.edit.parents) == 2


def test_crossover_disabled():
    events = list(engine_with(crossover_rate=1.0).run("nx", cfg(use_crossover=False)))
    assert all(c.edit.category != "crossover" for c in by_type(events, "candidate"))


# ------------------------------------------------------------------ policy
def test_policy_guidance_and_online_update():
    pol = PreferencePolicy("fake_add_hydroxyl")
    events = list(engine_with(policy=pol).run("pol", cfg(policy_guidance=1.0, use_crossover=False, generations=4)))
    muts = [c for c in by_type(events, "candidate")]
    assert muts and all(c.edit.chosen_by == "policy" for c in muts)
    hydroxyl = sum(c.edit.edit_id == "fake_add_hydroxyl" for c in muts)
    assert hydroxyl / len(muts) > 0.5  # softmax(8) on one of ~7 edit types
    pm = by_type(events, "policy_metrics")
    assert all(p.policy_choice_rate == 1.0 for p in pm)
    assert any(p.loss is not None for p in pm)
    assert pol.updates, "Policy.update was never called"
    parent_smiles, op, site, child, delta = pol.updates[0][0]
    assert hasattr(op, "apply") and isinstance(delta, float)


def test_zero_guidance_never_uses_policy():
    pol = PreferencePolicy("fake_add_hydroxyl")
    events = list(engine_with(policy=pol).run("rnd", cfg(policy_guidance=0.0, use_crossover=False)))
    assert all(c.edit.chosen_by == "random" for c in by_type(events, "candidate"))
    assert all(p.policy_choice_rate == 0.0 for p in by_type(events, "policy_metrics"))


def test_policy_failures_do_not_kill_the_run():
    class Broken(FakePolicy):
        def rank(self, *a):
            raise RuntimeError("boom")

        def update(self, t):
            raise RuntimeError("boom")

    events = list(engine_with(policy=Broken()).run("br", cfg(policy_guidance=1.0)))
    assert events[-1].type == "run_end"
    assert "policy" in (by_type(events, "policy_metrics")[0].note or "")


# ------------------------------------------------------------------ GA behaviour
def test_elitism_best_never_decreases_and_population_bounded():
    events = list(make_fake_engine().run("el", cfg(generations=8, population_size=6)))
    gens = by_type(events, "generation")
    bests = [g.best.composite for g in gens]
    assert bests == sorted(bests)
    assert all(len(g.population) <= 6 for g in gens)
    assert all(len({m.smiles for m in g.population}) == len(g.population) for g in gens)


def test_niching_prefers_diverse_members():
    mk = lambda s, q: _scored(s, q)  # noqa: E731
    a, b, c = mk("c1ccccc1CC", 0.9), mk("c1ccccc1CCC", 0.89), mk("OC1CCNCC1", 0.5)
    kept = Niching(threshold=0.5).survive([a, b, c], 2)
    assert {i.smiles for i in kept} == {a.smiles, c.smiles}
    assert {i.smiles for i in Niching(threshold=1.01).survive([a, b, c], 2)} == {a.smiles, b.smiles}


def _scored(smiles, q):
    ind = Individual(smiles=smiles)
    sc = FakeScorer().score(smiles, cfg())
    ind.set_score(sc.model_copy(update={"composite": q}))
    return ind


# ------------------------------------------------------------------ geometry
def test_embed_3d_aligns_child_to_parent():
    _, parent = geometry.embed_3d("CCOC(=O)c1ccccc1")
    _, child = geometry.embed_3d("CCNC(=O)c1ccccc1", ref_mol=parent)
    amap = geometry.atom_map(parent, child)
    assert len(amap) >= 8
    pc, cc = parent.GetConformer(), child.GetConformer()
    rms = (sum((pc.GetAtomPosition(p) - cc.GetAtomPosition(c)).LengthSq() for p, c in amap) / len(amap)) ** 0.5
    assert rms < 1.0


def test_embed_3d_rejects_garbage():
    with pytest.raises(ValueError):
        geometry.embed_3d("not_a_smiles")


# ------------------------------------------------------------------ acceptance
def test_acceptance_pop20_gen10_under_60s():
    config = RunConfig(seed_smiles=SEED, population_size=20, generations=10, rng_seed=0)
    eng = make_fake_engine()
    t = time.perf_counter()
    events = list(eng.run("acc", config))
    dt = time.perf_counter() - t
    assert events[-1].type == "run_end"
    for e in events:
        e.typed_payload()
    assert dt < 60
    print(f"\n[engine] pop20 x gen10: {dt:.1f}s, {eng.stats['children']:.0f} children, "
          f"{eng.stats['children_per_second']:.1f} children/s, {len(events)} events")
