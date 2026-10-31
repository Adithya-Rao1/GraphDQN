import types

import pytest

from studio.binding import BindingScorer, FakeBackend
from studio.binding.backends import CachedBackend
from studio.binding.scorer import binding_desirability, tanimoto
from studio.binding.stage import BindingEngine, binding_config
from studio.schema import RunConfig

SEED = "CC(=O)Nc1ccc(O)cc1"                 # paracetamol
SIMILAR = "CC(=O)Nc1ccc(OC)cc1"             # methoxy analogue
DISSIMILAR = "CCCCCCCCCC"                   # alkane, Tanimoto ~0
HEAVY = "CC(=O)Nc1ccc(OCCCCCCCCCCCCCCCCCCCCCC)cc1"  # same core, MW far over cap
STUB = types.SimpleNamespace(sequence="MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQ")


def provider(tid):
    assert tid == "stub"
    return STUB


def cfg(**kw):
    return RunConfig(seed_smiles=SEED, target_id="stub", **kw)


def scorer(overrides=None):
    return BindingScorer(FakeBackend(overrides), provider)


def test_seed_is_half_and_extras_labelled():
    r = scorer({SEED: 6.0}).score(SEED, cfg())
    assert r.composite == pytest.approx(0.5)
    assert r.extras["plapt_pkd"] == 6.0 and r.extras["plapt_affinity"] == 6.0
    assert r.extras["plapt_affinity_uM"] == pytest.approx(1.0)
    assert r.descriptors.mw > 100 and "binding" in r.components


def test_desirability_monotone_and_seed_relative():
    vals = [binding_desirability(p, 6.0) for p in (4, 5, 6, 7, 8)]
    assert vals == sorted(vals) and vals[2] == pytest.approx(0.5)
    s = scorer({SEED: 6.0, SIMILAR: 7.5})
    better, seed = s.score_many([SIMILAR, SEED], cfg())
    assert better.composite > seed.composite
    s2 = scorer({SEED: 6.0, SIMILAR: 5.0})
    assert s2.score(SIMILAR, cfg()).composite < 0.5


def test_similarity_floor_is_hard():
    s = scorer({SEED: 6.0, DISSIMILAR: 9.0})
    assert tanimoto(SEED, DISSIMILAR) < 0.4
    r = s.score(DISSIMILAR, cfg())
    assert r.composite == 0.0 and r.extras["guard_failed"] == 1.0 and "plapt_pkd" not in r.extras
    assert s.score(DISSIMILAR, cfg(min_similarity_to_seed=0.0)).extras["plapt_pkd"] == 9.0


def test_mw_cap_enforced():
    s = scorer({SEED: 6.0, HEAVY: 9.0})
    assert s.score(HEAVY, cfg(min_similarity_to_seed=0.0)).composite == 0.0
    assert s.score(HEAVY, cfg(min_similarity_to_seed=0.0, max_mw=2000)).extras["guard_failed"] == 0.0
    assert s.score(SIMILAR, cfg(max_mw=100)).composite == 0.0


def test_guarded_molecules_never_reach_backend():
    be = FakeBackend()
    BindingScorer(be, provider).score_many([DISSIMILAR, HEAVY], cfg())
    assert be.molecules_scored == 1  # only the seed


def test_new_alert_penalised():
    azo = "CC(=O)Nc1ccc(N=Nc2ccccc2)cc1"
    s = scorer({SEED: 6.0, azo: 6.0})
    r = s.score(azo, cfg(min_similarity_to_seed=0.0, max_mw=1000))
    assert r.extras["new_alerts"] > 0, "azo should trigger an alert in the scorer's alert sets"
    assert r.composite < 0.5


def test_cache_hit(tmp_path):
    inner = FakeBackend()
    cb = CachedBackend(inner, str(tmp_path / "c"))
    a = cb.predict(STUB.sequence, [SEED, SIMILAR])
    assert cb.misses == 2 and inner.molecules_scored == 2
    b = cb.predict(STUB.sequence, ["OC1=CC=C(NC(C)=O)C=C1", SIMILAR])  # non-canonical spelling of the seed
    assert b == a
    assert cb.hits == 2 and inner.molecules_scored == 2
    cb2 = CachedBackend(FakeBackend(), str(tmp_path / "c"))  # persistent across instances
    cb2.predict(STUB.sequence, [SEED])
    assert cb2.hits == 1 and cb2.inner.molecules_scored == 0


def test_fake_is_deterministic_and_labelled():
    a, b = FakeBackend(), FakeBackend()
    assert a.predict("AAAA", [SEED]) == b.predict("AAAA", [SEED])
    assert a.predict("AAAA", [SEED]) != a.predict("CCCC", [SEED])
    assert "Fake" in a.name and "not PLAPT" in a.name


def test_stage_forces_pure_ga():
    c = binding_config(cfg(policy_guidance=0.9))
    assert c.policy_guidance == 0.0 and c.stage == "binding"
    seen = []
    eng = types.SimpleNamespace(run=lambda rid, config: seen.append(config) or iter(()))
    list(BindingEngine(eng, None).run("r", cfg(policy_guidance=0.9)))
    assert seen[0].policy_guidance == 0.0


def test_engine_integration_small_run():
    from studio.wiring import build_engine
    s = scorer()
    eng = BindingEngine(build_engine(s), s)
    events = list(eng.run("t", cfg(generations=2, population_size=6)))
    kinds = {e.type for e in events}
    assert "run_end" in kinds and "error" not in kinds
