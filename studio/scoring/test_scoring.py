import statistics
import time

from studio.schema import ObjectiveSpec, RunConfig
from studio.scoring import PRESETS, RDKitScorer, desirability
from studio.scoring.descriptors import SA_SOURCE

SEED = "CCOC(=O)c1ccc(cc1)[N+](=O)[O-]"
NITRILE = "CCOC(=O)c1ccc(cc1)C#N"
AMIDE = "CCNC(=O)c1ccc(cc1)C#N"
AMIDE2 = "CCNC(=O)c1ccc(cc1)C(N)=O"


def cfg(**kw):
    return RunConfig(seed_smiles=SEED, **kw)


def test_range_monotone_falloff():
    s = ObjectiveSpec(direction="range", low=1, high=3)
    assert desirability("logp", 2, s) == 1.0
    assert desirability("logp", 3, s) == 1.0
    assert desirability("logp", 4, s) < desirability("logp", 3.5, s) < 1.0
    assert desirability("logp", -1, s) < desirability("logp", 0, s) < 1.0


def test_maximize_minimize_monotone():
    mx = ObjectiveSpec(direction="maximize", low=0.3, high=0.9)
    mn = ObjectiveSpec(direction="minimize", low=0, high=2)
    xs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    up = [desirability("qed", x, mx) for x in xs]
    assert up == sorted(up) and up[0] < up[-1]
    ys = [0, 1, 2, 3, 4]
    down = [desirability("alerts", y, mn) for y in ys]
    assert down == sorted(down, reverse=True) and down[0] > down[-1]
    assert all(0 <= v <= 1 for v in up + down)


def test_cache_hits():
    sc = RDKitScorer()
    c = cfg()
    a = sc.score(SEED, c)
    b = sc.score(SEED, c)
    assert a is b and sc._score_cache.hits >= 1
    # non-canonical spelling + different objectives: score miss, descriptor hit
    sc.score("O=C(OCC)c1ccc([N+](=O)[O-])cc1", cfg(objectives=PRESETS["more soluble"]))
    assert sc._desc_cache.hits >= 1


def test_alerts_drop_along_mock_path():
    sc = RDKitScorer()
    c = cfg()
    seed = sc.score(SEED, c)
    assert seed.descriptors.alerts > 0 and seed.alerts
    assert all(h.atom_indices and h.family in ("PAINS", "BRENK", "NIH") for h in seed.alerts)
    for smi in (NITRILE, AMIDE, AMIDE2):
        assert sc.score(smi, c).descriptors.alerts < seed.descriptors.alerts


def test_weights_change_composite():
    sc = RDKitScorer()
    a = {"alerts": ObjectiveSpec(weight=5, direction="minimize", low=0, high=2),
         "logp": ObjectiveSpec(weight=1, direction="range", low=5, high=6)}
    b = {"alerts": ObjectiveSpec(weight=1, direction="minimize", low=0, high=2),
         "logp": ObjectiveSpec(weight=5, direction="range", low=5, high=6)}
    assert sc.score(SEED, cfg(objectives=a)).composite != sc.score(SEED, cfg(objectives=b)).composite


def test_presets_valid_and_sa_reported():
    sc = RDKitScorer()
    for name, obj in PRESETS.items():
        r = sc.score(SEED, cfg(objectives=obj))
        assert 0 <= r.composite <= 1, name
    d = sc.descriptors(SEED)[0]
    assert (d.sa is not None) == (SA_SOURCE != "unavailable")


def test_speed():
    mols = ["CC(C)Cc1ccc(cc1)C(C)C(=O)O", "CN1CCC[C@H]1c1cccnc1", "COc1ccc2[nH]cc(CCNC(C)=O)c2c1",
            "CC(=O)Nc1ccc(O)cc1", "O=C(Nc1ccccc1)c1ccc(cc1)S(=O)(=O)N1CCCC1",
            "CCN(CC)C(=O)c1cccc(Cn2cnc3ccccc32)c1", "Cc1ccc(cc1)S(=O)(=O)Nc1nccs1",
            "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1"]
    sc = RDKitScorer()
    c = cfg()
    ts = []
    for _ in range(5):
        for m in mols:
            sc = RDKitScorer()
            t = time.perf_counter(); sc.score(m, c); ts.append((time.perf_counter() - t) * 1000)
    ts.sort()
    med, p95 = statistics.median(ts), ts[int(0.95 * len(ts)) - 1]
    print(f"uncached median {med:.1f} ms, p95 {p95:.1f} ms, SA={SA_SOURCE}")
    assert med < 20
