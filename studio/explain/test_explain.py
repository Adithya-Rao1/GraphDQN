import time

import pytest
from rdkit import Chem

from studio.explain.contributions import atom_contributions
from studio.explain.explainer import RDKitExplainer
from studio.explain.story import build_story_card
from studio.mock.build_mock import score
from studio.schema import Descriptors, EditRecord

PARENT = "CCOC(=O)c1ccc(cc1)[N+](=O)[O-]"
CHILD = "CCOC(=O)c1ccc(cc1)C#N"
EDIT = EditRecord(edit_id="nitro_to_nitrile", name="Nitro -> nitrile", category="bioisostere",
                  chosen_by="policy", parents=[PARENT])


class FakeOp:
    rationale = "Nitroaromatics are a mutagenicity alert."


@pytest.fixture(scope="module")
def scores():
    return score(PARENT), score(CHILD)


def explain(scores, op=None):
    return RDKitExplainer().explain(PARENT, CHILD, EDIT, scores[0], scores[1], op)


def test_deltas_exact(scores):
    ex = explain(scores)
    p, c = scores[0].descriptors, scores[1].descriptors
    for name in Descriptors.model_fields:
        if getattr(p, name) is None:
            assert name not in ex.descriptor_deltas
        else:
            assert ex.descriptor_deltas[name] == float(getattr(c, name)) - float(getattr(p, name))


def test_nitro_to_nitrile_alerts(scores):
    ex = explain(scores)
    removed = {a.name for a in ex.alerts_removed}
    assert {"nitro_group", "Oxygen-nitrogen_single_bond"} <= removed
    assert not ex.alerts_added
    pm = Chem.MolFromSmiles(PARENT)
    assert all(i < pm.GetNumAtoms() for a in ex.alerts_removed for i in a.atom_indices)
    assert "removes_alert" in ex.tags and "improves_qed" in ex.tags and "adds_alert" not in ex.tags
    rev = RDKitExplainer().explain(CHILD, PARENT, EDIT, scores[1], scores[0])
    assert {a.name for a in rev.alerts_added} == removed and "adds_alert" in rev.tags


def test_contributions_length():
    for smi in (PARENT, CHILD, "C", "[NH4+]", "c1ccccc1O"):
        m = Chem.MolFromSmiles(smi)
        c = atom_contributions(m)
        assert len(c.crippen_logp) == len(c.tpsa) == len(c.gasteiger) == m.GetNumAtoms()
    c = atom_contributions(Chem.MolFromSmiles(CHILD))
    assert sum(c.crippen_logp) == pytest.approx(score(CHILD).descriptors.logp)
    assert sum(c.tpsa) == pytest.approx(score(CHILD).descriptors.tpsa)


def test_rationale_has_true_numbers(scores):
    ex = explain(scores, FakeOp())
    d = ex.descriptor_deltas
    assert f"QED {d['qed']:+.2f}" in ex.rationale
    assert f"TPSA {d['tpsa']:+.0f}" in ex.rationale
    assert "removes 2 structural alerts (nitro_group, Oxygen-nitrogen_single_bond)" in ex.rationale
    assert "Why: Nitroaromatics" in ex.rationale
    assert ex.rationale.startswith("Nitro -> nitrile:")


def test_latency(scores):
    explain(scores)
    t = time.perf_counter()
    for _ in range(20):
        explain(scores)
    assert (time.perf_counter() - t) / 20 < 0.030


def test_story_card(scores):
    sc = build_story_card(PARENT, CHILD, scores[0], scores[1], ["c1"])
    assert sc.headline == f"Removed 2 alerts, QED {scores[0].descriptors.qed:.2f} -> {scores[1].descriptors.qed:.2f} in 1 edit"
    assert sc.descriptor_table["tpsa"] == (scores[0].descriptors.tpsa, scores[1].descriptors.tpsa)
