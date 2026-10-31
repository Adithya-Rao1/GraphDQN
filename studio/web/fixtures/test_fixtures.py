"""The ui-manual fixtures must stay valid against studio.schema (run: pytest studio/web/fixtures)."""
import json
from pathlib import Path

import pytest

from studio.schema import Event, ManualPreview, ScoreResult

HERE = Path(__file__).parent


@pytest.mark.parametrize("name", ["run_rdkit_local", "run_binding_dev"])
def test_run_fixtures_are_valid_event_streams(name):
    events = json.loads((HERE / f"{name}.json").read_text())
    assert [e["seq"] for e in events] == list(range(len(events)))
    for e in events:
        Event.model_validate(e).typed_payload()
    assert events[0]["type"] == "run_start" and events[-1]["type"] == "run_end"


def test_binding_fixture_is_stamped_mock_and_synthetic():
    events = json.loads((HERE / "run_binding_dev.json").read_text())
    assert events[0]["payload"]["source"] == "mock"
    assert "plapt_affinity" in events[0]["payload"]["seed_score"]["extras"]


def test_live_fixture_has_no_binding_extras():
    events = json.loads((HERE / "run_rdkit_local.json").read_text())
    assert not events[0]["payload"]["seed_score"].get("extras")


def test_manual_fixture_previews_valid_and_consistent():
    fx = json.loads((HERE / "manual_fixture.json").read_text())
    ScoreResult.model_validate(fx["seed"]["score"])
    assert fx["seed"]["smiles"] in fx["applicable"]
    for key, pv in fx["previews"].items():
        smiles, edit_id, site = key.split("|")
        m = ManualPreview.model_validate(pv)
        assert m.ok and m.molblock and m.child_smiles and m.atom_map
        assert smiles in fx["applicable"]
    for smiles, lst in fx["applicable"].items():
        for e in lst:  # every listed edit has a precomputed preview
            assert f'{smiles}|{e["edit_id"]}|{",".join(map(str, e["site"]))}' in fx["previews"]
