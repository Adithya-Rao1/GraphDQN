"""Wave-0 contract tests: mock run validates against the schema, and the stub server streams it."""
import json
from pathlib import Path

from fastapi.testclient import TestClient

from studio import contracts  # noqa: F401  (import check)
from studio.schema import Event, PAYLOAD_MODELS
from studio.server.app import app

MOCK = Path(__file__).resolve().parents[2] / "studio" / "mock" / "mock_run.json"


def test_mock_events_validate():
    events = [Event.model_validate(e) for e in json.loads(MOCK.read_text())]
    assert [e.seq for e in events] == list(range(len(events)))
    assert events[0].type == "run_start" and events[-1].type == "run_end"
    for e in events:
        assert isinstance(e.typed_payload(), PAYLOAD_MODELS[e.type])


def test_mock_run_is_labelled_mock():
    first = json.loads(MOCK.read_text())[0]
    assert first["payload"]["source"] == "mock"


def test_server_validate_and_stream():
    c = TestClient(app)
    assert c.get("/api/health").json()["ok"] is True
    ok = c.post("/api/molecules/validate", json={"smiles": "CCO"}).json()
    assert ok["ok"] and ok["heavy_atoms"] == 3
    assert not c.post("/api/molecules/validate", json={"smiles": "not-a-smiles"}).json()["ok"]
    run_id = c.post("/api/runs", json={"seed_smiles": "CCO"}).json()["run_id"]
    body = c.get(f"/api/runs/{run_id}/events?delay=0").text
    types = [ln.split(": ", 1)[1] for ln in body.splitlines() if ln.startswith("event: ")]
    assert types[0] == "run_start" and types[-1] == "run_end" and "candidate" in types
    assert c.get("/api/runs/nope/events").status_code == 404
