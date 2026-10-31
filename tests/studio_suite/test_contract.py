"""Wave-0 contract tests: mock run validates against the schema, and the stub server streams it."""
import json
from pathlib import Path

from fastapi.testclient import TestClient

from studio import contracts  # noqa: F401  (import check)
from studio.schema import Event, PAYLOAD_MODELS
from studio.server.app import create_app
from studio.server.core import Settings

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


def _mock_client():
    return TestClient(create_app(settings=Settings(force_mock=True, mock_delay=0.0, replay_delay=0.0)))


def test_import_does_not_build_engine():
    import studio.server.app as m
    assert m.app._app is None  # nothing built at import time
    assert isinstance(m.app, m._LazyApp)


def test_server_validate_and_stream():
    c = _mock_client()
    h = c.get("/api/health").json()
    assert h["ok"] is True and h["engine_source"] == "mock" and h["details"]["engine_source"] == "mock"
    ok = c.post("/api/molecules/validate", json={"smiles": "CCO"}).json()
    assert ok["ok"] and ok["heavy_atoms"] == 3
    assert not c.post("/api/molecules/validate", json={"smiles": "not-a-smiles"}).json()["ok"]
    run_id = c.post("/api/runs", json={"seed_smiles": "CCO"}).json()["run_id"]
    body = c.get(f"/api/runs/{run_id}/events?delay=0").text
    types = [ln.split(": ", 1)[1] for ln in body.splitlines() if ln.startswith("event: ")]
    assert types[0] == "run_start" and types[-1] == "run_end" and "candidate" in types
    assert c.get("/api/runs/nope/events").status_code == 404


def test_live_path_small_config():
    """Real engine (studio.wiring) with a tiny config; asserts live source and truthful health."""
    c = TestClient(create_app(settings=Settings(force_mock=False, replay_delay=0.0)))
    h = c.get("/api/health").json()
    d = h["details"]
    assert h["engine_source"] == "live" and d["registry_ops"] > 0
    assert d["scorer"] and d["explainer"] and d["embedder"] and d["policy"]["kind"] in ("dqn", "uniform")
    assert all(h["components"][k] for k in ("engine", "registry", "explain", "embed", "policy"))
    run_id = c.post("/api/runs", json={"seed_smiles": "CCOC(=O)c1ccccc1", "generations": 1,
                                       "population_size": 4, "use_reactions": False}).json()["run_id"]
    body = c.get(f"/api/runs/{run_id}/events").text
    assert '"source":"live"' in body and "event: run_end" in body
