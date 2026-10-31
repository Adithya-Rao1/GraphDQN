import json
import time
from pathlib import Path

from fastapi.testclient import TestClient

from studio.schema import Event, SCHEMA_VERSION
from studio.server.app import create_app
from studio.server.core import Settings

MOCK = json.loads((Path(__file__).resolve().parents[1] / "mock" / "mock_run.json").read_text())
SEED = "CCOC(=O)c1ccc([N+](=O)[O-])cc1"


class FakeEngine:
    """Stand-in for the real engine: replays the scripted events as 'live'."""
    def __init__(self, delay=0.0):
        self.delay, self.calls = delay, 0

    def run(self, run_id, config):
        self.calls += 1
        for i, ev in enumerate(MOCK):
            time.sleep(self.delay)
            ev = json.loads(json.dumps(ev))
            if ev["type"] == "run_start":
                ev["payload"]["source"] = "live"
            yield Event(type=ev["type"], run_id=run_id, seq=i, payload=ev["payload"])


def fast(**kw):
    base = dict(mock_delay=0.0, replay_delay=0.0, heartbeat=0.2)
    base.update(kw)
    return Settings(**base)


def client(engine=None, **kw):
    return TestClient(create_app(engine=engine, settings=fast(**kw)))


def read_sse(c, run_id, headers=None):
    events = []
    with c.stream("GET", f"/api/runs/{run_id}/events", headers=headers or {}) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        name = None
        for line in r.iter_lines():
            if line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: "):
                d = json.loads(line[6:])
                assert d["type"] == name
                events.append(d)
    return events


def start(c, **cfg):
    r = c.post("/api/runs", json={"seed_smiles": SEED, **cfg})
    return r


def test_health_and_presets():
    c = client()
    h = c.get("/api/health").json()
    assert h["ok"] and h["engine_source"] == "mock" and h["components"]["engine"] is False
    assert c.get("/api/presets").json() == [] or isinstance(c.get("/api/presets").json(), list)


def test_force_mock_overrides_engine():
    c = client(engine=FakeEngine(), force_mock=True)
    assert c.get("/api/health").json()["engine_source"] == "mock"


def test_validate_ok_with_3d():
    r = client().post("/api/molecules/validate", json={"smiles": "c1ccccc1O"}).json()
    assert r["ok"] and r["heavy_atoms"] == 7 and "M  END" in r["molblock"]
    assert " H " not in r["molblock"]


def test_validate_molfile_and_sdf():
    c = client()
    block = c.post("/api/molecules/validate", json={"smiles": "CCO"}).json()["molblock"]
    assert c.post("/api/molecules/validate", json={"molfile": block}).json()["canonical_smiles"] == "CCO"
    sdf = block + "\n$$$$\n" + block + "\n$$$$\n"
    assert c.post("/api/molecules/validate", json={"molfile": sdf}).json()["ok"]


def test_validate_rejections():
    c = client()
    bad = ["not a smiles", "C(C)(C)(C)(C)C", "", "CCO\nCCN", "C" * 61]
    for smi in bad:
        assert c.post("/api/molecules/validate", json={"smiles": smi}).json()["ok"] is False, smi
    assert not c.post("/api/molecules/validate", json={"molfile": "hello"}).json()["ok"]
    assert not c.post("/api/molecules/validate", json={"molfile": "\x80\x04\x95\x00 pickle"}).json()["ok"]
    assert not c.post("/api/molecules/validate", json={"smiles": "C\x00C"}).json()["ok"]
    assert not c.post("/api/molecules/validate", json={}).json()["ok"]


def test_validate_body_limit():
    c = client()
    r = c.post("/api/molecules/validate", json={"molfile": "A" * 250_000})
    assert r.status_code == 413


def test_validate_keeps_largest_fragment():
    r = client().post("/api/molecules/validate", json={"smiles": "CCO.[Na+].[Cl-]"}).json()
    assert r["ok"] and r["canonical_smiles"] == "CCO"


def test_mock_sse_ordering_and_label():
    c = client()
    rid = start(c).json()["run_id"]
    evs = read_sse(c, rid)
    assert [e["seq"] for e in evs] == list(range(len(evs)))
    assert evs[0]["type"] == "run_start" and evs[-1]["type"] == "run_end"
    assert evs[0]["payload"]["source"] == "mock"
    assert all(e["run_id"] == rid for e in evs)
    for e in evs:
        Event.model_validate(e).typed_payload()


def test_late_join_and_last_event_id():
    c = client()
    rid = start(c).json()["run_id"]
    first = read_sse(c, rid)
    again = read_sse(c, rid)  # late joiner replays from seq 0
    assert [e["seq"] for e in again] == [e["seq"] for e in first]
    tail = read_sse(c, rid, {"Last-Event-ID": "4"})
    assert tail[0]["seq"] == 5


def test_unknown_run_404():
    assert client().get("/api/runs/nope/events").status_code == 404


def test_run_validation_and_caps():
    c = client()
    assert start(c).status_code == 200
    assert c.post("/api/runs", json={"seed_smiles": "xxx"}).status_code == 422
    assert c.post("/api/runs", json={"seed_smiles": "CCO", "locked_atoms": [99]}).status_code == 422
    c2 = client()
    rid = start(c2, generations=9999, population_size=9999).json()["run_id"]
    cfg = c2.app.state.manager.get(rid).config  # config handed to the engine (the mock ignores it)
    assert cfg.generations == 15 and cfg.population_size == 40


def test_per_session_limit_429():
    c = client(mock_delay=0.3)
    assert start(c).status_code == 200
    r = start(c)
    assert r.status_code == 429 and int(r.headers["Retry-After"]) >= 1
    # a different session is fine
    assert start(client(mock_delay=0.3)).status_code == 200


def test_global_limit_and_queue_429():
    app = create_app(settings=fast(mock_delay=0.5, max_concurrent=1, max_queue=1))
    clients = [TestClient(app) for _ in range(3)]
    codes = [start(x).status_code for x in clients]
    assert codes == [200, 200, 429]
    assert "Retry-After" in start(clients[2]).headers


def test_rate_limit():
    c = client(runs_per_minute=2)
    for _ in range(2):
        rid = start(c).json()["run_id"]
        read_sse(c, rid)
    r = start(c)
    assert r.status_code == 429 and "Retry-After" in r.headers


def test_cancel():
    c = client(mock_delay=0.3)
    rid = start(c).json()["run_id"]
    assert c.delete(f"/api/runs/{rid}").json()["cancelled"] is True
    evs = read_sse(c, rid)
    assert evs[-1]["type"] == "error" and evs[-1]["payload"]["recoverable"]


def test_live_run_then_cache_replay():
    eng = FakeEngine()
    c = client(engine=eng)
    r1 = start(c).json()["run_id"]
    live = read_sse(c, r1)
    assert live[0]["payload"]["source"] == "live" and live[-1]["type"] == "run_end"
    time.sleep(0.1)
    # same seed (different but equivalent SMILES) + same config + different preset id -> cache hit
    r2 = c.post("/api/runs", json={"seed_smiles": "O=C(OCC)c1ccc(cc1)[N+](=O)[O-]", "preset_id": "x"})
    # config differs only by seed spelling -> canonicalised -> hit
    evs = read_sse(c, r2.json()["run_id"])
    assert eng.calls == 1
    assert evs[0]["payload"]["source"] == "replay"
    assert [e["type"] for e in evs] == [e["type"] for e in live]
    assert [e["seq"] for e in evs] == list(range(len(evs)))


def test_mock_runs_are_not_cached_as_replay():
    c = client()
    read_sse(c, start(c).json()["run_id"])
    evs = read_sse(c, start(c).json()["run_id"])
    assert evs[0]["payload"]["source"] == "mock"


def test_timeout():
    c = client(engine=FakeEngine(delay=0.4), run_timeout=0.7)
    evs = read_sse(c, start(c).json()["run_id"])
    assert evs[-1]["type"] == "error" and "time limit" in evs[-1]["payload"]["message"]
    assert evs[-1]["payload"]["recoverable"] is True
    assert len(evs) < len(MOCK)
    # timed-out runs are not cached
    assert not c.app.state.manager.cache


def test_engine_crash_reported():
    class Boom:
        def run(self, run_id, config):
            raise RuntimeError("secret detail")
            yield
    c = client(engine=Boom())
    evs = read_sse(c, start(c).json()["run_id"])
    assert evs[-1]["type"] == "error" and "secret" not in evs[-1]["payload"]["message"]


def test_heartbeat():
    c = client(engine=FakeEngine(delay=0.5), heartbeat=0.1, run_timeout=5)
    rid = start(c).json()["run_id"]
    with c.stream("GET", f"/api/runs/{rid}/events") as r:
        assert any(l.startswith(": heartbeat") for l in r.iter_lines())
