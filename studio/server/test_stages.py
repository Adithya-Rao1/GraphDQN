"""Tests for targets / data / stages / poses / manual / auth. Everything external is faked."""
import json
import time

import pytest
from fastapi.testclient import TestClient

from studio.schema import Event, PoseResult
from studio.server import poses
from studio.server.app import create_app
from studio.server.stages import Services
from studio.server.test_server import FakeEngine, SEED, fast

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"


class FakePoseClient:
    def __init__(self):
        self.calls, self.n = [], 0

    def dock(self, target_id, smiles):
        self.calls.append((target_id, smiles))
        return "r1"

    def job(self, rid):
        self.n += 1
        if self.n < 2:
            return {"status": "running", "progress": 0.5}
        return {"status": "done", "ligand_sdf": "FAKE-SDF", "protein_pdb": "FAKE-PDB",
                "pose_confidence": 0.8, "affinity": 6.5, "note": "fake client"}


class BindingEngine:
    def __init__(self):
        self.configs = []

    def run(self, run_id, config):
        self.configs.append(config)
        return FakeEngine().run(run_id, config)


class FakeBindingScorer:
    def score(self, smiles, config):
        from studio.scoring import RDKitScorer
        r = RDKitScorer().score(smiles, config)
        return r.model_copy(update={"extras": {"plapt_affinity": 6.0 + 0.01 * len(smiles)}})


@pytest.fixture(autouse=True)
def dirs(tmp_path, monkeypatch):
    monkeypatch.setenv("STUDIO_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("STUDIO_STRUCTURES_DIR", str(tmp_path / "structures"))
    monkeypatch.setenv("STUDIO_POSES_DIR", str(tmp_path / "poses"))
    monkeypatch.delenv("STUDIO_API_KEY", raising=False)
    # These tests exercise the fixture-target path; the real loader (studio.data) now exists on integrate.
    monkeypatch.setattr("studio.server.stages.Services._loader", lambda self: self._targets, raising=True)
    (tmp_path / "structures").mkdir()
    (tmp_path / "structures" / "stub.pdb").write_text("ATOM  FAKE\n")
    (tmp_path / "structures" / "notes.txt").write_text("nope")
    (tmp_path / "secret.pdb").write_text("outside")
    return tmp_path


def make(engine=None, svc=None, **kw):
    svc = svc or Services()
    return TestClient(create_app(engine=engine or FakeEngine(), settings=fast(**kw), services=svc)), svc


# ------------------------------------------------------------------ targets / data
def test_targets_fixture():
    c, _ = make()
    ts = c.get("/api/targets").json()
    assert [t["id"] for t in ts] == ["stub"] and "STUB" in ts[0]["name"]
    assert c.get("/api/targets/stub").json()["pdb_id"] == "0000"
    assert c.get("/api/targets/nope").status_code == 404
    assert c.get("/api/targets/..%2Fx").status_code == 404


def test_data_serving_and_traversal(dirs):
    c, _ = make()
    assert c.get("/api/data/stub.pdb").text.startswith("ATOM")
    assert c.get("/api/data/structures/stub.pdb").status_code == 200
    for bad in ["notes.txt", "missing.pdb", "../secret.pdb", "%2e%2e/secret.pdb", "..%2Fsecret.pdb",
                "a/../../secret.pdb", "%2Fetc%2Fpasswd", ".hidden.pdb", "stub.pdb%00.txt", "stub.pdb/"]:
        assert c.get(f"/api/data/{bad}").status_code == 404, bad


def test_data_symlink_escape(dirs):
    (dirs / "structures" / "link.pdb").symlink_to(dirs / "secret.pdb")
    c, _ = make()
    assert c.get("/api/data/link.pdb").status_code == 404


# ------------------------------------------------------------------ stages
def test_binding_503_when_unavailable(monkeypatch):
    c, svc = make()
    monkeypatch.setattr(Services, "binding_available", lambda self: False)
    r = c.post("/api/runs", json={"seed_smiles": SEED, "stage": "binding", "target_id": "stub"})
    assert r.status_code == 503 and "not available" in r.json()["detail"]


def test_binding_stage_routes_and_caps():
    be = BindingEngine()
    base = FakeEngine()
    c, _ = make(engine=base, svc=Services(binding_factory=lambda target_id: be))
    r = c.post("/api/runs", json={"seed_smiles": SEED, "stage": "binding", "target_id": "stub",
                                  "generations": 999, "population_size": 999, "policy_guidance": 0.9,
                                  "max_mw": 5000})
    assert r.status_code == 200
    run = c.app.state.manager.get(r.json()["run_id"])
    for _ in range(100):
        if run.finished:
            break
        time.sleep(0.05)
    assert base.calls == 0 and len(be.configs) == 1
    cfg = be.configs[0]
    assert cfg.policy_guidance == 0 and cfg.generations <= 10 and cfg.population_size <= 30 and cfg.max_mw == 700


def test_stage_validation():
    c, _ = make(svc=Services(binding_factory=lambda: BindingEngine()))
    assert c.post("/api/runs", json={"seed_smiles": SEED, "stage": "binding"}).status_code == 422
    assert c.post("/api/runs", json={"seed_smiles": SEED, "stage": "binding", "target_id": "zzz"}).status_code == 422
    assert c.post("/api/runs", json={"seed_smiles": SEED, "stage": "manual"}).status_code == 422
    assert c.post("/api/runs", json={"seed_smiles": SEED, "target_id": "zzz"}).status_code == 422
    assert c.post("/api/runs", json={"seed_smiles": SEED, "stage": "pharmacology", "target_id": "stub"}).status_code == 200


def test_binding_needs_live_engine():
    c = TestClient(create_app(settings=fast(force_mock=True), services=Services(binding_factory=lambda: None)))
    r = c.post("/api/runs", json={"seed_smiles": SEED, "stage": "binding", "target_id": "stub"})
    assert r.status_code == 503 and "mock" in r.json()["detail"]


def test_pharmacology_unchanged():
    c, _ = make()
    r = c.post("/api/runs", json={"seed_smiles": SEED})
    assert r.status_code == 200


# ------------------------------------------------------------------ auth
def test_auth(monkeypatch):
    monkeypatch.setenv("STUDIO_API_KEY", "s3cret")
    c, _ = make()
    assert c.get("/api/health").status_code == 200
    assert c.get("/api/health").json()["components"]["api_auth"] is True
    for path in ["/api/targets", "/api/presets", "/api/data/stub.pdb"]:
        assert c.get(path).status_code == 401
    assert c.get("/api/targets", headers={"X-Studio-Key": "wrong"}).status_code == 401
    assert c.post("/api/runs", json={"seed_smiles": SEED}).status_code == 401
    assert c.get("/api/targets", headers={"X-Studio-Key": "s3cret"}).status_code == 200
    assert c.post("/api/runs", json={"seed_smiles": SEED}, headers={"X-Studio-Key": "s3cret"}).status_code == 200


def test_no_auth_by_default():
    c, _ = make()
    assert c.get("/api/targets").status_code == 200


# ------------------------------------------------------------------ poses
def wait_pose(c, jid, want=("done", "failed")):
    for _ in range(200):
        j = c.get(f"/api/poses/{jid}").json()
        if j["status"] in want:
            return j
        time.sleep(0.02)
    raise AssertionError(j)


def test_pose_service_flow(monkeypatch):
    monkeypatch.setenv("DYNAMICBIND_POLL", "0.01")
    fc = FakePoseClient()
    c, _ = make(svc=Services(pose_client=fc))
    r = c.post("/api/poses", json={"target_id": "stub", "smiles": ASPIRIN})
    assert r.status_code == 200 and r.json()["status"] == "queued"
    j = wait_pose(c, r.json()["job_id"])
    assert j["status"] == "done" and j["source"] == "dynamicbind" and j["ligand_sdf"] == "FAKE-SDF"
    assert j["apo_pdb"].startswith("ATOM") and j["affinity"] == 6.5
    # same molecule again reuses the job, no second service call
    assert c.post("/api/poses", json={"target_id": "stub", "smiles": ASPIRIN}).json()["job_id"] == r.json()["job_id"]
    assert len(fc.calls) == 1


def test_pose_precomputed(dirs):
    from rdkit import Chem
    canon = Chem.MolToSmiles(Chem.MolFromSmiles(ASPIRIN))
    d = dirs / "poses" / "stub"
    d.mkdir(parents=True)
    pre = PoseResult(job_id="old", status="done", target_id="stub", smiles=canon, ligand_sdf="PRE-SDF",
                     protein_pdb="PRE-PDB", source="dynamicbind", note="from disk")
    (d / f"{poses.smiles_hash(canon)}.json").write_text(pre.model_dump_json())
    fc = FakePoseClient()
    c, _ = make(svc=Services(pose_client=fc))
    r = c.post("/api/poses", json={"target_id": "stub", "smiles": "OC(=O)c1ccccc1OC(C)=O"}).json()
    assert r["status"] == "done" and r["source"] == "precomputed" and r["ligand_sdf"] == "PRE-SDF"
    assert r["job_id"] != "old" and c.get(f"/api/poses/{r['job_id']}").json()["status"] == "done"
    assert fc.calls == []


def test_pose_service_unreachable(monkeypatch):
    monkeypatch.setenv("DYNAMICBIND_URL", "http://127.0.0.1:1")
    c, _ = make()
    r = c.post("/api/poses", json={"target_id": "stub", "smiles": ASPIRIN}).json()
    j = wait_pose(c, r["job_id"])
    assert j["status"] == "failed" and "unreachable" in j["note"] and j["ligand_sdf"] is None


def test_pose_service_reports_failure(monkeypatch):
    monkeypatch.setenv("DYNAMICBIND_POLL", "0.01")

    class Bad(FakePoseClient):
        def job(self, rid):
            return {"status": "failed", "note": "oom"}
    c, _ = make(svc=Services(pose_client=Bad()))
    j = wait_pose(c, c.post("/api/poses", json={"target_id": "stub", "smiles": ASPIRIN}).json()["job_id"])
    assert j["status"] == "failed" and "oom" in j["note"]


def test_pose_validation():
    c, _ = make(svc=Services(pose_client=FakePoseClient()))
    assert c.post("/api/poses", json={"target_id": "nope", "smiles": ASPIRIN}).status_code == 422
    assert c.post("/api/poses", json={"target_id": "stub", "smiles": "not a smiles"}).status_code == 422
    assert c.get("/api/poses/doesnotexist").status_code == 404


def test_pose_ttl(monkeypatch):
    monkeypatch.setenv("DYNAMICBIND_POLL", "0.01")
    c, _ = make(svc=Services(pose_client=FakePoseClient()))
    jid = c.post("/api/poses", json={"target_id": "stub", "smiles": ASPIRIN}).json()["job_id"]
    wait_pose(c, jid)
    c.app.state.poses.ttl = -1
    assert c.get(f"/api/poses/{jid}").status_code == 404


# ------------------------------------------------------------------ manual
def first_site(c, smiles, locked=""):
    apps = c.get("/api/manual/applicable", params={"smiles": smiles, "locked": locked}).json()
    assert apps, "registry should offer edits"
    return apps


def test_manual_applicable():
    c, _ = make()
    apps = first_site(c, SEED)
    assert set(apps[0]) == {"edit_id", "name", "category", "site", "rationale"}
    atoms = {i for a in apps for i in a["site"]}
    locked = ",".join(str(i) for i in sorted(atoms))
    assert c.get("/api/manual/applicable", params={"smiles": SEED, "locked": locked}).json() == []
    assert c.get("/api/manual/applicable", params={"smiles": "xx"}).status_code == 422
    assert c.get("/api/manual/applicable", params={"smiles": SEED, "locked": "a"}).status_code == 422


def test_manual_preview_and_session(dirs):
    c, _ = make()
    a = first_site(c, SEED)
    body = None
    for cand in a:  # first edit that yields a product
        r = c.post("/api/manual/preview", json={"smiles": SEED, "edit_id": cand["edit_id"], "site": cand["site"]}).json()
        if r["ok"]:
            body = (cand, r)
            break
    assert body, "at least one applicable edit must preview"
    cand, r = body
    assert r["child_smiles"] and "M  END" in r["molblock"] and r["atom_map"]
    assert r["score"]["descriptors"]["mw"] > 0 and r["explanation"]["rationale"]
    assert "tpsa" in r["explanation"]["descriptor_deltas"]

    s = c.post("/api/manual/sessions", json={"seed_smiles": SEED, "target_id": "stub"}).json()
    sid = s["session_id"]
    step = {"parent_smiles": SEED, "child_smiles": r["child_smiles"], "edit_id": cand["edit_id"], "site": cand["site"]}
    out = c.post(f"/api/manual/sessions/{sid}/steps", json=step).json()
    assert len(out["steps"]) == 1 and out["steps"][0]["reward_delta"] is not None
    c.post(f"/api/manual/sessions/{sid}/steps", json={**step, "reward_delta": 0.25})
    ex = c.get(f"/api/manual/sessions/{sid}/export")
    assert ex.headers["content-type"].startswith("application/x-ndjson") and "attachment" in ex.headers["content-disposition"]
    lines = [json.loads(x) for x in ex.text.splitlines()]
    assert len(lines) == 2 and lines[1]["reward"] == 0.25
    for ln in lines:
        assert {"parent_smiles", "edit_id", "site", "child_smiles", "reward"} <= set(ln) and ln["chosen_by"] == "user"
    logged = (dirs / "cache" / "manual" / f"{sid}.jsonl").read_text().splitlines()
    assert [json.loads(x) for x in logged] == lines


def test_manual_preview_failures():
    c, _ = make()
    cand = first_site(c, SEED)[0]
    ok = {"smiles": SEED, "edit_id": cand["edit_id"], "site": cand["site"]}
    assert c.post("/api/manual/preview", json={**ok, "edit_id": "nope"}).json()["ok"] is False
    assert c.post("/api/manual/preview", json={**ok, "site": [999]}).json()["reason"] == "site index out of range"
    assert c.post("/api/manual/preview", json={**ok, "smiles": "zz"}).status_code == 422
    assert c.post("/api/manual/preview", json={**ok, "target_id": "nope"}).status_code == 422


def test_manual_preview_binding_extras():
    c, _ = make(svc=Services(binding_scorer=FakeBindingScorer()))
    for cand in first_site(c, SEED):
        r = c.post("/api/manual/preview", json={"smiles": SEED, "edit_id": cand["edit_id"], "site": cand["site"],
                                                "target_id": "stub"}).json()
        if r["ok"]:
            assert "plapt_affinity" in r["score"]["extras"]
            return
    raise AssertionError("no edit previewed")


def test_manual_sessions_errors():
    c, _ = make()
    assert c.post("/api/manual/sessions", json={"seed_smiles": "zz"}).status_code == 422
    assert c.post("/api/manual/sessions", json={"seed_smiles": SEED, "target_id": "nope"}).status_code == 422
    step = {"parent_smiles": SEED, "child_smiles": SEED, "edit_id": "x", "site": [0]}
    assert c.post("/api/manual/sessions/none/steps", json=step).status_code == 404
    assert c.get("/api/manual/sessions/none/export").status_code == 404
    sid = c.post("/api/manual/sessions", json={"seed_smiles": SEED}).json()["session_id"]
    assert c.post(f"/api/manual/sessions/{sid}/steps", json={**step, "child_smiles": "zz"}).status_code == 422
