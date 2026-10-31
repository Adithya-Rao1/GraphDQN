"""Tests use the real registry. Transitions are SYNTHETIC (always-raise-TPSA scientist), see synthetic.py."""
import threading
import time

import pytest
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from studio.adapt import pipeline, store
from studio.adapt.dataset import clean
from studio.adapt.evaluate import evaluate, hit_prob, scores_for
from studio.adapt.finetune import Prepared, TooFewSteps, adapt, batch_scores, resolve_base
from studio.adapt.install import install
from studio.adapt.jobs import JobManager, QueueFull
from studio.adapt.policy_factory import load_adapted_policy
from studio.adapt.synthetic import make_transitions
from studio.policy.policy import DQNPolicy, UniformPolicy
from studio.registry import get_registry


@pytest.fixture(scope="module")
def reg():
    return get_registry()


@pytest.fixture(scope="module")
def synth(reg):
    return make_transitions(reg, 40)


@pytest.fixture(scope="module")
def base(reg):
    return resolve_base([o.id for o in reg.all()])


@pytest.fixture(autouse=True)
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("STUDIO_CACHE_DIR", str(tmp_path))


def test_invalid_steps_dropped_and_reported(reg, synth):
    good = synth[:6]
    bad = [
        {**good[0], "parent_smiles": "not-a-smiles"},
        {**good[1], "child_smiles": "C(("},
        {**good[2], "edit_id": "no_such_edit"},
        {**good[3], "site": [9999]},
        {**good[4], "child_smiles": "CCCCCCCC"},         # not what the edit produces
        {**good[0]},                                        # duplicate
        {"parent_smiles": "CC"},                            # malformed
        {**good[5], "reward": float("nan")},
    ]
    ex, rep = clean(good + bad, reg)
    assert rep.valid == len(ex) == 6 and rep.dropped == len(bad) and rep.received == 6 + len(bad)
    assert rep.reasons["invalid_parent"] == 1 and rep.reasons["invalid_child"] == 1
    assert rep.reasons["unknown_edit"] == 1 and rep.reasons["not_applicable"] == 1
    assert rep.reasons["child_mismatch"] == 1 and rep.reasons["duplicate"] == 1 and rep.reasons["malformed"] == 2


def test_fewer_than_five_refused(reg, synth, base):
    ex, _ = clean(synth[:4], reg)
    with pytest.raises(TooFewSteps, match="at least 5"):
        adapt(ex, base)
    with pytest.raises(TooFewSteps):
        pipeline.run(synth[:4], "o", registry=reg, base=base)


def test_batched_scores_match_qnet(reg, synth):
    ex, _ = clean(synth[:3], reg)
    pol = DQNPolicy([o.id for o in reg.all()], checkpoint=None)
    ref = pol.rank(ex[0].transition.parent_smiles, ex[0].candidates)
    with torch.no_grad():
        got = batch_scores(pol.net, Prepared(pol, ex[0])).tolist()
    assert got == pytest.approx(ref, abs=1e-4)


def test_hit_prob_ties_are_expected_value():
    assert hit_prob([0.0] * 10, 3, 1) == pytest.approx(0.1)
    assert hit_prob([0.0] * 10, 3, 3) == pytest.approx(0.3)
    assert hit_prob([1, 2, 3], 0, 1) == 0.0 and hit_prob([3, 2, 1], 0, 1) == 1.0


def test_adapted_beats_base_on_heldout_synthetic(reg, synth, base):
    ex, _ = clean(synth, reg)
    assert len(ex) == 40
    ev = evaluate(ex, base, steps=200)
    assert ev["method"] == "80/20 holdout" and ev["n"] == 8
    assert ev["adapted"]["top1"] > ev["base"]["top1"] + 0.2
    assert ev["adapted"]["top3"] > ev["random"]["top3"]
    assert "Small sample" in ev["note"] and "uniform fallback" in ev["note"]


def test_small_n_uses_leave_one_out(reg, synth, base):
    ex, _ = clean(synth[:12], reg)
    ev = evaluate(ex, base, steps=60)
    assert ev["method"] == "leave-one-out" and ev["n"] == 12 and "Small sample" in ev["note"]
    for k in ("base", "adapted", "random"):
        lo, hi = ev[k]["top1_ci95"]
        assert lo <= ev[k]["top1"] <= hi


def test_wallclock_cap_enforced(reg, synth, base):
    ex, _ = clean(synth[:10], reg)
    t0 = time.monotonic()
    r = adapt(ex, base, steps=300, max_seconds=0.5)
    assert r.stopped_early and r.steps_done < 300 and time.monotonic() - t0 < 5
    r2 = adapt(ex, base, steps=300, max_seconds=999, deadline=time.monotonic() + 0.3)
    assert r2.stopped_early
    assert adapt(ex, base, steps=10_000, max_seconds=0.01).steps_done <= 300


def test_training_does_not_touch_base_and_is_deterministic(reg, synth):
    ids = [o.id for o in reg.all()]
    b = resolve_base(ids)
    before = {k: v.clone() for k, v in b.start.net.state_dict().items()}
    ex, _ = clean(synth[:10], reg)
    a1, a2 = adapt(ex, b, steps=20), adapt(ex, b, steps=20)
    assert all(torch.equal(before[k], v) for k, v in b.start.net.state_dict().items())
    assert all(torch.equal(a1.policy.net.state_dict()[k], a2.policy.net.state_dict()[k]) for k in before)
    assert not all(torch.equal(before[k], a1.policy.net.state_dict()[k]) for k in before)


def test_store_roundtrip_and_fallback(reg, synth, base):
    ids = [o.id for o in reg.all()]
    res = pipeline.run(synth[:10], "owner-a", registry=reg, base=base, steps=20)
    aid = res["adapter_id"]
    meta = store.get_meta("owner-a", aid)
    assert meta["n_steps"] == 10 and meta["op_id_hash"] == store.op_hash(ids) and meta["base_kind"] == "uniform_fallback"
    pol = load_adapted_policy(aid, "owner-a", ids)
    assert isinstance(pol, DQNPolicy) and pol.adapter_info["used"]
    # another owner cannot load it; missing file falls back and says so
    other = load_adapted_policy(aid, "owner-b", ids)
    assert not other.adapter_info["used"] and "missing" in other.adapter_info["reason"]
    # op-id hash differs -> fallback to base
    changed = load_adapted_policy(aid, "owner-a", ids[:-1])
    assert not changed.adapter_info["used"] and "op-id hash" in changed.adapter_info["reason"]
    assert isinstance(changed, UniformPolicy) or isinstance(changed, DQNPolicy)
    assert not load_adapted_policy("../../etc/passwd", "owner-a", ids).adapter_info["used"]
    assert store.delete("owner-a", aid) and not store.delete("owner-a", aid)


def test_job_queue_limits_and_owner_scoping():
    gate = threading.Event()

    def slow(payload, owner, progress=None):
        gate.wait(5)
        return {"ok": True}

    m = JobManager(slow, max_queue=3)
    ids = [m.submit("o", i) for i in range(4)]  # 1 running + 3 queued (worker may not have picked first yet)
    time.sleep(0.2)
    m.submit("o", 5)  # worker holds one, so a fourth queued slot is still free
    with pytest.raises(QueueFull):
        m.submit("o", 6)
    assert m.get(ids[0], "intruder") is None
    gate.set()
    for _ in range(50):
        if m.get(ids[0], "o")["status"] == "done":
            break
        time.sleep(0.05)
    assert m.get(ids[0], "o")["status"] == "done"


def test_job_timeout_discards_result():
    def slow(payload, owner, progress=None):
        time.sleep(0.3)
        return {"x": 1}

    m = JobManager(slow, timeout_s=0.1)
    jid = m.submit("o", [])
    for _ in range(50):
        if m.get(jid, "o")["status"] != "queued" and m.get(jid, "o")["status"] != "running":
            break
        time.sleep(0.05)
    j = m.get(jid, "o")
    assert j["status"] == "failed" and "limit" in j["error"] and j["result"] is None


def test_http_routes(reg, synth, base):
    app = FastAPI()

    @app.middleware("http")
    async def sid(request, call_next):
        request.state.sid = request.headers.get("x-sid", "s1")
        return await call_next(request)

    install(app, JobManager(lambda p, o, progress=None: pipeline.run(p, o, registry=reg, base=base, steps=20)))
    c = TestClient(app)
    assert c.post("/api/adapt/jobs", json={"transitions": synth[:3]}).status_code == 422
    assert c.post("/api/adapt/jobs", json={}).status_code == 422
    assert c.post("/api/adapt/jobs", json={"transitions": [{}] * 2001}).status_code == 422
    jid = c.post("/api/adapt/jobs", json={"transitions": synth[:8]}).json()["job_id"]
    for _ in range(200):
        j = c.get(f"/api/adapt/jobs/{jid}").json()
        if j["status"] in ("done", "failed"):
            break
        time.sleep(0.1)
    assert j["status"] == "done", j
    assert c.get(f"/api/adapt/jobs/{jid}", headers={"x-sid": "other"}).status_code == 404
    aid = j["result"]["adapter_id"]
    lst = c.get("/api/adapt/adapters").json()["adapters"]
    assert [a["adapter_id"] for a in lst] == [aid] and "owner" not in lst[0]
    assert c.get("/api/adapt/adapters", headers={"x-sid": "other"}).json()["adapters"] == []
    assert c.delete(f"/api/adapt/adapters/{aid}").status_code == 200
    assert c.delete(f"/api/adapt/adapters/{aid}").status_code == 404


def test_manual_session_id_reads_log(reg, synth, tmp_path):
    import json
    d = tmp_path / "manual"
    d.mkdir()
    (d / "abc123XYZ.jsonl").write_text("".join(json.dumps(t) + "\n" for t in synth[:6]))
    app = FastAPI()
    seen = []
    mgr = JobManager(lambda p, o, progress=None: seen.append(len(p)) or {"adapter_id": None})
    install(app, mgr, owner_fn=lambda r: "o")
    c = TestClient(app)
    assert c.post("/api/adapt/jobs", json={"manual_session_id": "nope1234"}).status_code == 404
    assert c.post("/api/adapt/jobs", json={"manual_session_id": "../x"}).status_code == 422
    assert c.post("/api/adapt/jobs", json={"manual_session_id": "abc123XYZ"}).status_code == 200
    time.sleep(0.3)
    assert seen == [6]
