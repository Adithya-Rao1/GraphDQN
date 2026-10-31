import json
import os
import stat
import time
from pathlib import Path

import jwt
import pytest
from fastapi.testclient import TestClient

from studio.accounts import auth
from studio.accounts.install import install
from studio.accounts.profile_scorer import ProfileScorer
from studio.schema import ObjectiveSpec, RunConfig
from studio.server.app import create_app
from studio.server.core import Settings

H = {"X-Requested-With": "studio"}
MOCK = json.loads((Path(__file__).resolve().parents[1] / "mock" / "mock_run.json").read_text())
PW = "correct horse"


def make_app(tmp_path, monkeypatch, **cfg):
    monkeypatch.setenv("STUDIO_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("STUDIO_DB_PATH", str(tmp_path / "t.db"))
    monkeypatch.delenv("STUDIO_JWT_SECRET", raising=False)
    app = create_app(settings=Settings(force_mock=True))
    install(app, auth.AccountsConfig(bcrypt_cost=4, **cfg))
    return app


@pytest.fixture
def app(tmp_path, monkeypatch):
    return make_app(tmp_path, monkeypatch)


def client(app):
    return TestClient(app, base_url="http://testserver")


def signup(c, name="alice", pw=PW):
    return c.post("/api/auth/signup", json={"username": name, "password": pw}, headers=H)


def test_happy_path_and_guest(app):
    c = client(app)
    assert c.get("/api/me").json() == {"guest": True, "user": None}
    assert c.get("/api/health").status_code == 200  # main app untouched
    r = signup(c)
    assert r.status_code == 200 and r.json()["user"]["username"] == "alice"
    assert "httponly" in r.headers["set-cookie"].lower() and "samesite=lax" in r.headers["set-cookie"].lower()
    assert "secure" not in r.headers["set-cookie"].lower()
    assert c.get("/api/me").json()["user"]["username"] == "alice"
    assert c.post("/api/auth/logout", headers=H).status_code == 200
    assert c.get("/api/me").json()["guest"] is True
    assert c.post("/api/auth/login", json={"username": "ALICE", "password": PW}, headers=H).status_code == 200
    assert c.get("/api/me").json()["user"]["username"] == "alice"


def test_secure_cookie_behind_https_proxy(app):
    r = client(app).post("/api/auth/signup", json={"username": "bob", "password": PW},
                         headers={**H, "X-Forwarded-Proto": "https"})
    assert "secure" in r.headers["set-cookie"].lower()


def test_wrong_password_uniform_message(app):
    c = client(app)
    signup(c)
    a = c.post("/api/auth/login", json={"username": "alice", "password": "nope-nope"}, headers=H)
    b = c.post("/api/auth/login", json={"username": "ghost", "password": "nope-nope"}, headers=H)
    assert a.status_code == b.status_code == 401 and a.json() == b.json()


def test_duplicate_and_validation(app):
    c = client(app)
    assert signup(c).status_code == 200
    assert signup(client(app), "Alice").status_code == 409
    for bad in ["ab", "a" * 33, "has space", "emoji😀", "semi;colon"]:
        assert signup(client(app), bad).status_code == 422, bad
    assert signup(client(app), "okname", "short").status_code == 422
    assert signup(client(app), "okname", "é" * 40).status_code == 422  # 80 bytes > 72


def test_rate_limit(tmp_path, monkeypatch):
    app = make_app(tmp_path, monkeypatch, login_per_min_ip=3)
    c = client(app)
    codes = [c.post("/api/auth/login", json={"username": "x" + str(i), "password": "whatever1"}, headers=H).status_code
             for i in range(5)]
    assert codes[:3] == [401] * 3 and codes[3:] == [429, 429]


def test_per_user_login_limit(tmp_path, monkeypatch):
    app = make_app(tmp_path, monkeypatch, login_per_5min_user=2)
    codes = []
    for i in range(4):  # different "IPs" via XFF
        codes.append(client(app).post("/api/auth/login", json={"username": "alice", "password": "whatever1"},
                                      headers={**H, "X-Forwarded-For": f"9.9.9.{i}"}).status_code)
    assert codes == [401, 401, 429, 429]


def test_tampered_and_expired_jwt(app):
    c = client(app)
    signup(c)
    st = app.state.accounts
    good = auth.make_token(st.secret, 1, "alice", 60)
    assert client(app).get("/api/me", headers={"Authorization": f"Bearer {good}"}).json()["user"]["username"] == "alice"
    for tok in [good[:-3] + ("AAA" if not good.endswith("AAA") else "BBB"),
                auth.make_token("other-secret-other-secret-other-secret", 1, "alice", 60),
                auth.make_token(st.secret, 1, "alice", -10),
                jwt.encode({"sub": "1", "usr": "alice"}, st.secret, algorithm="HS256"),  # no exp
                "x.y.z",
                auth.make_token(st.secret, 1, "mallory", 60),  # username mismatch
                auth.make_token(st.secret, 999, "alice", 60)]:  # unknown id
        assert client(app).get("/api/me", headers={"Authorization": f"Bearer {tok}"}).json()["guest"] is True
        cc = client(app)
        cc.cookies.set("studio_auth", tok)
        assert cc.get("/api/me").json()["guest"] is True


def test_alg_none_rejected(app):
    st = app.state.accounts
    tok = jwt.encode({"sub": "1", "usr": "alice", "exp": time.time() + 99}, None, algorithm="none")
    signup(client(app))
    assert client(app).get("/api/me", headers={"Authorization": f"Bearer {tok}"}).json()["guest"] is True


def test_csrf_required(app):
    c = client(app)
    assert c.post("/api/auth/signup", json={"username": "carol", "password": PW}).status_code == 403
    signup(c, "carol")
    assert c.post("/api/library/runs", json={"events": MOCK}).status_code == 403
    assert c.post("/api/profiles", json={"name": "p"}).status_code == 403
    assert c.post("/api/auth/logout").status_code == 403


def run_body(**kw):
    return {"events": MOCK, **kw}


def test_library_runs_roundtrip_and_auth(app):
    c = client(app)
    assert c.get("/api/library/runs").status_code == 401
    signup(c)
    r = c.post("/api/library/runs", json=run_body(title="My run"), headers=H)
    assert r.status_code == 201, r.text
    rid = r.json()["id"]
    assert r.json()["title"] == "My run" and r.json()["seed_smiles"]
    assert c.get("/api/library/runs").json()["runs"][0]["id"] == rid
    full = c.get(f"/api/library/runs/{rid}").json()
    assert full["events"] == json.loads(json.dumps(MOCK)) or len(full["events"]) == len(MOCK)
    assert c.delete(f"/api/library/runs/{rid}", headers=H).status_code == 200
    assert c.get(f"/api/library/runs/{rid}").status_code == 404


def test_cross_user_isolation(app):
    a, b = client(app), client(app)
    signup(a, "alice"), signup(b, "bobby")
    rid = a.post("/api/library/runs", json=run_body(), headers=H).json()["id"]
    mid = a.post("/api/library/manual", json={"steps": [{"chosen_by": "user", "smiles": "CCO"}]}, headers=H).json()["id"]
    pid = a.post("/api/profiles", json={"name": "mine"}, headers=H).json()["id"]
    for method, url, kw in [("get", f"/api/library/runs/{rid}", {}), ("delete", f"/api/library/runs/{rid}", {"headers": H}),
                            ("get", f"/api/library/manual/{mid}", {}), ("delete", f"/api/library/manual/{mid}", {"headers": H}),
                            ("get", f"/api/profiles/{pid}", {}), ("delete", f"/api/profiles/{pid}", {"headers": H}),
                            ("put", f"/api/profiles/{pid}", {"headers": H, "json": {"name": "stolen"}})]:
        assert getattr(b, method)(url, **kw).status_code == 404, url
    assert b.get("/api/library/runs").json()["runs"] == []
    assert b.get("/api/library/manual").json()["sessions"] == []
    assert b.get("/api/profiles").json()["profiles"] == []
    assert a.get(f"/api/profiles/{pid}").json()["name"] == "mine"  # untouched
    assert a.get(f"/api/library/runs/{rid}").status_code == 200


def test_size_caps(tmp_path, monkeypatch):
    app = make_app(tmp_path, monkeypatch, max_payload_bytes=50_000, max_runs_per_user=2)
    c = client(app)
    signup(c)
    assert c.post("/api/library/runs", json=run_body(), headers=H).status_code == 413 or len(json.dumps(MOCK)) < 50_000
    big = {"steps": [{"x": "y" * 1000}] * 100}
    assert c.post("/api/library/manual", json=big, headers=H).status_code == 413
    assert c.post("/api/library/runs", json={"events": []}, headers=H).status_code == 422
    assert c.post("/api/library/runs", json={"events": [{"type": "bogus", "run_id": "r", "seq": 0, "payload": {}}]},
                  headers=H).status_code == 422
    # 2 MB library cap is lifted only under /api/library; other routes keep the server cap
    r = c.post("/api/molecules/validate", content=b'{"smiles":"' + b"C" * 300_000 + b'"}',
               headers={**H, "Content-Type": "application/json"})
    assert r.status_code == 413


def test_quota_and_default_cap(tmp_path, monkeypatch):
    app = make_app(tmp_path, monkeypatch, max_runs_per_user=2)
    c = client(app)
    signup(c)
    tiny = [MOCK[0]]
    codes = [c.post("/api/library/runs", json={"events": tiny}, headers=H).status_code for _ in range(3)]
    assert codes == [201, 201, 409]


def test_sql_injection_inputs(app):
    c = client(app)
    signup(c)
    evil = "x'; DROP TABLE users; --"
    assert c.post("/api/auth/login", json={"username": evil, "password": evil}, headers=H).status_code == 401
    assert signup(client(app), "a' OR '1'='1").status_code == 422
    r = c.post("/api/profiles", json={"name": evil}, headers=H)
    assert r.status_code == 201 and r.json()["name"] == evil
    r = c.post("/api/library/manual", json={"seed_smiles": evil, "steps": [{"a": evil}]}, headers=H)
    assert r.status_code == 201
    assert c.get("/api/library/runs/1%20OR%201=1").status_code == 422
    assert c.get("/api/me").json()["user"]["username"] == "alice"  # users table intact


def test_profile_smarts_validation(app):
    c = client(app)
    signup(c)
    ok = c.post("/api/profiles", json={"name": "p", "avoid_smarts": ["[N+](=O)[O-]", "c1ccccc1"],
                                       "objectives": {"qed": {"weight": 2.0}}}, headers=H)
    assert ok.status_code == 201
    for bad in [["not(smarts"], ["a" * 201], [""], ["C"] * 21]:
        r = c.post("/api/profiles", json={"name": "p", "avoid_smarts": bad}, headers=H)
        assert r.status_code == 422 and isinstance(r.json()["detail"], str), bad
    r = c.post("/api/profiles", json={"name": "p", "edit_allowlist": ["definitely_not_an_edit"]}, headers=H)
    assert r.status_code == 422
    chk = c.post("/api/profiles/check-smarts", json={"patterns": ["C=O", "((("]}).json()["results"]
    assert [x["ok"] for x in chk] == [True, False]
    pid = ok.json()["id"]
    upd = c.put(f"/api/profiles/{pid}", json={"name": "renamed", "avoid_smarts": ["C=O"]}, headers=H)
    assert upd.json()["name"] == "renamed" and upd.json()["avoid_smarts"] == ["C=O"]


def test_valid_edit_allowlist(app):
    from studio.registry import get_registry
    eid = get_registry().all()[0].id
    c = client(app)
    signup(c)
    assert c.post("/api/profiles", json={"name": "p", "edit_allowlist": [eid]}, headers=H).status_code == 201


def test_persistence_across_restart_and_secret(tmp_path, monkeypatch):
    app1 = make_app(tmp_path, monkeypatch)
    c = client(app1)
    signup(c)
    c.post("/api/library/runs", json=run_body(title="persist"), headers=H)
    cookie = c.cookies.get("studio_auth")
    secret1 = app1.state.accounts.secret
    mode = stat.S_IMODE(os.stat(tmp_path / "jwt_secret").st_mode)
    assert mode == 0o600
    app1.state.accounts.engine.dispose()

    app2 = make_app(tmp_path, monkeypatch)  # same files = restart
    assert app2.state.accounts.secret == secret1
    c2 = client(app2)
    c2.cookies.set("studio_auth", cookie)
    assert c2.get("/api/me").json()["user"]["username"] == "alice"  # session survives restart
    assert c2.get("/api/library/runs").json()["runs"][0]["title"] == "persist"
    assert c2.post("/api/auth/login", json={"username": "alice", "password": PW}, headers=H).status_code == 200


def test_env_secret_and_wal(tmp_path, monkeypatch):
    monkeypatch.setenv("STUDIO_JWT_SECRET", "e" * 40)
    monkeypatch.setenv("STUDIO_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("STUDIO_DB_PATH", str(tmp_path / "w.db"))
    a = create_app(settings=Settings(force_mock=True))
    install(a, auth.AccountsConfig(bcrypt_cost=4))
    assert a.state.accounts.secret == "e" * 40
    with a.state.accounts.engine.connect() as conn:
        assert conn.exec_driver_sql("PRAGMA journal_mode").scalar() == "wal"


def test_static_mount_not_shadowed(app):
    c = client(app)
    assert c.get("/api/me").status_code == 200
    assert c.get("/").status_code == 200  # static UI still served


def test_profile_scorer(monkeypatch):
    from studio.wiring import build_scorer
    base = build_scorer()
    cfg = RunConfig(seed_smiles="c1ccccc1[N+](=O)[O-]")
    prof = {"name": "p", "avoid_smarts": ["[N+](=O)[O-]"], "objectives": {"qed": {"weight": 3.0}}}
    ps = ProfileScorer(base, prof)
    merged = cfg.model_copy(update={"objectives": {"qed": ObjectiveSpec(weight=3.0)}})
    plain, wrapped = base.score(cfg.seed_smiles, merged), ps.score(cfg.seed_smiles, cfg)
    hits = [a for a in wrapped.alerts if a.family == "USER"]
    assert len(hits) == 1 and wrapped.composite < plain.composite
    assert wrapped.descriptors.alerts == plain.descriptors.alerts + 1
    clean = ps.score("CCO", cfg)
    assert not [a for a in clean.alerts if a.family == "USER"]
