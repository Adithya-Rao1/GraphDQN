"""Gateway tests: real uvicorn servers on ephemeral ports (fake SSE backend, studio.server.app in mock mode)."""
import importlib.util
import json
import socket
import sys
import threading
import time
from pathlib import Path

import httpx
import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

ROOT = Path(__file__).resolve().parents[2]
KEY = "sekrit-key-123"
COOKIE_ECHO = "backend-cookie-value-xyz"


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Srv:
    def __init__(self, app):
        self.port = _free_port()
        self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="error"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()
        for _ in range(100):
            if self.server.started:
                break
            time.sleep(0.05)
        self.url = f"http://127.0.0.1:{self.port}"

    def stop(self):
        self.server.should_exit = True
        self.thread.join(5)


def fake_backend():
    app = FastAPI()
    seen = {"keys": [], "cookies": [], "closed": 0}
    app.state.seen = seen

    @app.get("/api/health")
    def health(request: Request):
        return {"ok": True, "engine_source": "mock", "schema_version": "x"}

    @app.post("/api/echo")
    async def echo(request: Request):
        seen["keys"].append(request.headers.get("x-studio-key"))
        seen["cookies"].append(request.headers.get("cookie"))
        b = await request.body()
        resp = JSONResponse({"len": len(b), "key": request.headers.get("x-studio-key")})
        resp.set_cookie("studio_sid", COOKIE_ECHO)
        resp.headers["X-Studio-Key"] = KEY
        return resp

    @app.get("/api/slow")
    async def slow():
        await __import__("asyncio").sleep(3)
        return {}

    @app.get("/api/runs/{rid}/events")
    async def events(rid: str):
        async def gen():
            import asyncio
            try:
                yield "retry: 3000\n\n"
                for i in range(5):
                    yield f"id: {i}\nevent: candidate\ndata: {json.dumps({'seq': i})}\n\n"
                    await asyncio.sleep(0.15)
                await asyncio.sleep(0.6)  # idle gap -> gateway heartbeat
                yield "id: 5\nevent: run_end\ndata: {\"seq\":5}\n\n"
            finally:
                seen["closed"] += 1
        return StreamingResponse(gen(), media_type="text/event-stream")
    return app


def load_gateway(monkeypatch, tmp_path, backend_url, web=None, permissive=True, **env):
    monkeypatch.setenv("BACKEND_URL", backend_url or "")
    monkeypatch.setenv("STUDIO_API_KEY", KEY)
    monkeypatch.setenv("METIS_BACKEND_FILE", str(tmp_path / "backend_url.txt"))
    monkeypatch.setenv("GATEWAY_WEB_DIR", str(web or tmp_path / "noweb"))
    monkeypatch.setenv("GATEWAY_HEARTBEAT", "0.3")
    monkeypatch.setenv("ENABLE_PROXY", "1")
    for k, v in env.items():
        monkeypatch.setenv(k, str(v))
    spec = importlib.util.spec_from_file_location("gw_app_" + str(time.time_ns()), ROOT / "deploy/space/app.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if permissive:  # local fake backends are http://127.0.0.1; the real allowlist is tested separately
        mod._validate_backend = lambda raw: (raw.strip().splitlines() or [""])[0].rstrip("/") or None
    return mod


@pytest.fixture
def chain(monkeypatch, tmp_path):
    be = fake_backend()
    bsrv = Srv(be)
    gw = load_gateway(monkeypatch, tmp_path, bsrv.url)
    gsrv = Srv(gw.app)
    yield gsrv, bsrv, be, gw, tmp_path
    gsrv.stop()
    bsrv.stop()


def parse_sse(text):
    return [b for b in text.split("\n\n") if b.strip()]


def test_sse_order_heartbeat_and_incremental(chain):
    g, *_ = chain
    got, stamps = [], []
    with httpx.stream("GET", g.url + "/api/runs/r1/events", timeout=10) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        t0 = time.monotonic()
        for chunk in r.iter_text():
            got.append(chunk)
            stamps.append(time.monotonic() - t0)
    text = "".join(got)
    ids = [int(l.split(": ")[1]) for l in text.splitlines() if l.startswith("id: ")]
    assert ids == [0, 1, 2, 3, 4, 5]
    assert ": gw-heartbeat" in text
    assert text.index("id: 4") < text.index("gw-heartbeat") < text.index("id: 5")
    assert stamps[0] < 0.4 < stamps[-1]  # first bytes arrive well before the stream ends (unbuffered)


def test_client_disconnect_closes_upstream(chain):
    g, _, be, *_ = chain
    with httpx.stream("GET", g.url + "/api/runs/r1/events", timeout=10) as r:
        next(r.iter_text())
    for _ in range(40):
        if be.state.seen["closed"]:
            break
        time.sleep(0.1)
    assert be.state.seen["closed"] == 1


def test_key_sent_upstream_never_to_browser(chain):
    g, _, be, *_ = chain
    r = httpx.post(g.url + "/api/echo", json={"a": 1})
    assert r.status_code == 200
    assert be.state.seen["keys"] == [KEY]            # backend got the key
    blob = r.text + json.dumps(dict(r.headers)) + json.dumps(dict(r.cookies))
    # backend echoed the key in the JSON body in this fake; the headers/cookies must be clean
    assert KEY not in json.dumps(dict(r.headers)) and COOKIE_ECHO not in json.dumps(dict(r.headers))
    assert "x-studio-key" not in {k.lower() for k in r.headers}
    assert "studio_sid" not in r.cookies               # backend cookie is not forwarded
    assert "metis_gw" in r.cookies                     # gateway's own session cookie only
    # browser cannot choose the backend cookie / key
    r2 = httpx.post(g.url + "/api/echo", json={}, headers={"X-Studio-Key": "attacker", "Cookie": "studio_sid=evil"})
    assert be.state.seen["keys"][-1] == KEY
    assert "evil" not in be.state.seen["cookies"][-1]


def test_session_isolated_but_stable(chain):
    g, _, be, *_ = chain
    with httpx.Client(base_url=g.url) as c:
        c.post("/api/echo", json={}); c.post("/api/echo", json={})
    with httpx.Client(base_url=g.url) as c2:
        c2.post("/api/echo", json={})
    ck = be.state.seen["cookies"]
    assert ck[0] == ck[1] != ck[2]


def test_body_caps(chain):
    g, *_ = chain
    assert httpx.post(g.url + "/api/echo", content=b"x" * 300_000).status_code == 413
    s = socket.create_connection(("127.0.0.1", g.port))
    s.sendall(b"POST /api/echo HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n")
    assert b" 411 " in s.recv(200)
    s.close()


def test_request_timeout(monkeypatch, tmp_path):
    bsrv = Srv(fake_backend())
    gw = load_gateway(monkeypatch, tmp_path, bsrv.url, GATEWAY_REQUEST_TIMEOUT=1)
    g = Srv(gw.app)
    try:
        assert httpx.get(g.url + "/api/slow", timeout=10).status_code == 504
    finally:
        g.stop(); bsrv.stop()


def test_backend_down_fallback(monkeypatch, tmp_path):
    gw = load_gateway(monkeypatch, tmp_path, f"http://127.0.0.1:{_free_port()}")
    g = Srv(gw.app)
    try:
        h = httpx.get(g.url + "/api/health").json()
        assert h["backend"] == "down" and h["gateway"] == "ok"
        r = httpx.post(g.url + "/api/runs", json={})
        assert r.status_code == 503 and r.json()["backend"] == "down"
        assert httpx.get(g.url + "/api/runs/x/events").status_code == 503
        assert KEY not in r.text
    finally:
        g.stop()


def test_unconfigured_backend(monkeypatch, tmp_path):
    gw = load_gateway(monkeypatch, tmp_path, "")
    g = Srv(gw.app)
    try:
        assert httpx.get(g.url + "/api/health").json()["backend"] == "down"
        assert httpx.get(g.url + "/api/presets").status_code == 503
    finally:
        g.stop()


def test_bucket_file_overrides_and_rotates(monkeypatch, tmp_path):
    bsrv = Srv(fake_backend())
    gw = load_gateway(monkeypatch, tmp_path, f"http://127.0.0.1:{_free_port()}")   # dead variable
    g = Srv(gw.app)
    try:
        assert httpx.get(g.url + "/api/health").json()["backend"] == "down"
        (tmp_path / "backend_url.txt").write_text(bsrv.url + "\n")                 # rotate, no restart
        time.sleep(3.2)                                                            # probe cache TTL
        assert httpx.get(g.url + "/api/health").json()["backend"] == "up"
    finally:
        g.stop(); bsrv.stop()


def test_non_api_paths_not_proxied(chain):
    g, *_ = chain
    assert httpx.get(g.url + "/secret").status_code == 404


def test_full_chain_with_real_studio_mock_and_ui(monkeypatch, tmp_path):
    """browser -> gateway -> studio.server.app (mock mode), real UI served by the gateway."""
    monkeypatch.setenv("STUDIO_FORCE_MOCK", "1")
    monkeypatch.setenv("STUDIO_MOCK_DELAY", "0.01")
    from studio.server.app import create_app
    studio = Srv(create_app())
    web = tmp_path / "web"
    import shutil
    shutil.copytree(ROOT / "studio/web", web, ignore=shutil.ignore_patterns("screenshots"))
    (web / "mock").mkdir()
    shutil.copy(ROOT / "studio/mock/mock_run.json", web / "mock" / "mock_run.json")
    gw = load_gateway(monkeypatch, tmp_path, studio.url, web=web)
    g = Srv(gw.app)
    try:
        idx = httpx.get(g.url + "/live/")      # UI served by the backend, address bar stays on the gateway
        assert idx.status_code == 200 and "<html" in idx.text.lower()
        assert httpx.get(g.url + "/live/vendor/3Dmol-min.js").status_code == 200
        rp = httpx.get(g.url + "/replay/?mock=1")  # local static copy (offline fallback)
        assert rp.status_code == 200 and "<html" in rp.text.lower()
        assert httpx.get(g.url + "/replay/vendor/3Dmol-min.js").status_code == 200
        assert httpx.get(g.url + "/api/health").json()["backend"] == "up"
        with httpx.Client(base_url=g.url, timeout=30) as c:
            v = c.post("/api/molecules/validate", json={"smiles": "CC(=O)Oc1ccccc1C(=O)O"})
            assert v.status_code == 200 and v.json()["ok"]
            run = c.post("/api/runs", json={"seed_smiles": "CC(=O)Oc1ccccc1C(=O)O", "generations": 2,
                                            "population_size": 4})
            assert run.status_code == 200, run.text
            rid = run.json()["run_id"]
            types, seqs = [], []
            with c.stream("GET", f"/api/runs/{rid}/events") as r:
                assert r.headers["content-type"].startswith("text/event-stream")
                for line in r.iter_lines():
                    if line.startswith("event: "):
                        types.append(line[7:])
                    elif line.startswith("data: "):
                        seqs.append(json.loads(line[6:])["seq"])
            assert types[0] == "run_start" and types[-1] == "run_end"
            assert seqs == sorted(seqs) and len(seqs) > 3
            assert c.cookies.get("studio_sid") is None
    finally:
        g.stop(); studio.stop()


# ---------------------------------------------------------------- front door: redirect / allowlist / landing
GOOD = "https://quiet-river-1234.trycloudflare.com"


def _fd(monkeypatch, tmp_path, backend_url="", file_text=None, **env):
    if file_text is not None:
        (tmp_path / "backend_url.txt").write_text(file_text)
    gw = load_gateway(monkeypatch, tmp_path, backend_url, permissive=False, **env)
    monkeypatch.delenv("ENABLE_PROXY", raising=False) if "ENABLE_PROXY" not in env else None
    return gw


@pytest.mark.parametrize("raw,ok", [
    (GOOD, True), (GOOD + "/", True), (GOOD + "/x?y=1", True), ("https://trycloudflare.com", True),
    ("http://a.trycloudflare.com", False),                      # not https
    ("https://evil.com", False), ("https://trycloudflare.com.evil.com", False),
    ("https://eviltrycloudflare.com", False), ("https://a.trycloudflare.com@evil.com", False),
    ("https://user:pw@a.trycloudflare.com", False), ("https://a.trycloudflare.com:8443", False),
    ("//a.trycloudflare.com", False), ("javascript:alert(1)", False), ("https://a.trycloudflare.com\\@evil.com", False),
    ("", False), ("https://127.0.0.1", False),
])
def test_allowlist(monkeypatch, tmp_path, raw, ok):
    gw = _fd(monkeypatch, tmp_path)
    got = gw._validate_backend(raw)
    assert (got is not None) == ok
    if ok:
        assert got.startswith("https://") and got.count("/") == 2   # origin only


def test_allowlist_env_configurable(monkeypatch, tmp_path):
    gw = _fd(monkeypatch, tmp_path, ALLOWED_BACKEND_HOSTS="example.org, trycloudflare.com")
    assert gw._validate_backend("https://x.example.org") == "https://x.example.org"
    assert gw._validate_backend("https://a.trycloudflare.com")


def test_file_beats_env_first_line_only(monkeypatch, tmp_path):
    gw = _fd(monkeypatch, tmp_path, "https://env-one.trycloudflare.com",
             file_text="https://file-one.trycloudflare.com\nhttps://second-line.trycloudflare.com\n")
    assert gw._backend_url() == "https://file-one.trycloudflare.com"
    (tmp_path / "backend_url.txt").write_text("https://file-two.trycloudflare.com")   # change at request time
    assert gw._backend_url() == "https://file-two.trycloudflare.com"


def test_invalid_file_falls_to_env_then_none(monkeypatch, tmp_path):
    gw = _fd(monkeypatch, tmp_path, "https://env-one.trycloudflare.com", file_text="https://evil.com\n")
    assert gw._backend_url() == "https://env-one.trycloudflare.com"
    gw2 = _fd(monkeypatch, tmp_path, "http://env.trycloudflare.com", file_text="https://evil.com")
    assert gw2._backend_url() is None
    (tmp_path / "backend_url.txt").write_text("")
    assert _fd(monkeypatch, tmp_path, "")._backend_url() is None


class _FD:
    """Front-door app over httpx ASGI with the upstream probe faked (no network)."""
    def __init__(self, gw, up):
        self.calls = []
        def handler(req):
            self.calls.append(req)
            if up:
                return httpx.Response(200, json={"ok": True, "engine_source": "live", "schema_version": "0.1"})
            raise httpx.ConnectError("down")
        gw.app.state.client._transport = httpx.MockTransport(handler)
        self.app = gw.app


def _get(app, path, **kw):
    async def go():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
            return await c.get(path, follow_redirects=False, **kw)
    return __import__("asyncio").run(go())


def test_root_redirects_when_up(monkeypatch, tmp_path):
    gw = _fd(monkeypatch, tmp_path, GOOD)
    fd = _FD(gw, up=True)
    r = _get(fd.app, "/")
    assert r.status_code == 302 and r.headers["location"] == GOOD + "/"
    assert "set-cookie" not in r.headers and KEY not in json.dumps(dict(r.headers))
    assert fd.calls[0].headers["x-studio-key"] == KEY            # optional key goes only to the backend probe
    r2 = _get(fd.app, "/open")
    assert r2.status_code == 302 and r2.headers["location"] == GOOD + "/"
    h = _get(fd.app, "/api/health").json()
    assert h["backend"] == "up" and h["proxy"] is False and GOOD not in json.dumps(h)


def test_root_offline_page_when_down(monkeypatch, tmp_path):
    gw = _fd(monkeypatch, tmp_path, GOOD)
    fd = _FD(gw, up=False)
    r = _get(fd.app, "/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    assert "Metis Lead Studio" in r.text and "Open the live studio" in r.text and "offline" in r.text
    assert "/replay/" in r.text and "precomputed" in r.text
    assert GOOD not in r.text and KEY not in r.text and "set-cookie" not in r.headers
    assert _get(fd.app, "/api/health").json()["backend"] == "down"
    o = _get(fd.app, "/open")
    assert o.status_code == 302 and o.headers["location"] == "/landing"


def test_disallowed_backend_never_redirects_or_probes(monkeypatch, tmp_path):
    gw = _fd(monkeypatch, tmp_path, "https://evil.example.com", file_text="http://x.trycloudflare.com")
    fd = _FD(gw, up=True)                      # even if it "answers", it must be ignored
    r = _get(fd.app, "/")
    assert r.status_code == 200 and "offline" in r.text
    assert fd.calls == []                      # no request ever sent to a non-allowlisted host
    assert _get(fd.app, "/api/health").json()["backend"] == "down"


def test_landing_up_state_and_replay_assets(monkeypatch, tmp_path):
    web = tmp_path / "web"; (web / "mock").mkdir(parents=True)
    (web / "index.html").write_text("<html>ui</html>"); (web / "mock" / "mock_run.json").write_text("[]")
    gw = _fd(monkeypatch, tmp_path, GOOD)
    gw = load_gateway(monkeypatch, tmp_path, GOOD, web=web, permissive=False)
    fd = _FD(gw, up=True)
    assert "Live backend: up" in _get(fd.app, "/landing").text
    assert "<html>ui</html>" in _get(fd.app, "/replay/").text
    assert _get(fd.app, "/mock/mock_run.json").status_code == 200


def test_proxy_off_by_default(monkeypatch, tmp_path):
    monkeypatch.delenv("ENABLE_PROXY", raising=False)
    gw = _fd(monkeypatch, tmp_path, GOOD)
    monkeypatch.delenv("ENABLE_PROXY", raising=False)
    fd = _FD(gw, up=True)
    for path in ("/live/", "/live/api/presets", "/api/presets", "/api/runs/x/events"):
        assert _get(fd.app, path).status_code == 404
    assert [c for c in fd.calls if c.url.path != "/api/health"] == []
