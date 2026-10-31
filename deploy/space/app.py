"""Metis Lead Studio front door (Hugging Face Space, Gradio SDK).

Stable URL that sends visitors to the live studio, or shows the recorded replay when the backend is offline.

  GET /          backend up -> 302 to the backend origin; down -> landing page with the replay link
  GET /landing   landing page always (status chip polls /api/health)
  GET /open      the landing page's button: 302 to the backend if up, else back to the landing page
  GET /replay/   static UI from ./web (replay/mock mode, labelled in the UI); /mock serves the mock run
  GET /api/health  {"backend": "up"|"down"} (local; probes the backend with a 2 s timeout)
  /live/{path}, /api/*  OPTIONAL SSE-safe reverse proxy, OFF unless ENABLE_PROXY=1 (keeps the address bar here)

Backend URL: first line of /metis-data/backend_url.txt, else env BACKEND_URL, read at request time. A candidate is
used only if it is https and its host is (a subdomain of) an entry in ALLOWED_BACKEND_HOSTS (default
trycloudflare.com); only the origin is kept. Anything else is ignored -> offline page (open-redirect protection).
STUDIO_API_KEY (optional) is sent to the backend as X-Studio-Key by probes/proxy only, never to the browser.

Why uvicorn.run at the bottom: a Gradio-SDK Space just executes `python app.py` and routes port 7860, so any ASGI
server bound there works. A minimal Gradio Blocks is mounted on /gradio only so the Space is a recognisable
Gradio app (ZeroGPU/SDK checks); it is optional and skipped if gradio is not importable (local tests).
"""
from __future__ import annotations

try:  # ZeroGPU hardware aborts startup ("No @spaces.GPU function detected") without one decorated function.
    import spaces  # must be imported before torch/CUDA libs; preinstalled on Spaces, absent locally

    @spaces.GPU
    def _zerogpu_noop() -> str:  # never called; satisfies the ZeroGPU startup check. Harmless on cpu-basic.
        return "noop"
except Exception:
    pass

import asyncio
import contextlib
import hashlib
import hmac
import os
import secrets
import time
from pathlib import Path
from urllib.parse import urlsplit
from typing import AsyncIterator, Optional

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

HERE = Path(__file__).parent
WEB = Path(os.environ.get("GATEWAY_WEB_DIR", HERE / "web"))
BACKEND_FILE = Path(os.environ.get("METIS_BACKEND_FILE", "/metis-data/backend_url.txt"))

MAX_BODY = int(os.environ.get("GATEWAY_MAX_BODY", 256_000))        # bytes; backend caps similarly
REQUEST_TIMEOUT = float(os.environ.get("GATEWAY_REQUEST_TIMEOUT", 60))   # non-streaming round trip, s
STREAM_MAX_SECONDS = float(os.environ.get("GATEWAY_STREAM_MAX", 900))    # hard cap on one SSE connection
HEARTBEAT = float(os.environ.get("GATEWAY_HEARTBEAT", 15))        # idle seconds before a gateway comment
PROBE_TIMEOUT = 2.0
PROBE_TTL = 3.0

GW_COOKIE = "metis_gw"             # browser <-> gateway only
BACKEND_COOKIE = "studio_sid"      # cookie name the studio server uses for its session id
ALLOWED_METHODS = {"GET", "POST", "DELETE", "OPTIONS"}
FORWARD_REQ_HEADERS = {"content-type", "accept", "last-event-id", "content-length"}
DROP_RESP_HEADERS = {"set-cookie", "connection", "keep-alive", "transfer-encoding", "content-length",
                     "content-encoding", "server", "date", "x-studio-key"}

_SALT = os.environ.get("GATEWAY_SESSION_SALT") or secrets.token_hex(16)


def _allowed_hosts() -> list[str]:
    raw = os.environ.get("ALLOWED_BACKEND_HOSTS", "trycloudflare.com")
    return [h.strip().lower().lstrip(".") for h in raw.split(",") if h.strip()]


def _validate_backend(raw: str) -> Optional[str]:
    """Return the https origin if raw is an allowed backend URL, else None."""
    raw = (raw or "").strip()
    if not raw or any(c in raw for c in "\\ \t\r\n@"):
        return None
    try:
        u = urlsplit(raw)
        host, port = (u.hostname or "").lower(), u.port
    except ValueError:
        return None
    if u.scheme != "https" or not host or u.username or u.password or port not in (None, 443):
        return None
    if not any(host == h or host.endswith("." + h) for h in _allowed_hosts()):
        return None
    return f"https://{host}"


def _backend_url() -> Optional[str]:
    """First valid of: first line of the bucket file, then env BACKEND_URL."""
    cands = []
    try:
        if BACKEND_FILE.is_file():
            lines = BACKEND_FILE.read_text().strip().splitlines()
            cands.append(lines[0] if lines else "")
    except OSError:
        pass
    cands.append(os.environ.get("BACKEND_URL", ""))
    for c in cands:
        ok = _validate_backend(c)
        if ok:
            return ok
    return None


def _proxy_enabled() -> bool:
    return os.environ.get("ENABLE_PROXY", "") not in ("", "0", "false")


def _api_key() -> str:
    return os.environ.get("STUDIO_API_KEY", "")


def _backend_sid(browser_sid: str) -> str:
    """Stable per-browser session id for the backend (rate limits, run ownership) without exposing its cookie."""
    return hmac.new(_SALT.encode(), browser_sid.encode(), hashlib.sha256).hexdigest()[:32]


def _upstream_headers(request: Request, browser_sid: str) -> dict:
    h = {k: v for k, v in request.headers.items() if k.lower() in FORWARD_REQ_HEADERS}
    h.pop("content-length", None)  # httpx recomputes
    if _api_key():
        h["X-Studio-Key"] = _api_key()
    h["Cookie"] = f"{BACKEND_COOKIE}={_backend_sid(browser_sid)}"
    return h


def _clean_resp_headers(src) -> dict:
    return {k: v for k, v in src.items() if k.lower() not in DROP_RESP_HEADERS}


def _down(detail: str = "Backend unavailable") -> JSONResponse:
    return JSONResponse({"detail": detail, "backend": "down"}, status_code=503, headers={"Retry-After": "10"})


def create_app() -> FastAPI:
    client = httpx.AsyncClient(timeout=httpx.Timeout(REQUEST_TIMEOUT, connect=5.0),
                               limits=httpx.Limits(max_connections=64), follow_redirects=False)

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        await client.aclose()

    app = FastAPI(title="Metis Space gateway", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    probe = {"t": 0.0, "base": None, "up": False, "info": {}}
    app.state.client = client

    @app.middleware("http")
    async def session(request: Request, call_next):
        sid = request.cookies.get(GW_COOKIE)
        fresh = not sid or len(sid) > 64
        request.state.sid = secrets.token_urlsafe(16) if fresh else sid
        resp = await call_next(request)
        if fresh and _proxy_enabled() and request.url.path.startswith(("/api/", "/live")):
            secure = request.headers.get("x-forwarded-proto", request.url.scheme) == "https"
            resp.set_cookie(GW_COOKIE, request.state.sid, httponly=True, samesite="lax", secure=secure,
                            max_age=86400)
        return resp

    async def check_backend() -> tuple[Optional[str], bool, dict]:
        """(validated base or None, is_up, health info). Cached PROBE_TTL s per base URL."""
        base = _backend_url()
        now = time.monotonic()
        if base == probe["base"] and now - probe["t"] < PROBE_TTL:
            return base, probe["up"], probe["info"]
        up, info = False, {}
        if base:
            try:
                r = await client.get(f"{base}/api/health", timeout=PROBE_TIMEOUT,
                                     headers={"X-Studio-Key": _api_key()} if _api_key() else {})
                if r.status_code == 200:
                    up = True
                    j = r.json()
                    info = {k: j[k] for k in ("engine_source", "schema_version") if isinstance(j, dict) and k in j}
            except (httpx.HTTPError, ValueError):
                up = False
        probe.update(t=now, base=base, up=up, info=info)
        return base, up, info

    NO_STORE = {"Cache-Control": "no-store"}

    @app.get("/api/health")
    async def health():
        _, up, info = await check_backend()
        return JSONResponse({"backend": "up" if up else "down", "gateway": "ok",
                             "proxy": _proxy_enabled(), **info}, headers=NO_STORE)

    @app.get("/")
    async def root():
        base, up, _ = await check_backend()
        if base and up:
            return RedirectResponse(base + "/", status_code=302, headers=NO_STORE)
        return HTMLResponse(_landing(False), headers=NO_STORE)

    @app.get("/landing")
    async def landing():
        _, up, _ = await check_backend()
        return HTMLResponse(_landing(up), headers=NO_STORE)

    @app.get("/open")
    async def open_live():
        base, up, _ = await check_backend()
        if base and up:
            return RedirectResponse(base + "/", status_code=302, headers=NO_STORE)
        return RedirectResponse("/landing", status_code=302, headers=NO_STORE)

    async def forward(request: Request, upstream_path: str):
        base = _backend_url()
        if not base:
            return _down("Backend not configured")
        body = b""
        if request.method in ("POST", "PUT", "PATCH"):
            cl = request.headers.get("content-length")
            if cl is None:
                return JSONResponse({"detail": "Content-Length required"}, status_code=411)
            if not cl.isdigit() or int(cl) > MAX_BODY:
                return JSONResponse({"detail": f"Body too large (max {MAX_BODY // 1000} KB)"}, status_code=413)
            body = await request.body()
            if len(body) > MAX_BODY:
                return JSONResponse({"detail": "Body too large"}, status_code=413)
        url = f"{base}/{upstream_path.lstrip('/')}"
        headers = _upstream_headers(request, request.state.sid)
        params = list(request.query_params.multi_items())
        wants_stream = upstream_path.endswith("/events") or "text/event-stream" in request.headers.get("accept", "")
        try:
            if wants_stream:
                return await _stream(client, request, url, headers, params)
            r = await client.request(request.method, url, headers=headers, params=params, content=body)
        except httpx.TimeoutException:
            return JSONResponse({"detail": "Backend timed out", "backend": "slow"}, status_code=504)
        except httpx.HTTPError:
            probe["t"] = 0.0
            return _down()
        return Response(r.content, status_code=r.status_code, headers=_clean_resp_headers(r.headers),
                        media_type=r.headers.get("content-type"))

    def _off():
        return JSONResponse({"detail": "Proxy disabled on this deployment", "backend": "n/a"}, status_code=404)

    @app.api_route("/api/{path:path}", methods=sorted(ALLOWED_METHODS))
    async def api_proxy(path: str, request: Request):
        if not _proxy_enabled():
            return _off()
        return await forward(request, f"api/{path}")

    @app.api_route("/live", methods=sorted(ALLOWED_METHODS))
    @app.api_route("/live/{path:path}", methods=sorted(ALLOWED_METHODS))
    async def live_proxy(request: Request, path: str = ""):
        if not _proxy_enabled():
            return _off()
        return await forward(request, path)

    try:  # optional: makes this a recognisable Gradio-SDK app; must be mounted before static mounts
        import gradio as gr

        with gr.Blocks(title="Metis Lead Studio") as blocks:
            gr.Markdown("Metis Lead Studio front door. Open the studio at `/`.")
        gr.mount_gradio_app(app, blocks, path="/gradio")
    except Exception:  # gradio missing or incompatible: front door still fully works
        pass

    if WEB.exists():
        if (WEB / "mock").exists():  # the UI looks for ../mock/mock_run.json relative to /replay/
            app.mount("/mock", StaticFiles(directory=WEB / "mock"), name="mock")
        app.mount("/replay", StaticFiles(directory=WEB, html=True), name="replay")
    return app


REPLAY_HREF = os.environ.get("REPLAY_HREF", "/replay/?mock=1")

LANDING = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Metis Lead Studio</title>
<style>
:root{--bg:#0b1020;--fg:#e8ecf6;--mut:#9aa6c4;--card:#131a31;--line:#243056;--acc:#6ea8ff;--ok:#3ddc97;--bad:#ff6b6b}
@media (prefers-color-scheme:light){:root{--bg:#f4f6fb;--fg:#12182b;--mut:#56617f;--card:#fff;--line:#d9dff0;--acc:#2d6cdf;--ok:#12a366;--bad:#d64545}}
*{box-sizing:border-box}body{margin:0;min-height:100vh;display:grid;place-items:center;background:var(--bg);color:var(--fg);
font:16px/1.5 system-ui,-apple-system,Segoe UI,sans-serif;padding:16px}
main{max-width:620px;width:100%;background:var(--card);border:1px solid var(--line);border-radius:16px;padding:32px 28px}
h1{margin:0 0 6px;font-size:1.9rem;letter-spacing:-.02em}p{color:var(--mut);margin:.4rem 0 1.2rem}
.chip{display:inline-flex;align-items:center;gap:8px;border:1px solid var(--line);border-radius:999px;padding:4px 12px;font-size:.85rem}
.dot{width:9px;height:9px;border-radius:50%;background:var(--mut)}.up .dot{background:var(--ok)}.down .dot{background:var(--bad)}
.btn{display:inline-block;margin:6px 10px 6px 0;padding:11px 20px;border-radius:10px;font-weight:600;text-decoration:none;
background:var(--acc);color:#fff}.btn.alt{background:transparent;color:var(--acc);border:1px solid var(--acc)}
.btn[aria-disabled=true]{opacity:.45;pointer-events:none}
.note{border-left:3px solid var(--bad);padding:6px 12px;margin:16px 0;color:var(--fg);display:none}.down~.note{display:block}
small{color:var(--mut)}
</style></head><body><main>
<h1>Metis Lead Studio</h1>
<p>Watch a genetic algorithm and RL policy evolve a drug-like molecule in 3D, with every edit explained. Binding
numbers are model predictions, not lab measurements.</p>
<span class="chip __STATE__" id="chip" role="status" aria-live="polite"><span class="dot"></span><span id="txt">__TEXT__</span></span>
<div class="note" id="note">The live backend is offline right now. You can still watch the recorded replay (precomputed, not a live run).</div>
<div style="margin-top:18px">
<a class="btn" id="open" href="/open" aria-disabled="__DIS__">Open the live studio</a>
<a class="btn alt" href="__REPLAY__">Watch the recorded replay</a></div>
<small>This page is a stable front door; the live studio runs on a GPU box behind a temporary tunnel.</small>
</main><script>
const chip=document.getElementById("chip"),txt=document.getElementById("txt"),btn=document.getElementById("open"),note=document.getElementById("note");
function set(up){chip.className="chip "+(up?"up":"down");txt.textContent=up?"Live backend: up":"Live backend: offline";
btn.setAttribute("aria-disabled",up?"false":"true");note.style.display=up?"none":"block";}
async function poll(){try{const r=await fetch("/api/health",{cache:"no-store"});set((await r.json()).backend==="up")}catch(e){set(false)}}
set(__UPJS__);setInterval(poll,5000);poll();
</script></body></html>"""


def _landing(up: bool) -> str:
    return (LANDING.replace("__STATE__", "up" if up else "down")
            .replace("__TEXT__", "Live backend: up" if up else "Live backend: offline")
            .replace("__DIS__", "false" if up else "true").replace("__UPJS__", "true" if up else "false")
            .replace("__REPLAY__", REPLAY_HREF))


async def _stream(client: httpx.AsyncClient, request: Request, url: str, headers: dict, params) -> Response:
    """Unbuffered SSE passthrough with gateway heartbeats, a total-time cap and disconnect handling."""
    req = client.build_request("GET", url, headers=headers, params=params,
                               timeout=httpx.Timeout(None, connect=5.0))
    try:
        up = await client.send(req, stream=True)
    except httpx.TimeoutException:
        return JSONResponse({"detail": "Backend timed out", "backend": "slow"}, status_code=504)
    except httpx.HTTPError:
        return _down()
    if up.status_code != 200 or "text/event-stream" not in up.headers.get("content-type", ""):
        content = await up.aread()
        await up.aclose()
        return Response(content, status_code=up.status_code, headers=_clean_resp_headers(up.headers),
                        media_type=up.headers.get("content-type"))

    queue: asyncio.Queue = asyncio.Queue(maxsize=256)

    async def pump():
        try:
            async for chunk in up.aiter_raw():
                await queue.put(chunk)
        except Exception:
            await queue.put(b"event: error\ndata: {\"detail\":\"backend stream interrupted\"}\n\n")
        finally:
            await queue.put(None)

    task = asyncio.create_task(pump())

    async def gen() -> AsyncIterator[bytes]:
        deadline = time.monotonic() + STREAM_MAX_SECONDS
        try:
            while True:
                left = deadline - time.monotonic()
                if left <= 0:
                    yield b"event: error\ndata: {\"detail\":\"stream time cap reached\"}\n\n"
                    return
                try:
                    chunk = await asyncio.wait_for(queue.get(), timeout=min(HEARTBEAT, left))
                except asyncio.TimeoutError:
                    yield b": gw-heartbeat\n\n"
                    continue
                if chunk is None:
                    return
                yield chunk
        finally:  # runs on completion, cap, or client disconnect (generator cancelled/closed)
            task.cancel()
            await up.aclose()

    h = _clean_resp_headers(up.headers)
    h.update({"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"})
    return StreamingResponse(gen(), media_type="text/event-stream", headers=h)


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)), proxy_headers=True,
                forwarded_allow_ips="*", timeout_keep_alive=75)
