"""Metis Lead Studio gateway (Hugging Face Space, Gradio SDK).

Serves the static studio UI from ./web and proxies /api/* to the GPU backend (Lambda, via a cloudflared tunnel).
The backend URL comes from /metis-data/backend_url.txt (re-read on change, so a rotating tunnel URL needs no
restart) or the BACKEND_URL variable. The shared secret STUDIO_API_KEY is only ever sent *to* the backend.

Why uvicorn.run at the bottom: a Gradio-SDK Space just executes `python app.py` and routes port 7860, so any
ASGI server bound there works. A minimal Gradio Blocks is mounted on /gradio only so the Space is a recognisable
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
from typing import AsyncIterator, Optional

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
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


def _backend_url() -> Optional[str]:
    """File (bucket) overrides the variable; None when nothing usable is configured."""
    url = ""
    try:
        if BACKEND_FILE.is_file():
            url = BACKEND_FILE.read_text().strip()
    except OSError:
        url = ""
    url = url or os.environ.get("BACKEND_URL", "").strip()
    url = url.rstrip("/")
    return url if url.startswith(("http://", "https://")) else None


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
    probe = {"t": 0.0, "up": False, "info": {}}
    app.state.client = client

    @app.middleware("http")
    async def session(request: Request, call_next):
        sid = request.cookies.get(GW_COOKIE)
        fresh = not sid or len(sid) > 64
        request.state.sid = secrets.token_urlsafe(16) if fresh else sid
        resp = await call_next(request)
        if fresh and request.url.path.startswith("/api/"):
            resp.set_cookie(GW_COOKIE, request.state.sid, httponly=True, samesite="lax", secure=request.headers.get("x-forwarded-proto", request.url.scheme) == "https",
                            max_age=86400)
        return resp

    async def check_backend() -> tuple[bool, dict]:
        now = time.monotonic()
        if now - probe["t"] < PROBE_TTL:
            return probe["up"], probe["info"]
        base = _backend_url()
        up, info = False, {}
        if base:
            try:
                r = await client.get(f"{base}/api/health", timeout=PROBE_TIMEOUT,
                                     headers={"X-Studio-Key": _api_key()} if _api_key() else {})
                if r.status_code == 200:
                    up = True
                    j = r.json()
                    info = {k: j[k] for k in ("engine_source", "schema_version") if k in j}
            except (httpx.HTTPError, ValueError):
                up = False
        probe.update(t=now, up=up, info=info)
        return up, info

    @app.get("/api/health")
    async def health():
        up, info = await check_backend()
        return JSONResponse({"backend": "up" if up else "down", "gateway": "ok", **info},
                            headers={"Cache-Control": "no-store"})

    @app.api_route("/api/{path:path}", methods=sorted(ALLOWED_METHODS))
    async def proxy(path: str, request: Request):
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
        url = f"{base}/api/{path}"
        headers = _upstream_headers(request, request.state.sid)
        params = list(request.query_params.multi_items())
        wants_stream = path.endswith("/events") or "text/event-stream" in request.headers.get("accept", "")
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

    try:  # optional: makes this a recognisable Gradio-SDK app; must be mounted before the "/" static mount
        import gradio as gr

        with gr.Blocks(title="Metis Lead Studio") as blocks:
            gr.Markdown("Metis Lead Studio gateway. The studio UI is at `/`.")
        gr.mount_gradio_app(app, blocks, path="/gradio")
    except Exception:  # gradio missing or incompatible: gateway still fully works
        pass

    if WEB.exists():
        app.mount("/", StaticFiles(directory=WEB, html=True), name="web")
    else:
        @app.get("/")
        async def _no_ui():
            return JSONResponse({"detail": "UI not built: run deploy/build_space.sh"}, status_code=404)
    return app


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
