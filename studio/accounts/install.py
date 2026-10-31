"""One hook for the integrator:  `from studio.accounts.install import install; install(app)`.

Registers the account/library/profile routers *ahead of* the static "/" mount, creates the SQLite
tables, and lifts the body cap for /api/library/* only (other paths keep the server's original cap).
If anything here fails, the app is left untouched and /api/me simply does not exist (the UI hides itself).
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.routing import Mount

from studio.accounts import auth
from studio.accounts.db import make_engine
from studio.accounts.routes import State, router

log = logging.getLogger("studio.accounts")


def install(app: FastAPI, config: Optional[auth.AccountsConfig] = None) -> None:
    try:
        cfg = config or auth.AccountsConfig()
        engine = make_engine()
        app.state.accounts = State(engine=engine, secret=auth.load_jwt_secret(), cfg=cfg,
                                   limiter=auth.RateLimiter(), dummy_hash=auth.hash_password("x" * 12, cfg.bcrypt_cost))
        before = len(app.router.routes)
        app.include_router(router)
        new = app.router.routes[before:]
        rest = app.router.routes[:before]
        i = next((k for k, r in enumerate(rest) if isinstance(r, Mount) and r.path in ("", "/")), len(rest))
        app.router.routes[:] = rest[:i] + new + rest[i:]

        settings = getattr(app.state, "settings", None)
        orig_cap = getattr(settings, "max_body_bytes", None)
        if orig_cap is not None:
            settings.max_body_bytes = max(orig_cap, cfg.max_payload_bytes + 200_000)

            @app.middleware("http")  # added last => outermost: enforces the original cap outside the library
            async def _cap(request, call_next):
                if request.method in ("POST", "PUT", "PATCH") and not request.url.path.startswith("/api/library"):
                    cl = request.headers.get("content-length", "")
                    if cl.isdigit() and int(cl) > orig_cap:
                        return JSONResponse({"detail": f"Body too large (max {orig_cap // 1000} KB)"}, status_code=413)
                return await call_next(request)
    except Exception as exc:  # never threaten the main demo
        log.error("accounts disabled: %s", type(exc).__name__)
