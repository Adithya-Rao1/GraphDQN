"""Password hashing, JWT, cookie/bearer sessions, CSRF and rate limiting. Never logs passwords or tokens."""
from __future__ import annotations

import hmac
import os
import re
import secrets
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Optional

import bcrypt
import jwt

from studio.accounts.db import cache_dir

COOKIE = "studio_auth"
USERNAME_RE = re.compile(r"^[a-z0-9_-]{3,32}$")
JWT_ALG = "HS256"


@dataclass
class AccountsConfig:
    bcrypt_cost: int = 12
    token_ttl_s: int = 7 * 24 * 3600
    login_per_min_ip: int = 10
    signup_per_hour_ip: int = 10
    login_per_min_cookie: int = 10
    login_per_5min_user: int = 10
    max_runs_per_user: int = 200
    max_manual_per_user: int = 100
    max_profiles_per_user: int = 20
    max_payload_bytes: int = 2_000_000


def validate_username(name: str) -> Optional[str]:
    return None if USERNAME_RE.match(name or "") else "Username must be 3-32 characters of a-z, 0-9, _ or -"


def validate_password(pw: str) -> Optional[str]:
    if not isinstance(pw, str) or len(pw) < 8:
        return "Password must be at least 8 characters"
    if len(pw.encode("utf-8")) > 72:
        return "Password must be at most 72 bytes"
    return None


def hash_password(pw: str, cost: int = 12) -> str:
    return bcrypt.hashpw(pw.encode("utf-8"), bcrypt.gensalt(rounds=cost)).decode("ascii")


def check_password(pw: str, pw_hash: str) -> bool:
    try:
        return bcrypt.checkpw(pw.encode("utf-8")[:72], pw_hash.encode("ascii"))
    except ValueError:
        return False


def load_jwt_secret() -> str:
    """STUDIO_JWT_SECRET, else a secret generated ONCE and persisted (0600) in the cache dir."""
    env = os.environ.get("STUDIO_JWT_SECRET")
    if env:
        return env
    path = cache_dir() / "jwt_secret"
    try:
        s = path.read_text().strip()
        if len(s) >= 32:
            return s
    except OSError:
        pass
    secret = secrets.token_urlsafe(48)
    try:  # O_EXCL: if two workers race, the loser re-reads the winner's file
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(secret)
        return secret
    except FileExistsError:
        for _ in range(20):
            s = path.read_text().strip()
            if len(s) >= 32:
                return s
            time.sleep(0.05)
        raise


def make_token(secret: str, user_id: int, username: str, ttl_s: int) -> str:
    now = int(time.time())
    return jwt.encode({"sub": str(user_id), "usr": username, "iat": now, "exp": now + ttl_s}, secret, algorithm=JWT_ALG)


def read_token(secret: str, token: str) -> Optional[dict]:
    try:
        return jwt.decode(token, secret, algorithms=[JWT_ALG], options={"require": ["exp", "sub"]})
    except jwt.PyJWTError:
        return None


def consteq(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode(), b.encode())


class RateLimiter:
    """In-memory sliding window; bounded memory (stale keys pruned)."""

    def __init__(self) -> None:
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def hit(self, key: str, limit: int, window_s: float) -> int:
        """Record a hit. Returns 0 if allowed, else seconds to wait."""
        now = time.monotonic()
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] > window_s:
                q.popleft()
            if len(q) >= limit:
                return max(1, int(window_s - (now - q[0])) + 1)
            q.append(now)
            if len(self._hits) > 20000:
                for k in [k for k, v in self._hits.items() if not v or now - v[-1] > window_s]:
                    self._hits.pop(k, None)
            return 0


def client_ip(request) -> str:
    """Last X-Forwarded-For hop (appended by our own proxy) else the socket peer."""
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[-1].strip()[:64]
    return (request.client.host if request.client else "unknown")[:64]
