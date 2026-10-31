"""Account, library and profile routes (all under /api). Cross-user access returns 404, never 403."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from studio.accounts import auth
from studio.accounts.db import ManualSessionRow, Profile, SavedRun, User
from studio.accounts.profiles import MAX_SMARTS, ProfileBody, check_smarts
from studio.schema import Event

log = logging.getLogger("studio.accounts")
router = APIRouter(prefix="/api")

MAX_EVENTS = 5000
MAX_STEPS = 5000


@dataclass
class State:
    engine: Any
    secret: str
    cfg: auth.AccountsConfig
    limiter: auth.RateLimiter
    dummy_hash: str


def _st(request: Request) -> State:
    return request.app.state.accounts


def db(request: Request):
    with Session(_st(request).engine) as s:
        yield s


# ------------------------------------------------------------------ dependencies
def csrf(request: Request) -> None:
    if request.headers.get("x-requested-with") != "studio":
        raise HTTPException(403, "Missing X-Requested-With header")


def _token(request: Request) -> Optional[str]:
    h = request.headers.get("authorization", "")
    if h[:7].lower() == "bearer ":
        return h[7:].strip()
    return request.cookies.get(auth.COOKIE)


def current_user(request: Request) -> Optional[User]:
    tok = _token(request)
    if not tok:
        return None
    st = _st(request)
    claims = auth.read_token(st.secret, tok)
    if not claims or not str(claims["sub"]).isdigit():
        return None
    with Session(st.engine) as s:
        u = s.get(User, int(claims["sub"]))
        if u is None or u.username != claims.get("usr"):
            return None
        s.expunge(u)
        return u


def require_user(request: Request) -> User:
    u = current_user(request)
    if u is None:
        raise HTTPException(401, "Sign in required")
    return u


def _limit(request: Request, key: str, limit: int, window: float) -> None:
    wait = _st(request).limiter.hit(key, limit, window)
    if wait:
        raise HTTPException(429, "Too many attempts, try again later", headers={"Retry-After": str(wait)})


def _set_cookie(request: Request, resp: Response, token: str) -> None:
    secure = request.headers.get("x-forwarded-proto", "").split(",")[0].strip() == "https"
    resp.set_cookie(auth.COOKIE, token, httponly=True, samesite="lax", secure=secure,
                    max_age=_st(request).cfg.token_ttl_s, path="/")


# ----------------------------------------------------------------------- auth
class Creds(BaseModel):
    username: str = Field(max_length=200)
    password: str = Field(max_length=1000)


def _user_out(u: User) -> dict:
    return {"id": u.id, "username": u.username}


@router.post("/auth/signup", dependencies=[Depends(csrf)])
def signup(body: Creds, request: Request, response: Response, s: Session = Depends(db)):
    st = _st(request)
    ip = auth.client_ip(request)
    _limit(request, f"su:ip:{ip}", st.cfg.signup_per_hour_ip, 3600)
    sid = request.cookies.get("studio_sid")
    if sid:
        _limit(request, f"su:ck:{sid}", st.cfg.signup_per_hour_ip, 3600)
    name = body.username.strip().lower()
    err = auth.validate_username(name) or auth.validate_password(body.password)
    if err:
        raise HTTPException(422, err)
    user = User(username=name, pw_hash=auth.hash_password(body.password, st.cfg.bcrypt_cost))
    s.add(user)
    try:
        s.commit()
    except IntegrityError:
        s.rollback()
        raise HTTPException(409, "That username is not available")
    _set_cookie(request, response, auth.make_token(st.secret, user.id, user.username, st.cfg.token_ttl_s))
    return {"user": _user_out(user)}


@router.post("/auth/login", dependencies=[Depends(csrf)])
def login(body: Creds, request: Request, response: Response, s: Session = Depends(db)):
    st = _st(request)
    ip = auth.client_ip(request)
    name = body.username.strip().lower()[:64]
    _limit(request, f"li:ip:{ip}", st.cfg.login_per_min_ip, 60)
    sid = request.cookies.get("studio_sid")
    if sid:
        _limit(request, f"li:ck:{sid}", st.cfg.login_per_min_cookie, 60)
    _limit(request, f"li:us:{name}", st.cfg.login_per_5min_user, 300)
    user = s.scalar(select(User).where(User.username == name))
    # Always run one bcrypt check so unknown users and wrong passwords look identical (timing + message).
    ok = auth.check_password(body.password, user.pw_hash if user else st.dummy_hash)
    if not user or not ok:
        raise HTTPException(401, "Invalid username or password")
    _set_cookie(request, response, auth.make_token(st.secret, user.id, user.username, st.cfg.token_ttl_s))
    return {"user": _user_out(user)}


@router.post("/auth/logout", dependencies=[Depends(csrf)])
def logout(response: Response):
    response.delete_cookie(auth.COOKIE, path="/")
    return {"ok": True}


@router.get("/me")
def me(request: Request):
    u = current_user(request)
    return {"guest": u is None, "user": _user_out(u) if u else None}


# -------------------------------------------------------------------- library
def _iso(dt) -> str:
    return dt.isoformat() if dt else ""


class RunIn(BaseModel):
    title: Optional[str] = Field(default=None, max_length=120)
    stage: Optional[str] = Field(default=None, max_length=32)
    target_id: Optional[str] = Field(default=None, max_length=64)
    events: List[Event]


def _run_summary(r: SavedRun) -> dict:
    return {"id": r.id, "title": r.title, "stage": r.stage, "target_id": r.target_id,
            "seed_smiles": r.seed_smiles, "headline": r.headline, "created_at": _iso(r.created_at)}


def _cap(n: int, limit: int, what: str) -> None:
    if n >= limit:
        raise HTTPException(409, f"Library full: at most {limit} {what}. Delete some first.")


@router.get("/library/runs")
def list_runs(user: User = Depends(require_user), s: Session = Depends(db)):
    rows = s.scalars(select(SavedRun).where(SavedRun.user_id == user.id).order_by(SavedRun.id.desc()).limit(500))
    return {"runs": [_run_summary(r) for r in rows]}


@router.post("/library/runs", dependencies=[Depends(csrf)], status_code=201)
def save_run(body: RunIn, request: Request, user: User = Depends(require_user), s: Session = Depends(db)):
    cfg = _st(request).cfg
    if not body.events or len(body.events) > MAX_EVENTS:
        raise HTTPException(422, f"events must contain 1..{MAX_EVENTS} items")
    events = [e.model_dump(mode="json") for e in body.events]
    blob = json.dumps(events, separators=(",", ":"))
    if len(blob.encode("utf-8")) > cfg.max_payload_bytes:
        raise HTTPException(413, f"Run too large (max {cfg.max_payload_bytes // 1_000_000} MB)")
    _cap(s.scalar(select(_count(SavedRun.id)).where(SavedRun.user_id == user.id)) or 0,
         cfg.max_runs_per_user, "saved runs")
    start = next((e for e in events if e["type"] == "run_start"), None)
    end = next((e for e in reversed(events) if e["type"] == "run_end"), None)
    rc = (start or {}).get("payload", {}).get("config", {}) if start else {}
    seed = rc.get("seed_smiles") or (end or {}).get("payload", {}).get("seed_smiles")
    headline = (end or {}).get("payload", {}).get("headline")
    stage = body.stage or rc.get("stage") or "pharmacology"
    target = body.target_id or rc.get("target_id")
    title = (body.title or "").strip() or (headline or f"Run of {seed or 'molecule'}")
    row = SavedRun(user_id=user.id, title=title[:120], stage=str(stage)[:32],
                   target_id=str(target)[:64] if target else None,
                   seed_smiles=str(seed)[:500] if seed else None,
                   headline=str(headline)[:300] if headline else None, events_json=blob)
    s.add(row)
    s.commit()
    return _run_summary(row)


def _count(col):
    from sqlalchemy import func
    return func.count(col)


def _own(s: Session, model, row_id: int, user: User):
    row = s.scalar(select(model).where(model.id == row_id, model.user_id == user.id))
    if row is None:
        raise HTTPException(404, "Not found")
    return row


@router.get("/library/runs/{run_id}")
def get_run(run_id: int, user: User = Depends(require_user), s: Session = Depends(db)):
    r = _own(s, SavedRun, run_id, user)
    return {**_run_summary(r), "events": json.loads(r.events_json)}


@router.delete("/library/runs/{run_id}", dependencies=[Depends(csrf)])
def delete_run(run_id: int, user: User = Depends(require_user), s: Session = Depends(db)):
    s.delete(_own(s, SavedRun, run_id, user))
    s.commit()
    return {"ok": True}


class ManualIn(BaseModel):
    seed_smiles: Optional[str] = Field(default=None, max_length=500)
    target_id: Optional[str] = Field(default=None, max_length=64)
    steps: Optional[List[Dict[str, Any]]] = None
    jsonl: Optional[str] = Field(default=None, max_length=2_000_000)  # the manual export file, verbatim


def _manual_summary(m: ManualSessionRow) -> dict:
    return {"id": m.id, "seed_smiles": m.seed_smiles, "target_id": m.target_id, "created_at": _iso(m.created_at)}


@router.get("/library/manual")
def list_manual(user: User = Depends(require_user), s: Session = Depends(db)):
    rows = s.scalars(select(ManualSessionRow).where(ManualSessionRow.user_id == user.id)
                     .order_by(ManualSessionRow.id.desc()).limit(500))
    return {"sessions": [_manual_summary(m) for m in rows]}


@router.post("/library/manual", dependencies=[Depends(csrf)], status_code=201)
def save_manual(body: ManualIn, request: Request, user: User = Depends(require_user), s: Session = Depends(db)):
    cfg = _st(request).cfg
    steps = body.steps
    if steps is None and body.jsonl is not None:
        steps = []
        for line in body.jsonl.splitlines():
            if line.strip():
                try:
                    obj = json.loads(line)
                except ValueError:
                    raise HTTPException(422, "jsonl contains a line that is not valid JSON")
                if not isinstance(obj, dict):
                    raise HTTPException(422, "jsonl lines must be JSON objects")
                steps.append(obj)
    if not steps or len(steps) > MAX_STEPS:
        raise HTTPException(422, f"steps must contain 1..{MAX_STEPS} items")
    blob = json.dumps(steps, separators=(",", ":"))
    if len(blob.encode("utf-8")) > cfg.max_payload_bytes:
        raise HTTPException(413, f"Session too large (max {cfg.max_payload_bytes // 1_000_000} MB)")
    _cap(s.scalar(select(_count(ManualSessionRow.id)).where(ManualSessionRow.user_id == user.id)) or 0,
         cfg.max_manual_per_user, "manual sessions")
    seed = body.seed_smiles or next((x.get("seed_smiles") or x.get("parent_smiles") or x.get("smiles_in")
                                     for x in steps if isinstance(x, dict)), None)
    row = ManualSessionRow(user_id=user.id, seed_smiles=str(seed)[:500] if isinstance(seed, str) else None,
                           target_id=body.target_id, steps_json=blob)
    s.add(row)
    s.commit()
    return _manual_summary(row)


@router.get("/library/manual/{sid}")
def get_manual(sid: int, user: User = Depends(require_user), s: Session = Depends(db)):
    m = _own(s, ManualSessionRow, sid, user)
    return {**_manual_summary(m), "steps": json.loads(m.steps_json)}


@router.delete("/library/manual/{sid}", dependencies=[Depends(csrf)])
def delete_manual(sid: int, user: User = Depends(require_user), s: Session = Depends(db)):
    s.delete(_own(s, ManualSessionRow, sid, user))
    s.commit()
    return {"ok": True}


# ------------------------------------------------------------------- profiles
def _profile_out(p: Profile) -> dict:
    return {"id": p.id, "created_at": _iso(p.created_at), **json.loads(p.json)}


def _parse_profile(raw: dict) -> ProfileBody:
    try:
        return ProfileBody.model_validate(raw)
    except ValidationError as exc:
        e = exc.errors()[0]
        loc = ".".join(str(x) for x in e["loc"])
        msg = e["msg"].removeprefix("Value error, ")
        raise HTTPException(422, f"{loc}: {msg}" if loc else msg)


class SmartsCheck(BaseModel):
    patterns: List[str] = Field(max_length=MAX_SMARTS)


@router.post("/profiles/check-smarts")
def check_smarts_route(body: SmartsCheck, request: Request):
    """Live validation for the editor: [{ok, error}] per pattern. No auth, no state."""
    _limit(request, f"sm:{auth.client_ip(request)}", 120, 60)
    return {"results": [{"ok": (e := check_smarts(p)) is None, "error": e} for p in body.patterns]}


@router.get("/profiles")
def list_profiles(user: User = Depends(require_user), s: Session = Depends(db)):
    rows = s.scalars(select(Profile).where(Profile.user_id == user.id).order_by(Profile.id.desc()))
    return {"profiles": [_profile_out(p) for p in rows]}


@router.post("/profiles", dependencies=[Depends(csrf)], status_code=201)
def create_profile(raw: Dict[str, Any], request: Request, user: User = Depends(require_user),
                   s: Session = Depends(db)):
    body = _parse_profile(raw)
    _cap(s.scalar(select(_count(Profile.id)).where(Profile.user_id == user.id)) or 0,
         _st(request).cfg.max_profiles_per_user, "profiles")
    row = Profile(user_id=user.id, name=body.name, json=body.model_dump_json())
    s.add(row)
    s.commit()
    return _profile_out(row)


@router.get("/profiles/{pid}")
def get_profile(pid: int, user: User = Depends(require_user), s: Session = Depends(db)):
    return _profile_out(_own(s, Profile, pid, user))


@router.put("/profiles/{pid}", dependencies=[Depends(csrf)])
def update_profile(pid: int, raw: Dict[str, Any], user: User = Depends(require_user), s: Session = Depends(db)):
    row = _own(s, Profile, pid, user)
    body = _parse_profile(raw)
    row.name, row.json = body.name, body.model_dump_json()
    s.commit()
    return _profile_out(row)


@router.delete("/profiles/{pid}", dependencies=[Depends(csrf)])
def delete_profile(pid: int, user: User = Depends(require_user), s: Session = Depends(db)):
    s.delete(_own(s, Profile, pid, user))
    s.commit()
    return {"ok": True}
