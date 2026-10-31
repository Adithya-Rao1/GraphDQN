"""Adapter files: $STUDIO_CACHE_DIR/adapters/<owner>/<adapter_id>.pt + <adapter_id>.json sidecar."""
from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import time
from pathlib import Path
from typing import List, Optional, Sequence

import torch

ID_RE = re.compile(r"^[a-f0-9]{16}$")
MAX_ADAPTERS_PER_OWNER = 10


def cache_root() -> Path:
    return Path(os.environ.get("STUDIO_CACHE_DIR") or ".cache/studio") / "adapters"


def owner_dir(owner: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", owner)[:40] + "-" + hashlib.sha256(owner.encode()).hexdigest()[:8]
    return cache_root() / safe


def op_hash(op_ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(op_ids).encode()).hexdigest()[:16]


def save(owner: str, policy, *, n_steps: int, base_checkpoint: str, base_kind: str, evaluation: dict,
         training: dict) -> dict:
    op_ids = list(policy.net.op_ids)
    aid = secrets.token_hex(8)
    d = owner_dir(owner)
    d.mkdir(parents=True, exist_ok=True)
    existing = list_adapters(owner)
    for old in existing[MAX_ADAPTERS_PER_OWNER - 1:]:  # keep the newest MAX-1, then add this one
        delete(owner, old["adapter_id"])
    torch.save({"op_ids": op_ids, "state": policy.net.state_dict()}, d / f"{aid}.pt")
    meta = {"adapter_id": aid, "owner": owner, "created_at": time.time(), "n_steps": n_steps,
            "op_id_hash": op_hash(op_ids), "base_checkpoint": base_checkpoint, "base_kind": base_kind,
            "hit_rates": {k: evaluation.get(k) for k in ("base", "adapted", "random")},
            "eval_n": evaluation.get("n"), "eval_method": evaluation.get("method"), "training": training}
    (d / f"{aid}.json").write_text(json.dumps(meta))
    return meta


def get_meta(owner: str, adapter_id: str) -> Optional[dict]:
    if not ID_RE.match(adapter_id or ""):
        return None
    p = owner_dir(owner) / f"{adapter_id}.json"
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def weights_path(owner: str, adapter_id: str) -> Optional[Path]:
    if not ID_RE.match(adapter_id or ""):
        return None
    p = owner_dir(owner) / f"{adapter_id}.pt"
    return p if p.exists() else None


def list_adapters(owner: str) -> List[dict]:
    d = owner_dir(owner)
    out = []
    for p in d.glob("*.json") if d.exists() else []:
        try:
            out.append(json.loads(p.read_text()))
        except (OSError, ValueError):
            continue
    return sorted(out, key=lambda m: m.get("created_at", 0), reverse=True)


def delete(owner: str, adapter_id: str) -> bool:
    if not ID_RE.match(adapter_id or ""):
        return False
    d, hit = owner_dir(owner), False
    for ext in (".pt", ".json"):
        try:
            (d / f"{adapter_id}{ext}").unlink()
            hit = True
        except FileNotFoundError:
            pass
    return hit
