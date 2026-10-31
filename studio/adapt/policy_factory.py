"""Load an adapted policy for new runs; falls back to the base policy and says so (`policy.adapter_info`)."""
from __future__ import annotations

import logging
from typing import Optional, Sequence

import torch

from studio.adapt import store
from studio.policy.policy import DQNPolicy, load_policy

log = logging.getLogger("studio.adapt")


def _fallback(op_ids: Sequence[str], adapter_id: str, reason: str):
    log.warning("adapter %s not used: %s (falling back to the base policy)", adapter_id, reason)
    pol = load_policy(list(op_ids))
    pol.adapter_info = {"adapter_id": adapter_id, "used": False, "reason": reason}
    return pol


def load_adapted_policy(adapter_id: str, owner: str, op_ids: Optional[Sequence[str]] = None):
    """Policy with the owner's adapted weights. `op_ids` defaults to the live registry's op ids."""
    if op_ids is None:
        from studio.registry import get_registry
        op_ids = [o.id for o in get_registry().all()]
    op_ids = list(op_ids)
    path, meta = store.weights_path(owner, adapter_id), store.get_meta(owner, adapter_id)
    if path is None or meta is None:
        return _fallback(op_ids, adapter_id, "adapter file missing")
    if meta.get("op_id_hash") != store.op_hash(op_ids):
        return _fallback(op_ids, adapter_id, "edit set changed since the adapter was trained (op-id hash differs)")
    try:
        ck = torch.load(path, map_location="cpu", weights_only=True)
        if list(ck["op_ids"]) != op_ids:
            return _fallback(op_ids, adapter_id, "op ids differ")
        pol = DQNPolicy(op_ids, checkpoint=None)
        pol.net.load_state_dict(ck["state"])
        pol.net.eval()
    except Exception as exc:  # corrupt/incompatible file must never break a run
        return _fallback(op_ids, adapter_id, f"adapter unreadable ({type(exc).__name__})")
    pol.adapter_info = {"adapter_id": adapter_id, "used": True, "reason": "ok"}
    return pol
