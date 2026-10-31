"""Domain-profile schema + validation (shared by the routes and ProfileScorer)."""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator
from rdkit import Chem, RDLogger

from studio.schema import ObjectiveSpec

MAX_SMARTS, MAX_SMARTS_LEN, MAX_ALLOW, MAX_OBJECTIVES = 20, 200, 300, 20


def check_smarts(s: str) -> Optional[str]:
    """None if valid, else a short message. Never raises."""
    if not isinstance(s, str) or not s.strip():
        return "empty pattern"
    if len(s) > MAX_SMARTS_LEN:
        return f"longer than {MAX_SMARTS_LEN} characters"
    RDLogger.DisableLog("rdApp.*")
    try:
        q = Chem.MolFromSmarts(s)
    except Exception:
        q = None
    finally:
        RDLogger.EnableLog("rdApp.*")
    if q is None or q.GetNumAtoms() == 0:
        return "not a valid SMARTS pattern"
    return None


def known_edit_ids() -> Optional[set]:
    try:
        from studio.registry import get_registry
        return {op.id for op in get_registry().all()}
    except Exception:
        return None


class ProfileBody(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    target_id: Optional[str] = Field(default=None, max_length=64)
    objectives: Dict[str, ObjectiveSpec] = Field(default_factory=dict)
    avoid_smarts: List[str] = Field(default_factory=list)
    edit_allowlist: List[str] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("name is empty")
        return v

    @field_validator("objectives")
    @classmethod
    def _obj(cls, v):
        if len(v) > MAX_OBJECTIVES:
            raise ValueError(f"at most {MAX_OBJECTIVES} objectives")
        for k, spec in v.items():
            if not (1 <= len(k) <= 40):
                raise ValueError("objective key length")
            if not (0.0 <= spec.weight <= 10.0):
                raise ValueError("objective weight must be within 0..10")
        return v

    @field_validator("avoid_smarts")
    @classmethod
    def _smarts(cls, v):
        if len(v) > MAX_SMARTS:
            raise ValueError(f"at most {MAX_SMARTS} SMARTS patterns")
        for i, s in enumerate(v):
            err = check_smarts(s)
            if err:
                raise ValueError(f"avoid_smarts[{i}]: {err}")
        return v

    @field_validator("edit_allowlist")
    @classmethod
    def _allow(cls, v):
        if len(v) > MAX_ALLOW:
            raise ValueError(f"at most {MAX_ALLOW} edit ids")
        known = known_edit_ids()
        for e in v:
            if not isinstance(e, str) or len(e) > 80:
                raise ValueError("edit id too long")
            if known is not None and e not in known:
                raise ValueError(f"unknown edit id: {e[:40]}")
        return v
