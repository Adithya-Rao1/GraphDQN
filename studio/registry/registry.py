"""Registry implementation: ops from ops.py + metadata from edits.yaml."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import yaml

from studio.registry.ops import _BaseOp, build_ops

YAML_PATH = Path(__file__).with_name("edits.yaml")
DEMO_SAFE_PATH = Path(__file__).with_name("demo_safe.yaml")


class RegistryImpl:
    def __init__(self, yaml_path: Path = YAML_PATH, demo_safe_path: Path = DEMO_SAFE_PATH):
        meta = {e["id"]: e for e in yaml.safe_load(Path(yaml_path).read_text())["edits"]}
        ops = build_ops()
        ids = {o.id for o in ops}
        if ids != set(meta):
            raise ValueError(
                f"edits.yaml / ops mismatch: missing in yaml={sorted(ids - set(meta))}, "
                f"missing op={sorted(set(meta) - ids)}"
            )
        for op in ops:
            m = meta[op.id]
            op.name, op.category = m["name"], m["category"]
            op.tags = list(m.get("tags", []))
            op.rationale = m["rationale"]
            op.rationale_reviewed = bool(m.get("reviewed", False))
        safe = {e["id"]: e for e in yaml.safe_load(Path(demo_safe_path).read_text())["ops"]}
        if set(safe) != ids:
            raise ValueError(f"demo_safe.yaml / ops mismatch: {sorted(ids ^ set(safe))}")
        for op in ops:
            op.demo_safe = bool(safe[op.id]["demo_safe"])
            op.demo_safe_reason = safe[op.id]["reason"]
        self._ops: List[_BaseOp] = ops
        self._by_id: Dict[str, _BaseOp] = {o.id: o for o in ops}

    def all(self) -> List[_BaseOp]:
        return list(self._ops)

    def get(self, edit_id: str) -> _BaseOp:
        return self._by_id[edit_id]

    def applicable(self, smiles: str, locked_atoms: Sequence[int] = (),
                   demo_safe_only: Optional[bool] = None) -> List[Tuple[_BaseOp, List[int]]]:
        """Applicable (op, site) pairs. ``demo_safe_only`` defaults to True unless env STUDIO_ALL_EDITS=1."""
        if demo_safe_only is None:
            demo_safe_only = os.environ.get("STUDIO_ALL_EDITS") != "1"
        locked = set(locked_atoms)
        return [
            (op, site)
            for op in self._ops
            if op.demo_safe or not demo_safe_only
            for site in op.sites(smiles)
            if not locked.intersection(site)
        ]


_DEFAULT = None


def get_registry() -> RegistryImpl:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = RegistryImpl()
    return _DEFAULT
