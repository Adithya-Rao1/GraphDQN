"""Read-only access to the verified target repository under studio/data/."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import List

from studio.schema import TargetInfo

DATA_DIR = Path(__file__).resolve().parent
TARGETS_DIR = DATA_DIR / "targets"


def list_targets() -> List[dict]:
    """[{id, name}, ...] from index.json, in display order."""
    return json.loads((TARGETS_DIR / "index.json").read_text())


@lru_cache(maxsize=None)
def get_target(target_id: str) -> TargetInfo:
    path = TARGETS_DIR / f"{target_id}.json"
    if target_id == "index" or not path.is_file():
        raise KeyError(f"unknown target: {target_id}")
    return TargetInfo.model_validate_json(path.read_text())


def structure_path(target_id: str) -> Path:
    """Absolute path of the cleaned protein-only PDB for a target."""
    return DATA_DIR / get_target(target_id).structure_url


def dynamicbind_ligand_csv(target_id: str) -> Path:
    get_target(target_id)
    return DATA_DIR / "dynamicbind_inputs" / target_id / "ligand.csv"


def reference_ligand_path(target_id: str):
    """Co-crystal reference pose (SDF, same frame as the structure) or None."""
    get_target(target_id)
    p = DATA_DIR / "ligands" / f"{target_id}_reference.sdf"
    return p if p.is_file() else None
