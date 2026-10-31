"""Story card helper: headline + seed-vs-final table from real scores and the accepted edit path."""
from __future__ import annotations

from typing import List, Optional, Sequence

from studio.schema import ScoreResult, StoryCard

_TABLE_FIELDS = ("qed", "sa", "logp", "tpsa", "mw", "hbd", "hba", "rotb", "fsp3", "lipinski_violations", "alerts")


def build_story_card(
    seed_smiles: str,
    final_smiles: str,
    seed_score: ScoreResult,
    final_score: ScoreResult,
    edit_path: Sequence[str],
    suffix: Optional[str] = None,
) -> StoryCard:
    """`edit_path` = candidate ids from seed to final. `suffix` (e.g. "(MOCK DATA)") is appended verbatim."""
    s, f = seed_score.descriptors, final_score.descriptors
    table = {}
    for k in _TABLE_FIELDS:
        a, b = getattr(s, k), getattr(f, k)
        if a is not None and b is not None:
            table[k] = (float(a), float(b))

    n = len(edit_path)
    parts: List[str] = []
    da = s.alerts - f.alerts
    if da > 0:
        parts.append(f"Removed {da} alert{'s' if da != 1 else ''}")
    elif da < 0:
        parts.append(f"Added {-da} alert{'s' if da != -1 else ''}")
    parts.append(f"QED {s.qed:.2f} -> {f.qed:.2f}")
    headline = ", ".join(parts) + f" in {n} edit{'s' if n != 1 else ''}"
    if suffix:
        headline += f" {suffix}"
    return StoryCard(
        seed_smiles=seed_smiles,
        final_smiles=final_smiles,
        seed_score=seed_score,
        final_score=final_score,
        edit_path=list(edit_path),
        headline=headline,
        descriptor_table=table,
    )
