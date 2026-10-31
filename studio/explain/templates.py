"""Rationale text. Every number comes from the descriptor deltas / alert lists; nothing is invented."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

# field -> (label, unit, decimals, noise floor below which a change is not worth citing, scale for ranking)
_FIELDS = {
    "logp": ("logP", "", 1, 0.05, 1.0),
    "tpsa": ("TPSA", " Å²", 0, 0.5, 20.0),
    "qed": ("QED", "", 2, 0.005, 0.1),
    "mw": ("MW", " Da", 0, 0.5, 50.0),
    "hbd": ("HBD", "", 0, 0.5, 1.0),
    "hba": ("HBA", "", 0, 0.5, 2.0),
    "rotb": ("rotatable bonds", "", 0, 0.5, 2.0),
    "fsp3": ("Fsp3", "", 2, 0.005, 0.1),
    "sa": ("SA score", "", 1, 0.05, 1.0),
    "lipinski_violations": ("Lipinski violations", "", 0, 0.5, 1.0),
}


def _signed(v: float, decimals: int) -> str:
    s = f"{v:+.{decimals}f}"
    # avoid "-0.0" / "+0.0" after rounding
    return s.replace("-0.0", "0.0").replace("+0.0", "0.0") if float(s) == 0 else s


def top_changes(deltas: Dict[str, float], k: int = 3) -> List[str]:
    ranked = []
    for key, (label, unit, dec, floor, scale) in _FIELDS.items():
        d = deltas.get(key)
        if d is None or abs(d) < floor or float(f"{d:.{dec}f}") == 0:
            continue
        ranked.append((abs(d) / scale, f"{label} {_signed(d, dec)}{unit}"))
    ranked.sort(key=lambda t: -t[0])
    return [text for _, text in ranked[:k]]


def _alert_phrase(verb: str, names: Sequence[str]) -> str:
    n = len(names)
    return f"{verb} {n} structural alert{'s' if n != 1 else ''} ({', '.join(names)})"


def compose_rationale(
    edit_name: str,
    deltas: Dict[str, float],
    removed: Sequence[str],
    added: Sequence[str],
    registry_rationale: Optional[str] = None,
) -> str:
    parts = []
    if removed:
        parts.append(_alert_phrase("removes", removed))
    if added:
        parts.append(_alert_phrase("adds", added))
    changes = top_changes(deltas)
    if changes:
        parts.append(", ".join(changes))
    body = "; ".join(parts) if parts else "no change in the tracked descriptors"
    text = f"{edit_name}: {body}."
    if registry_rationale:
        text += f" Why: {registry_rationale.strip()}"
    return text
