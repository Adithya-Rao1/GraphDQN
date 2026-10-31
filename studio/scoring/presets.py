"""Named objective bundles. Keys of each bundle are schema.Descriptors fields."""
from __future__ import annotations

from typing import Dict

from studio.schema import ObjectiveSpec

O = ObjectiveSpec

PRESETS: Dict[str, Dict[str, ObjectiveSpec]] = {
    "remove toxic alerts": {
        "alerts": O(weight=3.0, direction="minimize", low=0, high=2),
        "qed": O(weight=0.5, direction="maximize", low=0.3, high=0.8),
        "mw": O(weight=0.5, direction="range", low=200, high=500),
    },
    "more drug-like": {
        "qed": O(weight=2.0, direction="maximize", low=0.3, high=0.9),
        "lipinski_violations": O(weight=1.5, direction="minimize", low=0, high=2),
        "mw": O(weight=1.0, direction="range", low=200, high=450),
        "logp": O(weight=1.0, direction="range", low=1, high=3),
        "tpsa": O(weight=1.0, direction="range", low=40, high=100),
        "alerts": O(weight=1.0, direction="minimize", low=0, high=2),
    },
    "more soluble": {
        "logp": O(weight=2.0, direction="range", low=0, high=2.5),
        "tpsa": O(weight=1.5, direction="range", low=60, high=120),
        "fsp3": O(weight=1.0, direction="maximize", low=0.2, high=0.6),
        "mw": O(weight=0.5, direction="range", low=150, high=450),
        "qed": O(weight=0.5, direction="maximize", low=0.3, high=0.8),
    },
}
