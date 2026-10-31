"""Sanity CLI:  python -m studio.explain.demo "<parent>" "<child>"

Uses the mock scorer from studio.mock.build_mock (RDKit descriptors + BRENK/PAINS alerts)
purely to feed the explainer; the real scorer is owned by the objectives stream."""
from __future__ import annotations

import sys
import time

from rdkit import Chem

from studio.explain.explainer import RDKitExplainer
from studio.mock.build_mock import score
from studio.schema import EditRecord


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print(__doc__)
        return 2
    parent, child = (Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in argv)
    ps, cs = score(parent), score(child)
    edit = EditRecord(edit_id="cli", name="Edit", category="functional_group", chosen_by="user", parents=[parent])
    t = time.perf_counter()
    ex = RDKitExplainer().explain(parent, child, edit, ps, cs)
    ms = (time.perf_counter() - t) * 1000
    print(ex.rationale)
    print("tags:", ", ".join(ex.tags) or "-")
    print("removed:", [a.name for a in ex.alerts_removed], "added:", [a.name for a in ex.alerts_added])
    for k, v in ex.descriptor_deltas.items():
        print(f"  {k:22s}{v:+.3f}")
    print(f"({ms:.1f} ms; scores from studio.mock.build_mock.score)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
