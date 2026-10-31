"""python -m studio.adapt.demo : end-to-end run with a SYNTHETIC scientist (always picks the highest-TPSA edit)."""
from __future__ import annotations

import json
import os
import tempfile

from studio.adapt import pipeline
from studio.adapt.synthetic import make_transitions


def main() -> None:
    os.environ.setdefault("STUDIO_CACHE_DIR", tempfile.mkdtemp(prefix="adapt-demo-"))
    from studio.registry import get_registry
    reg = get_registry()
    trans = make_transitions(reg, 40)
    print(f"SYNTHETIC scientist: {len(trans)} logged steps (not real user data)")
    res = pipeline.run(trans, "demo-owner", registry=reg, progress=lambda s, f: print(f"  {s} {f:.0%}", end="\r"))
    ev = res["evaluation"]
    print()
    print(json.dumps({k: res[k] for k in ("n_steps", "dataset", "training", "base_kind", "elapsed_s")}, indent=2))
    print(f"{ev['method']}  n={ev['n']}  mean candidates/step={ev['mean_candidates']}")
    for name in ("base", "adapted", "random"):
        r = ev[name]
        print(f"  {name:8s} top1={r['top1']:.3f} {r['top1_ci95']}  top3={r['top3']:.3f} {r['top3_ci95']}")
    print(f"  delta top1 (adapted-base) = {ev['delta_top1']['value']:+.3f} {ev['delta_top1']['ci95']}")
    print("NOTE:", ev["note"])


if __name__ == "__main__":
    main()
