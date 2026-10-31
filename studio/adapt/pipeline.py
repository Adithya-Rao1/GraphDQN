"""clean -> refuse if < 5 -> fine-tune on all valid steps -> held-out evaluation -> save adapter."""
from __future__ import annotations

import time
from typing import Any, Callable, Optional, Sequence

from studio.adapt import store
from studio.adapt.dataset import MIN_VALID_STEPS, clean
from studio.adapt.evaluate import evaluate
from studio.adapt.finetune import DEFAULT_STEPS, BaseInfo, TooFewSteps, adapt, resolve_base

JOB_BUDGET_S = 80.0     # everything must finish inside this (job timeout is 90 s)
FINAL_TRAIN_S = 30.0


def run(transitions: Sequence[Any], owner: str, *, registry, base: Optional[BaseInfo] = None,
        progress: Optional[Callable[[str, float], None]] = None, budget_s: float = JOB_BUDGET_S,
        steps: int = DEFAULT_STEPS, save: bool = True) -> dict:
    t0 = time.monotonic()
    deadline = t0 + budget_s
    prog = progress or (lambda stage, f: None)
    base = base or resolve_base([o.id for o in registry.all()])
    prog("validating", 0.0)
    examples, report = clean(transitions, registry)
    if len(examples) < MIN_VALID_STEPS:
        raise TooFewSteps(f"Need at least {MIN_VALID_STEPS} valid edits to adapt; {len(examples)} of "
                          f"{report.received} were usable (dropped: {report.reasons or 'none'}).")
    prog("training", 0.1)
    final = adapt(examples, base, steps=steps, max_seconds=FINAL_TRAIN_S, deadline=deadline,
                  progress=lambda f: prog("training", 0.1 + 0.4 * f))
    prog("evaluating", 0.5)
    ev = evaluate(examples, base, steps=steps, deadline=deadline, progress=lambda f: prog("evaluating", 0.5 + 0.5 * f))
    training = {"opt_steps": final.steps_done, "stopped_early": final.stopped_early, "seconds": round(final.seconds, 2),
                "final_loss": final.final_loss, "objective": "imitation (softmax cross-entropy over applicable candidates)"}
    meta = None
    if save:
        meta = store.save(owner, final.policy, n_steps=len(examples), base_checkpoint=base.checkpoint_name,
                          base_kind=base.kind, evaluation=ev, training=training)
    prog("done", 1.0)
    return {"adapter_id": meta["adapter_id"] if meta else None, "n_steps": len(examples), "dataset": report.as_dict(),
            "evaluation": ev, "training": training, "base_kind": base.kind,
            "elapsed_s": round(time.monotonic() - t0, 2),
            "label": "Evaluation is on held-out edits from this user's own log; the saved adapter is trained on all valid steps."}
