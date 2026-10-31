"""Stage-1 wiring: the real engine pieces with the BindingScorer, pure GA (policy_guidance forced to 0)."""
from __future__ import annotations

from typing import Callable, Iterator, Optional

from studio.binding.backends import BindingBackend, make_backend
from studio.binding.scorer import BindingScorer
from studio.schema import Event, RunConfig


def binding_config(config: RunConfig) -> RunConfig:
    """Force the stage-1 semantics: pure GA (RL is untrained), binding stage."""
    return config.model_copy(update={"policy_guidance": 0.0, "stage": "binding"})


class BindingEngine:
    """Wraps an Engine so every run is a pure-GA binding run, whatever the caller asked for."""

    def __init__(self, engine, scorer: BindingScorer):
        self.engine, self.scorer = engine, scorer

    def __getattr__(self, name):  # registry, explainer, stats, ...
        return getattr(self.engine, name)

    def run(self, run_id: str, config: RunConfig) -> Iterator[Event]:
        return self.engine.run(run_id, binding_config(config))


def build_binding_engine(backend: Optional[BindingBackend] = None,
                         target_provider: Optional[Callable[[str], object]] = None) -> BindingEngine:
    from studio.wiring import build_engine
    scorer = BindingScorer(backend or make_backend("plapt"), target_provider)
    return BindingEngine(build_engine(scorer), scorer)
