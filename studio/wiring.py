"""Integrator-owned: assemble the real components into the Engine the server runs.

Everything is constructed lazily and falls back honestly: a missing policy checkpoint
means UniformPolicy (the UI/health report it), a missing ChemBERTa means Morgan.
"""
from __future__ import annotations

import logging

from studio.embed import get_embedder
from studio.engine import Engine
from studio.explain.explainer import RDKitExplainer
from studio.policy.policy import load_policy
from studio.registry import get_registry
from studio.scoring import RDKitScorer

log = logging.getLogger("studio.wiring")


def build_scorer() -> RDKitScorer:
    return RDKitScorer()


def build_engine(scorer: RDKitScorer | None = None) -> Engine:
    registry = get_registry()
    policy = load_policy([op.id for op in registry.all()])
    log.info("policy=%s embedder=%s ops=%d", type(policy).__name__, get_embedder().method, len(registry.all()))
    return Engine(registry, scorer or build_scorer(), RDKitExplainer(), get_embedder(), policy)
