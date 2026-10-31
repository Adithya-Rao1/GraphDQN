"""Python interfaces the studio components code against.

Each Protocol has exactly one owner (see studio/OWNERSHIP.md). Owners must keep
their implementation compatible with these signatures; consumers must depend only
on these signatures, never on a concrete class, so streams can be built in parallel
and tested with fakes.
"""
from __future__ import annotations

from typing import Iterator, List, Optional, Protocol, Sequence, runtime_checkable

from studio.schema import (
    EditRecord,
    Event,
    Explanation,
    RunConfig,
    ScoreResult,
)


@runtime_checkable
class EditOp(Protocol):
    """One molecular edit from the registry (owner: registry)."""

    id: str
    name: str
    category: str  # one of schema.EditCategory (non-crossover)
    tags: Sequence[str]  # expected effects, e.g. ["lowers_logp", "adds_hbd"]
    rationale: str  # one-line med-chem reason shown in the UI (drafted; human-reviewed)

    def sites(self, smiles: str) -> List[List[int]]:
        """Atom-index sites where this edit can apply. Empty list if not applicable."""
        ...

    def apply(self, smiles: str, site: Sequence[int]) -> List[str]:
        """Return valid, canonical product SMILES (possibly empty). Never raises on bad input."""
        ...


@runtime_checkable
class Registry(Protocol):
    """Collection of edits (owner: registry)."""

    def all(self) -> List[EditOp]: ...

    def get(self, edit_id: str) -> EditOp: ...

    def applicable(self, smiles: str, locked_atoms: Sequence[int] = ()) -> List[tuple]:
        """Return [(EditOp, site), ...] that touch no locked atom."""
        ...


@runtime_checkable
class Scorer(Protocol):
    """RDKit-descriptor scoring (owner: scoring). Must be fast (<20 ms/molecule) and cached."""

    def score(self, smiles: str, config: RunConfig) -> ScoreResult: ...

    def score_many(self, smiles: Sequence[str], config: RunConfig) -> List[ScoreResult]: ...


@runtime_checkable
class Explainer(Protocol):
    """Exact, RDKit-based explanations (owner: explain)."""

    def explain(
        self,
        parent_smiles: str,
        child_smiles: str,
        edit: EditRecord,
        parent_score: ScoreResult,
        child_score: ScoreResult,
        edit_op: Optional[EditOp] = None,
    ) -> Explanation: ...


@runtime_checkable
class Embedder(Protocol):
    """Chemical-space coordinates for the population map (owner: embed)."""

    method: str  # "morgan_pca" | "chemberta_pca"

    def map_xy(self, smiles: Sequence[str], anchor: Optional[str] = None) -> List[tuple]:
        """2D coordinates, index-aligned with `smiles`. Falls back to Morgan+PCA if the model is unavailable."""
        ...


@runtime_checkable
class Policy(Protocol):
    """DQN edit-proposal prior (owner: policy). May be a uniform fake until a checkpoint exists."""

    def rank(self, smiles: str, candidates: Sequence[tuple]) -> List[float]:
        """Score each (EditOp, site) candidate; higher = more preferred."""
        ...

    def update(self, transitions: Sequence[tuple]) -> Optional[float]:
        """Online learning step on accepted transitions; returns loss if any."""
        ...


@runtime_checkable
class Engine(Protocol):
    """GA + policy loop (owner: engine). Yields schema Events for a run, including
    3D molblocks (via the shared geometry helper) and explanations."""

    def run(self, run_id: str, config: RunConfig) -> Iterator[Event]: ...
