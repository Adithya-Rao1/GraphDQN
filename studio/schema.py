"""Wire schema for Metis Lead Studio.

Single source of truth for every JSON shape that crosses a process boundary
(HTTP request/response bodies and SSE event payloads). Agents code against these
models; do not change a field without updating docs/event_contract.md and telling
the integrator.
"""
from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

SCHEMA_VERSION = "0.1"

EditCategory = Literal["atom", "bond", "bioisostere", "functional_group", "crossover", "reaction"]
ChosenBy = Literal["policy", "random", "crossover", "reaction", "user"]
EventType = Literal["run_start", "generation", "candidate", "policy_metrics", "run_end", "error"]


# ---------------------------------------------------------------- descriptors
class Descriptors(BaseModel):
    qed: float
    sa: Optional[float] = None
    logp: float
    tpsa: float
    mw: float
    hbd: int
    hba: int
    rotb: int
    fsp3: float
    heavy_atoms: int
    lipinski_violations: int
    alerts: int


class AlertHit(BaseModel):
    name: str
    family: str  # PAINS | BRENK | NIH | ...
    atom_indices: List[int]


class ScoreResult(BaseModel):
    descriptors: Descriptors
    components: Dict[str, float]  # per-descriptor desirability in [0, 1]
    composite: float  # weighted aggregate in [0, 1]
    alerts: List[AlertHit] = Field(default_factory=list)


# ------------------------------------------------------------------ objectives
class ObjectiveSpec(BaseModel):
    """One user-controllable objective. Desirability is 1 inside [low, high]."""
    weight: float = 1.0
    low: Optional[float] = None
    high: Optional[float] = None
    direction: Literal["maximize", "minimize", "range"] = "range"


class RunConfig(BaseModel):
    seed_smiles: str
    objectives: Dict[str, ObjectiveSpec] = Field(default_factory=dict)  # keys = Descriptors fields
    population_size: int = 20
    generations: int = 10
    rng_seed: int = 0
    policy_guidance: float = 0.5  # fraction of mutations chosen by the DQN policy (0 = pure random)
    use_crossover: bool = True
    use_reactions: bool = False  # SynGA-style reaction operator (stretch)
    locked_atoms: List[int] = Field(default_factory=list)  # seed atom indices that must not change
    preset_id: Optional[str] = None


# ------------------------------------------------------------------- explaining
class AtomContributions(BaseModel):
    """Per-atom decompositions of the CHILD molecule, index-aligned with its RDKit atoms."""
    crippen_logp: List[float]
    tpsa: List[float]
    gasteiger: List[float]


class Explanation(BaseModel):
    descriptor_deltas: Dict[str, float]  # child - parent
    alerts_added: List[AlertHit] = Field(default_factory=list)
    alerts_removed: List[AlertHit] = Field(default_factory=list)  # indices refer to the PARENT
    atom_contributions: Optional[AtomContributions] = None
    rationale: str
    tags: List[str] = Field(default_factory=list)  # e.g. ["lowers_logp", "removes_alert"]


class EditRecord(BaseModel):
    edit_id: str  # registry id, or "brics_crossover" / "reaction:<id>"
    name: str
    category: EditCategory
    site: List[int] = Field(default_factory=list)  # parent atom indices the edit touched
    chosen_by: ChosenBy
    parents: List[str] = Field(default_factory=list)  # parent SMILES (2 for crossover)


# ---------------------------------------------------------------- event payloads
class Candidate(BaseModel):
    id: str
    generation: int
    smiles: str
    parent_smiles: Optional[str] = None
    molblock: str  # 3D mol block of the child, aligned to the parent when possible
    parent_molblock: Optional[str] = None
    atom_map: List[Tuple[int, int]] = Field(default_factory=list)  # (parent_idx, child_idx)
    edit: Optional[EditRecord] = None
    score: ScoreResult
    explanation: Optional[Explanation] = None
    accepted: bool = False  # survived selection into the next generation


class PopulationMember(BaseModel):
    smiles: str
    composite: float
    candidate_id: Optional[str] = None


class GenerationPayload(BaseModel):
    generation: int
    best: PopulationMember
    mean_composite: float
    population: List[PopulationMember]
    map_xy: Optional[List[Tuple[float, float]]] = None  # 2D chemical-space map, index-aligned with population
    map_method: Optional[Literal["morgan_pca", "chemberta_pca"]] = None


class PolicyMetrics(BaseModel):
    generation: int
    policy_choice_rate: float
    mean_reward_policy_edits: Optional[float] = None
    mean_reward_random_edits: Optional[float] = None
    loss: Optional[float] = None
    note: Optional[str] = None


class StoryCard(BaseModel):
    seed_smiles: str
    final_smiles: str
    seed_score: ScoreResult
    final_score: ScoreResult
    edit_path: List[str]  # candidate ids from seed to final
    headline: str  # e.g. "Removed 2 alerts, QED 0.41 -> 0.72 in 7 edits"
    descriptor_table: Dict[str, Tuple[float, float]]  # name -> (seed, final)


class RunStartPayload(BaseModel):
    config: RunConfig
    seed_molblock: str
    seed_score: ScoreResult
    source: Literal["live", "replay", "mock"] = "live"
    schema_version: str = SCHEMA_VERSION


class ErrorPayload(BaseModel):
    message: str
    recoverable: bool = False


PAYLOAD_MODELS = {
    "run_start": RunStartPayload,
    "generation": GenerationPayload,
    "candidate": Candidate,
    "policy_metrics": PolicyMetrics,
    "run_end": StoryCard,
    "error": ErrorPayload,
}


class Event(BaseModel):
    type: EventType
    run_id: str
    seq: int
    payload: dict

    def typed_payload(self) -> BaseModel:
        return PAYLOAD_MODELS[self.type].model_validate(self.payload)


# ------------------------------------------------------------------- HTTP bodies
class ValidateRequest(BaseModel):
    smiles: Optional[str] = None
    molfile: Optional[str] = None  # MOL / SDF text


class ValidateResponse(BaseModel):
    ok: bool
    canonical_smiles: Optional[str] = None
    heavy_atoms: Optional[int] = None
    molblock: Optional[str] = None
    score: Optional[ScoreResult] = None
    error: Optional[str] = None


class RunCreated(BaseModel):
    run_id: str
