from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, EmailStr, Field, field_validator

from reward.multi_objective import ADMET_PROPERTIES


class SignupRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=72)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    token: str
    token_type: str = "bearer"
    id: Optional[int] = None
    email: Optional[str] = None


class UserOut(BaseModel):
    id: int
    email: str
    created_at: datetime

    model_config = {"from_attributes": True}


class AdmetPropertyOut(BaseModel):
    name: str
    default_direction: str  # "maximize" | "minimize"


class ExampleTargetOut(BaseModel):
    key: str
    name: str
    sequence: str


class MoleculeCreate(BaseModel):
    smiles: str
    label: Optional[str] = None


class MoleculePreviewRequest(BaseModel):
    smiles: str


class MoleculePreviewOut(BaseModel):
    valid: bool
    canonical_smiles: Optional[str] = None
    image_b64: Optional[str] = None
    error: Optional[str] = None


class MoleculeOut(BaseModel):
    id: int
    smiles: str
    canonical_smiles: str
    label: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class ProteinCreate(BaseModel):
    name: str
    sequence: str = Field(min_length=1)


class ProteinOut(BaseModel):
    id: int
    name: str
    sequence: str
    created_at: datetime

    model_config = {"from_attributes": True}


def _direction_str_to_int(value: str) -> int:
    if value not in ("maximize", "minimize"):
        raise ValueError(f"direction must be 'maximize' or 'minimize', got {value!r}")
    return 1 if value == "maximize" else -1


class OptimizationConfigCreate(BaseModel):
    name: str
    starting_molecule_id: int
    target_protein_id: int
    off_target_protein_id: Optional[int] = None

    admet_weight: float = 0.3
    binding_weight: float = 0.5
    synthetic_weight: float = 0.2
    selectivity_weight: float = 0.3

    admet_properties: List[str] = Field(default_factory=list)
    admet_directions: Dict[str, str] = Field(default_factory=dict)  # property -> "maximize"|"minimize"

    @field_validator("admet_properties")
    @classmethod
    def _validate_properties(cls, v):
        unknown = set(v) - set(ADMET_PROPERTIES)
        if unknown:
            raise ValueError(f"Unknown ADMET properties: {sorted(unknown)}")
        return v

    @field_validator("admet_directions")
    @classmethod
    def _validate_directions(cls, v):
        for direction in v.values():
            _direction_str_to_int(direction)
        return v

    def weights_valid(self) -> bool:
        weights = [self.admet_weight, self.binding_weight, self.synthetic_weight, self.selectivity_weight]
        return all(w >= 0 for w in weights) and any(w > 0 for w in weights)

    def admet_directions_as_int(self) -> Dict[str, int]:
        return {prop: _direction_str_to_int(d) for prop, d in self.admet_directions.items()}


class OptimizationConfigOut(BaseModel):
    id: int
    name: str
    starting_molecule_id: int
    target_protein_id: int
    off_target_protein_id: Optional[int] = None
    admet_weight: float
    binding_weight: float
    synthetic_weight: float
    selectivity_weight: float
    admet_properties: List[str]
    admet_directions: Dict[str, int]
    created_at: datetime

    model_config = {"from_attributes": True}


# ---- training runs ----

class TrainingRunCreate(BaseModel):
    config_id: int
    seed: int = 0
    num_episodes: int = 150
    max_steps: int = 40
    checkpoint_interval: int = 25


class TrainingRunOut(BaseModel):
    id: int
    config_id: int
    is_finetune: bool
    parent_run_id: Optional[int] = None
    finetune_alpha: Optional[float] = None
    seed: int
    num_episodes: Optional[int] = None
    max_steps: Optional[int] = None
    status: str
    progress_current: int
    progress_total: int
    checkpoint_dir: Optional[str] = None
    final_checkpoint_path: Optional[str] = None
    result_summary_json: Optional[dict] = None
    error_message: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class GenerationRequest(BaseModel):
    num_candidates: int = Field(default=10, ge=1, le=100)
    sampling_strategy: str = Field(default="boltzmann")  # "boltzmann" | "greedy"
    temperature: float = 1.0
    max_steps: Optional[int] = None

    @field_validator("sampling_strategy")
    @classmethod
    def _validate_strategy(cls, v):
        if v not in ("boltzmann", "greedy"):
            raise ValueError("sampling_strategy must be 'boltzmann' or 'greedy'")
        return v


class GenerationBatchOut(BaseModel):
    id: int
    training_run_id: int
    checkpoint_path_snapshot: str
    num_requested: int
    sampling_strategy: str
    temperature: Optional[float] = None
    status: str
    error_message: Optional[str] = None
    created_at: datetime
    finished_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class FineTuneUsageOut(BaseModel):
    training_run_id: int
    status: str
    alpha: float
    created_at: datetime


class TrajectoryStepOut(BaseModel):
    step_index: int
    smiles: str
    image_b64: Optional[str] = None
    molblock_3d: Optional[str] = None
    reward: float
    admet_score: Optional[float] = None
    binding_uM: Optional[float] = None
    sa_score: Optional[float] = None
    selectivity: Optional[float] = None


class TrajectoryOut(BaseModel):
    trajectory_index: int
    steps: List[TrajectoryStepOut]


class PromoteStepSelection(BaseModel):
    trajectory_index: int
    step_index: int


class PromoteStepsRequest(BaseModel):
    selections: List[PromoteStepSelection] = Field(min_length=1)


class CandidateOut(BaseModel):
    id: int
    generation_batch_id: int
    training_run_id: int
    smiles: str
    image_b64: Optional[str] = None
    molblock_3d: Optional[str] = None
    reward: float
    admet_score: Optional[float] = None
    binding_uM: Optional[float] = None
    sa_score: Optional[float] = None
    selectivity: Optional[float] = None
    user_rating: Optional[float] = None
    rating_notes: Optional[str] = None
    created_at: datetime

    config_id: Optional[int] = None
    config_name: str
    starting_molecule_id: Optional[int] = None
    starting_smiles: str
    target_protein_id: Optional[int] = None
    target_protein_name: str
    off_target_protein_id: Optional[int] = None
    off_target_protein_name: Optional[str] = None
    used_in_finetune_runs: List[FineTuneUsageOut] = Field(default_factory=list)

    model_config = {"from_attributes": True}


class CandidateScoreRequest(BaseModel):
    rating: float = Field(ge=0, le=100)
    notes: Optional[str] = None


class FineTuneRequest(BaseModel):
    candidate_ids: List[int] = Field(min_length=1)
    alpha: float = Field(default=0.3, ge=0, le=1)
    num_extra_episodes: int = Field(default=50, ge=1)
