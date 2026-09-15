from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    func,
)
from sqlalchemy.orm import relationship

from webapp.backend.db import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    created_at = Column(DateTime, server_default=func.now())


class StartingMolecule(Base):
    __tablename__ = "starting_molecules"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    smiles = Column(String, nullable=False)
    canonical_smiles = Column(String, nullable=False, index=True)
    label = Column(String, nullable=True)
    created_at = Column(DateTime, server_default=func.now())


class Protein(Base):
    __tablename__ = "proteins"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    name = Column(String, nullable=False)
    sequence = Column(Text, nullable=False)
    created_at = Column(DateTime, server_default=func.now())


class OptimizationConfig(Base):
    __tablename__ = "optimization_configs"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    name = Column(String, nullable=False)

    starting_molecule_id = Column(Integer, ForeignKey("starting_molecules.id"), nullable=False)
    target_protein_id = Column(Integer, ForeignKey("proteins.id"), nullable=False)
    off_target_protein_id = Column(Integer, ForeignKey("proteins.id"), nullable=True)

    admet_weight = Column(Float, nullable=False, default=0.3)
    binding_weight = Column(Float, nullable=False, default=0.5)
    synthetic_weight = Column(Float, nullable=False, default=0.2)
    selectivity_weight = Column(Float, nullable=False, default=0.3)

    admet_properties = Column(JSON, nullable=False)  # list[str]
    admet_directions = Column(JSON, nullable=False)  # dict[str, int] (1 or -1)

    created_at = Column(DateTime, server_default=func.now())

    starting_molecule = relationship("StartingMolecule")
    target_protein = relationship("Protein", foreign_keys=[target_protein_id])
    off_target_protein = relationship("Protein", foreign_keys=[off_target_protein_id])


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    config_id = Column(Integer, ForeignKey("optimization_configs.id"), nullable=False)

    is_finetune = Column(Boolean, nullable=False, default=False)
    parent_run_id = Column(Integer, ForeignKey("training_runs.id"), nullable=True)
    finetune_alpha = Column(Float, nullable=True)

    seed = Column(Integer, nullable=False, default=0)
    num_episodes = Column(Integer, nullable=True)
    max_steps = Column(Integer, nullable=True)
    checkpoint_interval = Column(Integer, nullable=True)

    status = Column(String, nullable=False, default="pending")  # pending|running|completed|failed|cancelled
    progress_current = Column(Integer, nullable=False, default=0)
    progress_total = Column(Integer, nullable=False, default=0)

    checkpoint_dir = Column(String, nullable=True)
    final_checkpoint_path = Column(String, nullable=True)
    result_summary_json = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)

    created_at = Column(DateTime, server_default=func.now())
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)

    config = relationship("OptimizationConfig")


class GenerationBatch(Base):
    __tablename__ = "generation_batches"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    training_run_id = Column(Integer, ForeignKey("training_runs.id"), nullable=False)

    checkpoint_path_snapshot = Column(String, nullable=False)
    num_requested = Column(Integer, nullable=False)
    sampling_strategy = Column(String, nullable=False, default="boltzmann")
    temperature = Column(Float, nullable=True)

    status = Column(String, nullable=False, default="pending")
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    finished_at = Column(DateTime, nullable=True)

    trajectories = Column(JSON, nullable=True)


class GeneratedCandidate(Base):
    __tablename__ = "generated_candidates"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    generation_batch_id = Column(Integer, ForeignKey("generation_batches.id"), nullable=False, index=True)
    training_run_id = Column(Integer, ForeignKey("training_runs.id"), nullable=False, index=True)

    smiles = Column(String, nullable=False)
    reward = Column(Float, nullable=False)
    admet_score = Column(Float, nullable=True)
    binding_uM = Column(Float, nullable=True)
    sa_score = Column(Float, nullable=True)
    selectivity = Column(Float, nullable=True)

    user_rating = Column(Float, nullable=True)
    rating_notes = Column(Text, nullable=True)

    config_id = Column(Integer, ForeignKey("optimization_configs.id"), nullable=True)
    config_name = Column(String, nullable=False)
    starting_molecule_id = Column(Integer, ForeignKey("starting_molecules.id"), nullable=True)
    starting_smiles = Column(String, nullable=False)
    target_protein_id = Column(Integer, ForeignKey("proteins.id"), nullable=True)
    target_protein_name = Column(String, nullable=False)
    off_target_protein_id = Column(Integer, ForeignKey("proteins.id"), nullable=True)
    off_target_protein_name = Column(String, nullable=True)

    created_at = Column(DateTime, server_default=func.now())


class FineTuneFeedback(Base):
    __tablename__ = "finetune_feedback"

    id = Column(Integer, primary_key=True)
    training_run_id = Column(Integer, ForeignKey("training_runs.id"), nullable=False, index=True)
    candidate_id = Column(Integer, ForeignKey("generated_candidates.id"), nullable=False)
    user_rating_snapshot = Column(Float, nullable=False)
