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

    admet_properties = Column(JSON, nullable=False)  
    admet_directions = Column(JSON, nullable=False) 

    generated_by_pareto_sweep_id = Column(Integer, ForeignKey("pareto_sweep_runs.id"), nullable=True)

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


class EditOutcomeLog(Base):
    __tablename__ = "edit_outcome_logs"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    training_run_id = Column(Integer, ForeignKey("training_runs.id"), nullable=False, index=True)

    step_index = Column(Integer, nullable=False)
    parent_smiles = Column(String, nullable=False)
    applied_edit_ids = Column(JSON, nullable=False)  
    pre_edit_states = Column(JSON, nullable=False) 
    k_edits_used = Column(Integer, nullable=False)
    edit_count_mode = Column(String, nullable=False)  # "fixed" | "random" | "learned"
    resulting_smiles = Column(String, nullable=False)

    admet_score = Column(Float, nullable=True)
    binding_uM = Column(Float, nullable=True)
    sa_score = Column(Float, nullable=True)
    selectivity = Column(Float, nullable=True)
    reward = Column(Float, nullable=False)
    reward_vector = Column(JSON, nullable=False)  # [admet, binding, sa, selectivity]

    consumed_by_finetune = Column(Boolean, nullable=False, default=False)

    created_at = Column(DateTime, server_default=func.now())


class ParetoSweepRun(Base):
    __tablename__ = "pareto_sweep_runs"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    base_config_id = Column(Integer, ForeignKey("optimization_configs.id"), nullable=False)

    seed = Column(Integer, nullable=False, default=0)
    population_size = Column(Integer, nullable=False)
    concentration_alpha = Column(Float, nullable=False)
    num_rounds = Column(Integer, nullable=False)
    episodes_per_round = Column(Integer, nullable=False)
    eval_episodes_per_round = Column(Integer, nullable=False)
    max_steps = Column(Integer, nullable=False)

    edit_count_mode = Column(String, nullable=False, default="fixed")  # "fixed" | "random" | "learned"
    fixed_edit_count = Column(Integer, nullable=True)
    edit_count_range_min = Column(Integer, nullable=True)
    edit_count_range_max = Column(Integer, nullable=True)
    k_max = Column(Integer, nullable=True)

    use_llm = Column(Boolean, nullable=False, default=False)
    llm_model_name = Column(String, nullable=True)
    use_predictor = Column(Boolean, nullable=False, default=True)  # False = round-robin baseline
    concurrent = Column(Boolean, nullable=False, default=False)
    max_concurrent_members = Column(Integer, nullable=True)

    status = Column(String, nullable=False, default="pending")  # pending|running|completed|failed|cancelled
    progress_current_round = Column(Integer, nullable=False, default=0)
    progress_total_rounds = Column(Integer, nullable=False, default=0)
    members_snapshot_json = Column(JSON, nullable=True)

    result_summary_json = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)

    created_at = Column(DateTime, server_default=func.now())
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)

    base_config = relationship("OptimizationConfig", foreign_keys=[base_config_id])


class PGMORLPerformanceRecord(Base):
    __tablename__ = "pgmorl_performance_records"

    id = Column(Integer, primary_key=True)
    pareto_sweep_id = Column(Integer, ForeignKey("pareto_sweep_runs.id"), nullable=False, index=True)
    population_member_config_id = Column(Integer, ForeignKey("optimization_configs.id"), nullable=False)

    objective_vector_before = Column(JSON, nullable=False)  # [admet, binding, sa, selectivity]
    weight_vector_used = Column(JSON, nullable=False)
    training_steps_this_round = Column(Integer, nullable=False)
    objective_vector_after = Column(JSON, nullable=False)

    created_at = Column(DateTime, server_default=func.now())


class LLMAdapterCheckpoint(Base):
    __tablename__ = "llm_adapter_checkpoints"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    base_model_name = Column(String, nullable=False)
    adapter_path = Column(String, nullable=False)
    trained_on_pareto_sweep_ids = Column(JSON, nullable=False)  
    num_training_examples = Column(Integer, nullable=False)

    created_at = Column(DateTime, server_default=func.now())
