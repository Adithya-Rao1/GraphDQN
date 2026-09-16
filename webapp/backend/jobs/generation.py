import threading
from datetime import datetime, timezone

from webapp.backend.candidate_provenance import provenance_kwargs
from webapp.backend.db import SessionLocal
from webapp.backend.jobs.training import _reward_config_from
from webapp.backend.models import GeneratedCandidate, GenerationBatch, OptimizationConfig, ParetoSweepRun, TrainingRun
from webapp.backend.rl_bridge.candidate_generator import generate_dqn_candidates
from webapp.backend.rl_bridge.pgmorl_candidate_generator import generate_pgmorl_candidates


def _run_dqn_generation(db, batch: GenerationBatch, cancel_event: threading.Event):
    run = db.get(TrainingRun, batch.training_run_id)
    config = run.config
    reward_config = _reward_config_from(config)

    results, trajectories = generate_dqn_candidates(
        checkpoint_path=batch.checkpoint_path_snapshot,
        start_smiles=config.starting_molecule.canonical_smiles,
        target_seq=config.target_protein.sequence,
        off_target_seq=config.off_target_protein.sequence if config.off_target_protein else None,
        reward_config=reward_config,
        num_candidates=batch.num_requested,
        max_steps=run.max_steps,
        sampling=batch.sampling_strategy,
        temperature=batch.temperature or 1.0,
        cancel_event=cancel_event,
    )
    return config, results, trajectories, dict(training_run_id=run.id)


def _run_pgmorl_generation(db, batch: GenerationBatch, cancel_event: threading.Event):
    sweep = db.get(ParetoSweepRun, batch.pareto_sweep_id)
    config = db.get(OptimizationConfig, batch.population_member_config_id)
    reward_config = _reward_config_from(config)

    results, trajectories = generate_pgmorl_candidates(
        checkpoint_path=batch.checkpoint_path_snapshot,
        reward_config=reward_config,
        target_seq=config.target_protein.sequence,
        off_target_seq=config.off_target_protein.sequence if config.off_target_protein else None,
        init_mol=config.starting_molecule.canonical_smiles,
        max_steps=sweep.max_steps,
        edit_count_mode=sweep.edit_count_mode,
        fixed_edit_count=sweep.fixed_edit_count,
        edit_count_range=(
            (sweep.edit_count_range_min, sweep.edit_count_range_max)
            if sweep.edit_count_mode == "random" else None
        ),
        k_max=sweep.k_max,
        use_llm=sweep.use_llm,
        llm_model_name=sweep.llm_model_name,
        num_candidates=batch.num_requested,
        sampling=batch.sampling_strategy,
        temperature=batch.temperature or 1.0,
        cancel_event=cancel_event,
    )
    return config, results, trajectories, dict(training_run_id=None)


def run_generation_job(batch_id: int, cancel_event: threading.Event) -> None:
    db = SessionLocal()
    try:
        batch = db.get(GenerationBatch, batch_id)
        batch.status = "running"
        db.commit()

        if batch.training_run_id is not None:
            config, results, trajectories, candidate_kwargs = _run_dqn_generation(db, batch, cancel_event)
        else:
            config, results, trajectories, candidate_kwargs = _run_pgmorl_generation(db, batch, cancel_event)
        batch.trajectories = trajectories

        for result in results:
            db.add(GeneratedCandidate(
                user_id=batch.user_id,
                generation_batch_id=batch.id,
                smiles=result["smiles"],
                reward=result["reward"],
                admet_score=result["admet_score"],
                binding_uM=result["binding_uM"],
                sa_score=result["sa_score"],
                selectivity=result["selectivity"],
                **candidate_kwargs,
                **provenance_kwargs(config),
            ))

        batch.status = "completed"
        batch.finished_at = datetime.now(timezone.utc)
        db.commit()
    except Exception as e:  # noqa: BLE001 -- job boundary
        batch = db.get(GenerationBatch, batch_id)
        batch.status = "failed"
        batch.error_message = str(e)
        batch.finished_at = datetime.now(timezone.utc)
        db.commit()
    finally:
        db.close()
