import threading
from datetime import datetime, timezone

from webapp.backend.candidate_provenance import provenance_kwargs
from webapp.backend.db import SessionLocal
from webapp.backend.jobs.training import _reward_config_from
from webapp.backend.models import GeneratedCandidate, GenerationBatch, TrainingRun
from webapp.backend.rl_bridge.candidate_generator import generate_dqn_candidates


def run_generation_job(batch_id: int, cancel_event: threading.Event) -> None:
    db = SessionLocal()
    try:
        batch = db.get(GenerationBatch, batch_id)
        batch.status = "running"
        db.commit()

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
        batch.trajectories = trajectories

        for result in results:
            db.add(GeneratedCandidate(
                user_id=batch.user_id,
                generation_batch_id=batch.id,
                training_run_id=run.id,
                smiles=result["smiles"],
                reward=result["reward"],
                admet_score=result["admet_score"],
                binding_uM=result["binding_uM"],
                sa_score=result["sa_score"],
                selectivity=result["selectivity"],
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
