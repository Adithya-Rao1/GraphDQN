import os
import shutil
import threading
import time
from datetime import datetime, timezone

from webapp.backend.config import CHECKPOINT_ROOT
from webapp.backend.db import SessionLocal
from webapp.backend.jobs.manager import job_manager
from webapp.backend.models import FineTuneFeedback, GeneratedCandidate, OptimizationConfig, TrainingRun
from webapp.backend.rl_bridge.dqn_runner import TrainingCancelled, train_dqn
from webapp.backend.rl_bridge.finetune_dqn import finetune_dqn
from reward.multi_objective import RewardConfig


def _reward_config_from(config: OptimizationConfig) -> RewardConfig:
    return RewardConfig(
        admet_weight=config.admet_weight,
        binding_weight=config.binding_weight,
        synthetic_weight=config.synthetic_weight,
        selectivity_weight=config.selectivity_weight,
        admet_properties=list(config.admet_properties),
        admet_directions=dict(config.admet_directions),
    )


def _finish_cancelled(db, run_id: int, error: Exception) -> None:
    run = db.get(TrainingRun, run_id)
    if job_manager.should_discard(run_id):
        checkpoint_dir = run.checkpoint_dir
        db.query(FineTuneFeedback).filter(FineTuneFeedback.training_run_id == run_id).delete()
        db.delete(run)
        db.commit()
        if checkpoint_dir:
            shutil.rmtree(checkpoint_dir, ignore_errors=True)
    else:
        run.status = "cancelled"
        run.error_message = str(error)
        run.finished_at = datetime.now(timezone.utc)
        db.commit()


def run_training_job(run_id: int, cancel_event: threading.Event) -> None:
    db = SessionLocal()
    try:
        run = db.get(TrainingRun, run_id)
        run.status = "running"
        run.started_at = datetime.now(timezone.utc)
        run.progress_total = run.num_episodes
        db.commit()

        config = run.config
        reward_config = _reward_config_from(config)

        run_id_str = f"run{run.id}_{int(time.time())}"
        checkpoint_dir = os.path.join(CHECKPOINT_ROOT, run_id_str)
        run.checkpoint_dir = checkpoint_dir
        db.commit()

        def progress_cb(current, total, _reward):
            run.progress_current = current
            run.progress_total = total
            db.commit()

        summary = train_dqn(
            reward_config=reward_config,
            target_seq=config.target_protein.sequence,
            off_target_seq=config.off_target_protein.sequence if config.off_target_protein else None,
            init_mol=config.starting_molecule.canonical_smiles,
            seed=run.seed,
            num_episodes=run.num_episodes,
            max_steps=run.max_steps,
            checkpoint_interval=run.checkpoint_interval,
            run_id=run_id_str,
            checkpoint_root=CHECKPOINT_ROOT,
            progress_cb=progress_cb,
            cancel_event=cancel_event,
        )

        run.status = "completed"
        run.checkpoint_dir = summary["checkpoint_dir"]
        run.final_checkpoint_path = summary["final_checkpoint_path"]
        run.result_summary_json = summary
        run.finished_at = datetime.now(timezone.utc)
        db.commit()
    except TrainingCancelled as e:
        _finish_cancelled(db, run_id, e)
    except Exception as e:  # noqa: BLE001 --> job boundary, must not raise into the thread pool silently
        run = db.get(TrainingRun, run_id)
        run.status = "failed"
        run.error_message = str(e)
        run.finished_at = datetime.now(timezone.utc)
        db.commit()
    finally:
        db.close()


def run_finetune_job(run_id: int, cancel_event: threading.Event) -> None:
    db = SessionLocal()
    try:
        run = db.get(TrainingRun, run_id)
        run.status = "running"
        run.started_at = datetime.now(timezone.utc)
        run.progress_total = run.num_episodes
        db.commit()

        parent_run = db.get(TrainingRun, run.parent_run_id)
        config = run.config
        reward_config = _reward_config_from(config)

        feedback_rows = (
            db.query(FineTuneFeedback, GeneratedCandidate)
            .join(GeneratedCandidate, FineTuneFeedback.candidate_id == GeneratedCandidate.id)
            .filter(FineTuneFeedback.training_run_id == run.id)
            .all()
        )
        scored_candidates = [
            {"smiles": candidate.smiles, "user_rating": feedback.user_rating_snapshot}
            for feedback, candidate in feedback_rows
        ]

        run_id_str = f"run{run.id}_finetune_{int(time.time())}"
        checkpoint_dir = os.path.join(CHECKPOINT_ROOT, run_id_str)
        run.checkpoint_dir = checkpoint_dir
        db.commit()

        def progress_cb(current, total, _reward):
            run.progress_current = current
            run.progress_total = total
            db.commit()

        summary = finetune_dqn(
            parent_checkpoint_path=parent_run.final_checkpoint_path,
            reward_config=reward_config,
            target_seq=config.target_protein.sequence,
            off_target_seq=config.off_target_protein.sequence if config.off_target_protein else None,
            init_mol=config.starting_molecule.canonical_smiles,
            seed=run.seed,
            num_extra_episodes=run.num_episodes,
            alpha=run.finetune_alpha,
            scored_candidates=scored_candidates,
            checkpoint_interval=run.checkpoint_interval,
            run_id=run_id_str,
            checkpoint_root=CHECKPOINT_ROOT,
            max_steps=run.max_steps,
            progress_cb=progress_cb,
            cancel_event=cancel_event,
        )

        run.status = "completed"
        run.checkpoint_dir = summary["checkpoint_dir"]
        run.final_checkpoint_path = summary["final_checkpoint_path"]
        run.result_summary_json = summary
        run.finished_at = datetime.now(timezone.utc)
        db.commit()
    except TrainingCancelled as e:
        _finish_cancelled(db, run_id, e)
    except Exception as e:  # noqa: BLE001
        run = db.get(TrainingRun, run_id)
        run.status = "failed"
        run.error_message = str(e)
        run.finished_at = datetime.now(timezone.utc)
        db.commit()
    finally:
        db.close()
