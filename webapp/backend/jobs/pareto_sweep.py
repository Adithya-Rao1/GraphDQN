import shutil
import threading
from datetime import datetime, timezone

from webapp.backend.config import PARETO_SWEEP_CHECKPOINT_ROOT
from webapp.backend.db import SessionLocal
from webapp.backend.jobs.manager import job_manager
from webapp.backend.jobs.training import _reward_config_from
from webapp.backend.models import OptimizationConfig, ParetoSweepRun, PGMORLPerformanceRecord
from webapp.backend.rl_bridge.pgmorl_runner import run_pareto_sweep
from ppo.pgmorl_population import SweepCancelled


def _finish_cancelled(db, sweep_id: int, error: Exception) -> None:
    sweep = db.get(ParetoSweepRun, sweep_id)
    if job_manager.should_discard_pareto_sweep(sweep_id):
        checkpoint_dir = None
        result = sweep.result_summary_json
        if result:
            checkpoint_dir = result.get("checkpoint_dir")
        db.query(PGMORLPerformanceRecord).filter(PGMORLPerformanceRecord.pareto_sweep_id == sweep_id).delete()
        db.query(OptimizationConfig).filter(
            OptimizationConfig.generated_by_pareto_sweep_id == sweep_id
        ).delete()
        db.delete(sweep)
        db.commit()
        if checkpoint_dir:
            shutil.rmtree(checkpoint_dir, ignore_errors=True)
    else:
        sweep.status = "cancelled"
        sweep.error_message = str(error)
        sweep.finished_at = datetime.now(timezone.utc)
        db.commit()


def run_pareto_sweep_job(sweep_id: int, cancel_event: threading.Event) -> None:
    db = SessionLocal()
    try:
        sweep = db.get(ParetoSweepRun, sweep_id)
        sweep.status = "running"
        sweep.started_at = datetime.now(timezone.utc)
        sweep.progress_total_rounds = sweep.num_rounds
        db.commit()

        base_config = sweep.base_config
        reward_config = _reward_config_from(base_config)

        member_config_ids: dict[int, int] = {}  # member_index -> OptimizationConfig.id

        def on_member_created(member_index: int, weight_vector: list) -> None:
            member_config = OptimizationConfig(
                user_id=sweep.user_id,
                name=f"{base_config.name} (sweep #{sweep.id}, member {member_index})",
                starting_molecule_id=base_config.starting_molecule_id,
                target_protein_id=base_config.target_protein_id,
                off_target_protein_id=base_config.off_target_protein_id,
                admet_weight=weight_vector[0],
                binding_weight=weight_vector[1],
                synthetic_weight=weight_vector[2],
                selectivity_weight=weight_vector[3],
                admet_properties=list(base_config.admet_properties),
                admet_directions=dict(base_config.admet_directions),
                generated_by_pareto_sweep_id=sweep.id,
            )
            db.add(member_config)
            db.commit()
            db.refresh(member_config)
            member_config_ids[member_index] = member_config.id

        def progress_cb(current_round: int, total_rounds: int, members_snapshot: list, new_records: list) -> None:
            # members_snapshot entries are keyed by member_id == f"member_{i}"
            # (see rl_bridge/pgmorl_runner.py's own docstring on this coupling).
            for entry in members_snapshot:
                member_index = int(entry["member_id"].rsplit("_", 1)[-1])
                entry["config_id"] = member_config_ids.get(member_index)
            sweep.progress_current_round = current_round
            sweep.progress_total_rounds = total_rounds
            sweep.members_snapshot_json = members_snapshot

            for record in new_records:
                member_index = int(record["member_id"].rsplit("_", 1)[-1])
                db.add(PGMORLPerformanceRecord(
                    pareto_sweep_id=sweep.id,
                    population_member_config_id=member_config_ids[member_index],
                    objective_vector_before=record["objective_vector_before"],
                    weight_vector_used=record["weight_vector_used"],
                    training_steps_this_round=record["training_steps_this_round"],
                    objective_vector_after=record["objective_vector_after"],
                ))

            db.commit()

        summary = run_pareto_sweep(
            reward_config=reward_config,
            target_seq=base_config.target_protein.sequence,
            off_target_seq=base_config.off_target_protein.sequence if base_config.off_target_protein else None,
            init_mol=base_config.starting_molecule.canonical_smiles,
            seed=sweep.seed,
            population_size=sweep.population_size,
            concentration_alpha=sweep.concentration_alpha,
            num_rounds=sweep.num_rounds,
            episodes_per_round=sweep.episodes_per_round,
            eval_episodes_per_round=sweep.eval_episodes_per_round,
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
            use_predictor=sweep.use_predictor,
            concurrent=sweep.concurrent,
            max_concurrent_members=sweep.max_concurrent_members,
            run_id=f"sweep{sweep.id}",
            checkpoint_root=PARETO_SWEEP_CHECKPOINT_ROOT,
            on_member_created=on_member_created,
            progress_cb=progress_cb,
            cancel_event=cancel_event,
        )

        for member_out in summary["members"]:
            member_index = int(member_out["member_id"].rsplit("_", 1)[-1])
            member_out["config_id"] = member_config_ids.get(member_index)

        sweep.status = "completed"
        sweep.result_summary_json = summary
        sweep.members_snapshot_json = summary["members"]
        sweep.finished_at = datetime.now(timezone.utc)
        db.commit()
    except SweepCancelled as e:
        _finish_cancelled(db, sweep_id, e)
    except Exception as e:  # noqa: BLE001 -- job boundary, must not raise into the thread pool silently
        sweep = db.get(ParetoSweepRun, sweep_id)
        sweep.status = "failed"
        sweep.error_message = str(e)
        sweep.finished_at = datetime.now(timezone.utc)
        db.commit()
    finally:
        db.close()
