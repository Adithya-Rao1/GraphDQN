import shutil
import threading
from datetime import datetime, timezone

from molecular_modifications.macro_actions import build_edit_catalog
from ppo.llm_finetune import EditOutcome
from ppo.pgmorl_population import SweepCancelled
from webapp.backend.config import LLM_ADAPTER_CHECKPOINT_ROOT, PARETO_SWEEP_CHECKPOINT_ROOT
from webapp.backend.db import SessionLocal
from webapp.backend.jobs.manager import job_manager
from webapp.backend.jobs.training import _reward_config_from
from webapp.backend.models import (
    EditOutcomeLog,
    LLMAdapterCheckpoint,
    OptimizationConfig,
    ParetoSweepRun,
    PGMORLPerformanceRecord,
)
from webapp.backend.rl_bridge.pgmorl_runner import run_pareto_sweep


def _finish_cancelled(db, sweep_id: int, error: Exception) -> None:
    sweep = db.get(ParetoSweepRun, sweep_id)
    if job_manager.should_discard_pareto_sweep(sweep_id):
        checkpoint_dir = None
        result = sweep.result_summary_json
        if result:
            checkpoint_dir = result.get("checkpoint_dir")
        db.query(EditOutcomeLog).filter(EditOutcomeLog.pareto_sweep_id == sweep_id).delete()
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
        step_counters: dict[str, int] = {}  # member_id -> running EditOutcomeLog step counter
        step_lock = threading.Lock()  # on_step_scored can fire from multiple concurrent-sweep threads
        edit_id_to_index = {op.id: i for i, op in enumerate(build_edit_catalog())}

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

        def on_step_scored(member_id: str, entry) -> None:
            member_index = int(member_id.rsplit("_", 1)[-1])
            with step_lock:
                step_idx = step_counters.get(member_id, 0)
                step_counters[member_id] = step_idx + 1
                db.add(EditOutcomeLog(
                    user_id=sweep.user_id,
                    pareto_sweep_id=sweep.id,
                    population_member_config_id=member_config_ids.get(member_index),
                    step_index=step_idx,
                    parent_smiles=entry.initial_state,
                    applied_edit_ids=list(entry.edit_ids),
                    pre_edit_states=list(entry.pre_edit_states),
                    k_edits_used=entry.k_used,
                    edit_count_mode=sweep.edit_count_mode,
                    resulting_smiles=entry.final_smiles,
                    reward=entry.reward,
                    reward_vector=list(entry.reward_vector),
                ))
                db.commit()

        def progress_cb(current_round: int, total_rounds: int, members_snapshot: list,
                         new_records: list, round_steps: int):
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

            finetune_batch = None
            if sweep.use_llm and sweep.llm_finetune_interval_steps:
                sweep.cumulative_steps_since_finetune += round_steps
                if sweep.cumulative_steps_since_finetune >= sweep.llm_finetune_interval_steps:
                    rows = (
                        db.query(EditOutcomeLog)
                        .filter(EditOutcomeLog.user_id == sweep.user_id,
                                EditOutcomeLog.consumed_by_finetune.is_(False))
                        .all()
                    )
                    if rows:
                        finetune_batch = [
                            (row.id, EditOutcome(
                                pre_edit_states=list(row.pre_edit_states),
                                edit_indices=[
                                    edit_id_to_index[eid] for eid in row.applied_edit_ids
                                    if eid in edit_id_to_index
                                ],
                                reward=row.reward,
                            ))
                            for row in rows
                        ]
                    sweep.cumulative_steps_since_finetune = 0

            db.commit()
            return finetune_batch

        def on_finetune_result(result, row_ids: list, adapter_path) -> None:
            if adapter_path:
                db.add(LLMAdapterCheckpoint(
                    user_id=sweep.user_id,
                    base_model_name=sweep.llm_model_name,
                    adapter_path=adapter_path,
                    trained_on_pareto_sweep_ids=[sweep.id],
                    num_training_examples=result.num_examples,
                ))
            if row_ids:
                db.query(EditOutcomeLog).filter(EditOutcomeLog.id.in_(row_ids)).update(
                    {"consumed_by_finetune": True}, synchronize_session=False,
                )
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
            on_step_scored=on_step_scored,
            on_finetune_result=on_finetune_result,
            adapter_checkpoint_root=LLM_ADAPTER_CHECKPOINT_ROOT,
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
