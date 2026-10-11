import os
import time
from datetime import datetime, timezone

import torch

from molecular_modifications.macro_actions import build_edit_catalog
from ppo.gnn_llm_actor_critic import GNNLLMActor, build_shared_llm_backbone, default_qwen2_lora_config
from ppo.llm_finetune import EditOutcome, run_llm_finetune_cycle
from webapp.backend.config import LLM_ADAPTER_CHECKPOINT_ROOT
from webapp.backend.db import SessionLocal
from webapp.backend.models import EditOutcomeLog, LLMAdapterCheckpoint, LLMFinetuneRun


def _descriptor_fn(smiles: str) -> str:
    return f"Molecule {smiles} against target sequence (pooled fine-tune batch from manual sessions)."


def run_llm_finetune_from_sessions_job(llm_finetune_run_id: int) -> None:
    db = SessionLocal()
    try:
        run = db.get(LLMFinetuneRun, llm_finetune_run_id)
        run.status = "running"
        run.started_at = datetime.now(timezone.utc)
        db.commit()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        catalog = build_edit_catalog()
        edit_id_to_index = {op.id: i for i, op in enumerate(catalog)}

        rows = (
            db.query(EditOutcomeLog)
            .filter(
                EditOutcomeLog.manual_edit_session_id.in_(run.manual_edit_session_ids),
                EditOutcomeLog.consumed_by_finetune.is_(False),
            )
            .all()
        )
        if not rows:
            run.status = "failed"
            run.error_message = "No unconsumed edit outcomes found for the selected sessions"
            run.finished_at = datetime.now(timezone.utc)
            db.commit()
            return

        outcomes = [
            EditOutcome(
                pre_edit_states=list(row.pre_edit_states),
                edit_indices=[edit_id_to_index[eid] for eid in row.applied_edit_ids if eid in edit_id_to_index],
                reward=row.reward,
            )
            for row in rows
        ]

        tokenizer, peft_model = build_shared_llm_backbone(
            run.llm_model_name, {"actor": default_qwen2_lora_config()},
            torch_dtype=torch.bfloat16 if device.type == "cuda" else None,
        )
        for p in peft_model.parameters():
            p.requires_grad = False

        if run.warm_start:
            latest_checkpoint = (
                db.query(LLMAdapterCheckpoint)
                .filter(LLMAdapterCheckpoint.base_model_name == run.llm_model_name)
                .order_by(LLMAdapterCheckpoint.created_at.desc())
                .first()
            )
            if latest_checkpoint is not None:
                peft_model.load_adapter(latest_checkpoint.adapter_path, adapter_name="actor")

        actor = GNNLLMActor(
            num_edit_ops=len(catalog), hidden_dim=64, use_llm=True,
            shared_llm_backbone=(tokenizer, peft_model), adapter_name="actor",
            edit_count_mode="fixed",
        ).to(device)

        result = run_llm_finetune_cycle(
            actor=actor, adapter_name="actor", outcomes=outcomes,
            descriptor_fn=_descriptor_fn, device=device,
        )

        adapter_dir = os.path.join(LLM_ADAPTER_CHECKPOINT_ROOT, f"manual_finetune{run.id}_{int(time.time())}")
        os.makedirs(adapter_dir, exist_ok=True)
        actor.llm.save_pretrained(adapter_dir, selected_adapters=["actor"])

        db.add(LLMAdapterCheckpoint(
            user_id=run.user_id,
            base_model_name=run.llm_model_name,
            adapter_path=adapter_dir,
            trained_on_pareto_sweep_ids=[],
            trained_on_manual_session_ids=run.manual_edit_session_ids,
            num_training_examples=result.num_examples,
        ))
        db.query(EditOutcomeLog).filter(EditOutcomeLog.id.in_([r.id for r in rows])).update(
            {"consumed_by_finetune": True}, synchronize_session=False,
        )

        run.status = "completed"
        run.progress_current = 1
        run.result_summary_json = {
            "num_examples": result.num_examples,
            "rl_loss": result.rl_loss,
            "kl_loss": result.kl_loss,
            "total_loss": result.total_loss,
            "num_updates": result.num_updates,
            "adapter_path": adapter_dir,
        }
        run.finished_at = datetime.now(timezone.utc)
        db.commit()
    except Exception as e:  # noqa: BLE001 -- job boundary
        run = db.get(LLMFinetuneRun, llm_finetune_run_id)
        run.status = "failed"
        run.error_message = str(e)
        run.finished_at = datetime.now(timezone.utc)
        db.commit()
    finally:
        db.close()
