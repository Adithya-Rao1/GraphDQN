from datetime import datetime, timezone

import torch

from ADMET.model import ADMETModel
from binding_module.binding_affinity.plapt import Plapt
from synthetic_accessibility.sa_score import SyntheticAccessibility
from reward.multi_objective import compute_reward_batch
from webapp.backend.db import SessionLocal
from webapp.backend.models import EditOutcomeLog, ManualEditSession, StartingMolecule


def _before_after_summary(results: list) -> dict:
    """results: compute_reward_batch's raw output over the full sequence
    (index 0 = starting molecule, before any edits; index -1 = after the
    last edit) -- NOT the per-edit EditOutcomeLog rows, which only cover
    results[1:]."""
    first, last = results[0], results[-1]
    return {
        "reward_before": first["reward"], "reward_after": last["reward"],
        "reward_delta": last["reward"] - first["reward"],
        "admet_before": first["admet"], "admet_after": last["admet"],
        "binding_uM_before": first["binding_uM"], "binding_uM_after": last["binding_uM"],
        "sa_score_before": first["sa_score"], "sa_score_after": last["sa_score"],
        "selectivity_before": first["selectivity"], "selectivity_after": last["selectivity"],
    }


def run_manual_edit_finish_job(session_id: int) -> None:
    db = SessionLocal()
    try:
        session = db.get(ManualEditSession, session_id)
        molecule = db.get(StartingMolecule, session.starting_molecule_id)
        target_protein = session.target_protein
        off_target_protein = session.off_target_protein

        sequence = [molecule.canonical_smiles] + [e["resulting_smiles"] for e in session.pending_edits]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        admet_model = ADMETModel(device)
        binding_model = Plapt(device=str(device))
        sa_model = SyntheticAccessibility()

        results = compute_reward_batch(
            sequence, target_protein.sequence, device,
            off_target_seq=off_target_protein.sequence if off_target_protein else None,
            admet_model=admet_model, binding_model=binding_model, sa_model=sa_model,
        )

        for i, edit in enumerate(session.pending_edits):
            result = results[i + 1]
            db.add(EditOutcomeLog(
                user_id=session.user_id,
                manual_edit_session_id=session.id,
                step_index=i,
                parent_smiles=sequence[i],
                applied_edit_ids=[edit["edit_id"]],
                pre_edit_states=[sequence[i]],
                k_edits_used=1,
                edit_count_mode="manual",
                resulting_smiles=sequence[i + 1],
                admet_score=result["admet"],
                binding_uM=result["binding_uM"],
                sa_score=result["sa_score"],
                selectivity=result["selectivity"],
                reward=result["reward"],
                reward_vector=result["reward_vector"],
            ))

        session.result_summary_json = _before_after_summary(results)
        session.status = "completed"
        session.finished_at = datetime.now(timezone.utc)
        db.commit()
    except Exception as e:  # noqa: BLE001 -- job boundary
        session = db.get(ManualEditSession, session_id)
        session.status = "failed"
        session.error_message = str(e)
        session.finished_at = datetime.now(timezone.utc)
        db.commit()
    finally:
        db.close()
