from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from webapp.backend.db import get_db
from webapp.backend.deps import get_current_user
from webapp.backend.models import FineTuneFeedback, GeneratedCandidate, TrainingRun, User
from webapp.backend.schemas import CandidateOut, CandidateScoreRequest, FineTuneUsageOut
from webapp.backend.mol_render import mol_image_base64, mol_to_molblock_3d

router = APIRouter(prefix="/api/candidates", tags=["candidates"])


def _to_out(db: Session, candidate: GeneratedCandidate) -> CandidateOut:
    out = CandidateOut.model_validate(candidate)
    out.image_b64 = mol_image_base64(candidate.smiles)
    out.molblock_3d = mol_to_molblock_3d(candidate.smiles)

    usage_rows = (
        db.query(FineTuneFeedback, TrainingRun)
        .join(TrainingRun, FineTuneFeedback.training_run_id == TrainingRun.id)
        .filter(FineTuneFeedback.candidate_id == candidate.id)
        .all()
    )
    out.used_in_finetune_runs = [
        FineTuneUsageOut(
            training_run_id=run.id,
            status=run.status,
            alpha=run.finetune_alpha,
            created_at=run.created_at,
        )
        for _feedback, run in usage_rows
    ]
    return out


@router.get("", response_model=List[CandidateOut])
def list_candidates(
    generation_batch_id: Optional[int] = None,
    training_run_id: Optional[int] = None,
    target_protein_id: Optional[int] = None,
    config_id: Optional[int] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    query = db.query(GeneratedCandidate).filter(GeneratedCandidate.user_id == current_user.id)
    if generation_batch_id is not None:
        query = query.filter(GeneratedCandidate.generation_batch_id == generation_batch_id)
    if training_run_id is not None:
        query = query.filter(GeneratedCandidate.training_run_id == training_run_id)
    if target_protein_id is not None:
        query = query.filter(GeneratedCandidate.target_protein_id == target_protein_id)
    if config_id is not None:
        query = query.filter(GeneratedCandidate.config_id == config_id)
    candidates = query.order_by(GeneratedCandidate.reward.desc()).all()
    return [_to_out(db, c) for c in candidates]


@router.post("/{candidate_id}/score", response_model=CandidateOut)
def score_candidate(candidate_id: int, payload: CandidateScoreRequest, db: Session = Depends(get_db),
                     current_user: User = Depends(get_current_user)):
    candidate = db.get(GeneratedCandidate, candidate_id)
    if candidate is None or candidate.user_id != current_user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Candidate not found")

    candidate.user_rating = payload.rating
    candidate.rating_notes = payload.notes
    db.commit()
    db.refresh(candidate)
    return _to_out(db, candidate)
