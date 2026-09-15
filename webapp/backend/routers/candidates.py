from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from webapp.backend.db import get_db
from webapp.backend.deps import get_current_user
from webapp.backend.models import GeneratedCandidate, GenerationBatch, User
from webapp.backend.schemas import CandidateOut, CandidateScoreRequest
from experiments.visualize_agent import mol_image_base64

router = APIRouter(prefix="/api/candidates", tags=["candidates"])


def _to_out(candidate: GeneratedCandidate) -> CandidateOut:
    out = CandidateOut.model_validate(candidate)
    out.image_b64 = mol_image_base64(candidate.smiles)
    return out


@router.get("", response_model=List[CandidateOut])
def list_candidates(generation_batch_id: Optional[int] = None, training_run_id: Optional[int] = None,
                     db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    query = db.query(GeneratedCandidate).filter(GeneratedCandidate.user_id == current_user.id)
    if generation_batch_id is not None:
        query = query.filter(GeneratedCandidate.generation_batch_id == generation_batch_id)
    if training_run_id is not None:
        query = query.join(GenerationBatch, GeneratedCandidate.generation_batch_id == GenerationBatch.id).filter(
            GenerationBatch.training_run_id == training_run_id
        )
    candidates = query.order_by(GeneratedCandidate.reward.desc()).all()
    return [_to_out(c) for c in candidates]


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
    return _to_out(candidate)
