from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from webapp.backend.db import get_db
from webapp.backend.deps import get_current_user
from webapp.backend.jobs.manager import job_manager
from webapp.backend.models import FineTuneFeedback, GeneratedCandidate, OptimizationConfig, TrainingRun, User
from webapp.backend.schemas import FineTuneRequest, TrainingRunCreate, TrainingRunOut

router = APIRouter(prefix="/api/runs", tags=["runs"])


def _owned_run_or_404(db: Session, run_id: int, user_id: int) -> TrainingRun:
    run = db.get(TrainingRun, run_id)
    if run is None or run.user_id != user_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Run not found")
    return run


@router.post("", response_model=TrainingRunOut)
def start_run(payload: TrainingRunCreate, db: Session = Depends(get_db),
              current_user: User = Depends(get_current_user)):
    config = db.get(OptimizationConfig, payload.config_id)
    if config is None or config.user_id != current_user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Config not found")

    run = TrainingRun(
        user_id=current_user.id,
        config_id=config.id,
        is_finetune=False,
        seed=payload.seed,
        num_episodes=payload.num_episodes,
        max_steps=payload.max_steps,
        checkpoint_interval=payload.checkpoint_interval,
        status="pending",
        progress_total=payload.num_episodes,
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    job_manager.submit_training(run.id)
    return run


@router.get("", response_model=List[TrainingRunOut])
def list_runs(config_id: Optional[int] = None, db: Session = Depends(get_db),
              current_user: User = Depends(get_current_user)):
    query = db.query(TrainingRun).filter(TrainingRun.user_id == current_user.id)
    if config_id is not None:
        query = query.filter(TrainingRun.config_id == config_id)
    return query.order_by(TrainingRun.created_at.desc()).all()


@router.get("/{run_id}", response_model=TrainingRunOut)
def get_run(run_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    return _owned_run_or_404(db, run_id, current_user.id)


@router.post("/{run_id}/cancel", response_model=TrainingRunOut)
def cancel_run(run_id: int, discard: bool = False, db: Session = Depends(get_db),
                current_user: User = Depends(get_current_user)):
    run = _owned_run_or_404(db, run_id, current_user.id)
    job_manager.cancel(run_id, discard=discard)
    return run


@router.post("/{run_id}/finetune", response_model=TrainingRunOut)
def finetune_run(run_id: int, payload: FineTuneRequest, db: Session = Depends(get_db),
                  current_user: User = Depends(get_current_user)):
    parent_run = _owned_run_or_404(db, run_id, current_user.id)
    if parent_run.status != "completed" or not parent_run.final_checkpoint_path:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                             "Parent run must be completed with a checkpoint before fine-tuning")

    candidates = (
        db.query(GeneratedCandidate)
        .filter(GeneratedCandidate.id.in_(payload.candidate_ids))
        .all()
    )
    if len(candidates) != len(payload.candidate_ids):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "One or more candidates not found")
    for candidate in candidates:
        if candidate.user_id != current_user.id:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "One or more candidates not found")
        if candidate.user_rating is None:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                                 f"Candidate {candidate.id} has not been scored yet")

    new_run = TrainingRun(
        user_id=current_user.id,
        config_id=parent_run.config_id,
        is_finetune=True,
        parent_run_id=parent_run.id,
        finetune_alpha=payload.alpha,
        seed=parent_run.seed,
        num_episodes=payload.num_extra_episodes,
        max_steps=parent_run.max_steps,
        checkpoint_interval=parent_run.checkpoint_interval,
        status="pending",
        progress_total=payload.num_extra_episodes,
    )
    db.add(new_run)
    db.flush()  # assign new_run.id without committing yet

    for candidate in candidates:
        db.add(FineTuneFeedback(
            training_run_id=new_run.id,
            candidate_id=candidate.id,
            user_rating_snapshot=candidate.user_rating,
        ))

    db.commit()
    db.refresh(new_run)

    job_manager.submit_finetune(new_run.id)
    return new_run
