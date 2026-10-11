from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ppo.pgmorl_agent import DEFAULT_LLM_MODEL_NAME
from webapp.backend.db import get_db
from webapp.backend.deps import get_current_user
from webapp.backend.jobs.manager import job_manager
from webapp.backend.models import LLMFinetuneRun, ManualEditSession, User
from webapp.backend.schemas import LLMFinetuneFromSessionsRequest, LLMFinetuneRunOut

router = APIRouter(prefix="/api/llm-finetune-runs", tags=["llm_finetune"])


def _owned_run_or_404(db: Session, run_id: int, user_id: int) -> LLMFinetuneRun:
    run = db.get(LLMFinetuneRun, run_id)
    if run is None or run.user_id != user_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "LLM fine-tune run not found")
    return run


@router.post("", response_model=LLMFinetuneRunOut)
def create_llm_finetune_run(payload: LLMFinetuneFromSessionsRequest, db: Session = Depends(get_db),
                             current_user: User = Depends(get_current_user)):
    sessions = (
        db.query(ManualEditSession)
        .filter(ManualEditSession.id.in_(payload.manual_edit_session_ids))
        .all()
    )
    if len(sessions) != len(payload.manual_edit_session_ids):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "One or more sessions not found")
    for session in sessions:
        if session.user_id != current_user.id:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "One or more sessions not found")
        if session.status != "completed":
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                                 f"Session {session.id} has not finished scoring yet")

    run = LLMFinetuneRun(
        user_id=current_user.id,
        manual_edit_session_ids=payload.manual_edit_session_ids,
        llm_model_name=payload.llm_model_name or DEFAULT_LLM_MODEL_NAME,
        warm_start=payload.warm_start,
        status="pending",
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    job_manager.submit_llm_finetune_from_sessions(run.id)
    return run


@router.get("", response_model=List[LLMFinetuneRunOut])
def list_llm_finetune_runs(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    return (
        db.query(LLMFinetuneRun)
        .filter(LLMFinetuneRun.user_id == current_user.id)
        .order_by(LLMFinetuneRun.created_at.desc())
        .all()
    )


@router.get("/{run_id}", response_model=LLMFinetuneRunOut)
def get_llm_finetune_run(run_id: int, db: Session = Depends(get_db),
                          current_user: User = Depends(get_current_user)):
    return _owned_run_or_404(db, run_id, current_user.id)
