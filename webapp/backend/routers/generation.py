from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from webapp.backend.db import get_db
from webapp.backend.deps import get_current_user
from webapp.backend.jobs.manager import job_manager
from webapp.backend.models import GenerationBatch, TrainingRun, User
from webapp.backend.schemas import GenerationBatchOut, GenerationRequest

router = APIRouter(tags=["generation"])


@router.post("/api/runs/{run_id}/generate", response_model=GenerationBatchOut)
def generate_candidates(run_id: int, payload: GenerationRequest, db: Session = Depends(get_db),
                         current_user: User = Depends(get_current_user)):
    run = db.get(TrainingRun, run_id)
    if run is None or run.user_id != current_user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Run not found")
    if run.status != "completed" or not run.final_checkpoint_path:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                             "Run must be completed with a checkpoint before generating candidates")

    batch = GenerationBatch(
        user_id=current_user.id,
        training_run_id=run.id,
        checkpoint_path_snapshot=run.final_checkpoint_path,
        num_requested=payload.num_candidates,
        sampling_strategy=payload.sampling_strategy,
        temperature=payload.temperature,
        status="pending",
    )
    db.add(batch)
    db.commit()
    db.refresh(batch)

    job_manager.submit_generation(batch.id)
    return batch


@router.get("/api/generation-batches/{batch_id}", response_model=GenerationBatchOut)
def get_generation_batch(batch_id: int, db: Session = Depends(get_db),
                          current_user: User = Depends(get_current_user)):
    batch = db.get(GenerationBatch, batch_id)
    if batch is None or batch.user_id != current_user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Generation batch not found")
    return batch
