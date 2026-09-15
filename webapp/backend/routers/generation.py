from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from webapp.backend.candidate_provenance import provenance_kwargs
from webapp.backend.db import get_db
from webapp.backend.deps import get_current_user
from webapp.backend.jobs.manager import job_manager
from webapp.backend.mol_render import mol_image_base64, mol_to_molblock_3d
from webapp.backend.models import GeneratedCandidate, GenerationBatch, TrainingRun, User
from webapp.backend.schemas import (
    CandidateOut,
    GenerationBatchOut,
    GenerationRequest,
    PromoteStepsRequest,
    TrajectoryOut,
    TrajectoryStepOut,
)

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


@router.get("/api/generation-batches", response_model=List[GenerationBatchOut])
def list_generation_batches(training_run_id: int, db: Session = Depends(get_db),
                             current_user: User = Depends(get_current_user)):
    return (
        db.query(GenerationBatch)
        .filter(GenerationBatch.user_id == current_user.id, GenerationBatch.training_run_id == training_run_id)
        .order_by(GenerationBatch.created_at.desc())
        .all()
    )


@router.get("/api/generation-batches/{batch_id}", response_model=GenerationBatchOut)
def get_generation_batch(batch_id: int, db: Session = Depends(get_db),
                          current_user: User = Depends(get_current_user)):
    batch = db.get(GenerationBatch, batch_id)
    if batch is None or batch.user_id != current_user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Generation batch not found")
    return batch


@router.get("/api/generation-batches/{batch_id}/trajectories", response_model=List[TrajectoryOut])
def get_trajectories(batch_id: int, db: Session = Depends(get_db),
                      current_user: User = Depends(get_current_user)):
    batch = db.get(GenerationBatch, batch_id)
    if batch is None or batch.user_id != current_user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Generation batch not found")
    if not batch.trajectories:
        return []

    out = []
    for traj_idx, steps in enumerate(batch.trajectories):
        step_outs = [
            TrajectoryStepOut(
                step_index=step["step_index"],
                smiles=step["smiles"],
                image_b64=mol_image_base64(step["smiles"]),
                molblock_3d=mol_to_molblock_3d(step["smiles"]),
                reward=step["reward"],
                admet_score=step.get("admet_score"),
                binding_uM=step.get("binding_uM"),
                sa_score=step.get("sa_score"),
                selectivity=step.get("selectivity"),
            )
            for step in steps
        ]
        out.append(TrajectoryOut(trajectory_index=traj_idx, steps=step_outs))
    return out


@router.post("/api/generation-batches/{batch_id}/promote", response_model=List[CandidateOut])
def promote_steps(batch_id: int, payload: PromoteStepsRequest, db: Session = Depends(get_db),
                   current_user: User = Depends(get_current_user)):
    batch = db.get(GenerationBatch, batch_id)
    if batch is None or batch.user_id != current_user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Generation batch not found")
    if not batch.trajectories:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "This batch has no step data to promote from")

    run = db.get(TrainingRun, batch.training_run_id)
    config = run.config

    created = []
    for sel in payload.selections:
        if not (0 <= sel.trajectory_index < len(batch.trajectories)):
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                                 f"trajectory_index {sel.trajectory_index} out of range")
        trajectory = batch.trajectories[sel.trajectory_index]
        if not (0 <= sel.step_index < len(trajectory)):
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                                 f"step_index {sel.step_index} out of range for trajectory {sel.trajectory_index}")
        step = trajectory[sel.step_index]

        candidate = GeneratedCandidate(
            user_id=current_user.id,
            generation_batch_id=batch.id,
            training_run_id=run.id,
            smiles=step["smiles"],
            reward=step["reward"],
            admet_score=step.get("admet_score"),
            binding_uM=step.get("binding_uM"),
            sa_score=step.get("sa_score"),
            selectivity=step.get("selectivity"),
            **provenance_kwargs(config),
        )
        db.add(candidate)
        created.append(candidate)

    db.commit()
    results = []
    for c in created:
        db.refresh(c)
        out = CandidateOut.model_validate(c)
        out.image_b64 = mol_image_base64(c.smiles)
        out.molblock_3d = mol_to_molblock_3d(c.smiles)
        results.append(out)
    return results
