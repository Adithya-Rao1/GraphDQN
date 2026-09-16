from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from webapp.backend.db import get_db
from webapp.backend.deps import get_current_user
from webapp.backend.jobs.manager import job_manager
from webapp.backend.models import OptimizationConfig, ParetoSweepRun, User
from webapp.backend.schemas import ParetoSweepCreate, ParetoSweepOut

router = APIRouter(prefix="/api/pareto-sweeps", tags=["pareto-sweeps"])


def _owned_sweep_or_404(db: Session, sweep_id: int, user_id: int) -> ParetoSweepRun:
    sweep = db.get(ParetoSweepRun, sweep_id)
    if sweep is None or sweep.user_id != user_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Pareto sweep not found")
    return sweep


@router.post("", response_model=ParetoSweepOut)
def start_pareto_sweep(payload: ParetoSweepCreate, db: Session = Depends(get_db),
                        current_user: User = Depends(get_current_user)):
    base_config = db.get(OptimizationConfig, payload.base_config_id)
    if base_config is None or base_config.user_id != current_user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Config not found")

    error = payload.edit_count_config_valid()
    if error:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, error)
    if payload.use_llm and not payload.llm_model_name:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                             "llm_model_name is required when use_llm is true")

    sweep = ParetoSweepRun(
        user_id=current_user.id,
        base_config_id=base_config.id,
        seed=payload.seed,
        population_size=payload.population_size,
        concentration_alpha=payload.concentration_alpha,
        num_rounds=payload.num_rounds,
        episodes_per_round=payload.episodes_per_round,
        eval_episodes_per_round=payload.eval_episodes_per_round,
        max_steps=payload.max_steps,
        edit_count_mode=payload.edit_count_mode,
        fixed_edit_count=payload.fixed_edit_count,
        edit_count_range_min=payload.edit_count_range_min,
        edit_count_range_max=payload.edit_count_range_max,
        k_max=payload.k_max,
        use_llm=payload.use_llm,
        llm_model_name=payload.llm_model_name,
        use_predictor=payload.use_predictor,
        concurrent=payload.concurrent,
        max_concurrent_members=payload.max_concurrent_members,
        status="pending",
        progress_total_rounds=payload.num_rounds,
    )
    db.add(sweep)
    db.commit()
    db.refresh(sweep)

    job_manager.submit_pareto_sweep(sweep.id)
    return sweep


@router.get("", response_model=List[ParetoSweepOut])
def list_pareto_sweeps(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    return (
        db.query(ParetoSweepRun)
        .filter(ParetoSweepRun.user_id == current_user.id)
        .order_by(ParetoSweepRun.created_at.desc())
        .all()
    )


@router.get("/{sweep_id}", response_model=ParetoSweepOut)
def get_pareto_sweep(sweep_id: int, db: Session = Depends(get_db),
                      current_user: User = Depends(get_current_user)):
    return _owned_sweep_or_404(db, sweep_id, current_user.id)


@router.post("/{sweep_id}/cancel", response_model=ParetoSweepOut)
def cancel_pareto_sweep(sweep_id: int, discard: bool = False, db: Session = Depends(get_db),
                         current_user: User = Depends(get_current_user)):
    sweep = _owned_sweep_or_404(db, sweep_id, current_user.id)
    job_manager.cancel_pareto_sweep(sweep_id, discard=discard)
    return sweep
