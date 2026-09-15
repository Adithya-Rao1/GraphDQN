from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from webapp.backend.db import get_db
from webapp.backend.deps import get_current_user
from webapp.backend.models import OptimizationConfig, Protein, StartingMolecule, User
from webapp.backend.schemas import OptimizationConfigCreate, OptimizationConfigOut

router = APIRouter(prefix="/api/configs", tags=["configs"])


def _owned_or_404(db: Session, model, obj_id: int, user_id: int, label: str):
    obj = db.get(model, obj_id)
    if obj is None or obj.user_id != user_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"{label} not found")
    return obj


@router.post("", response_model=OptimizationConfigOut)
def create_config(payload: OptimizationConfigCreate, db: Session = Depends(get_db),
                   current_user: User = Depends(get_current_user)):
    if not payload.weights_valid():
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                             "All weights must be >= 0 and at least one must be > 0")
    if set(payload.admet_directions) - set(payload.admet_properties):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                             "admet_directions keys must be a subset of admet_properties")

    _owned_or_404(db, StartingMolecule, payload.starting_molecule_id, current_user.id, "Starting molecule")
    _owned_or_404(db, Protein, payload.target_protein_id, current_user.id, "Target protein")
    if payload.off_target_protein_id is not None:
        _owned_or_404(db, Protein, payload.off_target_protein_id, current_user.id, "Off-target protein")

    config = OptimizationConfig(
        user_id=current_user.id,
        name=payload.name,
        starting_molecule_id=payload.starting_molecule_id,
        target_protein_id=payload.target_protein_id,
        off_target_protein_id=payload.off_target_protein_id,
        admet_weight=payload.admet_weight,
        binding_weight=payload.binding_weight,
        synthetic_weight=payload.synthetic_weight,
        selectivity_weight=payload.selectivity_weight,
        admet_properties=payload.admet_properties,
        admet_directions=payload.admet_directions_as_int(),
    )
    db.add(config)
    db.commit()
    db.refresh(config)
    return config


@router.get("", response_model=List[OptimizationConfigOut])
def list_configs(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    return (
        db.query(OptimizationConfig)
        .filter(OptimizationConfig.user_id == current_user.id)
        .order_by(OptimizationConfig.created_at.desc())
        .all()
    )


@router.get("/{config_id}", response_model=OptimizationConfigOut)
def get_config(config_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    return _owned_or_404(db, OptimizationConfig, config_id, current_user.id, "Config")


@router.delete("/{config_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_config(config_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    config = _owned_or_404(db, OptimizationConfig, config_id, current_user.id, "Config")
    db.delete(config)
    db.commit()
