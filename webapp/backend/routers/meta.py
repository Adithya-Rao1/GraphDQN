from fastapi import APIRouter, Depends
from typing import List

from webapp.backend.deps import get_current_user
from webapp.backend.models import User
from webapp.backend.schemas import AdmetPropertyOut, ExampleTargetOut
from reward.multi_objective import ADMET_PROPERTIES, ADMET_OPTIM_DIRECTIONS
from experiments.data.targets import TARGETS

router = APIRouter(prefix="/api/meta", tags=["meta"])


@router.get("/admet-properties", response_model=List[AdmetPropertyOut])
def admet_properties(current_user: User = Depends(get_current_user)):
    return [
        AdmetPropertyOut(name=prop, default_direction="maximize" if direction == 1 else "minimize")
        for prop, direction in zip(ADMET_PROPERTIES, ADMET_OPTIM_DIRECTIONS)
    ]


@router.get("/example-targets", response_model=List[ExampleTargetOut])
def example_targets(current_user: User = Depends(get_current_user)):
    return [ExampleTargetOut(key=key, name=key, sequence=seq) for key, seq in TARGETS.items()]
