from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from webapp.backend.db import get_db
from webapp.backend.deps import get_current_user
from webapp.backend.models import Protein, User
from webapp.backend.schemas import ProteinCreate, ProteinOut

router = APIRouter(prefix="/api/proteins", tags=["proteins"])


@router.post("", response_model=ProteinOut)
def create_protein(payload: ProteinCreate, db: Session = Depends(get_db),
                    current_user: User = Depends(get_current_user)):
    protein = Protein(user_id=current_user.id, name=payload.name, sequence=payload.sequence.strip())
    db.add(protein)
    db.commit()
    db.refresh(protein)
    return protein


@router.get("", response_model=List[ProteinOut])
def list_proteins(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    return (
        db.query(Protein)
        .filter(Protein.user_id == current_user.id)
        .order_by(Protein.created_at.desc())
        .all()
    )


@router.delete("/{protein_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_protein(protein_id: int, db: Session = Depends(get_db),
                    current_user: User = Depends(get_current_user)):
    protein = db.get(Protein, protein_id)
    if protein is None or protein.user_id != current_user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Protein not found")
    db.delete(protein)
    db.commit()
