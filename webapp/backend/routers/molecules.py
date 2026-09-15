from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from rdkit import Chem
from sqlalchemy.orm import Session

from webapp.backend.db import get_db
from webapp.backend.deps import get_current_user
from webapp.backend.models import StartingMolecule, User
from webapp.backend.schemas import (
    MoleculeCreate,
    MoleculeOut,
    MoleculePreviewOut,
    MoleculePreviewRequest,
)
from experiments.visualize_agent import mol_image_base64

router = APIRouter(prefix="/api/molecules", tags=["molecules"])


def _canonicalize(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    return Chem.MolToSmiles(mol)


@router.post("/preview", response_model=MoleculePreviewOut)
def preview_molecule(payload: MoleculePreviewRequest, current_user: User = Depends(get_current_user)):
    try:
        canonical = _canonicalize(payload.smiles)
    except ValueError as e:
        return MoleculePreviewOut(valid=False, error=str(e))
    return MoleculePreviewOut(valid=True, canonical_smiles=canonical, image_b64=mol_image_base64(canonical))


@router.post("", response_model=MoleculeOut)
def create_molecule(payload: MoleculeCreate, db: Session = Depends(get_db),
                     current_user: User = Depends(get_current_user)):
    try:
        canonical = _canonicalize(payload.smiles)
    except ValueError as e:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(e))

    molecule = StartingMolecule(
        user_id=current_user.id, smiles=payload.smiles, canonical_smiles=canonical, label=payload.label,
    )
    db.add(molecule)
    db.commit()
    db.refresh(molecule)
    return molecule


@router.get("", response_model=List[MoleculeOut])
def list_molecules(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    return (
        db.query(StartingMolecule)
        .filter(StartingMolecule.user_id == current_user.id)
        .order_by(StartingMolecule.created_at.desc())
        .all()
    )


@router.delete("/{molecule_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_molecule(molecule_id: int, db: Session = Depends(get_db),
                     current_user: User = Depends(get_current_user)):
    molecule = db.get(StartingMolecule, molecule_id)
    if molecule is None or molecule.user_id != current_user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Molecule not found")
    db.delete(molecule)
    db.commit()
