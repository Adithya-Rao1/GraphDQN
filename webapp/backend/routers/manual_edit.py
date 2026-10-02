from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from rdkit import Chem
from sqlalchemy.orm import Session

from molecular_modifications.chemistry_constants import find_unsupported_elements
from molecular_modifications.macro_actions import build_edit_catalog
from webapp.backend.db import get_db
from webapp.backend.deps import get_current_user
from webapp.backend.jobs.manager import job_manager
from webapp.backend.mol_render import mol_image_base64, mol_to_molblock_3d
from webapp.backend.models import ManualEditSession, Protein, StartingMolecule, User
from webapp.backend.schemas import (
    ApplyEditRequest,
    ApplyEditResponse,
    EditCatalogEntryOut,
    ManualEditSessionCreate,
    ManualEditSessionOut,
)

router = APIRouter(prefix="/api/manual-sessions", tags=["manual_edit"])

CATALOG = build_edit_catalog()
CATALOG_BY_ID = {op.id: op for op in CATALOG}


def _get_owned_session(session_id: int, db: Session, current_user: User) -> ManualEditSession:
    session = db.get(ManualEditSession, session_id)
    if session is None or session.user_id != current_user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Manual edit session not found")
    return session


def _current_head_smiles(session: ManualEditSession, db: Session) -> str:
    if session.pending_edits:
        return session.pending_edits[-1]["resulting_smiles"]
    molecule = db.get(StartingMolecule, session.starting_molecule_id)
    return molecule.canonical_smiles


@router.post("", response_model=ManualEditSessionOut)
def create_session(payload: ManualEditSessionCreate, db: Session = Depends(get_db),
                    current_user: User = Depends(get_current_user)):
    molecule = db.get(StartingMolecule, payload.starting_molecule_id)
    if molecule is None or molecule.user_id != current_user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Starting molecule not found")
    target_protein = db.get(Protein, payload.target_protein_id)
    if target_protein is None or target_protein.user_id != current_user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Target protein not found")
    if payload.off_target_protein_id is not None:
        off_target = db.get(Protein, payload.off_target_protein_id)
        if off_target is None or off_target.user_id != current_user.id:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Off-target protein not found")

    session = ManualEditSession(
        user_id=current_user.id,
        starting_molecule_id=payload.starting_molecule_id,
        target_protein_id=payload.target_protein_id,
        off_target_protein_id=payload.off_target_protein_id,
        pending_edits=[],
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


@router.get("", response_model=List[ManualEditSessionOut])
def list_sessions(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    return (
        db.query(ManualEditSession)
        .filter(ManualEditSession.user_id == current_user.id)
        .order_by(ManualEditSession.created_at.desc())
        .all()
    )


@router.get("/{session_id}", response_model=ManualEditSessionOut)
def get_session(session_id: int, db: Session = Depends(get_db),
                 current_user: User = Depends(get_current_user)):
    return _get_owned_session(session_id, db, current_user)


@router.get("/{session_id}/catalog", response_model=List[EditCatalogEntryOut])
def get_catalog(session_id: int, db: Session = Depends(get_db),
                 current_user: User = Depends(get_current_user)):
    _get_owned_session(session_id, db, current_user)
    return [EditCatalogEntryOut(id=op.id, category=op.category, description=op.description) for op in CATALOG]


@router.get("/{session_id}/current", response_model=ApplyEditResponse)
def get_current_state(session_id: int, db: Session = Depends(get_db),
                        current_user: User = Depends(get_current_user)):
    session = _get_owned_session(session_id, db, current_user)
    smiles = _current_head_smiles(session, db)
    return ApplyEditResponse(
        smiles=smiles,
        image_b64=mol_image_base64(smiles),
        molblock_3d=mol_to_molblock_3d(smiles),
    )


@router.post("/{session_id}/apply-edit", response_model=ApplyEditResponse)
def apply_edit(session_id: int, payload: ApplyEditRequest, db: Session = Depends(get_db),
                current_user: User = Depends(get_current_user)):
    session = _get_owned_session(session_id, db, current_user)
    if session.status != "in_progress":
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                             f"Session is {session.status!r}, no further edits can be applied")

    edit_op = CATALOG_BY_ID.get(payload.edit_id)
    if edit_op is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown edit id: {payload.edit_id!r}")

    head_smiles = _current_head_smiles(session, db)
    mol = Chem.MolFromSmiles(head_smiles)
    if mol is None:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Current molecule state is invalid")

    result_smiles = edit_op.apply(mol)
    if not result_smiles:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                             "This edit could not be applied to the current molecule -- try a different one")

    result_mol = Chem.MolFromSmiles(result_smiles)
    if result_mol is None or find_unsupported_elements(result_mol):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                             "This edit produced an invalid or unsupported molecule -- try a different one")

    session.pending_edits = list(session.pending_edits) + [{
        "edit_id": edit_op.id,
        "category": edit_op.category,
        "description": edit_op.description,
        "pre_smiles": head_smiles,
        "resulting_smiles": result_smiles,
    }]
    db.commit()

    return ApplyEditResponse(
        smiles=result_smiles,
        image_b64=mol_image_base64(result_smiles),
        molblock_3d=mol_to_molblock_3d(result_smiles),
    )


@router.post("/{session_id}/undo", response_model=ManualEditSessionOut)
def undo_last_edit(session_id: int, db: Session = Depends(get_db),
                    current_user: User = Depends(get_current_user)):
    session = _get_owned_session(session_id, db, current_user)
    if session.status != "in_progress":
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                             f"Session is {session.status!r}, nothing to undo")
    if not session.pending_edits:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "No edits to undo")

    session.pending_edits = list(session.pending_edits)[:-1]
    db.commit()
    db.refresh(session)
    return session


@router.post("/{session_id}/finish", response_model=ManualEditSessionOut)
def finish_session(session_id: int, db: Session = Depends(get_db),
                    current_user: User = Depends(get_current_user)):
    session = _get_owned_session(session_id, db, current_user)
    if session.status != "in_progress":
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                             f"Session is already {session.status!r}")
    if not session.pending_edits:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                             "This session has no edits to score yet")

    session.status = "scoring"
    db.commit()
    db.refresh(session)

    job_manager.submit_manual_edit_finish(session.id)
    return session
