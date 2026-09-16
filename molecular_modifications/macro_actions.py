from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from rdkit import Chem

from molecular_modifications.atom_optimization import ModifyAtom
from molecular_modifications.bond_optimization import ModifyBond
from molecular_modifications.bioisosteres_optimization import ModifyBioisosteres
from molecular_modifications.functional_group_optimization import ModifyFunctionalGroup
from molecular_modifications.chemistry_constants import find_unsupported_elements

_FG_REMOVE_GROUPS = [
    'methyl', 'hydroxyl', 'amino', 'carboxyl', 'carbonyl', 'aldehyde',
    'ketone', 'ether', 'ester', 'amide', 'nitro', 'cyano', 'thiol',
    'halogen', 'azide', 'sulfonamide',
]
_FG_MODIFY_PAIRS = [
    ('hydroxyl', 'amino'), ('hydroxyl', 'thiol'), ('hydroxyl', 'methyl'), ('hydroxyl', 'halogen'),
    ('amino', 'hydroxyl'), ('amino', 'amide'),
    ('carboxyl', 'ester'), ('carboxyl', 'amide'),
    ('aldehyde', 'ketone'), ('aldehyde', 'carboxyl'),
    ('thiol', 'hydroxyl'), ('cyano', 'carboxyl'), ('nitro', 'amino'),
    ('ester', 'carboxyl'), ('amide', 'carboxyl'),
]
_FG_ADD_GROUPS = ['methyl', 'hydroxyl', 'amino', 'halogen', 'cyano', 'carboxyl']

_BIOISOSTERE_MODIFICATIONS = [
    ('carboxylic_acid', 'tetrazole'),
    ('carboxylic_acid', 'phosphonic_acid'),
    ('amine', None),
    ('amide', 'sulfonamide'),
    ('amide', 'retroamide'),
    ('phenyl', 'pyridyl'),
    ('phenyl', 'thiophene'),
    ('phenyl', 'furan'),
    ('phenyl', 'pyrrole'),
]


@dataclass(frozen=True)
class EditOp:
    id: str
    category: str  # "atom" | "bond" | "bioisostere" | "functional_group"
    description: str
    apply: Callable[[Chem.Mol], Optional[str]]  


def build_edit_catalog(mol_logger=None) -> List[EditOp]:
    modify_atom = ModifyAtom(mol_logger)
    modify_bond = ModifyBond(mol_logger)
    modify_bio = ModifyBioisosteres(mol_logger)
    modify_fg = ModifyFunctionalGroup(mol_logger)

    catalog: List[EditOp] = []

    catalog.append(EditOp(
        id="atom_add", category="atom",
        description="Add a new heavy atom bonded to an existing modification site.",
        apply=lambda mol: modify_atom.add_atom(mol, 0),
    ))
    catalog.append(EditOp(
        id="atom_remove", category="atom",
        description="Remove a non-ring (or ring-but-non-carbon) heavy atom.",
        apply=lambda mol: modify_atom.remove_atom(mol, 0),
    ))
    catalog.append(EditOp(
        id="atom_modify", category="atom",
        description="Substitute a heavy atom for a chemically similar element "
                     "(electronegativity/covalent-radius balanced).",
        apply=lambda mol: modify_atom.modify_atom(mol, 0),
    ))

    catalog.append(EditOp(
        id="bond_add", category="bond",
        description="Add a new bond between two currently unbonded, valence-compatible atoms.",
        apply=lambda mol: modify_bond.optimize_bond(mol, 0),
    ))
    catalog.append(EditOp(
        id="bond_remove", category="bond",
        description="Remove an existing bond (subject to ring-strain/valence rules).",
        apply=lambda mol: modify_bond.optimize_bond(mol, 1),
    ))
    catalog.append(EditOp(
        id="bond_modify", category="bond",
        description="Change an existing bond's order (single/double/triple).",
        apply=lambda mol: modify_bond.optimize_bond(mol, 2),
    ))

    def _make_bio_apply(fg_type: str, replacement: Optional[str]):
        return lambda mol: modify_bio.apply_modification(mol, fg_type, replacement)

    for fg_type, replacement in _BIOISOSTERE_MODIFICATIONS:
        suffix = f"{fg_type}_to_{replacement}" if replacement else fg_type
        replacement_desc = f"'{replacement}'" if replacement else "its standard bioisostere"
        catalog.append(EditOp(
            id=f"bio_{suffix}", category="bioisostere",
            description=f"Replace a '{fg_type}' group with {replacement_desc}.",
            apply=_make_bio_apply(fg_type, replacement),
        ))

    # --- Functional-group removal (16) ---
    def _make_fg_remove_apply(fg: str):
        return lambda mol: modify_fg.remove_functional_group(mol, fg)

    for fg in _FG_REMOVE_GROUPS:
        catalog.append(EditOp(
            id=f"fg_remove_{fg}", category="functional_group",
            description=f"Remove a '{fg}' functional group.",
            apply=_make_fg_remove_apply(fg),
        ))

    def _make_fg_modify_apply(source_fg: str, target_fg: str):
        return lambda mol: modify_fg.modify_functional_group(mol, source_fg, target_fg)

    for source_fg, target_fg in _FG_MODIFY_PAIRS:
        catalog.append(EditOp(
            id=f"fg_modify_{source_fg}_to_{target_fg}", category="functional_group",
            description=f"Convert a '{source_fg}' group to '{target_fg}'.",
            apply=_make_fg_modify_apply(source_fg, target_fg),
        ))

    def _make_fg_add_apply(fg: str):
        def _apply(mol: Chem.Mol) -> Optional[str]:
            sites = modify_fg.functional_group_add_sites(mol, fg)
            if not sites:
                return None
            return modify_fg.add_functional_group(mol, fg, sites[0])
        return _apply

    for fg in _FG_ADD_GROUPS:
        catalog.append(EditOp(
            id=f"fg_add_{fg}", category="functional_group",
            description=f"Add a '{fg}' group at an available attachment site.",
            apply=_make_fg_add_apply(fg),
        ))

    ids = [op.id for op in catalog]
    assert len(ids) == len(set(ids)), "macro_actions catalog has duplicate ids"

    return catalog


@dataclass
class MacroActionResult:
    final_smiles: str
    requested_edit_ids: List[str]
    applied_edit_ids: List[str]
    intermediate_smiles: List[str] 
    fully_applied: bool


def compose_macro_action(
    start_smiles: str,
    edit_ids: Sequence[str],
    catalog_by_id: Dict[str, EditOp],
) -> Optional[MacroActionResult]:
    mol = Chem.MolFromSmiles(start_smiles)
    if mol is None:
        return None

    current_smiles = start_smiles
    applied: List[str] = []
    intermediates: List[str] = []

    for edit_id in edit_ids:
        edit_op = catalog_by_id.get(edit_id)
        if edit_op is None:
            break

        current_mol = Chem.MolFromSmiles(current_smiles)
        if current_mol is None:
            break

        result_smiles = edit_op.apply(current_mol)
        if not result_smiles:
            break

        result_mol = Chem.MolFromSmiles(result_smiles)
        if result_mol is None:
            break

        if find_unsupported_elements(result_mol):
            break

        current_smiles = result_smiles
        applied.append(edit_id)
        intermediates.append(current_smiles)

    if not applied:
        return None

    return MacroActionResult(
        final_smiles=current_smiles,
        requested_edit_ids=list(edit_ids),
        applied_edit_ids=applied,
        intermediate_smiles=intermediates,
        fully_applied=(len(applied) == len(edit_ids)),
    )
