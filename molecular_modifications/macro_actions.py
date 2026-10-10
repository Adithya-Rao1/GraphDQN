from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from rdkit import Chem

from molecular_modifications.atom_optimization import ModifyAtom
from molecular_modifications.bond_optimization import ModifyBond
from molecular_modifications.bioisosteres_optimization import ModifyBioisosteres
from molecular_modifications.functional_group_optimization import ModifyFunctionalGroup
from molecular_modifications.chemistry_constants import find_unsupported_elements
from molecular_modifications.site_selection import SiteCandidate

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

_BOND_TYPES_BY_NAME = {
    "SINGLE": Chem.BondType.SINGLE,
    "DOUBLE": Chem.BondType.DOUBLE,
    "TRIPLE": Chem.BondType.TRIPLE,
    "AROMATIC": Chem.BondType.AROMATIC,
}


@dataclass(frozen=True)
class EditOp:
    id: str
    category: str  # "atom" | "bond" | "bioisostere" | "functional_group"
    description: str
    apply: Callable[[Chem.Mol, Optional[dict]], Optional[str]]
    enumerate_sites: Callable[[Chem.Mol], List[SiteCandidate]]


def _atom_add_sites(modify_atom: ModifyAtom):
    def _enum(mol: Chem.Mol) -> List[SiteCandidate]:
        candidates = []
        for atom_idx, element in modify_atom.enumerate_site_element_candidates(mol):
            symbol = mol.GetAtomWithIdx(atom_idx).GetSymbol()
            candidates.append(SiteCandidate(
                site_key={"atom_idx": atom_idx, "target_element": element},
                group_label=f"Add {element}",
                label=f"Add {element} bonded to atom {atom_idx} ({symbol})",
                pre_highlight_atoms=[atom_idx],
            ))
        return candidates
    return _enum


def _atom_add_apply(modify_atom: ModifyAtom):
    def _apply(mol: Chem.Mol, site: Optional[dict] = None) -> Optional[str]:
        s = site or {}
        return modify_atom.add_atom(mol, 0, atom_idx=s.get("atom_idx"), target_element=s.get("target_element"))
    return _apply


def _atom_remove_sites(modify_atom: ModifyAtom):
    def _enum(mol: Chem.Mol) -> List[SiteCandidate]:
        candidates = []
        for atom_idx in modify_atom.get_valid_atom_sites(mol):
            atom = mol.GetAtomWithIdx(atom_idx)
            group = "Ring atom" if atom.IsInRing() else "Chain atom"
            candidates.append(SiteCandidate(
                site_key={"atom_idx": atom_idx},
                group_label=group,
                label=f"Remove atom {atom_idx} ({atom.GetSymbol()})",
                pre_highlight_atoms=[atom_idx],
            ))
        return candidates
    return _enum


def _atom_remove_apply(modify_atom: ModifyAtom):
    def _apply(mol: Chem.Mol, site: Optional[dict] = None) -> Optional[str]:
        s = site or {}
        return modify_atom.remove_atom(mol, 0, atom_idx=s.get("atom_idx"))
    return _apply


def _atom_modify_sites(modify_atom: ModifyAtom):
    def _enum(mol: Chem.Mol) -> List[SiteCandidate]:
        candidates = []
        for atom_idx, element in modify_atom.enumerate_site_element_candidates(mol):
            symbol = mol.GetAtomWithIdx(atom_idx).GetSymbol()
            candidates.append(SiteCandidate(
                site_key={"atom_idx": atom_idx, "target_element": element},
                group_label=f"{symbol} to {element}",
                label=f"Atom {atom_idx}: {symbol} to {element}",
                pre_highlight_atoms=[atom_idx],
            ))
        return candidates
    return _enum


def _atom_modify_apply(modify_atom: ModifyAtom):
    def _apply(mol: Chem.Mol, site: Optional[dict] = None) -> Optional[str]:
        s = site or {}
        return modify_atom.modify_atom(mol, 0, atom_idx=s.get("atom_idx"), target_element=s.get("target_element"))
    return _apply


def _bond_add_sites(modify_bond: ModifyBond):
    def _enum(mol: Chem.Mol) -> List[SiteCandidate]:
        candidates = []
        for i, j, bond_type in modify_bond.get_valid_bond_add_sites(mol):
            type_name = bond_type.name
            candidates.append(SiteCandidate(
                site_key={"atom_i": i, "atom_j": j, "bond_type": type_name},
                group_label=f"New {type_name.lower()} bond",
                label=f"Add {type_name.lower()} bond: atom {i} ({mol.GetAtomWithIdx(i).GetSymbol()}) "
                      f"- atom {j} ({mol.GetAtomWithIdx(j).GetSymbol()})",
                pre_highlight_atoms=[i, j],
            ))
        return candidates
    return _enum


def _bond_remove_sites(modify_bond: ModifyBond):
    def _enum(mol: Chem.Mol) -> List[SiteCandidate]:
        candidates = []
        for i, j in modify_bond.get_valid_bond_remove_sites(mol):
            bond = mol.GetBondBetweenAtoms(i, j)
            type_name = bond.GetBondType().name if bond else "?"
            candidates.append(SiteCandidate(
                site_key={"atom_i": i, "atom_j": j},
                group_label=f"Remove {type_name.lower()} bond",
                label=f"Remove bond: atom {i} ({mol.GetAtomWithIdx(i).GetSymbol()}) "
                      f"- atom {j} ({mol.GetAtomWithIdx(j).GetSymbol()})",
                pre_highlight_atoms=[i, j],
            ))
        return candidates
    return _enum


def _bond_modify_sites(modify_bond: ModifyBond):
    def _enum(mol: Chem.Mol) -> List[SiteCandidate]:
        candidates = []
        for i, j, new_type in modify_bond.get_valid_bond_modify_sites(mol):
            bond = mol.GetBondBetweenAtoms(i, j)
            current_name = bond.GetBondType().name if bond else "?"
            new_name = new_type.name
            candidates.append(SiteCandidate(
                site_key={"atom_i": i, "atom_j": j, "bond_type": new_name},
                group_label=f"{current_name.lower()} to {new_name.lower()}",
                label=f"Atom {i}-{j}: {current_name.lower()} to {new_name.lower()}",
                pre_highlight_atoms=[i, j],
            ))
        return candidates
    return _enum


def _make_bond_apply(modify_bond: ModifyBond, action: int):
    def _apply(mol: Chem.Mol, site: Optional[dict] = None) -> Optional[str]:
        s = site or {}
        if "atom_i" in s and "atom_j" in s:
            bond_site = (s["atom_i"], s["atom_j"])
            target_type = _BOND_TYPES_BY_NAME.get(s["bond_type"]) if s.get("bond_type") else None
            return modify_bond.optimize_bond(mol, action, bond_site=bond_site, target_bond_type=target_type)
        return modify_bond.optimize_bond(mol, action)
    return _apply


def _bio_sites(modify_bio: ModifyBioisosteres, fg_type: str):
    def _enum(mol: Chem.Mol) -> List[SiteCandidate]:
        candidates = []
        for idx, match in enumerate(modify_bio.get_valid_bioisostere_matches(mol, fg_type)):
            candidates.append(SiteCandidate(
                site_key={"match_index": idx},
                group_label=fg_type,
                label=f"Instance {idx} (atoms {list(match)})",
                pre_highlight_atoms=list(match),
            ))
        return candidates
    return _enum


def _make_bio_apply(modify_bio: ModifyBioisosteres, fg_type: str, replacement: Optional[str]):
    def _apply(mol: Chem.Mol, site: Optional[dict] = None) -> Optional[str]:
        s = site or {}
        return modify_bio.apply_modification(mol, fg_type, replacement, match_index=s.get("match_index"))
    return _apply


def _fg_instance_sites(modify_fg: ModifyFunctionalGroup, fg_name: str, label_prefix: str):
    def _enum(mol: Chem.Mol) -> List[SiteCandidate]:
        candidates = []
        for instance_idx, match_atoms in modify_fg.get_fg_instances(mol, fg_name):
            candidates.append(SiteCandidate(
                site_key={"instance_idx": instance_idx},
                group_label=fg_name,
                label=f"{label_prefix} instance {instance_idx} (atoms {list(match_atoms)})",
                pre_highlight_atoms=list(match_atoms),
            ))
        return candidates
    return _enum


def _make_fg_remove_apply(modify_fg: ModifyFunctionalGroup, fg: str):
    def _apply(mol: Chem.Mol, site: Optional[dict] = None) -> Optional[str]:
        s = site or {}
        return modify_fg.remove_functional_group(mol, fg, instance_idx=s.get("instance_idx", 0))
    return _apply


def _make_fg_modify_apply(modify_fg: ModifyFunctionalGroup, source_fg: str, target_fg: str):
    def _apply(mol: Chem.Mol, site: Optional[dict] = None) -> Optional[str]:
        s = site or {}
        return modify_fg.modify_functional_group(mol, source_fg, target_fg, instance_idx=s.get("instance_idx", 0))
    return _apply


def _fg_add_sites(modify_fg: ModifyFunctionalGroup, fg_name: str):
    def _enum(mol: Chem.Mol) -> List[SiteCandidate]:
        candidates = []
        for site_idx in modify_fg.functional_group_add_sites(mol, fg_name):
            atom = mol.GetAtomWithIdx(site_idx)
            candidates.append(SiteCandidate(
                site_key={"site_idx": site_idx},
                group_label=fg_name,
                label=f"Attach at atom {site_idx} ({atom.GetSymbol()})",
                pre_highlight_atoms=[site_idx],
            ))
        return candidates
    return _enum


def _make_fg_add_apply(modify_fg: ModifyFunctionalGroup, fg: str):
    def _apply(mol: Chem.Mol, site: Optional[dict] = None) -> Optional[str]:
        s = site or {}
        if "site_idx" in s:
            return modify_fg.add_functional_group(mol, fg, s["site_idx"])
        sites = modify_fg.functional_group_add_sites(mol, fg)
        if not sites:
            return None
        return modify_fg.add_functional_group(mol, fg, sites[0])
    return _apply


def build_edit_catalog(mol_logger=None) -> List[EditOp]:
    modify_atom = ModifyAtom(mol_logger)
    modify_bond = ModifyBond(mol_logger)
    modify_bio = ModifyBioisosteres(mol_logger)
    modify_fg = ModifyFunctionalGroup(mol_logger)

    catalog: List[EditOp] = []

    catalog.append(EditOp(
        id="atom_add", category="atom",
        description="Add a new heavy atom bonded to an existing modification site.",
        apply=_atom_add_apply(modify_atom),
        enumerate_sites=_atom_add_sites(modify_atom),
    ))
    catalog.append(EditOp(
        id="atom_remove", category="atom",
        description="Remove a non-ring (or ring-but-non-carbon) heavy atom.",
        apply=_atom_remove_apply(modify_atom),
        enumerate_sites=_atom_remove_sites(modify_atom),
    ))
    catalog.append(EditOp(
        id="atom_modify", category="atom",
        description="Substitute a heavy atom for a chemically similar element "
                     "(electronegativity/covalent-radius balanced).",
        apply=_atom_modify_apply(modify_atom),
        enumerate_sites=_atom_modify_sites(modify_atom),
    ))

    catalog.append(EditOp(
        id="bond_add", category="bond",
        description="Add a new bond between two currently unbonded, valence-compatible atoms.",
        apply=_make_bond_apply(modify_bond, 0),
        enumerate_sites=_bond_add_sites(modify_bond),
    ))
    catalog.append(EditOp(
        id="bond_remove", category="bond",
        description="Remove an existing bond (subject to ring-strain/valence rules).",
        apply=_make_bond_apply(modify_bond, 1),
        enumerate_sites=_bond_remove_sites(modify_bond),
    ))
    catalog.append(EditOp(
        id="bond_modify", category="bond",
        description="Change an existing bond's order (single/double/triple).",
        apply=_make_bond_apply(modify_bond, 2),
        enumerate_sites=_bond_modify_sites(modify_bond),
    ))

    for fg_type, replacement in _BIOISOSTERE_MODIFICATIONS:
        suffix = f"{fg_type}_to_{replacement}" if replacement else fg_type
        replacement_desc = f"'{replacement}'" if replacement else "its standard bioisostere"
        catalog.append(EditOp(
            id=f"bio_{suffix}", category="bioisostere",
            description=f"Replace a '{fg_type}' group with {replacement_desc}.",
            apply=_make_bio_apply(modify_bio, fg_type, replacement),
            enumerate_sites=_bio_sites(modify_bio, fg_type),
        ))

    # --- Functional-group removal (16) ---
    for fg in _FG_REMOVE_GROUPS:
        catalog.append(EditOp(
            id=f"fg_remove_{fg}", category="functional_group",
            description=f"Remove a '{fg}' functional group.",
            apply=_make_fg_remove_apply(modify_fg, fg),
            enumerate_sites=_fg_instance_sites(modify_fg, fg, "Remove"),
        ))

    for source_fg, target_fg in _FG_MODIFY_PAIRS:
        catalog.append(EditOp(
            id=f"fg_modify_{source_fg}_to_{target_fg}", category="functional_group",
            description=f"Convert a '{source_fg}' group to '{target_fg}'.",
            apply=_make_fg_modify_apply(modify_fg, source_fg, target_fg),
            enumerate_sites=_fg_instance_sites(modify_fg, source_fg, "Convert"),
        ))

    for fg in _FG_ADD_GROUPS:
        catalog.append(EditOp(
            id=f"fg_add_{fg}", category="functional_group",
            description=f"Add a '{fg}' group at an available attachment site.",
            apply=_make_fg_add_apply(modify_fg, fg),
            enumerate_sites=_fg_add_sites(modify_fg, fg),
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

        if len(Chem.GetMolFrags(result_mol, asMols=False)) > 1:
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
