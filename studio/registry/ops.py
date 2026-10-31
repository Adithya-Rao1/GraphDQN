"""EditOp implementations wrapping `molecular_modifications/` (which stays untouched).

Conventions
- `sites(smiles)` returns raw structural candidates (cheap, cached per SMILES). A listed site is
  *plausible*, not guaranteed to produce a valid product; `apply` is the arbiter.
- `apply(smiles, site)` never raises; returns [] or a single canonical, single-fragment product
  that differs from the parent. Deterministic: every wrapped call passes explicit sites/targets,
  so none of the wrapped classes' `random` paths are reached.
- `_raw(smiles, site)` returns the wrapped class's raw output (str or None) for the self-check.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from rdkit import Chem, RDLogger

from molecular_modifications.atom_optimization import ModifyAtom
from molecular_modifications.bioisosteres_optimization import ModifyBioisosteres
from molecular_modifications.bond_optimization import ModifyBond
from molecular_modifications.chemistry_constants import find_unsupported_elements
from molecular_modifications.functional_group_optimization import (
    FG_FRAGMENT_SMILES,
    ModifyFunctionalGroup,
)

RDLogger.DisableLog("rdApp.*")

_LOG = logging.getLogger("studio.registry")
_ATOM = ModifyAtom(_LOG)
_BOND = ModifyBond(_LOG)
_BIO = ModifyBioisosteres(_LOG)
_FG = ModifyFunctionalGroup(_LOG)

MAX_HEAVY_ATOMS = 120  # guard against pathological inputs, not a chemistry rule
# Bond-add sites are limited to atoms 3..7 bonds apart (new ring of 4..8 atoms). Registry policy:
# the wrapped class would also join atoms 1-2 bonds apart (3-membered rings) or far apart
# (macrocycles / cage bridges), which are rarely sensible med-chem edits.
BOND_ADD_MIN_DIST, BOND_ADD_MAX_DIST = 3, 7

_BOND_TYPES = {
    "single": Chem.BondType.SINGLE,
    "double": Chem.BondType.DOUBLE,
    "triple": Chem.BondType.TRIPLE,
    "aromatic": Chem.BondType.AROMATIC,
}


# ---------------------------------------------------------------- helpers
@lru_cache(maxsize=4096)
def _parse(smiles: str) -> Optional[Chem.Mol]:
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception:
        return None
    if mol is None or mol.GetNumHeavyAtoms() == 0 or mol.GetNumHeavyAtoms() > MAX_HEAVY_ATOMS:
        return None
    if find_unsupported_elements(mol):
        return None
    return mol


def canonical(smiles: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    return Chem.MolToSmiles(mol) if mol is not None else None


def _validate_product(parent: str, product: Optional[str]) -> Optional[str]:
    """Canonical product SMILES if it is a valid, connected, supported, changed molecule."""
    if not product:
        return None
    try:
        mol = Chem.MolFromSmiles(product)
        if mol is None or mol.GetNumHeavyAtoms() == 0:
            return None
        if len(Chem.GetMolFrags(mol)) != 1 or find_unsupported_elements(mol):
            return None
        can = Chem.MolToSmiles(mol)
        parent_can = canonical(parent)
        return None if can == parent_can else can
    except Exception:
        return None


def _mol_to_smiles(mol: Optional[Chem.Mol]) -> Optional[str]:
    return Chem.MolToSmiles(mol) if mol is not None else None


def _dedupe(sites: List[List[int]]) -> List[List[int]]:
    seen, out = set(), []
    for s in sites:
        key = tuple(s)
        if key not in seen:
            seen.add(key)
            out.append(list(s))
    return out


# ---------------------------------------------------------------- base
class _BaseOp:
    """Common behaviour. Subclasses implement `_sites(mol)` and `_raw_apply(mol, site)`."""

    def __init__(self, id: str, name: str, category: str, tags: Sequence[str] = (), rationale: str = ""):
        self.id, self.name, self.category = id, name, category
        self.tags = list(tags)
        self.rationale = rationale
        self.rationale_reviewed = False

    def __repr__(self) -> str:
        return f"<EditOp {self.id}>"

    def sites(self, smiles: str) -> List[List[int]]:
        mol = _parse(smiles) if isinstance(smiles, str) else None
        if mol is None:
            return []
        try:
            return _dedupe([list(map(int, s)) for s in self._sites(mol)])
        except Exception:
            return []

    def _raw(self, smiles: str, site: Sequence[int]) -> Optional[str]:
        mol = _parse(smiles) if isinstance(smiles, str) else None
        if mol is None:
            return None
        try:
            site = [int(i) for i in site]
            if not site or any(i < 0 or i >= mol.GetNumAtoms() for i in site):
                return None
            return self._raw_apply(Chem.Mol(mol), site)
        except Exception:
            return None

    def apply(self, smiles: str, site: Sequence[int]) -> List[str]:
        out = _validate_product(smiles, self._raw(smiles, site))
        return [out] if out else []

    def _sites(self, mol: Chem.Mol) -> List[List[int]]:  # pragma: no cover
        raise NotImplementedError

    def _raw_apply(self, mol: Chem.Mol, site: List[int]) -> Optional[str]:  # pragma: no cover
        raise NotImplementedError


# ---------------------------------------------------------------- atom ops
class AtomSubstituteOp(_BaseOp):
    """Replace one non-ring-carbon atom by `element` (ModifyAtom.modify_atom)."""

    def __init__(self, element: str, **kw):
        super().__init__(id=f"atom_sub_{element.lower()}", name=f"Atom swap to {element}", category="atom", **kw)
        self.element = element

    def _sites(self, mol):
        return [[i] for i in _atom_element_candidates(mol).get(self.element, [])]

    def _raw_apply(self, mol, site):
        return _ATOM.modify_atom(mol, 0, atom_idx=site[0], target_element=self.element)


class AtomAddOp(_BaseOp):
    """Attach a new `element` atom to an atom (ModifyAtom.add_atom)."""

    def __init__(self, element: str, **kw):
        super().__init__(id=f"atom_add_{element.lower()}", name=f"Add {element} atom", category="atom", **kw)
        self.element = element

    def _sites(self, mol):
        # ModifyAtom validates the *replacement* of the anchor by `element`, not the addition,
        # so this list is a superset of what will succeed; apply() decides.
        return [[i] for i in _atom_element_candidates(mol).get(self.element, [])]

    def _raw_apply(self, mol, site):
        return _ATOM.add_atom(mol, 0, atom_idx=site[0], target_element=self.element)


class AtomRemoveOp(_BaseOp):
    """Delete one non-ring-carbon atom (ModifyAtom.remove_atom); rejects disconnecting removals."""

    def __init__(self, **kw):
        super().__init__(id="atom_remove", name="Remove atom", category="atom", **kw)

    def _sites(self, mol):
        return [[i] for i in _ATOM.get_valid_atom_sites(mol)]

    def _raw_apply(self, mol, site):
        return _ATOM.remove_atom(mol, 0, atom_idx=site[0])


@lru_cache(maxsize=512)
def _atom_element_candidates_cached(smiles: str) -> Dict[str, Tuple[int, ...]]:
    mol = Chem.MolFromSmiles(smiles)
    out: Dict[str, List[int]] = {}
    for idx, el in _ATOM.enumerate_site_element_candidates(mol):
        out.setdefault(el, []).append(idx)
    return {k: tuple(v) for k, v in out.items()}


def _atom_element_candidates(mol: Chem.Mol) -> Dict[str, Tuple[int, ...]]:
    return _atom_element_candidates_cached(Chem.MolToSmiles(mol))


# ---------------------------------------------------------------- bond ops
@lru_cache(maxsize=512)
def _bond_sites_cached(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    dm = Chem.GetDistanceMatrix(mol)
    add = [
        (i, j, bt)
        for i, j, bt in _BOND.get_valid_bond_add_sites(mol)
        if BOND_ADD_MIN_DIST <= dm[i][j] <= BOND_ADD_MAX_DIST
    ]
    return add, list(_BOND.get_valid_bond_remove_sites(mol)), list(_BOND.get_valid_bond_modify_sites(mol))


class BondAddOp(_BaseOp):
    def __init__(self, bond_type: str, **kw):
        super().__init__(id=f"bond_add_{bond_type}", name=f"Add {bond_type} bond (ring closure)", category="bond", **kw)
        self.bond_type = bond_type

    def _sites(self, mol):
        bt = _BOND_TYPES[self.bond_type]
        return [[i, j] for i, j, t in _bond_sites_cached(Chem.MolToSmiles(mol))[0] if t == bt]

    def _raw_apply(self, mol, site):
        if len(site) != 2:
            return None
        return _BOND.optimize_bond(mol, 0, bond_site=tuple(site), target_bond_type=_BOND_TYPES[self.bond_type])


class BondRemoveOp(_BaseOp):
    def __init__(self, **kw):
        super().__init__(id="bond_remove", name="Remove bond (ring opening)", category="bond", **kw)

    def _sites(self, mol):
        return [[i, j] for i, j in _bond_sites_cached(Chem.MolToSmiles(mol))[1]]

    def _raw_apply(self, mol, site):
        if len(site) != 2:
            return None
        return _BOND.optimize_bond(mol, 1, bond_site=tuple(site))


class BondModifyOp(_BaseOp):
    def __init__(self, bond_type: str, **kw):
        super().__init__(id=f"bond_to_{bond_type}", name=f"Change bond order to {bond_type}", category="bond", **kw)
        self.bond_type = bond_type

    def _sites(self, mol):
        bt = _BOND_TYPES[self.bond_type]
        return [[i, j] for i, j, t in _bond_sites_cached(Chem.MolToSmiles(mol))[2] if t == bt]

    def _raw_apply(self, mol, site):
        if len(site) != 2:
            return None
        return _BOND.optimize_bond(mol, 2, bond_site=tuple(site), target_bond_type=_BOND_TYPES[self.bond_type])


# ---------------------------------------------------------------- bioisostere ops
class BioisostereOp(_BaseOp):
    """ModifyBioisosteres.apply_modification(fg, replacement). Site = matched atoms of the group."""

    def __init__(self, id: str, name: str, fg: str, replacement: Optional[str], **kw):
        super().__init__(id=id, name=name, category="bioisostere", **kw)
        self.fg, self.replacement = fg, replacement

    def _matches(self, mol) -> List[Tuple[int, ...]]:
        return list(_BIO.get_valid_bioisostere_matches(mol, self.fg))

    def _sites(self, mol):
        out = []
        for m in self._matches(mol):
            site = list(m)
            if self.replacement == "retroamide":
                # retroamide also rewires the amide N's external substituent
                n_ext = [n.GetIdx() for n in mol.GetAtomWithIdx(m[0]).GetNeighbors() if n.GetIdx() != m[1]]
                if len(n_ext) != 1:
                    continue
                site.append(n_ext[0])
            out.append(site)
        return out

    def _raw_apply(self, mol, site):
        for k, m in enumerate(self._matches(mol)):
            if site[: len(m)] == list(m):
                return _BIO.apply_modification(mol, self.fg, self.replacement, match_index=k)
        return None


# ---------------------------------------------------------------- functional-group ops
def _fg_instance_sites(mol: Chem.Mol, fg: str) -> List[List[int]]:
    """[match atoms..., attachment atom] for each singly-attached instance, in class order."""
    out = []
    for inst_idx, match_atoms in _FG.get_fg_instances(mol, fg):
        res = _FG._match_single_attachment(mol, fg, inst_idx)
        if res is not None:
            out.append(list(match_atoms) + [res[1]])
    return out


def _fg_instance_index(mol: Chem.Mol, fg: str, site: Sequence[int]) -> Optional[int]:
    for inst_idx, match_atoms in _FG.get_fg_instances(mol, fg):
        res = _FG._match_single_attachment(mol, fg, inst_idx)
        if res is not None and set(site) == set(match_atoms) | {res[1]}:
            return inst_idx
    return None


class FGRemoveOp(_BaseOp):
    def __init__(self, fg: str, **kw):
        super().__init__(id=f"fg_remove_{fg}", name=f"Remove {fg}", category="functional_group", **kw)
        self.fg = fg

    def _sites(self, mol):
        return _fg_instance_sites(mol, self.fg)

    def _raw_apply(self, mol, site):
        k = _fg_instance_index(mol, self.fg, site)
        return None if k is None else _FG.remove_functional_group(mol, self.fg, k)


class FGModifyOp(_BaseOp):
    def __init__(self, source: str, target: str, **kw):
        super().__init__(id=f"fg_{source}_to_{target}", name=f"{source} -> {target}", category="functional_group", **kw)
        self.source, self.target = source, target

    def _sites(self, mol):
        return _fg_instance_sites(mol, self.source)

    def _raw_apply(self, mol, site):
        k = _fg_instance_index(mol, self.source, site)
        return None if k is None else _FG.modify_functional_group(mol, self.source, self.target, k)


class FGAddOp(_BaseOp):
    """Attach a fragment (FG_FRAGMENT_SMILES) at an atom that carries >= 1 hydrogen."""

    def __init__(self, fg: str, **kw):
        super().__init__(id=f"fg_add_{fg}", name=f"Add {fg}", category="functional_group", **kw)
        self.fg = fg

    def _sites(self, mol):
        return [[i] for i in _FG.functional_group_add_sites(mol, self.fg)]

    def _raw_apply(self, mol, site):
        return _FG.add_functional_group(mol, self.fg, site[0])


# ---------------------------------------------------------------- catalogue
ATOM_ELEMENTS = list(_ATOM.substitution_atoms)  # C N O S P F Cl Br I B Si

# Subset of ModifyFunctionalGroup used by dqn/dqn_env.py (remove 16, modify 15, add 6), plus
# registry extras for add (ether/methoxy, ketone/acetyl, amide, sulfonamide, sulfone) whose
# fragments already exist in FG_FRAGMENT_SMILES.
FG_REMOVE = [
    "methyl", "hydroxyl", "amino", "carboxyl", "carbonyl", "aldehyde", "ketone", "ether",
    "ester", "amide", "nitro", "cyano", "thiol", "halogen", "azide", "sulfonamide",
]
FG_MODIFY = [
    ("hydroxyl", "amino"), ("hydroxyl", "thiol"), ("hydroxyl", "methyl"), ("hydroxyl", "halogen"),
    ("amino", "hydroxyl"), ("amino", "amide"), ("carboxyl", "ester"), ("carboxyl", "amide"),
    ("aldehyde", "ketone"), ("aldehyde", "carboxyl"), ("thiol", "hydroxyl"), ("cyano", "carboxyl"),
    ("nitro", "amino"), ("ester", "carboxyl"), ("amide", "carboxyl"),
]
FG_ADD_DQN = ["methyl", "hydroxyl", "amino", "halogen", "cyano", "carboxyl"]
FG_ADD_EXTRA = ["ether", "ketone", "amide", "sulfonamide", "sulfone"]

BIOISOSTERES = [
    ("bio_acid_to_tetrazole", "Carboxylic acid -> tetrazole", "carboxylic_acid", "tetrazole"),
    ("bio_acid_to_phosphonic", "Carboxylic acid -> phosphonic acid", "carboxylic_acid", "phosphonic_acid"),
    ("bio_amine_to_pyridyl", "Primary/secondary amine -> pyridyl", "amine", None),
    ("bio_amide_to_sulfonamide", "Amide -> sulfonamide", "amide", "sulfonamide"),
    ("bio_amide_to_retroamide", "Amide -> retro-amide", "amide", "retroamide"),
    ("bio_phenyl_to_pyridyl", "Phenyl -> pyridyl", "phenyl", "pyridyl"),
    ("bio_phenyl_to_thiophene", "Phenyl -> thiophene", "phenyl", "thiophene"),
    ("bio_phenyl_to_furan", "Phenyl -> furan", "phenyl", "furan"),
    ("bio_phenyl_to_pyrrole", "Phenyl -> pyrrole", "phenyl", "pyrrole"),
]


def build_ops() -> List[_BaseOp]:
    ops: List[_BaseOp] = []
    ops += [AtomSubstituteOp(el) for el in ATOM_ELEMENTS]
    ops += [AtomAddOp(el) for el in ATOM_ELEMENTS]
    ops.append(AtomRemoveOp())
    ops += [BondAddOp(t) for t in _BOND_TYPES]
    ops.append(BondRemoveOp())
    ops += [BondModifyOp(t) for t in _BOND_TYPES]
    ops += [BioisostereOp(i, n, fg, rep) for i, n, fg, rep in BIOISOSTERES]
    ops += [FGRemoveOp(fg) for fg in FG_REMOVE]
    ops += [FGModifyOp(s, t) for s, t in FG_MODIFY]
    ops += [FGAddOp(fg) for fg in FG_ADD_DQN + FG_ADD_EXTRA]
    for fg in FG_ADD_DQN + FG_ADD_EXTRA:
        assert fg in FG_FRAGMENT_SMILES
    return ops
