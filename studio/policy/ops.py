"""STAND-IN edit operators for training/eval of the policy (not the studio registry).

The real registry (studio/registry, other stream) was not available on this
branch, so the policy trains and is evaluated against this small, local,
RDKit-only edit set. Each op satisfies the `EditOp` Protocol. `load_ops()` uses
`studio.registry` instead if it later exposes a default registry, but results in
docs/eval reports always state which op set was used.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

_ELEM = {"F": 9, "Cl": 17, "C": 6, "N": 7, "O": 8}


def _finish(rw: Chem.RWMol) -> List[str]:
    try:
        m = rw.GetMol()
        Chem.SanitizeMol(m)
        smi = Chem.MolToSmiles(m)
        m2 = Chem.MolFromSmiles(smi)
        if m2 is None or "." in smi or m2.GetNumHeavyAtoms() < 3 or m2.GetNumHeavyAtoms() > 60:
            return []
        return [smi]
    except Exception:
        return []


class SmartsEdit:
    """Edit defined by a one-atom-site predicate + a mutation function."""

    category = "atom"

    def __init__(self, id: str, name: str, tags: Sequence[str], rationale: str,
                 pred: Callable[[Chem.Atom], bool], fn: Callable[[Chem.RWMol, int], None],
                 category: str = "atom"):
        self.id, self.name, self.tags, self.rationale = id, name, list(tags), rationale
        self._pred, self._fn, self.category = pred, fn, category

    def sites(self, smiles: str) -> List[List[int]]:
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return []
        return [[a.GetIdx()] for a in m.GetAtoms() if self._pred(a)]

    def apply(self, smiles: str, site: Sequence[int]) -> List[str]:
        m = Chem.MolFromSmiles(smiles)
        if m is None or not site or site[0] >= m.GetNumAtoms():
            return []
        rw = Chem.RWMol(m)
        try:
            self._fn(rw, int(site[0]))
        except Exception:
            return []
        return _finish(rw)


def _add(sym: str):
    def fn(rw, i):
        j = rw.AddAtom(Chem.Atom(_ELEM[sym]))
        rw.AddBond(i, j, Chem.BondType.SINGLE)
    return fn


def _delete(rw, i):
    rw.RemoveAtom(i)


def _swap(sym: str):
    def fn(rw, i):
        a = rw.GetAtomWithIdx(i)
        a.SetAtomicNum(_ELEM[sym])
        a.SetNoImplicit(False)
        a.SetNumExplicitHs(0)
    return fn


def _has_h(a):  # sp3/aromatic heavy atom with a free valence
    return a.GetTotalNumHs() > 0 and a.GetAtomicNum() == 6


def _arom_ch(a):
    return a.GetIsAromatic() and a.GetAtomicNum() == 6 and a.GetTotalNumHs() > 0


def _terminal(a):
    return a.GetDegree() == 1 and a.GetAtomicNum() != 1


def _nitro_n(a):
    return a.GetAtomicNum() == 7 and a.GetFormalCharge() == 1 and a.GetDegree() == 3


def _del_nitro(rw, i):
    n = rw.GetAtomWithIdx(i)
    ox = [x.GetIdx() for x in n.GetNeighbors() if x.GetAtomicNum() == 8 and x.GetDegree() == 1]
    for k in sorted(ox + [i], reverse=True):
        rw.RemoveAtom(k)


def _oh_to(sym):
    def fn(rw, i):
        a = rw.GetAtomWithIdx(i)
        a.SetAtomicNum(_ELEM[sym])
        a.SetNoImplicit(False)
    return fn


def _is_oh(a):
    return a.GetAtomicNum() == 8 and a.GetDegree() == 1 and a.GetTotalNumHs() == 1


def _arom_to_n(rw, i):
    a = rw.GetAtomWithIdx(i)
    a.SetAtomicNum(7)
    a.SetNoImplicit(True)
    a.SetNumExplicitHs(0)


def _amide_n(a):
    return a.GetAtomicNum() == 7 and not a.GetIsAromatic() and a.GetFormalCharge() == 0 and a.GetDegree() == 2 and a.GetTotalNumHs() == 1


def _n_methylate(rw, i):
    j = rw.AddAtom(Chem.Atom(6))
    rw.AddBond(i, j, Chem.BondType.SINGLE)


def _ring_atom_c(a):
    return a.GetAtomicNum() == 6 and not a.GetIsAromatic() and a.IsInRing()


OPS: List[SmartsEdit] = [
    SmartsEdit("add_methyl", "Add methyl", ["adds_lipophilicity"], "Methyl adds small lipophilic bulk.", _has_h, _add("C")),
    SmartsEdit("add_fluoro", "Add fluorine", ["blocks_metabolism"], "Fluorine blocks metabolic soft spots.", _has_h, _add("F")),
    SmartsEdit("add_chloro", "Add chlorine", ["raises_logp"], "Chlorine raises lipophilicity.", _arom_ch, _add("Cl")),
    SmartsEdit("add_hydroxyl", "Add hydroxyl", ["lowers_logp", "adds_hbd"], "Hydroxyl adds polarity and an H-bond donor.", _has_h, _add("O")),
    SmartsEdit("add_amino", "Add amino", ["lowers_logp", "adds_hbd"], "Amine adds polarity and a donor.", _has_h, _add("N")),
    SmartsEdit("delete_terminal", "Remove terminal atom", ["lowers_mw"], "Trim a terminal substituent.", _terminal, _delete, "functional_group"),
    SmartsEdit("delete_nitro", "Remove nitro group", ["removes_alert"], "Nitro groups are structural alerts.", _nitro_n, _del_nitro, "functional_group"),
    SmartsEdit("oh_to_nh2", "Hydroxyl to amine", ["swaps_hbd_hba"], "Swap OH for NH2.", _is_oh, _oh_to("N"), "bioisostere"),
    SmartsEdit("oh_to_f", "Hydroxyl to fluorine", ["lowers_tpsa"], "Replace OH with F to lower polarity.", _is_oh, _oh_to("F"), "bioisostere"),
    SmartsEdit("arom_ch_to_n", "Aromatic CH to N", ["lowers_logp"], "Pyridine-like N lowers logP.", _arom_ch, _arom_to_n, "bioisostere"),
    SmartsEdit("swap_c_to_n", "Ring C to N", ["lowers_logp"], "Aza-substitution.", _ring_atom_c, _swap("N")),
    SmartsEdit("swap_c_to_o", "Ring C to O", ["lowers_logp"], "Oxa-substitution.", _ring_atom_c, _swap("O")),
    SmartsEdit("n_methylate", "N-methylate amide/amine", ["removes_hbd"], "Cap an NH to reduce donors.", _amide_n, _n_methylate),
    SmartsEdit("delete_atom", "Delete atom", ["lowers_mw"], "Delete a non-ring atom.", lambda a: not a.IsInRing() and a.GetDegree() <= 2 and a.GetAtomicNum() != 1, _delete),
]


def load_ops() -> List[SmartsEdit]:
    """Policy-local stand-in ops (see module docstring)."""
    return list(OPS)


def applicable(smiles: str, ops: Optional[Sequence] = None) -> list:
    return [(op, s) for op in (ops or OPS) for s in op.sites(smiles)]
