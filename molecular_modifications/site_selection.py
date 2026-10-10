from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

from rdkit import Chem


@dataclass(frozen=True)
class SiteCandidate:
    site_key: dict
    group_label: str
    label: str
    pre_highlight_atoms: List[int] = field(default_factory=list)


@dataclass(frozen=True)
class HighlightedEditResult:
    resulting_smiles: str
    pre_highlight_atoms: List[int]
    post_highlight_atoms: List[int]


def apply_with_highlights(
    apply_fn: Callable[..., Optional[str]],
    mol: Chem.Mol,
    pre_highlight_atoms: Sequence[int],
    **kwargs,
) -> Optional[HighlightedEditResult]:
    tagged = Chem.Mol(mol)
    for atom in tagged.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)

    tagged_result_smiles = apply_fn(tagged, **kwargs)
    if not tagged_result_smiles:
        return None

    tagged_result_mol = Chem.MolFromSmiles(tagged_result_smiles)
    if tagged_result_mol is None:
        return None

    pre_set = set(pre_highlight_atoms)
    post_highlight_atoms = []
    for atom in tagged_result_mol.GetAtoms():
        map_num = atom.GetAtomMapNum()
        if map_num == 0 or (map_num - 1) in pre_set:
            post_highlight_atoms.append(atom.GetIdx())

    clean_mol = Chem.Mol(tagged_result_mol)
    for atom in clean_mol.GetAtoms():
        atom.SetAtomMapNum(0)
    resulting_smiles = Chem.MolToSmiles(clean_mol)

    return HighlightedEditResult(
        resulting_smiles=resulting_smiles,
        pre_highlight_atoms=sorted(pre_set),
        post_highlight_atoms=sorted(set(post_highlight_atoms)),
    )
