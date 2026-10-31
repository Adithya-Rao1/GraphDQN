"""Exact, RDKit-based explainer implementing contracts.Explainer. No ML, no network."""
from __future__ import annotations

from typing import Dict, List, Optional

from rdkit import Chem

from studio.explain.contributions import atom_contributions
from studio.explain.templates import compose_rationale
from studio.schema import Descriptors, EditRecord, Explanation, ScoreResult

# tag -> (descriptor, threshold on |delta|); sign gives the direction
_THRESHOLDS = {
    "logp": 0.25,
    "tpsa": 5.0,
    "mw": 10.0,
    "qed": 0.02,
    "sa": 0.2,
    "hbd": 1,
    "hba": 1,
    "rotb": 1,
    "fsp3": 0.05,
}


def descriptor_deltas(parent: Descriptors, child: Descriptors) -> Dict[str, float]:
    """child - parent for every numeric Descriptors field (fields None on either side are skipped)."""
    out: Dict[str, float] = {}
    for name in Descriptors.model_fields:
        p, c = getattr(parent, name), getattr(child, name)
        if p is None or c is None:
            continue
        out[name] = float(c) - float(p)
    return out


def derive_tags(deltas: Dict[str, float], n_removed: int, n_added: int) -> List[str]:
    tags: List[str] = []
    for key, thr in _THRESHOLDS.items():
        d = deltas.get(key)
        if d is None or abs(d) < thr:
            continue
        if key == "qed":
            tags.append("improves_qed" if d > 0 else "worsens_qed")
        elif key == "sa":  # lower SA score = easier synthesis
            tags.append("improves_sa" if d < 0 else "worsens_sa")
        else:
            tags.append(("raises_" if d > 0 else "lowers_") + key)
    if n_removed:
        tags.append("removes_alert")
    if n_added:
        tags.append("adds_alert")
    lip = deltas.get("lipinski_violations", 0)
    if lip < 0:
        tags.append("fewer_lipinski_violations")
    elif lip > 0:
        tags.append("more_lipinski_violations")
    return tags


class RDKitExplainer:
    """Alerts are compared by name: indices of removed alerts are on the parent, added on the child."""

    def explain(
        self,
        parent_smiles: str,
        child_smiles: str,
        edit: EditRecord,
        parent_score: ScoreResult,
        child_score: ScoreResult,
        edit_op=None,
    ) -> Explanation:
        deltas = descriptor_deltas(parent_score.descriptors, child_score.descriptors)
        parent_names = {a.name for a in parent_score.alerts}
        child_names = {a.name for a in child_score.alerts}
        removed = [a for a in parent_score.alerts if a.name not in child_names]
        added = [a for a in child_score.alerts if a.name not in parent_names]

        child_mol = Chem.MolFromSmiles(child_smiles)
        contribs = atom_contributions(child_mol) if child_mol is not None else None

        rationale = compose_rationale(
            edit.name,
            deltas,
            _unique([a.name for a in removed]),
            _unique([a.name for a in added]),
            getattr(edit_op, "rationale", None) if edit_op is not None else None,
        )
        return Explanation(
            descriptor_deltas=deltas,
            alerts_added=added,
            alerts_removed=removed,
            atom_contributions=contribs,
            rationale=rationale,
            tags=derive_tags(deltas, len(removed), len(added)),
        )


def _unique(names: List[str]) -> List[str]:
    return list(dict.fromkeys(names))


def explain(*args, **kwargs) -> Explanation:
    return RDKitExplainer().explain(*args, **kwargs)
