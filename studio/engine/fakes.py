"""FAKES for developing and testing the engine before the real components land.

Everything here is a stand-in: ~7 trivial atom-level edits, a QED-only scorer, a uniform policy, a template
explainer and a Morgan+PCA embedder. None of it is the real registry / scorer / explainer / policy; outputs
produced with these must not be presented as such.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence

import numpy as np
from rdkit import Chem
from rdkit.Chem import Crippen, QED, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem import Descriptors as RDDescriptors

from studio.schema import Descriptors, EditRecord, Explanation, RunConfig, ScoreResult


# ----------------------------------------------------------------------------- registry
class FakeEdit:
    """Edit defined by a SMARTS for the site atom (first atom of the match) and an RWMol action."""

    def __init__(self, id: str, name: str, category: str, smarts: str, action: Callable[[Chem.RWMol, int], None],
                 tags: Sequence[str] = (), rationale: str = "FAKE edit (template rationale)."):
        self.id, self.name, self.category, self.tags, self.rationale = id, name, category, list(tags), rationale
        self._patt = Chem.MolFromSmarts(smarts)
        self._action = action

    def sites(self, smiles: str) -> List[List[int]]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []
        return [[m[0]] for m in mol.GetSubstructMatches(self._patt)]

    def apply(self, smiles: str, site: Sequence[int]) -> List[str]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []
        rw = Chem.RWMol(mol)
        try:
            self._action(rw, site[0])
            Chem.SanitizeMol(rw)
            return [Chem.MolToSmiles(rw)]
        except Exception:
            return []


def _attach(atomic_num: int):
    def act(rw: Chem.RWMol, idx: int) -> None:
        new = rw.AddAtom(Chem.Atom(atomic_num))
        rw.AddBond(idx, new, Chem.BondType.SINGLE)
    return act


def _to_nitrogen(rw: Chem.RWMol, idx: int) -> None:
    a = rw.GetAtomWithIdx(idx)
    a.SetAtomicNum(7)
    a.SetNoImplicit(False)
    a.SetNumExplicitHs(0)


def _remove_atom(rw: Chem.RWMol, idx: int) -> None:
    rw.RemoveAtom(idx)


def _methyl_to_hydroxymethyl(rw: Chem.RWMol, idx: int) -> None:
    _attach(8)(rw, idx)


class FakeRegistry:
    def __init__(self) -> None:
        self._edits = [
            FakeEdit("fake_add_methyl", "Add methyl (aromatic CH)", "atom", "[cH]", _attach(6), ["adds_lipophilicity"]),
            FakeEdit("fake_add_hydroxyl", "Add hydroxyl (aromatic CH)", "functional_group", "[cH]", _attach(8), ["lowers_logp"]),
            FakeEdit("fake_add_fluoro", "Add fluorine (aromatic CH)", "atom", "[cH]", _attach(9), ["blocks_metabolism"]),
            FakeEdit("fake_ch_to_n", "Ring CH -> N", "bioisostere", "[cH]", _to_nitrogen, ["lowers_logp"]),
            FakeEdit("fake_remove_methyl", "Remove terminal methyl", "atom", "[CH3;D1]", _remove_atom, ["lowers_mw"]),
            FakeEdit("fake_remove_halogen", "Remove halogen", "atom", "[F,Cl,Br;D1]", _remove_atom, ["lowers_mw"]),
            FakeEdit("fake_hydroxylate_methyl", "Methyl -> hydroxymethyl", "functional_group", "[CH3;D1]",
                     _methyl_to_hydroxymethyl, ["lowers_logp"]),
        ]

    def all(self):
        return list(self._edits)

    def get(self, edit_id: str):
        return next(e for e in self._edits if e.id == edit_id)

    def applicable(self, smiles: str, locked_atoms: Sequence[int] = ()):
        locked = set(locked_atoms)
        return [(e, s) for e in self._edits for s in e.sites(smiles) if not locked.intersection(s)]


# ----------------------------------------------------------------------------- scorer
def _descriptors(mol: Chem.Mol) -> Descriptors:
    lip = (int(RDDescriptors.MolWt(mol) > 500) + int(Crippen.MolLogP(mol) > 5)
           + int(rdMolDescriptors.CalcNumHBD(mol) > 5) + int(rdMolDescriptors.CalcNumHBA(mol) > 10))
    return Descriptors(
        qed=QED.qed(mol), sa=None, logp=Crippen.MolLogP(mol), tpsa=rdMolDescriptors.CalcTPSA(mol),
        mw=RDDescriptors.MolWt(mol), hbd=rdMolDescriptors.CalcNumHBD(mol), hba=rdMolDescriptors.CalcNumHBA(mol),
        rotb=rdMolDescriptors.CalcNumRotatableBonds(mol), fsp3=rdMolDescriptors.CalcFractionCSP3(mol),
        heavy_atoms=mol.GetNumHeavyAtoms(), lipinski_violations=lip, alerts=0)  # FAKE: no alert catalogue


class FakeScorer:
    """QED-only composite (FAKE: ignores config.objectives and structural alerts)."""

    def score(self, smiles: str, config: RunConfig) -> ScoreResult:
        d = _descriptors(Chem.MolFromSmiles(smiles))
        return ScoreResult(descriptors=d, components={"qed": d.qed}, composite=d.qed, alerts=[])

    def score_many(self, smiles: Sequence[str], config: RunConfig) -> List[ScoreResult]:
        return [self.score(s, config) for s in smiles]


# ----------------------------------------------------------------------------- policy
class FakePolicy:
    """Uniform policy: all candidates equally preferred. update() counts transitions and returns no loss."""
    note = "FAKE uniform policy (no learning)"

    def __init__(self) -> None:
        self.updates: List[tuple] = []

    def rank(self, smiles: str, candidates: Sequence[tuple]) -> List[float]:
        return [0.0] * len(candidates)

    def update(self, transitions: Sequence[tuple]) -> Optional[float]:
        self.updates.append(tuple(transitions))
        return None


class PreferencePolicy(FakePolicy):
    """Test helper: strongly prefers one edit id and reports a deterministic fake loss."""
    note = "FAKE preference policy (test helper)"

    def __init__(self, preferred_edit_id: str):
        super().__init__()
        self.preferred = preferred_edit_id

    def rank(self, smiles, candidates):
        return [8.0 if op.id == self.preferred else 0.0 for op, _ in candidates]

    def update(self, transitions):
        super().update(transitions)
        return 1.0 / (1 + len(self.updates))


# ----------------------------------------------------------------------------- explainer
class FakeExplainer:
    """Template explanations from descriptor deltas (FAKE: no alerts, no atom contributions)."""

    def explain(self, parent_smiles, child_smiles, edit: EditRecord, parent_score: ScoreResult,
                child_score: ScoreResult, edit_op=None) -> Explanation:
        p, c = parent_score.descriptors, child_score.descriptors
        deltas = {k: float(getattr(c, k) - getattr(p, k)) for k in ("qed", "logp", "tpsa", "mw", "hbd", "hba", "alerts")}
        tags = [t for t in (edit_op.tags if edit_op is not None else [])]
        return Explanation(
            descriptor_deltas=deltas, rationale=f"[FAKE] {edit.name}: QED {deltas['qed']:+.3f}, logP {deltas['logp']:+.2f}",
            tags=list(tags))


# ----------------------------------------------------------------------------- embedder
class FakeEmbedder:
    """Morgan fingerprints -> top-2 principal components via SVD (FAKE-light: labelled morgan_pca)."""
    method = "morgan_pca"

    def map_xy(self, smiles: Sequence[str], anchor: Optional[str] = None) -> List[tuple]:
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
        X = np.array([gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles], dtype=float)
        if len(X) < 2:
            return [(0.0, 0.0)] * len(X)
        X -= X.mean(axis=0)
        u, s, _ = np.linalg.svd(X, full_matrices=False)
        xy = u[:, :2] * s[:2]
        if xy.shape[1] < 2:
            xy = np.hstack([xy, np.zeros((len(xy), 1))])
        return [(float(a), float(b)) for a, b in xy]


def make_fake_engine(**kwargs):
    from studio.engine.engine import Engine
    return Engine(FakeRegistry(), FakeScorer(), FakeExplainer(), FakeEmbedder(), FakePolicy(), **kwargs)
