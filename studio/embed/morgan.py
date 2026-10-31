"""Morgan fingerprint + PCA(2) chemical-space map (method="morgan_pca").

The map is a visual aid only. It is stable within a run: the PCA is refit on the
union of all molecules seen so far, then Procrustes-aligned (rotation/reflection
only, no scaling) to the previous projection of the shared molecules so points
do not flip between generations.
"""
from __future__ import annotations

import threading
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

N_BITS = 2048
RADIUS = 2

_gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)


def canon(smiles: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    return Chem.MolToSmiles(mol) if mol is not None else None


def morgan_bitvect(smiles: str):
    """RDKit ExplicitBitVect, or None if the SMILES does not parse."""
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    return _gen.GetFingerprint(mol) if mol is not None else None


def tanimoto(a: str, b: str) -> float:
    """Morgan(2, 2048) Tanimoto similarity; 0.0 if either SMILES is invalid."""
    fa, fb = morgan_bitvect(a), morgan_bitvect(b)
    if fa is None or fb is None:
        return 0.0
    return float(DataStructs.TanimotoSimilarity(fa, fb))


def _fp_array(smiles: str) -> np.ndarray:
    fp = morgan_bitvect(smiles)
    arr = np.zeros(N_BITS, dtype=np.float32)
    if fp is not None:
        DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def pca2(x: np.ndarray) -> np.ndarray:
    """Deterministic PCA(2) via SVD with sign convention (largest |loading| positive)."""
    n = x.shape[0]
    if n == 0:
        return np.zeros((0, 2))
    xc = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(xc.astype(np.float64), full_matrices=False)
    comps = np.zeros((2, x.shape[1]))
    k = min(2, vt.shape[0])
    comps[:k] = vt[:k]
    for i in range(k):
        j = np.argmax(np.abs(comps[i]))
        if comps[i, j] < 0:
            comps[i] = -comps[i]
    return xc @ comps.T


def procrustes_rotation(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Orthogonal R (2x2, may reflect) minimising ||src @ R - dst||."""
    u, _, vt = np.linalg.svd(src.T @ dst)
    return u @ vt


class StableProjector:
    """Shared by Morgan and ChemBERTa embedders: incremental, orientation-stable PCA(2)."""

    def __init__(self) -> None:
        self._feats: Dict[str, np.ndarray] = {}
        self._prev: Dict[str, np.ndarray] = {}
        self._lock = threading.Lock()

    def project(self, keys: Sequence[str], feats: Dict[str, np.ndarray]) -> np.ndarray:
        """keys: canonical SMILES (may repeat). Returns (len(keys), 2)."""
        with self._lock:
            for k in keys:
                if k not in self._feats:
                    self._feats[k] = feats[k]
            order = sorted(self._feats)  # sorted -> independent of call order
            x = np.stack([self._feats[k] for k in order])
            xy = pca2(x)
            shared = [i for i, k in enumerate(order) if k in self._prev]
            if len(shared) >= 2:
                src = xy[shared]
                dst = np.stack([self._prev[order[i]] for i in shared])
                c_src, c_dst = src.mean(0), dst.mean(0)
                r = procrustes_rotation(src - c_src, dst - c_dst)
                xy = (xy - c_src) @ r + c_dst
            self._prev = {k: xy[i] for i, k in enumerate(order)}
            idx = {k: i for i, k in enumerate(order)}
            return np.stack([xy[idx[k]] for k in keys]) if keys else np.zeros((0, 2))


def _map(projector: StableProjector, smiles: Sequence[str], anchor: Optional[str], featurise) -> List[Tuple[float, float]]:
    """Common index-aligned map; unparsable SMILES get (0.0, 0.0) rather than raising."""
    smiles = list(smiles)
    canon_list = [canon(s) for s in smiles]
    keys = [c for c in canon_list if c is not None]
    anchor_c = canon(anchor) if anchor else None
    extra = [anchor_c] if anchor_c else []
    all_keys = list(dict.fromkeys(extra + keys))
    if not all_keys:
        return [(0.0, 0.0)] * len(smiles)
    feats = featurise(all_keys)
    xy = projector.project(all_keys, feats)
    pos = {k: xy[i] for i, k in enumerate(all_keys)}
    if anchor_c:  # anchor sits at the origin so "distance from seed" reads off the map
        off = pos[anchor_c]
        pos = {k: v - off for k, v in pos.items()}
    return [tuple(float(t) for t in pos[c]) if c is not None else (0.0, 0.0) for c in canon_list]


class MorganEmbedder:
    method = "morgan_pca"

    def __init__(self) -> None:
        RDLogger.DisableLog("rdApp.*")
        self._proj = StableProjector()

    def features(self, keys: Sequence[str]) -> Dict[str, np.ndarray]:
        return {k: _fp_array(k) for k in keys}

    def map_xy(self, smiles: Sequence[str], anchor: Optional[str] = None) -> List[tuple]:
        return _map(self._proj, smiles, anchor, self.features)
