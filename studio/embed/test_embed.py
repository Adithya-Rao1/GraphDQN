import builtins
import time

import numpy as np
import pytest

from studio.embed import distance_from_seed, get_embedder
from studio.embed.morgan import MorganEmbedder, tanimoto

SEED = "CCOC(=O)c1ccc([N+](=O)[O-])cc1"
MOLS = [SEED, "CCOC(=O)c1ccc(N)cc1", "CCOC(=O)c1ccc(O)cc1", "c1ccccc1", "CC(=O)Nc1ccc(O)cc1", "CCN(CC)CC"]


def test_index_aligned_and_bad_smiles():
    e = MorganEmbedder()
    xy = e.map_xy(MOLS + ["not_a_smiles"], anchor=SEED)
    assert len(xy) == len(MOLS) + 1
    assert xy[0] == (0.0, 0.0) and xy[-1] == (0.0, 0.0)  # anchor at origin; invalid -> origin
    assert xy[1] != xy[2]


def test_duplicates_share_coordinates():
    xy = MorganEmbedder().map_xy([MOLS[1], MOLS[2], MOLS[1]])
    assert xy[0] == xy[2]


def test_deterministic():
    a = MorganEmbedder().map_xy(MOLS, anchor=SEED)
    b = MorganEmbedder().map_xy(MOLS, anchor=SEED)
    assert np.allclose(a, b)
    # independent of input order
    c = MorganEmbedder().map_xy(list(reversed(MOLS)), anchor=SEED)
    assert np.allclose(a, c[::-1], atol=1e-6)


def test_stable_across_calls():
    # Procrustes alignment keeps the map from flipping/rotating as the population grows.
    # (PCA axes can still swap in very small populations; see handoff.)
    subs = ["N", "O", "F", "Cl", "C(=O)O", "C#N", "OC", "S", "C(F)(F)F", "N(C)C"]
    smi = [f"{'C' * i}c1ccc({sub})cc1" for i in range(1, 11) for sub in subs]
    e = MorganEmbedder()
    e.map_xy(smi[:60])
    first = np.array(e.map_xy(smi[:70]))
    second = np.array(e.map_xy(smi[:80]))[:70]
    drift = np.linalg.norm(first - second, axis=1).mean() / np.linalg.norm(first - first.mean(0), axis=1).mean()
    assert drift < 0.2


def test_morgan_speed_100():
    subs = ["N", "O", "F", "Cl", "C(=O)O", "C#N", "OC", "S", "C(F)(F)F", "N(C)C"]
    smi = [f"{'C' * i}c1ccc({sub})cc1" for i in range(1, 11) for sub in subs]
    assert len(smi) == 100
    e = MorganEmbedder()
    t = time.perf_counter()
    e.map_xy(smi, anchor=SEED)
    assert time.perf_counter() - t < 0.1


def test_default_is_morgan(monkeypatch):
    monkeypatch.delenv("STUDIO_EMBED", raising=False)
    assert get_embedder().method == "morgan_pca"


def test_fallback_when_transformers_absent(monkeypatch):
    real = builtins.__import__

    def fake(name, *a, **k):
        if name.split(".")[0] in ("transformers", "torch"):
            raise ImportError("simulated absence")
        return real(name, *a, **k)

    monkeypatch.setenv("STUDIO_EMBED", "chemberta")
    monkeypatch.setattr(builtins, "__import__", fake)
    assert get_embedder().method == "morgan_pca"


def test_fallback_when_model_missing(monkeypatch):
    pytest.importorskip("transformers")
    from studio.embed import chemberta

    missing = (("DeepChem/this-checkpoint-does-not-exist", "0" * 40),)
    monkeypatch.setattr(chemberta, "CHECKPOINTS", missing)
    monkeypatch.setattr(chemberta.ChemBERTaEmbedder.__init__, "__defaults__", (missing,))
    monkeypatch.setenv("STUDIO_EMBED", "chemberta")
    with pytest.raises(chemberta.ChemBERTaUnavailable):
        chemberta.ChemBERTaEmbedder()
    assert get_embedder().method == "morgan_pca"


def test_distance_from_seed():
    e = MorganEmbedder()
    e.map_xy(MOLS, anchor=SEED)
    same = distance_from_seed(e, SEED, SEED)
    far = distance_from_seed(e, "CCN(CC)CC", SEED)
    assert same["embedding_distance"] == 0.0 and same["tanimoto"] == 1.0
    assert far["embedding_distance"] > 0 and far["tanimoto"] < 0.5 and far["method"] == "morgan_pca"
    assert tanimoto("bad", SEED) == 0.0
