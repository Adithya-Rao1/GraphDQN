"""Chemical-space map for the GA population (owner: embed).

`get_embedder()` never raises. Selection: env STUDIO_EMBED=morgan (default) | chemberta.
If ChemBERTa is requested but cannot load, Morgan is returned and `method`
says "morgan_pca" so the UI labels the map truthfully.
"""
from __future__ import annotations

import logging
import os
from typing import Dict, Optional, Sequence

import numpy as np

from .morgan import MorganEmbedder, canon, tanimoto

log = logging.getLogger(__name__)

__all__ = ["get_embedder", "MorganEmbedder", "distance_from_seed", "tanimoto"]


def get_embedder(prefer: Optional[str] = None):
    choice = (prefer or os.environ.get("STUDIO_EMBED", "morgan")).strip().lower()
    if choice == "chemberta":
        try:
            from .chemberta import ChemBERTaEmbedder

            return ChemBERTaEmbedder()
        except Exception as e:
            log.warning("ChemBERTa unavailable, falling back to Morgan+PCA: %s", e)
    return MorganEmbedder()


def distance_from_seed(embedder, smiles: str, seed_smiles: str) -> Dict[str, float]:
    """Story-card numbers: Euclidean distance from the seed in the embedder's PCA map
    (embedder.method), and Morgan Tanimoto similarity (always Morgan, 2048 bits, r=2)."""
    xy = embedder.map_xy([smiles], anchor=seed_smiles)[0]  # anchor sits at the origin
    return {
        "embedding_distance": float(np.hypot(xy[0], xy[1])),
        "tanimoto": tanimoto(smiles, seed_smiles),
        "method": embedder.method,
    }
