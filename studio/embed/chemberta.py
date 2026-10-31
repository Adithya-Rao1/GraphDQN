"""ChemBERTa embeddings + PCA(2) map (method="chemberta_pca"). Optional upgrade.

Used ONLY for the 2D population map and embedding-space similarity. It explains
nothing about why an edit helps; all explanations come from RDKit descriptors.

Weights are pickle-format (pytorch_model.bin): we load only the allow-listed repos
below, at the pinned revision hash, never from an arbitrary path or id.
"""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
from rdkit import RDLogger

from .morgan import StableProjector, _map, canon

log = logging.getLogger(__name__)

# (repo id, pinned revision as listed on the Hub on 2026-10-30), in preference order.
CHECKPOINTS = (
    ("DeepChem/ChemBERTa-5M-MLM", "695672a601a8077ace5d0a6fc2529e93cc9cfa9c"),
    ("DeepChem/ChemBERTa-77M-MLM", "ed8a5374f2024ec8da53760af91a33fb8f6a15ff"),
    ("seyonec/ChemBERTa-zinc-base-v1", "761d6a18cf99db371e0b43baf3e2d21b3e865a20"),
)
BATCH = 32
MAX_LEN = 256


class ChemBERTaUnavailable(RuntimeError):
    pass


class ChemBERTaEmbedder:
    method = "chemberta_pca"

    def __init__(self, checkpoints=CHECKPOINTS) -> None:
        """Loads the model eagerly; raises ChemBERTaUnavailable if no checkpoint loads."""
        RDLogger.DisableLog("rdApp.*")
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except Exception as e:  # ImportError, or broken binary install
            raise ChemBERTaUnavailable(f"transformers/torch unavailable: {e}") from e
        self._torch = torch
        errors = []
        for repo, rev in checkpoints:
            try:
                # HF_HOME (e.g. /metis-data/hf-cache) is honoured by transformers itself.
                tok = AutoTokenizer.from_pretrained(repo, revision=rev)
                model = AutoModel.from_pretrained(repo, revision=rev)
                model.eval()
                self._tok, self._model, self.checkpoint = tok, model, f"{repo}@{rev[:8]}"
                break
            except Exception as e:
                errors.append(f"{repo}: {type(e).__name__}: {e}")
        else:
            raise ChemBERTaUnavailable("no checkpoint loadable: " + " | ".join(errors))
        self._cache: Dict[str, np.ndarray] = {}
        self._proj = StableProjector()

    def features(self, keys: Sequence[str]) -> Dict[str, np.ndarray]:
        """Mean-pooled last hidden state per canonical SMILES, batched and cached."""
        todo = [k for k in keys if k not in self._cache]
        torch = self._torch
        with torch.no_grad():
            for i in range(0, len(todo), BATCH):
                batch = todo[i : i + BATCH]
                enc = self._tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN)
                hid = self._model(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).to(hid.dtype)
                pooled = ((hid * mask).sum(1) / mask.sum(1).clamp(min=1)).cpu().numpy()
                for k, v in zip(batch, pooled):
                    self._cache[k] = v.astype(np.float64)
        return {k: self._cache[k] for k in keys}

    def map_xy(self, smiles: Sequence[str], anchor: Optional[str] = None) -> List[tuple]:
        return _map(self._proj, smiles, anchor, self.features)
