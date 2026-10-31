"""Affinity backends: real PLAPT, a deterministic fake for tests/offline dev, and a persistent cache wrapper.

Unit of every backend: `neg_log10_affinity_M` (pKd-like, higher = tighter predicted binding), the model's own
output unit (see `PredictionModule.convert_to_affinity` in binding_module/binding_affinity/plapt.py).
"""
from __future__ import annotations

import hashlib
import logging
import os
import threading
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Sequence, runtime_checkable

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
log = logging.getLogger("studio.binding")

UNIT = "neg_log10_affinity_M"  # pKd-like


def cache_dir() -> Path:
    return Path(os.environ.get("STUDIO_CACHE_DIR", "./.cache/studio"))


def canon(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    return Chem.MolToSmiles(mol)


def seq_hash(sequence: str) -> str:
    return hashlib.sha1(sequence.strip().upper().encode()).hexdigest()[:16]


@runtime_checkable
class BindingBackend(Protocol):
    name: str  # recorded in cache keys and shown in the UI
    unit: str

    def predict(self, sequence: str, smiles_list: Sequence[str]) -> List[float]:
        """Predicted pKd-like affinity for each SMILES against one protein sequence (index-aligned)."""
        ...


class FakeBackend:
    """DETERMINISTIC STAND-IN, NOT A PLAPT PREDICTION. Hash of (sequence, canonical SMILES) mapped to 4..9.

    For tests and offline dev only. `overrides` pins specific canonical SMILES to chosen values.
    """
    name = "FakeBackend(deterministic-hash, not PLAPT)"
    unit = UNIT

    def __init__(self, overrides: Optional[Dict[str, float]] = None):
        self.overrides = {canon(k): v for k, v in (overrides or {}).items()}
        self.calls = 0  # number of predict() calls
        self.molecules_scored = 0

    def predict(self, sequence: str, smiles_list: Sequence[str]) -> List[float]:
        self.calls += 1
        out = []
        for s in smiles_list:
            c = canon(s)
            self.molecules_scored += 1
            if c in self.overrides:
                out.append(float(self.overrides[c]))
                continue
            h = hashlib.sha1((seq_hash(sequence) + "|" + c).encode()).digest()
            out.append(4.0 + 5.0 * int.from_bytes(h[:4], "big") / 0xFFFFFFFF)
        return out


class PlaptBackend:
    """Real PLAPT via the repo's `binding_module.binding_affinity.plapt.Plapt` (imported read-only).

    Models are fetched from the official Hugging Face Hub on first use. The protein embedding is computed once per
    sequence and kept in memory (and in Plapt's own disk cache). Device: CUDA if available, else CPU.
    """
    unit = UNIT

    def __init__(self, embedding_cache_dir: Optional[str] = None, device: str = "cuda"):
        self._dir = embedding_cache_dir or str(cache_dir() / "plapt_embeddings")
        self._device = device
        self._plapt = None
        self._prot: Dict[str, object] = {}
        self._lock = threading.Lock()
        self.name = "PLAPT(prot_bert+ChemBERTa+onnx)"

    def _load(self):
        if self._plapt is None:
            from binding_module.binding_affinity.plapt import Plapt  # heavy: torch, transformers, onnxruntime
            Path(self._dir).mkdir(parents=True, exist_ok=True)
            self._plapt = Plapt(device=self._device, cache_dir=self._dir)
            self.name = f"PLAPT(prot_bert+ChemBERTa+onnx, {self._plapt.device.type})"
            log.info("loaded %s", self.name)
        return self._plapt

    def predict(self, sequence: str, smiles_list: Sequence[str]) -> List[float]:
        import torch
        if not smiles_list:
            return []
        with self._lock:
            p = self._load()
            key = seq_hash(sequence)
            if key not in self._prot:
                self._prot[key] = p.encode_proteins([sequence.strip()], 1)  # once per sequence
            prot = self._prot[key]
            mol = p.encode_molecules([canon(s) for s in smiles_list], 16)
            feats = torch.cat((prot.repeat(len(smiles_list), 1), mol), dim=1).cpu().numpy()
            return [float(r["neg_log10_affinity_M"]) for r in p.prediction_module.predict(feats)]


class CachedBackend:
    """Persistent (diskcache) memo keyed by (backend name, sequence hash, canonical SMILES)."""

    def __init__(self, inner: BindingBackend, directory: Optional[str] = None):
        from diskcache import Cache
        self.inner = inner
        self.unit = inner.unit
        self._cache = Cache(directory or str(cache_dir() / "binding_pkd"))
        self.hits = 0
        self.misses = 0

    @property
    def name(self) -> str:
        return self.inner.name

    def predict(self, sequence: str, smiles_list: Sequence[str]) -> List[float]:
        sh = seq_hash(sequence)
        cans = [canon(s) for s in smiles_list]
        keys = [(self.inner.name.split(",")[0], sh, c) for c in cans]  # strip device so CPU/GPU share entries
        vals: Dict[str, float] = {}
        todo: List[str] = []
        for k, c in zip(keys, cans):
            v = self._cache.get(k)
            if v is not None:
                vals[c] = v
                self.hits += 1
            elif c not in todo:
                todo.append(c)
        if todo:
            for c, v in zip(todo, self.inner.predict(sequence, todo)):
                vals[c] = float(v)
                self._cache.set((self.inner.name.split(",")[0], sh, c), float(v))
                self.misses += 1
        return [vals[c] for c in cans]


def make_backend(kind: str = "plapt", cached: bool = True) -> BindingBackend:
    inner: BindingBackend = FakeBackend() if kind == "fake" else PlaptBackend()
    return CachedBackend(inner) if cached else inner
