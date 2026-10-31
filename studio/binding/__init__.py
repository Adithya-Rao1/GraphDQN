"""Stage 1 (binding): PLAPT-predicted affinity as GA fitness, with guards against gaming it.

All affinities here are PREDICTED by PLAPT (ProtBERT + ChemBERTa + ONNX head) as -log10(Kd/M) ("pKd-like").
They are model outputs, not measurements.
"""
from studio.binding.backends import BindingBackend, CachedBackend, FakeBackend, PlaptBackend, make_backend
from studio.binding.scorer import BindingScorer

__all__ = ["BindingBackend", "CachedBackend", "FakeBackend", "PlaptBackend", "make_backend", "BindingScorer"]
