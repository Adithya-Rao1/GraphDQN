"""Upload validation and 3D helpers for the server (RDKit only, no network)."""
from __future__ import annotations

import importlib
import logging
from typing import Optional, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")
log = logging.getLogger("studio.server")

MAX_HEAVY_ATOMS = 60
MAX_SMILES_CHARS = 2000
MAX_MOLFILE_CHARS = 150_000


class InvalidMolecule(ValueError):
    pass


def _clean_text(text: str, limit: int) -> str:
    if "\x00" in text:
        raise InvalidMolecule("Binary content is not accepted (SMILES / MOL / SDF text only)")
    if len(text) > limit:
        raise InvalidMolecule("Input too long")
    return text


def parse_molecule(smiles: Optional[str] = None, molfile: Optional[str] = None) -> Chem.Mol:
    """Parse + sanitize + size-check. Raises InvalidMolecule with a user-facing message."""
    if smiles and smiles.strip():
        s = _clean_text(smiles.strip(), MAX_SMILES_CHARS)
        if any(c in s for c in "\r\n\t "):
            raise InvalidMolecule("SMILES must be a single token without whitespace")
        mol = Chem.MolFromSmiles(s)  # sanitizes
    elif molfile and molfile.strip():
        text = _clean_text(molfile, MAX_MOLFILE_CHARS).replace("\r\n", "\n")
        text = text.split("$$$$")[0]  # SDF: first record only
        if "M  END" not in text:
            raise InvalidMolecule("Not a MOL/SDF file (missing 'M  END')")
        mol = Chem.MolFromMolBlock(text, sanitize=True, removeHs=True)
    else:
        raise InvalidMolecule("Provide a SMILES string or a MOL/SDF file")
    if mol is None:
        raise InvalidMolecule("Could not parse molecule (RDKit sanitization failed)")
    frags = Chem.GetMolFrags(mol, asMols=True)
    if len(frags) > 1:  # keep the largest component (salts / counter-ions are dropped)
        mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
    n = mol.GetNumHeavyAtoms()
    if n == 0:
        raise InvalidMolecule("Molecule has no heavy atoms")
    if n > MAX_HEAVY_ATOMS:
        raise InvalidMolecule(f"Molecule too large ({n} heavy atoms, max {MAX_HEAVY_ATOMS})")
    return mol


def canonical(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(mol)


def _local_molblock(smiles: str) -> Optional[str]:
    """Fallback 3D embedding (ETKDG + MMFF, fixed seed); heavy atoms only."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xC0FFEE
    params.maxIterations = 200
    if AllChem.EmbedMolecule(mol, params) != 0:
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(mol, params) != 0:
            return None
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
    except Exception:
        pass
    return Chem.MolToMolBlock(Chem.RemoveHs(mol))


def molblock_3d(smiles: str) -> Tuple[Optional[str], str]:
    """Return (molblock, provider). Prefers studio.engine.geometry when it exposes a known helper."""
    try:
        geo = importlib.import_module("studio.engine.geometry")
        for name in ("molblock_3d", "smiles_to_molblock", "embed_3d", "embed_smiles"):
            fn = getattr(geo, name, None)
            if callable(fn):
                out = fn(smiles)
                if isinstance(out, str) and out:
                    return out, "studio.engine.geometry"
    except Exception:
        pass
    return _local_molblock(smiles), "local"
