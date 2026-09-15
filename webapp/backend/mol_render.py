import base64
from typing import Optional, Sequence

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D

_TEXT = "#e8e8ec"
_HIGHLIGHT = "#e8c66b"
_HETERO_COLORS = {
    7: "#7aa2e8",   # N - soft blue
    8: "#e8817a",   # O - soft coral
    9: "#7ad9a5",   # F - soft green
    15: "#c98fe0",  # P - soft violet
    16: "#e8c66b",  # S - soft gold
    17: "#7ad9a5",  # Cl - soft green
    35: "#7ad9a5",  # Br - soft green
    53: "#7ad9a5",  # I - soft green
}


def _hex_to_rgba(hex_color: str, alpha: float = 1.0):
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return (r, g, b, alpha)


def mol_image_base64(
    smiles: str,
    highlight_atoms: Optional[Sequence[int]] = None,
    size: tuple = (300, 300),
) -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    drawer = rdMolDraw2D.MolDraw2DCairo(*size)
    opts = drawer.drawOptions()
    opts.setBackgroundColour(_hex_to_rgba("#000000", 0.0))  # transparent
    opts.setHighlightColour(_hex_to_rgba(_HIGHLIGHT, 0.55))
    opts.bondLineWidth = 2
    palette = {0: _hex_to_rgba(_TEXT), 6: _hex_to_rgba(_TEXT)}
    palette.update({num: _hex_to_rgba(color) for num, color in _HETERO_COLORS.items()})
    opts.setAtomPalette(palette)

    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol, highlightAtoms=list(highlight_atoms or []))
    drawer.FinishDrawing()
    return base64.b64encode(drawer.GetDrawingText()).decode("ascii")


def mol_to_molblock_3d(smiles: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    if AllChem.EmbedMolecule(mol, params) != 0:
        if AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=42) != 0:
            return None

    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        try:
            AllChem.UFFOptimizeMolecule(mol)
        except Exception:
            pass  # fall back to the unoptimized embedded geometry

    return Chem.MolToMolBlock(mol)
