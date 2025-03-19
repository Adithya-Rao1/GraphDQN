from rdkit import Chem
from rdkit.Chem import Descriptors
from molecular_modifications.bioisosteres_optimization import ModifyBioisosteres
from molecular_modifications.atom_optimization import ModifyAtom
from molecular_modifications.bond_optimization import ModifyBond
from molecular_modifications.functional_group_optimization import ModifyFunctionalGroup
from molecular_modifications.logger import setup_molecule_logger
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

env = list(set(["C", "O"]))

modify_atom = ModifyAtom(setup_molecule_logger())
modify_bond = ModifyBond(setup_molecule_logger())
modify_bio = ModifyBioisosteres(setup_molecule_logger())
modify_fg = ModifyFunctionalGroup(setup_molecule_logger())

actions = set()
from synthetic_accessibility.sa_score import SyntheticAccessibility


"""for item in env:
    mol = Chem.MolFromSmiles(item)
    # Atom actions
    actions.add(
        modify_atom.add_atom(mol, 5)
    )
    actions.add(
        modify_atom.remove_atom(mol, 5)
    )
    actions.add(
        modify_atom.modify_atom(mol, 5)
    )

    # Bond actions
    actions.add(
        modify_bond.optimize_bond(mol, 0)
    )
    actions.add(
        modify_bond.optimize_bond(mol, 1)
    )
    actions.add(
        modify_bond.optimize_bond(mol, 2)
    )

    # Bioisosteric groups actions
    actions.add(
        modify_bio.apply_modification(mol, 'carboxylic_acid', 'tetrazole')
    )
    actions.add(
        modify_bio.apply_modification(mol, 'carboxylic_acid', 'phosphonic_acid')
    )
    actions.add(
        modify_bio.apply_modification(mol, 'amine', None)
    )
    actions.add(
        modify_bio.apply_modification(mol, 'amide', 'sulfonamide')
    )
    actions.add(
        modify_bio.apply_modification(mol, 'amide', 'retroamide')
    )
    actions.add(
        modify_bio.apply_modification(mol, 'phenyl', 'pyridyl')
    )
    actions.add(
        modify_bio.apply_modification(mol, 'phenyl', 'thiophene')
    )
    actions.add(
        modify_bio.apply_modification(mol, 'phenyl', 'furan')
    )
    actions.add(
        modify_bio.apply_modification(mol, 'phenyl', 'pyrrole')
    )

    actions = {smiles for smiles in actions if smiles}

print(actions)"""

from dqn.utils import penalized_logp, largest_ring_size
print(penalized_logp(Chem.MolFromSmiles('N#CC1=C(C(F)(F)F)C([N+]([O-])=O)=C(C2=CC=CC([N+]([O-])=O)=C2)NC1=O')))
print(Descriptors.MolLogP(Chem.MolFromSmiles('N#CC1=C(C(F)(F)F)C([N+]([O-])=O)=C(C2=CC=CC([N+]([O-])=O)=C2)NC1=O')))
print(SyntheticAccessibility().calculateScore(Chem.MolFromSmiles('N#CC1=C(C(F)(F)F)C([N+]([O-])=O)=C(C2=CC=CC([N+]([O-])=O)=C2)NC1=O')))
print(largest_ring_size(Chem.MolFromSmiles('N#CC1=C(C(F)(F)F)C([N+]([O-])=O)=C(C2=CC=CC([N+]([O-])=O)=C2)NC1=O')))
print(len('MDVFMKGLSKAKEGVVAAAEKTKQGVAEAAGKTKEGVLYVGSKTKEGVVHGVATVAEKTKEQVTNVGGAVVTGVTAVAQKTVEGAGSIAAATGFVKKDQLGKNEEGAPQEGILEDMPVDPDNEAYEMPSEEGYQDYEPEA'))
from rdkit.Chem import AllChem, rdMolDescriptors

mol = Chem.MolFromSmiles("NOc1cc(Cl)ccc1NC(=O)c1cc(O)c2c(c1)C(=O)c1cccc(O)c1C2=O")  # Load your ligand file
mol = Chem.AddHs(mol)
AllChem.EmbedMolecule(mol, AllChem.ETKDG())

Rg = rdMolDescriptors.CalcRadiusOfGyration(mol)
print(f"Radius of Gyration for Novel Molecule: {Rg} Å")

mol = Chem.MolFromSmiles("N#CC1=C(C(F)(F)F)C([N+]([O-])=O)=C(C2=CC=CC([N+]([O-])=O)=C2)NC1=O")  # Load your ligand file
mol = Chem.AddHs(mol)
AllChem.EmbedMolecule(mol, AllChem.ETKDG())

Rg = rdMolDescriptors.CalcRadiusOfGyration(mol)
print(f"Radius of Gyration for Synuclean-D: {Rg} Å")