from pymol import cmd
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors

# Load receptor and ligand
cmd.load("3zosA.pdb", "receptor")
cmd.load("ligand.pdb", "ligand")

# Define binding site selection
cmd.select("binding_site", "resi 50-100")  # Adjust residues as needed
center = cmd.get_center("binding_site")

# Load ligand in RDKit to compute Rg
ligand_mol = Chem.MolFromMolFile("ligand.mol")
AllChem.EmbedMolecule(ligand_mol)
AllChem.UFFOptimizeMolecule(ligand_mol)
Rg = rdMolDescriptors.CalcRadiusOfGyration(ligand_mol)

# Calculate box size
box_size = 2.9 * Rg
size_x, size_y, size_z = box_size, box_size, box_size

# Create config.txt
config_content = f"""receptor = 3zosA_prepared.pdbqt
ligand = ligand.pdbqt

#binding_pocket
center_x = {center[0]:.2f}
center_y = {center[1]:.2f}
center_z = {center[2]:.2f}

size_x = {size_x:.2f}
size_y = {size_y:.2f}
size_z = {size_z:.2f}
"""

with open("config.txt", "w") as f:
    f.write(config_content)

print("config.txt generated successfully with optimized box size!")