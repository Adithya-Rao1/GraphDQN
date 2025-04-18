from rdkit import Chem

mol = Chem.MolFromSmiles("CCO")
f_group = ["[OH]", ['O', 'C']]
for atom in mol.GetAtoms():
    if atom.GetSymbol() in f_group[1]:
        print("Applies")
