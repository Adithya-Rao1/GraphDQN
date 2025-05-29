from rdkit import Chem

mol = Chem.MolFromSmiles("CCO")
f_group = ["[OH]", ['O', 'C']]
for atom in mol.GetAtoms():
    if atom.GetSymbol() in f_group[1]:
        print("Applies")

import torch
a = torch.randn(1,3)
print(a)
print(torch.flip(a, (1,)))