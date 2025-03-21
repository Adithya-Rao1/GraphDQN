from rdkit import Chem
import numpy as np
import networkx as nx
import torch
from torch_geometric.utils.convert import from_networkx
from torch_geometric.data import Dataset

"""
TODO
- graphs should work with torch_geometric layers
- encode node features 
    - (convert categorical data to some sort of encoding, scaling continuous variables)
- encode edge features
    - Encode categorical bond types
- graph_to_pyg() func --> PyG expects edge_features to be in form of tensor 
and node features as 2d tensor (each row representend feature vector of node)
"""

def mol_to_graph(mol):
    G = nx.Graph()

    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(), 
                   atomic_num=atom.GetAtomicNum(),
                   formal_charge=atom.GetFormalCharge(),
                   chiral_rag=atom.GetChiralTag(),
                   hybridization=atom.GetHybridization(),
                   num_explicit_hs=atom.GetNumExplicitHs(),
                   is_aromatic=atom.GetIsAromatic(),
                   )

    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(), 
                   bond.GetEndAtomIdx(),
                   bond_type=bond.GetBondType()
                   )

    return G

def graph_to_mol(G):
    mol = Chem.RWMol()
    
    atomic_nums = nx.get_node_attributes(G, 'atomic_num')
    formal_charges = nx.get_node_attributes(G, 'formal_charge')
    chiral_rags = nx.get_node_attributes(G, 'chiral_rag')
    hybridizations = nx.get_node_attributes(G, 'hybridization')
    num_explicit_hs = nx.get_node_attributes(G, 'num_explicit_hs')
    is_aromatic = nx.get_node_attributes(G, 'is_aromatic')

    node_to_idx = {}

    for node in G.nodes():
        a=Chem.Atom(atomic_nums[node])
        a.SetFormalCharge(formal_charges[node])
        a.SetChiralTag(chiral_rags[node])
        a.SetHybridization(hybridizations[node])
        a.SetNumExplicitHs(num_explicit_hs[node])
        a.SetIsAromatic(is_aromatic[node])
        node_to_idx[node] = mol.AddAtom(a)

    bond_types = nx.get_edge_attributes(G, 'bond_type')

    for edge in G.edges():
        begin_idx, end_idx = edge
        ibegin = node_to_idx[begin_idx]
        iend = node_to_idx[end_idx]
        bond_type = bond_types[begin_idx, end_idx]
        mol.AddBond(ibegin, iend, bond_type)

    Chem.SanitizeMol(mol)
    return mol

def graph_to_pyg(G):
    pyg_graph = from_networkx(G)
    
    # Encode node features into a tensor (make sure you are using suitable encoding for categorical features)
    node_attrs = ['atomic_num', 'formal_charge', 'chiral_rag', 'hybridization', 'num_explicit_hs', 'is_aromatic']
    
    node_features = []
    for node in G.nodes():
        features = [G.nodes[node].get(attr, 0) for attr in node_attrs]
        node_features.append(features)
    
    pyg_graph.x = torch.tensor(node_features, dtype=torch.float)  # Node features tensor

    # Encode edge features (bond_type) into a tensor
    edge_attr = []
    for edge in G.edges():
        edge_attr.append([G.edges[edge].get('bond_type', 0)])  # Bond type as a numeric value (or one-hot if needed)
    
    pyg_graph.edge_attr = torch.tensor(edge_attr, dtype=torch.float)  # Edge features tensor
    
    return pyg_graph

class GraphDataset(Dataset):
    def __init__(self, smiles):
        self.smiles = smiles
        self.mols = [Chem.MolFromSmiles(smile) for smile in smiles]
        self.graphs = [graph_to_pyg(mol_to_graph(mol)) for mol in self.mols]

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        return self.graphs[idx]
    
"""mol = Chem.MolFromSmiles('CC1=CC=CC=C1')
G = mol_to_graph(mol)
mol2 = graph_to_mol(G)
pyg = graph_to_pyg(G)
print(pyg)"""