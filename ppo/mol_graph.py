import torch
import torch.nn.functional as F
from rdkit import Chem
import rdkit.Chem.rdchem as rdc
import numpy as np
import networkx as nx
import torch
from torch_geometric.utils.convert import from_networkx
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
import torch_geometric.datasets as datasets
import torch_geometric.transforms as transforms

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

# TURN MOL DIRECTLY INTO PYG OBJECT

CHIRAL_TYPES = [rdc.ChiralType.CHI_UNSPECIFIED,
                        rdc.ChiralType.CHI_TETRAHEDRAL_CW,
                        rdc.ChiralType.CHI_TETRAHEDRAL_CCW,
                        rdc.ChiralType.CHI_SQUAREPLANAR,
                        rdc.ChiralType.CHI_OTHER,]
        
HYBRIDIZATION_TYPES = [rdc.HybridizationType.SP,
                    rdc.HybridizationType.SP2,
                    rdc.HybridizationType.SP3,
                    rdc.HybridizationType.SP3D,
                    rdc.HybridizationType.SP3D2,
                    rdc.HybridizationType.UNSPECIFIED]

BOND_TYPES = [rdc.BondType.SINGLE,
            rdc.BondType.DOUBLE,
            rdc.BondType.TRIPLE,
            rdc.BondType.AROMATIC]

ATOM_TYPES = ['C', 'N', 'O', 'F', 'Si', 'S', 'Cl', 'Br']

FORMAL_CHARGES = [-3, -2, -1, 0, 1, 2, 3]

AROMATIC_TYPES = [True, False]

EXPLICIT_HS = [0, 1, 2, 3]


class GraphDataset(Dataset):
    def __init__(self, data=None):
        self.data = data or []
        self.mols = [Chem.MolFromSmiles(compound) if isinstance(compound, str) else compound for compound in self.data]
        self.graphs = [self.graph_to_pyg(self.mol_to_graph(mol)) for mol in self.mols]
    
    def categorical_encoding(self, x, possible_values):
        values_to_idx = {value: idx for idx, value in enumerate(possible_values)}
        return values_to_idx[x]
    
    def one_hot_encoding(self, x, allowed_items):
        one_hot_vec= (np.array(allowed_items) == x).astype(int).tolist()
        return torch.tensor(one_hot_vec)
    
    def find_feature(self, one_hot_vec):
        return torch.argmax(one_hot_vec)
    
    def mol_to_graph(self, mol):
        G = nx.Graph()

        for atom in mol.GetAtoms():
            G.add_node(atom.GetIdx(), 
                    atomic_sym=self.one_hot_encoding(atom.GetSymbol(), ATOM_TYPES),
                    formal_charge=self.one_hot_encoding(atom.GetFormalCharge(), FORMAL_CHARGES), 
                    chiral_tag=self.one_hot_encoding(atom.GetChiralTag(), CHIRAL_TYPES),
                    hybridization=self.one_hot_encoding(atom.GetHybridization(), HYBRIDIZATION_TYPES),
                    num_explicit_hs=self.one_hot_encoding(atom.GetNumExplicitHs(), EXPLICIT_HS),
                    is_aromatic=self.one_hot_encoding(atom.GetIsAromatic(), AROMATIC_TYPES),
                    )

        for bond in mol.GetBonds():
            G.add_edge(bond.GetBeginAtomIdx(), 
                    bond.GetEndAtomIdx(),
                    bond_type=self.one_hot_encoding(bond.GetBondType(), BOND_TYPES)
                    )

        return G

    def graph_to_mol(self, G):
        mol = Chem.RWMol()
        
        atomic_syms = nx.get_node_attributes(G, 'atomic_sym')
        formal_charges = nx.get_node_attributes(G, 'formal_charge')
        chiral_tags = nx.get_node_attributes(G, 'chiral_tag')
        hybridizations = nx.get_node_attributes(G, 'hybridization')
        num_explicit_hs = nx.get_node_attributes(G, 'num_explicit_hs')
        is_aromatic = nx.get_node_attributes(G, 'is_aromatic')

        for node in G.nodes():
            a=Chem.Atom(ATOM_TYPES[self.find_feature(atomic_syms[node])])
            a.SetFormalCharge(FORMAL_CHARGES[self.find_feature(formal_charges[node])])
            a.SetChiralTag(CHIRAL_TYPES[self.find_feature(chiral_tags[node])])
            a.SetHybridization(HYBRIDIZATION_TYPES[self.find_feature(hybridizations[node])])
            a.SetNumExplicitHs(EXPLICIT_HS[self.find_feature(num_explicit_hs[node])])
            a.SetIsAromatic(AROMATIC_TYPES[self.find_feature(is_aromatic[node])])
            mol.AddAtom(a)

        bond_types = nx.get_edge_attributes(G, 'bond_type')

        for edge in G.edges():
            begin_idx, end_idx = edge
            bond_type = BOND_TYPES[self.find_feature(bond_types[edge])]
            mol.AddBond(begin_idx, end_idx, bond_type)

        Chem.SanitizeMol(mol)
        return mol

    def graph_to_pyg(self, G):
        pyg_graph = from_networkx(G)

        node_attrs = ['atomic_sym', 'formal_charge', 'chiral_tag', 'hybridization', 'num_explicit_hs', 'is_aromatic']
        node_features = [
            torch.cat([G.nodes[node][attr] for attr in node_attrs]).float()
            for node in G.nodes()
        ]
        pyg_graph.x = torch.stack(node_features, dim=0) if node_features else torch.zeros((0, self.node_feature_dim))
        
        edge_features = [G.edges[edge]['bond_type'].float() for edge in G.edges()]
        pyg_graph.edge_attr = torch.stack(edge_features, dim=0) if edge_features else torch.zeros((0, len(BOND_TYPES)))

        return pyg_graph

    def mol_to_pyg(self, mol):
        return self.graph_to_pyg(self.mol_to_graph(mol))

    @property
    def node_feature_dim(self):
        return len(ATOM_TYPES) + len(FORMAL_CHARGES) + len(CHIRAL_TYPES) + len(HYBRIDIZATION_TYPES) + len(EXPLICIT_HS) + len(AROMATIC_TYPES)

    def mol_to_state_vector(self, mol):
        graph = self.mol_to_graph(mol)
        if graph.number_of_nodes() == 0:
            return torch.zeros(self.node_feature_dim)

        pyg_graph = self.graph_to_pyg(graph)
        return pyg_graph.x.mean(dim=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.graphs[idx]

if __name__ == "__main__":
    smiles_list = ["CCO"]

    dataset = GraphDataset(smiles_list)
    loader = DataLoader(dataset, batch_size=1)

    for batch in loader:
        print(batch)

    mol = Chem.MolFromSmiles("CCO")
    graph = dataset.mol_to_graph(mol)
    pygraph = dataset.graph_to_pyg(graph)
    mol2 = dataset.graph_to_mol(graph)
    print(Chem.MolToSmiles(mol2))
    

