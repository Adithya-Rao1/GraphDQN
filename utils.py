import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors
import logging
import os
import json

from ADMET.model import ADMETModel
from binding_module.binding_affinity.plapt import Plapt, run_predictions
from binding_module.selectivity.compare import compare_affinities
from synthetic_accessibility.sa_score import SyntheticAccessibility

def one_hot_encoding(x, allowed_items):
    """
    One hot encodes elements x. 
    Defaults to last element of allowed_atoms if x is not in allowed_atoms
    
    Args:
        x (str): Item to one hot encode
        allowed_atoms (list): List of allowed items

    Returns:
        list: One hot encoded element
    """
    

    if x not in allowed_items:
        x = allowed_items[-1]
    
    one_hot_vec= (np.array(allowed_items) == x).astype(int).tolist()
    
    return one_hot_vec

def get_atom_features(atom,
                      chirality=True,
                      imp_hs=True):
    """
    Returns a 1D array of atom features for a given RDKit atom.
    
    Args:
        atom (Chem.Atom): RDKit atom to get features for.
        chirality (bool): Whether to include chirality information.
        imp_hs (bool): Whether to include implicit hydrogen information.
    
    Returns:
        np.ndarray: 1D array of atom features."""
    
    allowed_atoms = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'OutOfRange']

    if imp_hs == False:
        allowed_atoms.append('H')

    # Atom features

    atom_enc = one_hot_encoding(atom.GetSymbol(), allowed_atoms)
    heavy_nei_enc = one_hot_encoding(int(atom.GetDegree()), [0, 1, 2, 3, 4, "OutOfRange"])
    formal_charge_enc = one_hot_encoding(atom.GetFormalCharge(), [-3, -2, -1, 0, 1, 2, 3, "OutOfRange"])
    hybridization_type_enc = one_hot_encoding(str(atom.GetHybridization()), ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OutOfRange"])
    in_ring_enc = [int(atom.IsInRing())]
    is_aromatic_enc = [int(atom.GetIsAromatic())]
    vdw_radius_scaled = [float((Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) - 1.5)/0.6)]
    covalent_radius_scaled = [float((Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()) - 0.64)/0.76)]
    
    atom_feature_vec = atom_enc + heavy_nei_enc + formal_charge_enc + hybridization_type_enc + in_ring_enc + is_aromatic_enc + vdw_radius_scaled + covalent_radius_scaled

    if chirality == True:
        atom_feature_vec += one_hot_encoding(str(atom.GetChiralTag()), ["CHI_UNSPECIFIED, CHI_TETRAHEDRAL_CW, CHI_TETRAHEDRAL_CCW, CHI_OTHER"])

    if imp_hs == True:
        atom_feature_vec += one_hot_encoding(int(atom.GetTotalNumHs()), [0, 1, 2, 3, 4, "OutOfRange"])

    return np.array(atom_feature_vec)

def get_bond_features(bond,
                      stereochemical=True):
    """
    Returns a 1D array of bond features for a given RDKit bond.
    
    Args:
        bond (Chem.Bond): RDKit bond to get features for.
        stereochemical (bool): Whether to include stereochemical information.
    
    Returns:
        np.ndarray: 1D array of bond features."""
    
    allowed_bonds = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ]

    bond_enc = one_hot_encoding(bond.GetBondType(), allowed_bonds)
    conj_bond_enc = [int(bond.GetIsConjugated())]
    in_ring_enc = [int(bond.IsInRing())]

    bond_feature_vec = bond_enc + conj_bond_enc + in_ring_enc

    if stereochemical == True:
        bond_feature_vec += one_hot_encoding(str(bond.GetStereo()), ["STEREOZ", "STEREOE", "STEREOANY", "STEREONONE"])

    return np.array(bond_feature_vec)

def create_graph(smiles):
    graphs_list = []   

    for smile in smiles:
        mol = Chem.MolFromSmiles(smile)
        
        n_nodes = mol.GetNumAtoms()
        n_edges = 2 * mol.GetNumBonds()
        
        node_features = np.zeros((n_nodes, 42))

        for atom in mol.GetAtoms():
            node_features[atom.GetIdx(), :] = get_atom_features(atom)

        node_features = torch.tensor(node_features, dtype=torch.float32)

        (rows, cols) = np.nonzero(Chem.rdmolops.GetAdjacencyMatrix(mol))
        torch_rows = torch.from_numpy(rows.astype(np.int64)).to(torch.long)
        torch_cols = torch.from_numpy(cols.astype(np.int64)).to(torch.long)
        edges = torch.stack([torch_rows, torch_cols], dim=0)

        edge_features = np.zeros((n_edges, 10))

        for (k, (i, j)) in enumerate(zip(rows, cols)):
            edge_features[k] = get_bond_features(mol.GetBondBetweenAtoms(int(i), int(j)))
        
        edge_features = torch.tensor(edge_features, dtype=torch.float32)

        # Create Data object without the label (y)
        graphs_list.append(Data(x=node_features, edge_index=edges, edge_attr=edge_features))

    return graphs_list

def collate_fn(batch):
    return Batch.from_data_list(batch)

def setup_dqn_logger(log_dir='./logs', log_filename='dqn.log'):
    """Set up a logger that writes logs to a file and console."""
    # Create the log directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Define the log file path
    log_filepath = os.path.join(log_dir, log_filename)

    # Set up the logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Set global log level (INFO by default)

    # Create file handler for logging to a file
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setLevel(logging.INFO)  # Log INFO level and higher to file

    # Create console handler for logging to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # Log INFO level and higher to console

    # Create a formatter and set it for the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def obs_to_loader(obs, batch_size):
    loader = DataLoader(obs, batch_size, collate_fn=collate_fn, drop_last=True)
    return loader 

def largest_ring_size(inp_mol):
    if isinstance(inp_mol, str):
        inp_mol = Chem.MolFromSmiles(inp_mol)
    
    cycles = inp_mol.GetRingInfo().AtomRings()
    if cycles:
        cycle_len = max([len(cycle) for cycle in cycles])
    else:
        cycle_len = 0

    return cycle_len

def penalized_logp(inp_mol):
    if isinstance(inp_mol, str):
        inp_mol = Chem.MolFromSmiles(inp_mol)
    
    log_p = Descriptors.MolLogP(inp_mol)
    sa_score = SyntheticAccessibility().calculateScore(inp_mol)
    ring_size = largest_ring_size(inp_mol)
    cycle_score = max(ring_size-6, 0)

    return log_p - sa_score - cycle_score

def track(first: bool, items: list, file_name: str, sort_by_values: str = False, output_dir: str = './dqn_results'):
    os.makedirs(output_dir, exist_ok=True)

    if sort_by_values == "high":
        items = sorted(items, key=lambda x: x[1], reverse=True) 
    elif sort_by_values == "low":
        items = sorted(items, key=lambda x: x[1])  

    os.makedirs(output_dir, exist_ok=True)

    tracker_file = os.path.join(output_dir, f'{file_name}.json')

    if first==True:
        with open(tracker_file, 'w') as f:
            json.dump({file_name: items}, f, indent=4)

    else:
        with open(tracker_file, 'r') as f:
            data = json.load(f)

        data[file_name] = data.get(file_name) + items

        with open(tracker_file, 'w') as f:
            json.dump(data, f, indent=4)

def calc_multi_obj_properties(smile, target_seq, device, off_target_seq=None):
    admet_properties = ['QED',
                        'Lipinski',
                        'Bioavailability_Ma',
                        'BBB_Martins',
                        'DILI',
                        'Clearance_Hepatocyte_AZ',
                        'Clearance_Microsome_AZ',
                        'Half_Life_Obach',
                        'hERG',
                        'ClinTox',
                        'LD50_Zhu']
    
    admet_optim_directions = [1, 1, 1, 1, -1, 1, 1, 1, -1, -1, -1]
    admet_preds = ADMETModel(device=device).predict(smile)
    admet_values = [admet_preds[prop] for prop in admet_properties]

    admet_reward = sum(
            admet_values[j] if admet_optim_directions[j] == 1 else (1 / admet_values[j])
            for j in range(len(admet_properties))
        )
    binding_reward = run_predictions(Plapt(), target_seq, [smile])[0]
    sa_reward = SyntheticAccessibility().calculateScore(Chem.MolFromSmiles(smile))
    if off_target_seq:
        off_target_pred = run_predictions(Plapt(), off_target_seq, [smile])[0]
        selectivity_reward = compare_affinities(binding_reward, off_target_pred)
    
        return [admet_reward, binding_reward, sa_reward, selectivity_reward]
    
    return [admet_reward, binding_reward, sa_reward]


"""dummy_smiles = 'O=O'
dummy_mol = Chem.MolFromSmiles(dummy_smiles)
n_node_features = len(get_atom_features(dummy_mol.GetAtomWithIdx(0)))
n_edge_features = len(get_bond_features(dummy_mol.GetBondBetweenAtoms(0,1)))"""

"""from dkdqn_network import DKDQNNetwork
model = DKDQNNetwork(15)
model.to('cuda')"""

"""smiles_list = ['N#CC1=C(C(F)(F)F)C([N+]([O-])=O)=C(C2=CC=CC([N+]([O-])=O)=C2)NC1=O',
                    'O=C(NC1=CC=C(NC(NC2=CC=CC3=C2C=CN3)=O)C=C1)NC4=C(C=CN5)C5=CC=C4',
                    'O=C1N(C)C(CNCC)=NC2=C1C(Cl)=CC(Cl)=C2O.Br',
                    'O=C(C(C=C1C2=O)=CC(O)=C1C(C3=C2C=CC=C3O)=O)NC4=CC=C(Cl)C=C4O',
                    'OC1=CC=C(CC2=CC=C(C(CC3=CC=C(C(CC4=CC=C(C=C4)O)=C3)O)=C2)O)C=C1']

loader = obs_to_loader(create_graph(smiles_list), 2)
print(len(loader.dataset))"""
"""q_values = []
for batch in loader:
    q_value = model.forward(batch.to('cuda'))
    q_values.append(q_value)

q_values = torch.cat(q_values, dim=0)
q_value = torch.max(q_values, dim=0)[0]

action = torch.argmax(q_value).item()

print(action)"""