import os
import sys
import random
from typing import List, Dict, Tuple, Optional, Union, Set
from rdkit import Chem
from rdkit.Chem import AllChem

# TODO: INCREASE SET OF POSSIBLE BONDING SITES FOR EACH FG

FUNCTIONAL_GROUPS = {
    # Format -- name: (SMARTS pattern, possible atoms/groups to bond to)
    'methyl': ('[CH3]', ['C', 'N', 'S', 'O']),
    'hydroxyl': ('[OH]', 1),
    'amino': ('[NH2]', 1),
    'carboxyl': ('[C](=O)[OH]', 1),
    'carbonyl': ('[C]=O', 2),
    'aldehyde': ('[CH]=O', 1),
    'ketone': ('[C](=O)[#6]', 1),
    'ether': ('[OD2]([#6])[#6]', 1),
    'ester': ('[C](=O)[O][#6]', 1),
    'amide': ('[C](=O)[N]', 1),
    'nitro': ('[N+](=O)[O-]', 1),
    'cyano': ('[C]#N', 1),
    'alkene': ('[C]=[C]', 1),
    'alkyne': ('[C]#[C]', 1),
    'thiol': ('[SH]', 1),
    'thioether': ('[SD2]([#6])[#6]', 1),
    'halogen': ('[F,Cl,Br,I]', 1),
    'phosphate': ('[P](=O)([O])[O]', 1),
    'sulfonamide': ('[S](=O)(=O)[N]', 1),
    'sulfone': ('[S](=O)(=O)[#6]', 1),
    'sulfoxide': ('[S](=O)[#6]', 1),
    'azide': ('[N-]=[N+]=[N]', 1),
    'isocyanate': ('[N]=[C]=[O]', 1),
    'isothiocyanate': ('[N]=[C]=[S]', 1),
    'methylenedioxy': ('[O][C]([#6])[O]', 1)
}

FG_PROPERTIES = {
    'methyl': {'electron_donating': True, 'hydrophobic': True, 'acidic': False},
    'hydroxyl': {'electron_donating': True, 'hydrophilic': True, 'acidic': True},
    'amino': {'electron_donating': True, 'hydrophilic': True, 'basic': True},
    'carboxyl': {'electron_withdrawing': True, 'hydrophilic': True, 'acidic': True},
    'carbonyl': {'electron_withdrawing': True, 'reactive': True},
    'aldehyde': {'electron_withdrawing': True, 'reactive': True, 'hydrophilic': True},
    'ketone': {'electron_withdrawing': True, 'reactive': True, 'hydrophilic': True},
    'ether': {'electron_donating': True, 'hydrophobic': True},
    'ester': {'electron_withdrawing': True, 'hydrophobic': True, 'reactive': True},
    'amide': {'resonance': True, 'hydrophilic': True, 'h_bonding': True},
    'nitro': {'electron_withdrawing': True, 'reactive': True},
    'cyano': {'electron_withdrawing': True, 'hydrophilic': True},
    'alkene': {'electron_donating': True, 'reactive': True},
    'alkyne': {'electron_withdrawing': True, 'reactive': True},
    'thiol': {'electron_donating': True, 'hydrophobic': True, 'acidic': True},
    'thioether': {'electron_donating': True, 'hydrophobic': True},
    'halogen': {'electron_withdrawing': True, 'hydrophobic': True},
    'phosphate': {'electron_withdrawing': True, 'hydrophilic': True, 'acidic': True},
    'sulfonamide': {'electron_withdrawing': True, 'hydrophilic': True, 'acidic': True},
    'sulfone': {'electron_withdrawing': True, 'hydrophilic': True},
    'sulfoxide': {'electron_withdrawing': True, 'hydrophilic': True},
    'azide': {'electron_withdrawing': True, 'reactive': True},
    'isocyanate': {'electron_withdrawing': True, 'reactive': True},
    'isothiocyanate': {'electron_withdrawing': True, 'reactive': True},
    'methylenedioxy': {'electron_donating': True, 'hydrophobic': True}
}

# 0: Incompatible, 1: Possible issues, 2: Compatible
FG_COMPATIBILITY = {
    'hydroxyl': {'carboxyl': 1, 'carbonyl': 1, 'aldehyde': 1, 'ester': 1, 'isocyanate': 0},
    'amino': {'carboxyl': 1, 'carbonyl': 1, 'aldehyde': 1, 'ketone': 1, 'ester': 1, 'isocyanate': 0},
    'carboxyl': {'hydroxyl': 1, 'amino': 1, 'aldehyde': 1, 'thiol': 1},
    'thiol': {'carboxyl': 1, 'carbonyl': 1, 'alkene': 1, 'alkyne': 1},
    'nitro': {'amino': 1},
    'isocyanate': {'hydroxyl': 0, 'amino': 0, 'carboxyl': 0, 'thiol': 0, 'water': 0}
}

REACTION_OUTCOMES = {
    ('hydroxyl', 'carboxyl'): 'ester_formation',
    ('amino', 'carboxyl'): 'amide_formation',
    ('hydroxyl', 'isocyanate'): 'urethane_formation',
    ('amino', 'carbonyl'): 'imine_formation'
}

class ModifyFunctionalGroup:
    def __init__(self, 
                 logger=None,
                 functional_groups: Optional[Dict[str, Tuple[str, int]]] = None,
                 modification_strategy: str = 'balanced',
                 log: bool = False):
        self.logger = logger
        self.log = log
        
        self.functional_groups = functional_groups or FUNCTIONAL_GROUPS
        self.modification_strategy = modification_strategy
        
        self.drug_like_priority = [
            'carboxyl', 'amino', 'amide', 'sulfonamide', 'hydroxyl',
            'methyl', 'halogen', 'ether', 'ketone', 'cyano'
        ]

    def identify_functional_groups(self, mol: Chem.Mol) -> Dict[str, List[int]]:
        if not mol:
            if self.log and self.logger:
                self.logger.error("Invalid molecule provided")
            return {}
            
        result = {}
        
        for fg_name, (smarts, _) in self.functional_groups.items():
            pattern = Chem.MolFromSmarts(smarts)
            if not pattern:
                if self.log and self.logger:
                    self.logger.error(f"Invalid SMARTS pattern for {fg_name}")
                continue
                
            matches = mol.GetSubstructMatches(pattern)
            if matches:
                central_atoms = []
                for match in matches:
                    central_atoms.append(match[0])
                
                result[fg_name] = central_atoms
                
        return result
    
    def functional_group_add_sites(self, mol: Union[Chem.Mol, Chem.RWMol], fg_name: str) -> List[int]:
        if not mol:
            if self.log and self.logger:
                self.logger.error("Invalid molecule provided")
            return []
            
        if fg_name not in self.functional_groups:
            if self.log and self.logger:
                self.logger.error(f"Unknown functional group: {fg_name}")
            return []
        
        rwmol = Chem.RWMol(mol)
        attachment_sites = []
        for atom in rwmol.GetAtoms():
            if atom.GetSymbol() in self.functional_groups[fg_name][1]:
                attachment_sites.append(atom.GetIdx())
        
        return attachment_sites

    def add_functional_group(self, mol: Union[Chem.Mol, Chem.RWMol], fg_name: str, site_idx: int) -> Optional[str]:
        if not mol:
            if self.log and self.logger:
                self.logger.error("Invalid molecule provided")
            return None
            
        if fg_name not in self.functional_groups:
            if self.log and self.logger:
                self.logger.error(f"Unknown functional group: {fg_name}")
            return None
        
        rwmol = Chem.RWMol(mol)
        
        try:
            site_atom = rwmol.GetAtomWithIdx(site_idx)
            site_symbol = site_atom.GetSymbol()
            
            current_valence = site_atom.GetExplicitValence() + site_atom.GetImplicitValence()
            available_valence = site_atom.GetTotalValence() - current_valence
            valence_required = self.functional_groups[fg_name][1]
            
            if available_valence < valence_required:
                if self.log and self.logger:
                    self.logger.error(f"Insufficient valence at site {site_idx} for {fg_name}")
                return Chem.MolToSmiles(mol)
            
            fg_smarts = self.functional_groups[fg_name][0]
            fg_mol = None
            
            fg_mapping = {
                'methyl': '[*:1]-[CH3]',
                'hydroxyl': '[*:1]-[OH]',
                'amino': '[*:1]-[NH2]',
                'carboxyl': '[*:1]-C(=O)O',
                'carbonyl': '[*:1]C=O',
                'aldehyde': '[*:1]-C=O',
                'ketone': '[*:1]-C(=O)-[#6]',
                'ether': '[*:1]-O-[CH3]',
                'ester': '[*:1]-C(=O)O[CH3]',
                'amide': '[*:1]-C(=O)N',
                'nitro': '[*:1]-[N+](=O)[O-]',
                'cyano': '[*:1]-C#N',
                'thiol': '[*:1]-[SH]',
                'halogen': '[*:1]-Cl', 
                'azide': '[*:1]-N=[N+]=[N-]',
                'sulfonamide': '[*:1]-S(=O)(=O)N',
                'sulfone': '[*:1]-S(=O)(=O)-[#6]',
                'phosphate': '[*:1]-P(=O)(O)O'
            }
            
            if fg_name in fg_mapping:
                rxn_smarts = fg_mapping[fg_name]
                rxn = AllChem.ReactionFromSmarts(f"[*:2][{site_symbol}:1]>>[{site_symbol}:1]{rxn_smarts.replace('[*:1]', '')}")
                products = rxn.RunReactants((rwmol,))
                
                if products and len(products) > 0 and len(products[0]) > 0:
                    modified_mol = products[0][0]
                    
                    try:
                        Chem.SanitizeMol(modified_mol)
                        Chem.AssignStereochemistry(modified_mol, cleanIt=True, force=True)
                        if self.log and self.logger:
                            self.logger.info(f"Successfully added {fg_name} at site {site_idx}")
                        return Chem.MolToSmiles(modified_mol)
                    except Exception as e:
                        if self.log and self.logger:
                            self.logger.error(f"Failed to sanitize after adding {fg_name}: {str(e)}")
                        return Chem.MolToSmiles(mol)
                else:
                    if self.log and self.logger:
                        self.logger.error(f"Reaction failed for adding {fg_name}")
                    return Chem.MolToSmiles(mol)
            else:
                if self.log and self.logger:
                    self.logger.error(f"No mapping available for functional group {fg_name}")
                return Chem.MolToSmiles(mol)
                
        except Exception as e:
            if self.log and self.logger:
                self.logger.error(f"Error adding functional group: {str(e)}")
            return Chem.MolToSmiles(mol)
            
    def remove_functional_group(self, mol: Union[Chem.Mol, Chem.RWMol], fg_name: str, instance_idx: int = 0) -> Optional[str]:
        if not mol:
            if self.log and self.logger:
                self.logger.error("Invalid molecule provided")
            return None
            
        if fg_name not in self.functional_groups:
            if self.log and self.logger:
                self.logger.error(f"Unknown functional group: {fg_name}")
            return None
        
        fg_matches = self.identify_functional_groups(mol)
        
        if fg_name not in fg_matches or instance_idx >= len(fg_matches[fg_name]):
            if self.log and self.logger:
                self.logger.error(f"Functional group {fg_name} instance {instance_idx} not found")
            return Chem.MolToSmiles(mol)
        
        fg_central_atom = fg_matches[fg_name][instance_idx]

        rwmol = Chem.RWMol(mol)
        
        try:
            removal_mapping = {
                'methyl': self._remove_simple_group,
                'hydroxyl': self._remove_simple_group,
                'amino': self._remove_simple_group,
                'carboxyl': self._remove_complex_group,
                'carbonyl': self._remove_complex_group,
                'aldehyde': self._remove_complex_group,
                'ketone': self._remove_complex_group,
                'ether': self._remove_complex_group,
                'ester': self._remove_complex_group,
                'amide': self._remove_complex_group,
                'nitro': self._remove_simple_group,
                'cyano': self._remove_simple_group,
                'thiol': self._remove_simple_group,
                'halogen': self._remove_simple_group,
                'azide': self._remove_simple_group,
                'sulfonamide': self._remove_complex_group
            }
            
            if fg_name in removal_mapping:
                modified_smiles = removal_mapping[fg_name](rwmol, fg_name, fg_central_atom)
            else:
                modified_smiles = self._remove_and_add_hydrogen(rwmol, fg_central_atom)
                
            if self.log and self.logger and modified_smiles != Chem.MolToSmiles(mol):
                self.logger.info(f"Successfully removed {fg_name} instance {instance_idx}")
                
            return modified_smiles
            
        except Exception as e:
            if self.log and self.logger:
                self.logger.error(f"Error removing functional group: {str(e)}")
            return Chem.MolToSmiles(mol)
    
    def _remove_simple_group(self, rwmol: Chem.RWMol, fg_name: str, atom_idx: int) -> str:
        smarts = self.functional_groups[fg_name][0]
        atom = rwmol.GetAtomWithIdx(atom_idx)
        
        pattern = Chem.MolFromSmarts(smarts)
        matches = rwmol.GetSubstructMatches(pattern)
        
        target_match = None
        for match in matches:
            if atom_idx in match:
                target_match = match
                break
        
        if target_match is None:
            return Chem.MolToSmiles(rwmol)
            
        mol_with_map = Chem.RWMol(rwmol)
        for i, idx in enumerate(target_match):
            mol_with_map.GetAtomWithIdx(idx).SetProp("molAtomMapNumber", str(i+1))
        
        attachment_idx = None
        for idx in target_match:
            atom = mol_with_map.GetAtomWithIdx(idx)
            for neighbor in atom.GetNeighbors():
                if neighbor.GetIdx() not in target_match:
                    attachment_idx = idx
                    break
            if attachment_idx is not None:
                break
        
        if attachment_idx is None:
            return Chem.MolToSmiles(rwmol)
        
        reactant_smiles = Chem.MolToSmiles(mol_with_map)
        rxn = AllChem.ReactionFromSmarts(f"{reactant_smiles}>>[*:1][H]")
        products = rxn.RunReactants((rwmol,))
        
        if products and len(products) > 0 and len(products[0]) > 0:
            modified_mol = products[0][0]
            try:
                Chem.SanitizeMol(modified_mol)
                Chem.AssignStereochemistry(modified_mol, cleanIt=True, force=True)
                return Chem.MolToSmiles(modified_mol)
            except Exception:
                return Chem.MolToSmiles(rwmol)
        else:
            return self._remove_and_add_hydrogen(rwmol, atom_idx)
    
    def _remove_complex_group(self, rwmol: Chem.RWMol, fg_name: str, atom_idx: int) -> str:
        removal_patterns = {
            'carboxyl': ['[C:1](=O)[OH]>>[*:1][H]', '[C](=O)[OH:1]>>[*:1][H]'],
            'carbonyl': ['[C:1]=O>>[*:1][H]', '[C]=O>>[H]'],
            'ester': ['[C:1](=O)[O][#6]>>[*:1][H]', '[C](=O)[O:1][#6]>>[*:1][H]'],
            'amide': ['[C:1](=O)[N]>>[*:1][H]', '[C](=O)[N:1]>>[*:1][H]'],
            'sulfonamide': ['[S:1](=O)(=O)[N]>>[*:1][H]', '[S](=O)(=O)[N:1]>>[*:1][H]']
        }
        
        if fg_name not in removal_patterns:
            return self._remove_and_add_hydrogen(rwmol, atom_idx)
        
        for pattern in removal_patterns[fg_name]:
            rxn = AllChem.ReactionFromSmarts(pattern)
            products = rxn.RunReactants((rwmol,))
            
            if products and len(products) > 0 and len(products[0]) > 0:
                modified_mol = products[0][0]
                try:
                    Chem.SanitizeMol(modified_mol)
                    Chem.AssignStereochemistry(modified_mol, cleanIt=True, force=True)
                    return Chem.MolToSmiles(modified_mol)
                except Exception:
                    continue
        
        return self._remove_and_add_hydrogen(rwmol, atom_idx)
    
    def _remove_and_add_hydrogen(self, rwmol: Chem.RWMol, atom_idx: int) -> str:
        original_mol = Chem.Mol(rwmol)
        
        atom = rwmol.GetAtomWithIdx(atom_idx)
        attachments = []
        
        for neighbor in atom.GetNeighbors():
            attachments.append(neighbor.GetIdx())
        
        for attach_idx in attachments:
            h_atom = Chem.Atom('H')
            new_idx = rwmol.AddAtom(h_atom)
            
            rwmol.AddBond(attach_idx, new_idx, Chem.BondType.SINGLE)
        
        try:
            rwmol.RemoveAtom(atom_idx)
            Chem.SanitizeMol(rwmol)
            Chem.AssignStereochemistry(rwmol, cleanIt=True, force=True)
            return Chem.MolToSmiles(rwmol)
        except Exception:
            return Chem.MolToSmiles(original_mol)
    
    def modify_functional_group(self, mol: Union[Chem.Mol, Chem.RWMol], 
                                 source_fg: str, target_fg: str, 
                                 instance_idx: int = 0) -> Optional[str]:
        if not mol:
            if self.log and self.logger:
                self.logger.error("Invalid molecule provided")
            return None
            
        if source_fg not in self.functional_groups or target_fg not in self.functional_groups:
            if self.log and self.logger:
                self.logger.error(f"Unknown functional group: {source_fg} or {target_fg}")
            return None
        
        fg_matches = self.identify_functional_groups(mol)
        
        if source_fg not in fg_matches or instance_idx >= len(fg_matches[source_fg]):
            if self.log and self.logger:
                self.logger.error(f"Functional group {source_fg} instance {instance_idx} not found")
            return Chem.MolToSmiles(mol)
        
        source_atom_idx = fg_matches[source_fg][instance_idx]
        
        compatibility = self._check_fg_compatibility(mol, source_fg, target_fg)
        if compatibility == 0:  
            if self.log and self.logger:
                self.logger.error(f"Incompatible transformation: {source_fg} to {target_fg}")
            return Chem.MolToSmiles(mol)
        
        transform_mapping = {
            # Format -- (source, target): reaction SMARTS
            ('hydroxyl', 'amino'): '[O:1][H]>>[N:1][H][H]',
            ('hydroxyl', 'thiol'): '[O:1][H]>>[S:1][H]',
            ('hydroxyl', 'methyl'): '[O:1][H]>>[C:1][H][H][H]',
            ('hydroxyl', 'halogen'): '[O:1][H]>>[Cl:1]',
            ('amino', 'hydroxyl'): '[N:1][H][H]>>[O:1][H]',
            ('amino', 'amide'): '[N:1][H][H]>>[N:1]C(=O)[H]',
            ('carboxyl', 'ester'): '[C:1](=O)[O][H]>>[C:1](=O)[O][C][H][H][H]',
            ('carboxyl', 'amide'): '[C:1](=O)[O][H]>>[C:1](=O)[N][H][H]',
            ('aldehyde', 'ketone'): '[C:1](=O)[H]>>[C:1](=O)[C][H][H][H]',
            ('aldehyde', 'carboxyl'): '[C:1](=O)[H]>>[C:1](=O)[O][H]',
            ('ketone', 'alcohol'): '[C:1](=O)[#6]>>[C:1]([O][H])[#6]',
            ('thiol', 'hydroxyl'): '[S:1][H]>>[O:1][H]',
            ('cyano', 'carboxyl'): '[C:1]#N>>[C:1](=O)[O][H]',
            ('nitro', 'amino'): '[N+:1](=O)[O-]>>[N:1][H][H]',
            ('ester', 'carboxyl'): '[C:1](=O)[O][#6]>>[C:1](=O)[O][H]',
            ('amide', 'carboxyl'): '[C:1](=O)[N]>>[C:1](=O)[O][H]'
        }
        
        key = (source_fg, target_fg)
        
        if key in transform_mapping:
            rxn_smarts = transform_mapping[key]
            rxn = AllChem.ReactionFromSmarts(rxn_smarts)
            
            products = rxn.RunReactants((mol,))
            if products and len(products) > 0 and len(products[0]) > 0:
                modified_mol = products[0][0]
                try:
                    Chem.SanitizeMol(modified_mol)
                    Chem.AssignStereochemistry(modified_mol, cleanIt=True, force=True)
                    if self.log and self.logger:
                        self.logger.info(f"Successfully transformed {source_fg} to {target_fg}")
                    return Chem.MolToSmiles(modified_mol)
                except Exception as e:
                    if self.log and self.logger:
                        self.logger.error(f"Failed to sanitize after transformation: {str(e)}")
        
        rwmol = Chem.RWMol(mol)
        
        attachment_idx = None
        source_atom = rwmol.GetAtomWithIdx(source_atom_idx)
        
        for neighbor in source_atom.GetNeighbors():
            if neighbor.GetIdx() not in fg_matches.get(source_fg, []):
                attachment_idx = neighbor.GetIdx()
                break
        
        if attachment_idx is None:
            if self.log and self.logger:
                self.logger.error("Could not identify attachment point for functional group")
            return Chem.MolToSmiles(mol)
        
        temp_mol = Chem.Mol(rwmol)
        removed_smiles = self.remove_functional_group(temp_mol, source_fg, instance_idx)
        if removed_smiles == Chem.MolToSmiles(mol):
            if self.log and self.logger:
                self.logger.error(f"Failed to remove {source_fg}")
            return Chem.MolToSmiles(mol)
            
        temp_mol = Chem.MolFromSmiles(removed_smiles)
        if not temp_mol:
            return Chem.MolToSmiles(mol)
            
        atom_mapping = {}
        for idx in range(min(mol.GetNumAtoms(), temp_mol.GetNumAtoms())):
            if idx < mol.GetNumAtoms() and idx < temp_mol.GetNumAtoms():
                atom1 = mol.GetAtomWithIdx(idx)
                atom2 = temp_mol.GetAtomWithIdx(idx)
                if atom1.GetSymbol() == atom2.GetSymbol():
                    atom_mapping[idx] = idx
        
        new_attachment_idx = None
        for old_idx, new_idx in atom_mapping.items():
            if old_idx == attachment_idx:
                new_attachment_idx = new_idx
                break
                
        if new_attachment_idx is None:
            if self.log and self.logger:
                self.logger.error("Could not find equivalent attachment point in modified molecule")
            return Chem.MolToSmiles(mol)
            
        final_smiles = self.add_functional_group(temp_mol, target_fg, new_attachment_idx)
        
        if final_smiles is None:
            if self.log and self.logger:
                self.logger.error(f"Failed to add {target_fg}")
            return Chem.MolToSmiles(mol)
            
        try:
            final_mol = Chem.MolFromSmiles(final_smiles)
            Chem.SanitizeMol(final_mol)
            if self.log and self.logger:
                self.logger.info(f"Successfully converted {source_fg} to {target_fg}")
            return final_smiles
        except Exception as e:
            if self.log and self.logger:
                self.logger.error(f"Final molecule invalid: {str(e)}")
            return Chem.MolToSmiles(mol)
    
    def _check_fg_compatibility(self, mol: Chem.Mol, source_fg: str, target_fg: str) -> int:
        compatibility = 2
        
        if source_fg in FG_COMPATIBILITY and target_fg in FG_COMPATIBILITY[source_fg]:
            compatibility = min(compatibility, FG_COMPATIBILITY[source_fg][target_fg])
        elif target_fg in FG_COMPATIBILITY and source_fg in FG_COMPATIBILITY[target_fg]:
            compatibility = min(compatibility, FG_COMPATIBILITY[target_fg][source_fg])
            
        fg_matches = self.identify_functional_groups(mol)
        
        for other_fg in fg_matches:
            if other_fg == source_fg:
                continue
                
            if target_fg in FG_COMPATIBILITY and other_fg in FG_COMPATIBILITY[target_fg]:
                compatibility = min(compatibility, FG_COMPATIBILITY[target_fg][other_fg])
            elif other_fg in FG_COMPATIBILITY and target_fg in FG_COMPATIBILITY[other_fg]:
                compatibility = min(compatibility, FG_COMPATIBILITY[other_fg][target_fg])
                
        return compatibility
    
    def suggest_modifications(self, mol: Chem.Mol, 
                              target_property: str = None, 
                              num_suggestions: int = 3) -> List[Tuple[str, str, str]]:
        if not mol:
            if self.log and self.logger:
                self.logger.error("Invalid molecule provided")
            return []
            
        suggestions = []
        
        fg_matches = self.identify_functional_groups(mol)
        existing_fgs = list(fg_matches.keys())
        
        carbon_indices = [atom.GetIdx() for atom in mol.GetAtoms() 
                         if atom.GetSymbol() == 'C' and atom.GetDegree() < 4]
        
        if target_property == 'solubility':
            # Add hydrophilic groups
            hydrophilic_fgs = ['hydroxyl', 'amino', 'carboxyl', 'amide', 'sulfone']
            for fg in hydrophilic_fgs:
                if len(suggestions) < num_suggestions and fg not in existing_fgs and carbon_indices:
                    suggestions.append(('add', fg, f"Adding {fg} group will increase water solubility"))
                    
            # Replace hydrophobic groups with hydrophilic
            hydrophobic_fgs = ['methyl', 'halogen', 'alkene', 'alkyne']
            for fg in hydrophobic_fgs:
                if fg in existing_fgs and len(suggestions) < num_suggestions:
                    target = random.choice(hydrophilic_fgs)
                    suggestions.append(('modify', f"{fg} to {target}", 
                                      f"Converting {fg} to {target} will increase hydrophilicity"))
                    
        elif target_property == 'stability':
            # Add groups that increase stability
            stability_fgs = ['methyl', 'ether', 'amide']
            for fg in stability_fgs:
                if len(suggestions) < num_suggestions and fg not in existing_fgs and carbon_indices:
                    suggestions.append(('add', fg, f"Adding {fg} group will increase chemical stability"))
                    
            # Remove reactive groups
            reactive_fgs = ['aldehyde', 'alkene', 'alkyne', 'azide', 'isocyanate']
            for fg in reactive_fgs:
                if fg in existing_fgs and len(suggestions) < num_suggestions:
                    suggestions.append(('remove', fg, f"Removing {fg} will reduce reactivity and increase stability"))
                    
        elif target_property == 'reactivity':
            # Add reactive groups
            reactive_fgs = ['hydroxyl', 'amino', 'carbonyl', 'carboxyl']
            for fg in reactive_fgs:
                if len(suggestions) < num_suggestions and fg not in existing_fgs and carbon_indices:
                    suggestions.append(('add', fg, f"Adding {fg} group will increase reactivity"))
        
        elif target_property == 'drug-like':
            # Add common drug-like groups
            for fg in self.drug_like_priority:
                if len(suggestions) < num_suggestions and fg not in existing_fgs and carbon_indices:
                    suggestions.append(('add', fg, f"Adding {fg} group is common in drug molecules"))
                    
            mol_weight = Chem.Descriptors.MolWt(mol)
            if mol_weight > 500 and len(suggestions) < num_suggestions:
                suggestions.append(('general', 'reduce size', 
                                  "Molecule may be too large for drug-likeness; consider simplifying structure"))
                                  
        while len(suggestions) < num_suggestions:
            if not existing_fgs and not carbon_indices:
                break
                
            if existing_fgs and random.random() < 0.7:
                source_fg = random.choice(existing_fgs)
                target_options = [fg for fg in self.functional_groups if fg != source_fg]
                if target_options:
                    target_fg = random.choice(target_options)
                    suggestions.append(('modify', f"{source_fg} to {target_fg}", 
                                      f"Changing {source_fg} to {target_fg} will alter molecular properties"))
            elif carbon_indices:
                fg = random.choice(list(self.functional_groups.keys()))
                if fg not in existing_fgs:
                    suggestions.append(('add', fg, f"Adding {fg} group will introduce new functionality"))
                    
        return suggestions[:num_suggestions]
    
    def apply_suggestion(self, mol: Chem.Mol, suggestion: Tuple[str, str, str]) -> Optional[str]:
        if not mol:
            if self.log and self.logger:
                self.logger.error("Invalid molecule provided")
            return None
            
        mod_type, fg_info, _ = suggestion
        
        if mod_type == 'add':
            carbon_indices = [atom.GetIdx() for atom in mol.GetAtoms() 
                             if atom.GetSymbol() == 'C' and atom.GetDegree() < 4]
            
            if not carbon_indices:
                if self.log and self.logger:
                    self.logger.error("No suitable carbon attachment sites found")
                return Chem.MolToSmiles(mol)
                
            site_idx = random.choice(carbon_indices)
            return self.add_functional_group(mol, fg_info, site_idx)
            
        elif mod_type == 'remove':
            fg_matches = self.identify_functional_groups(mol)
            if fg_info not in fg_matches:
                if self.log and self.logger:
                    self.logger.error(f"Functional group {fg_info} not found")
                return Chem.MolToSmiles(mol)
                
            return self.remove_functional_group(mol, fg_info)
            
        elif mod_type == 'modify':
            parts = fg_info.split(' to ')
            if len(parts) != 2:
                if self.log and self.logger:
                    self.logger.error(f"Invalid modification format: {fg_info}")
                return Chem.MolToSmiles(mol)
                
            source_fg, target_fg = parts
            
            fg_matches = self.identify_functional_groups(mol)
            if source_fg not in fg_matches:
                if self.log and self.logger:
                    self.logger.error(f"Source functional group {source_fg} not found")
                return Chem.MolToSmiles(mol)
                
            return self.modify_functional_group(mol, source_fg, target_fg)
            
        return Chem.MolToSmiles(mol)

    def batch_process(self, smiles_list: List[str], modifications: List[Tuple[str, str, int]]) -> List[str]:
        results = []
        
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                if self.log and self.logger:
                    self.logger.error(f"Invalid SMILES: {smiles}")
                results.append(smiles)
                continue
                
            curr_mol = mol
            
            for operation, fg, param in modifications:
                if operation == 'add':
                    new_smiles = self.add_functional_group(curr_mol, fg, param)
                    if new_smiles:
                        curr_mol = Chem.MolFromSmiles(new_smiles)
                        
                elif operation == 'remove':
                    new_smiles = self.remove_functional_group(curr_mol, fg, param)
                    if new_smiles:
                        curr_mol = Chem.MolFromSmiles(new_smiles)
                        
                elif operation == 'modify':
                    new_smiles = self.modify_functional_group(curr_mol, fg, param)
                    if new_smiles:
                        curr_mol = Chem.MolFromSmiles(new_smiles)
                        
            results.append(Chem.MolToSmiles(curr_mol))
            
        return results


def main():
    import logging
    
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('FunctionalGroupModifier')
    
    modifier = ModifyFunctionalGroup(logger=logger, log=True)
    
    aspirin = 'CC(=O)OC1=CC=CC=C1C(=O)O'  # Aspirin
    caffeine = 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'  # Caffeine
    ibuprofen = 'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O'  # Ibuprofen
    
    print("Testing functional group identification:")
    mol = Chem.MolFromSmiles(aspirin)
    fg_matches = modifier.identify_functional_groups(mol)
    print(f"Found functional groups in aspirin: {fg_matches}")
    
    print("\nTesting addition of functional group:")
    new_smiles = modifier.add_functional_group(mol, 'amino', 1)
    print(f"Added amino group: {new_smiles}")
    
    print("\nTesting removal of functional group:")
    new_smiles = modifier.remove_functional_group(mol, 'carboxyl')
    print(f"Removed carboxyl group: {new_smiles}")
    
    print("\nTesting modification of functional group:")
    new_smiles = modifier.modify_functional_group(mol, 'carboxyl', 'amide')
    print(f"Modified carboxyl to amide: {new_smiles}")
    
    print("\nTesting suggestion of modifications:")
    suggestions = modifier.suggest_modifications(mol, target_property='solubility')
    for mod_type, fg_info, rationale in suggestions:
        print(f"Suggestion: {mod_type} {fg_info} - {rationale}")
        
    print("\nBatch processing example:")
    modifications = [('add', 'hydroxyl', 5), ('remove', 'carboxyl', 0)]
    results = modifier.batch_process([aspirin, caffeine, ibuprofen], modifications)
    for i, result in enumerate(results):
        print(f"Molecule {i+1} modified: {result}")

if __name__ == "__main__":
    main()