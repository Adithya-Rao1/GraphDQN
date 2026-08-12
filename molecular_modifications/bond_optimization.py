from molecular_modifications.modification_imports import *
from molecular_modifications.logger import setup_molecule_logger

class ModifyBond:
    def __init__(self, logger, log: bool = False):
        self.logger = logger
        self.log = log

        self.bond_types = [Chem.BondType.SINGLE,
                      Chem.BondType.DOUBLE,
                      Chem.BondType.TRIPLE,
                      Chem.BondType.AROMATIC]
        
    def optimize_bond(self, mol, action, batch=False):
        rwmol = Chem.RWMol(mol)
        atom_indices = self.get_optimal_bond_sites(rwmol)
        bond_indices = self.get_bond_indices(rwmol)

        action_map = {
            0: 'add',
            1: 'remove',
            2: 'modify',
            3: 'random'
        }

        modification_strategies = {
            'modify': self._modify_bond,
            'add': self._addbond,
            'remove': self._removebond,
            'random': random.choice([
                self._modify_bond,
                self._addbond,
                self._removebond
            ])
        }

        chosen_action = action_map.get(action)

        strategy = modification_strategies.get(chosen_action, modification_strategies['random'])

        for atom_idx_pair in atom_indices:
                for bond_idx in bond_indices:
                    try:
                        if strategy == self._addbond:
                            modified_mol = strategy(mol, rwmol, atom_idx_pair)
                        else:
                            modified_mol = strategy(mol, rwmol, atom_idx_pair, bond_indices, batch)
                        if modified_mol != mol:
                            if self.log:
                                self.logger.info(f"Successful bond {chosen_action} operation.")
                            return Chem.MolToSmiles(modified_mol)
                    except Exception as e:
                        if self.log:
                            self.logger.warning(f"{chosen_action.capitalize()} failed for atom_idx_pair {atom_idx_pair}, bond_idx {bond_idx}: {e}")
                        return Chem.MolToSmiles(mol)
        
        if self.log:
            self.logger.error("All bond modification attempts failed.")
        return Chem.MolToSmiles(mol)

    def _modify_bond(self, mol, rwmol, atom_idx_pair, bond_indices, batch):
        for bond in self.bond_types:
            new_bond_type = bond
            if batch:
                successful_modifications = 0
                for bond_idx in bond_indices:
                    bond = rwmol.GetBondWithIdx(bond_idx)
                    if self.bond_rule_fn and not self.bond_rule_fn(bond):
                        continue
                    try:
                        rwmol.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
                        rwmol.AddBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), new_bond_type)
                        Chem.SanitizeMol(rwmol)
                        successful_modifications += 1
                    except Exception as e:
                        if self.log:
                            self.logger.warning(f"Modification to bond {bond_idx} caused an error: {e}. Final molecule not fully sanitized.")
                        return mol
                    
                if successful_modifications == 0:
                    if self.log:
                        self.logger.error("No bond modifications were successful.")
                    return mol
            
            else:
                bond = self._get_bond_by_indices(rwmol, atom_idx_pair)
                if bond:
                    bond_idx = bond.GetIdx()
                    if not self.bond_rule_fn or self.bond_rule_fn(bond):  
                        try:
                            rwmol.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
                            rwmol.AddBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), new_bond_type)
                            Chem.SanitizeMol(rwmol)
                            return Chem.Mol(rwmol)
                        except Exception as e:
                            if self.log:
                                self.logger.warning(f"Modification to bond {bond_idx} caused an error: {e}. Final molecule not fully sanitized.")
                            return mol
                            
    def _addbond(self, mol, rwmol, atom_idx_pair):
        for bond in self.bond_types:
            new_bond_type = bond

            try:
                rwmol = self._add_bond(mol, rwmol, atom_idx_pair[0], atom_idx_pair[1], new_bond_type)
                return Chem.Mol(rwmol)
            except Exception as e:
                if self.log:
                    self.logger.error(f"Bond addition failed due to error {e}.")
                return mol
            
    def _removebond(self, mol, rwmol, atom_idx_pair, bond_indices, batch):
        if batch:
            for bond_idx in bond_indices:
                bond = rwmol.GetBondWithIdx(bond_idx)
                if self.bond_rule_fn and not self.bond_rule_fn(bond):  
                    continue
                rwmol = self._remove_bond(mol, rwmol, bond_idx)
                return Chem.Mol(rwmol)
        else:
            bond = self._get_bond_by_indices(rwmol, atom_idx_pair)
            if bond:
                if not self.bond_rule_fn or self.bond_rule_fn(bond):  
                    rwmol = self._remove_bond(mol, rwmol, bond.GetIdx())
                    return Chem.Mol(rwmol)

    def get_optimal_bond_sites(self, mol, allowed_bond_types=None, exclude_hydrogens=True):
        if allowed_bond_types is None:
            allowed_bond_types = self.bond_types

        optimal_sites = []

        for atom_i in mol.GetAtoms():
            if exclude_hydrogens and atom_i.GetAtomicNum() == 1:
                continue
            for atom_j in mol.GetAtoms():
                if atom_j.GetIdx() <= atom_i.GetIdx():  
                    continue
                if exclude_hydrogens and atom_j.GetAtomicNum() == 1:
                    continue

                bond = mol.GetBondBetweenAtoms(atom_i.GetIdx(), atom_j.GetIdx())
                if bond:
                    if bond.GetBondType() in allowed_bond_types:
                        optimal_sites.append((atom_i.GetIdx(), atom_j.GetIdx()))
                else:
                    valence_i = VALENCE_ELECTRON_COUNTS.get(atom_i.GetSymbol(), None)
                    valence_j = VALENCE_ELECTRON_COUNTS.get(atom_j.GetSymbol(), None)
                    if valence_i and valence_j:
                        if atom_i.GetExplicitValence() < valence_i and atom_j.GetExplicitValence() < valence_j:
                            optimal_sites.append((atom_i.GetIdx(), atom_j.GetIdx()))

        return optimal_sites
    
    def get_bond_indices(self, mol, atom_indices=None, bond_order=None, atom_types=None, aromatic=None, rule_fn=None):
        if not mol or not isinstance(mol, Chem.Mol):
            if self.log:
                self.logger.error("Invalid molecule object provided.")
            return []

        bond_indices = []
        
        for bond in mol.GetBonds():
            bond_idx = bond.GetIdx()
            atom1 = bond.GetBeginAtom()
            atom2 = bond.GetEndAtom()
            atom1_idx = atom1.GetIdx()
            atom2_idx = atom2.GetIdx()

            if atom_indices:
                if tuple(sorted(atom_indices)) != tuple(sorted((atom1_idx, atom2_idx))):
                    continue

            if bond_order and bond.GetBondType() != bond_order:
                continue

            if atom_types:
                if atom1.GetSymbol() not in atom_types and atom2.GetSymbol() not in atom_types:
                    continue

            if aromatic is not None:
                if bond.GetIsAromatic() != aromatic:
                    continue

            if rule_fn and not rule_fn(bond):
                continue

            bond_indices.append(bond_idx)

        if not bond_indices:
            if self.log:
                self.logger.error("No bonds matched the filtering criteria.")
            return []
        
        return bond_indices

    def _get_bond_by_indices(self, mol, atom_indices):
        if not isinstance(atom_indices, (list, tuple)) or len(atom_indices) != 2:
            if self.log:
                self.logger.error("atom_indices must be a tuple of two integers.")
            return None
        if any(idx < 0 or idx >= mol.GetNumAtoms() for idx in atom_indices):
            if self.log:
                self.logger.error("Atom indices are out of bounds for the molecule.")
            return None
        
        bond = mol.GetBondBetweenAtoms(atom_indices[0], atom_indices[1])
        if bond is None:
            if self.log:
                self.logger.error(f"No bond exists between atoms {atom_indices[0]} and {atom_indices[1]}.")
        
        return bond

    def _add_bond(self, mol, rwmol, atom1_idx, atom2_idx, bond_type):
        if rwmol.GetBondBetweenAtoms(atom1_idx, atom2_idx) is not None:
            if self.log:
                self.logger.error(f"A bond already exists between atoms {atom1_idx} and {atom2_idx}.")
            return mol
        if not isinstance(bond_type, Chem.BondType):
            if self.log:
                self.logger.error("bond_type must be an rdkit.Chem.BondType.")
            return mol

        try:
            rwmol.AddBond(atom1_idx, atom2_idx, bond_type)

            for atom in rwmol.GetAtoms():
                explicit_valence = atom.GetExplicitValence()
                total_valence = atom.GetTotalValence()

                if explicit_valence > total_valence:
                    if self.log:
                        num_missing_h = explicit_valence - total_valence
                        self.logger.error(
                            f"Atom {atom.GetIdx()} has invalid valence. "
                            f"Adding {num_missing_h} hydrogen(s) to correct."
                        )
                    rwmol = Chem.AddHs(rwmol, onlyOnAtoms=[atom.GetIdx()])

            try:
                Chem.SanitizeMol(rwmol)
                return rwmol
            except Exception as e: 
                if self.log:   
                    self.logger.warning("Final molecule may not be fully sanitized.")
                return mol
            
        except Exception as e:
            if self.log:
                self.logger.error(f"Failed to add bond between atoms {atom1_idx} and {atom2_idx}: {e}")
            return mol
        
    def _remove_bond(self, mol, rwmol, bond_idx):
        if bond_idx < 0 or bond_idx >= rwmol.GetNumBonds():
            if self.log:
                self.logger.error(f"Bond index {bond_idx} is out of bounds for the molecule.")
            return mol
        
        bond = rwmol.GetBondWithIdx(bond_idx)
        if not bond:
            if self.log:
                self.logger.error(f"No bond exists with index {bond_idx}.")
            return mol
        
        try:
            rwmol.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            
            for atom in rwmol.GetAtoms():
                explicit_valence = atom.GetExplicitValence()
                total_valence = atom.GetTotalValence()

                if explicit_valence > total_valence:
                    num_missing_h = explicit_valence - total_valence
                    self.logger.error(
                        f"Atom {atom.GetIdx()} has invalid valence. "
                        f"Adding {num_missing_h} hydrogen(s) to correct."
                    )
                    rwmol = Chem.AddHs(rwmol, onlyOnAtoms=[atom.GetIdx()])
                
            try:
                Chem.SanitizeMol(rwmol)
                return rwmol
            except Exception:
                if self.log:
                    self.logger.warning("Final molecule may not be fully sanitized.")
                return mol
            
        except Exception as e:
            if self.log:
                self.logger.error(f"Failed to remove bond with index {bond_idx}: {e}")
            return mol
        
    def _bond_order(bond_type):
        if bond_type == rdchem.BondType.SINGLE:
            return 1
        elif bond_type == rdchem.BondType.DOUBLE:
            return 2
        elif bond_type == rdchem.BondType.TRIPLE:
            return 3
        elif bond_type == rdchem.BondType.AROMATIC:
            return 1.5
        return 0
    
    def bond_rule_fn(self, bond):
        atom1 = bond.GetBeginAtom()
        atom2 = bond.GetEndAtom()
        symbol1, symbol2 = atom1.GetSymbol(), atom2.GetSymbol()
        mol = bond.GetOwningMol()

        def _electronegativity_check():
            def calculate_bond_polarity(en1, en2):
                return abs(en1 - en2)
            
            en1 = ELECTRONEGATIVITY.get(symbol1, None)
            en2 = ELECTRONEGATIVITY.get(symbol2, None)
            
            if en1 is None or en2 is None:
                return False 
            
            polarity = calculate_bond_polarity(en1, en2)
            
            return 0.5 <= polarity <= 2.0

        def _steric_hindrance_check():
            radius1 = COVALENT_RADII.get(symbol1)
            radius2 = COVALENT_RADII.get(symbol2)
            
            hybridization_factor = {
                HybridizationType.SP: 1.2,
                HybridizationType.SP2: 1.1,
                HybridizationType.SP3: 1.0
            }
            
            h1_factor = hybridization_factor.get(atom1.GetHybridization())
            h2_factor = hybridization_factor.get(atom2.GetHybridization())
            
            adjusted_radius = (radius1 * h1_factor + radius2 * h2_factor) / 2
            
            if atom1.GetDegree() >= 4 or atom2.GetDegree() >= 4:
                return False
            
            return adjusted_radius > 0.5

        def _ring_strain_analysis():
            if not bond.IsInRing():
                return True
            
            ring_info = mol.GetRingInfo()
            ring_size = max(len(ring) for ring in ring_info.AtomRings() 
                            if atom1.GetIdx() in ring and atom2.GetIdx() in ring)
            
            ring_strain_limits = {
                3: 60,   # Cyclopropane (very strained)
                4: 45,   # Cyclobutane (high strain)
                5: 15,   # Cyclopentane (moderate strain)
                6: 5,    # Cyclohexane (minimal strain)
                7: 10    # Cycloheptane (slight strain)
            }
            
            return ring_size > 4 or ring_strain_limits.get(ring_size, 100) < 30

        def _valence_compliance():
            def check_atom_bond_counts(atom):
                valence = VALENCE_ELECTRON_COUNTS.get(atom.GetSymbol())
                return atom.GetDegree() <= valence
            
            return check_atom_bond_counts(atom1) and check_atom_bond_counts(atom2)

        def _bond_type_compatibility():
            current_bond_type = bond.GetBondType()
            
            if current_bond_type in [Chem.BondType.TRIPLE, Chem.BondType.AROMATIC]:
                return False
            
            forbidden_bonds = {
                frozenset(['H', 'H']): True,
                frozenset(['H', 'F']): True,
                frozenset(['F', 'F']): True,
                frozenset(['O', 'O']): True
            }
            
            return not forbidden_bonds.get(frozenset([symbol1, symbol2]), False)

        rule_checks = [
            _electronegativity_check(),
            _steric_hindrance_check(),
            _ring_strain_analysis(),
            _valence_compliance(),
            _bond_type_compatibility()
        ]
        
        return all(rule_checks)

if __name__ == "__main__":
    smiles_list = ['C1CCCCC1', 'C1=CC=C(C=C1)O', 'CC(=O)OC1=CC=CC=C1C(=O)O',
            'N#CC1=C(C(F)(F)F)C([N+]([O-])=O)=C(C2=CC=CC([N+]([O-])=O)=C2)NC1=O',
            'O=C(NC1=CC=C(NC(NC2=CC=CC3=C2C=CN3)=O)C=C1)NC4=C(C=CN5)C5=CC=C4',
            'O=C1N(C)C(CNCC)=NC2=C1C(Cl)=CC(Cl)=C2O.Br',
            'O=C(C(C=C1C2=O)=CC(O)=C1C(C3=C2C=CC=C3O)=O)NC4=CC=C(Cl)C=C4O',
            'OC1=CC=C(CC2=CC=C(C(CC3=CC=C(C(CC4=CC=C(C=C4)O)=C3)O)=C2)O)C=C1',
            'C[C@]1(CS(=O)(=O)N(C(=N1)N)C)C2=C(C=CC(=C2)NC(=O)C3=NC=C(C=C3)F)F',
            'CC#CC1=CC(=CN=C1)C2=CC3=C(CC4(C35N=C(C(=N5)N)C)CCC(CC4)OC)C=C2',
            'C[C@]1(C=CSC(=N1)N)C2=C(C=CC(=C2)NC(=O)C3=NC=C(C=C3)C#N)F',
            'C[C@@H]1[C@H]2CSC(=N[C@]2(CO1)C3=C(C=CC(=C3)NC(=O)C4=NC=C(N=C4)C(F)F)F)N',
            'O=C(C(C1=CN2CCN(C(N3CCCCC3)=O)CC4=CC(F)=CC1=C42)=C5C6=CN=C7C=CC=CN76)NC5=O',
            'FC(F)(C1=CC(CSC2=NN=C(C3=CC4=C(N=CS4)C=C3)O2)=CC=C1OC)F',
            'O=C(C1CC1)NC2=NC=CC(C3=CC=C(C4=NOC=N4)S3)=C2',
            'O=C(N(SC1=O)C2=C3C=CC=CC3=CC=C2)N1CC4=CC=CC=C4',
        ]
    
    modified_mols = []
    modify_bond = ModifyBond(setup_logger())
    for smile in smiles_list:
        mol = Chem.MolFromSmiles(smile)
        
        atom_indices = modify_bond.get_optimal_bond_sites(mol)
        bond_indices = modify_bond.get_bond_indices(mol)
        print(atom_indices, bond_indices)
        
        modified_mol = modify_bond.optimize_bond(mol, 1, False)
        modified_mols.append(Chem.MolToSmiles(modified_mol))
        print(modify_bond.get_bond_indices(mol))
        
        bond_by_idx = modify_bond._get_bond_by_indices(mol, (0,1))
        print(bond_by_idx)

    print(modified_mols)
    from collections import Counter
    if Counter(smiles_list) == Counter(modified_mols):
        print("Lists are same")
    else:
        print("Lists are not same")
