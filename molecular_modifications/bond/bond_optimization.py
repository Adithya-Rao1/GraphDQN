import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from molecular_modifications.modification_imports import *
from molecular_modifications.logger import setup_molecule_logger

class ModifyBond:
    def __init__(self, logger, log: bool = False):
        """
        Initialize the ModifyBond class.

        Args:
            logger (logging.Logger): The logger for logging messages.
        """
        self.logger = logger
        self.log = log

        self.bond_types = [Chem.BondType.SINGLE,
                      Chem.BondType.DOUBLE,
                      Chem.BondType.TRIPLE,
                      Chem.BondType.AROMATIC]
        
    def optimize_bond(self, mol, action, batch=False):
        """
        Modify bonds in a molecule by adding, removing, or changing them, with chemical validation.

        Args:
            mol (rdkit.Chem.Mol): The input molecule.
            action (str): 'modify', 'add', or 'remove'. Default is 'modify'.
            batch (bool): If True, perform batch modifications/removals.
            
        Returns:
            rdkit.Chem.Mol: The modified molecule.
        """
        rwmol = Chem.RWMol(mol)
        atom_indices = self.get_optimal_bond_sites(rwmol)
        bond_indices = self.get_bond_indices(rwmol)

        action_map = {
            0: 'add',
            1: 'remove',
            2: 'modify',
            3: 'random'
        }

        # Choose strategy
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
        """
        Modify bonds in a molecule by changing their type, with chemical validation.

        Args:
            rwmol (rdkit.Chem.RWMol): The input molecule.
            atom_idx_pair (tuple): Tuple of two atom indices to modify the bond between.

        Returns:
            rdkit.Chem.Mol: The modified molecule.
        """
        for bond in self.bond_types:
            new_bond_type = bond
            if batch:
                successful_modifications = 0
                for bond_idx in bond_indices:
                    bond = rwmol.GetBondWithIdx(bond_idx)
                    if self.bond_rule_fn and not self.bond_rule_fn(bond):
                        continue
                    try:
                        # Remove the old bond
                        rwmol.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
                        # Add the new bond with the new type
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
                    if not self.bond_rule_fn or self.bond_rule_fn(bond):  # Only modify if rule_fn allows
                        try:
                            # Remove the old bond
                            rwmol.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
                            # Add the new bond with the new type
                            rwmol.AddBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), new_bond_type)
                            Chem.SanitizeMol(rwmol)
                            return Chem.Mol(rwmol)
                        except Exception as e:
                            if self.log:
                                self.logger.warning(f"Modification to bond {bond_idx} caused an error: {e}. Final molecule not fully sanitized.")
                            return mol
                            
    def _addbond(self, mol, rwmol, atom_idx_pair):
        """
        Add bonds in a molecule with chemical validation.

        Args:
            rwmol (rdkit.Chem.RWMol): The input molecule.
            atom_idx_pair (tuple): Tuple of two atom indices to add the bond between.

        Returns:
            rdkit.Chem.Mol: The modified molecule.
        """
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
        """
        Remove bonds in a molecule with chemical validation.

        Args:
            rwmol (rdkit.Chem.RWMol): The input molecule.
            atom_idx_pair (tuple): Tuple of two atom indices to remove the bond between.

        Returns:
            rdkit.Chem.Mol: The modified molecule.
        """
        if batch:
            for bond_idx in bond_indices:
                bond = rwmol.GetBondWithIdx(bond_idx)
                if self.bond_rule_fn and not self.bond_rule_fn(bond):  # Only remove if rule_fn allows
                    continue
                rwmol = self._remove_bond(mol, rwmol, bond_idx)
                return Chem.Mol(rwmol)
        else:
            bond = self._get_bond_by_indices(rwmol, atom_idx_pair)
            if bond:
                if not self.bond_rule_fn or self.bond_rule_fn(bond):  # Only remove if rule_fn allows
                    rwmol = self._remove_bond(mol, rwmol, bond.GetIdx())
                    return Chem.Mol(rwmol)

    def get_optimal_bond_sites(self, mol, allowed_bond_types=None, exclude_hydrogens=True):
        """
        Identifies optimal atom pairs in a molecule for bonding, either as existing bonds or potential bonds.

        Args:
            mol (rdkit.Chem.Mol): The input RDKit molecule.
            allowed_bond_types (list): List of allowed bond types (e.g., [Chem.BondType.SINGLE, Chem.BondType.DOUBLE]).
                                    If None, all bond types are considered.
            exclude_hydrogens (bool): If True, excludes hydrogen atoms from consideration.

        Returns:
            list of tuples: A list of atom index pairs (i, j) representing optimal bonding sites.
        """
        if allowed_bond_types is None:
            allowed_bond_types = self.bond_types

        optimal_sites = []

        for atom_i in mol.GetAtoms():
            if exclude_hydrogens and atom_i.GetAtomicNum() == 1:
                continue
            for atom_j in mol.GetAtoms():
                if atom_j.GetIdx() <= atom_i.GetIdx():  # Avoid duplicate pairs
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
        """
        Get a list of bond indices from the molecule based on filtering criteria.

        Args:
            mol (rdkit.Chem.Mol): The molecule object to analyze.
            atom_indices (tuple, optional): A tuple of atom indices (e.g., (atom1_idx, atom2_idx)) to filter specific bonds.
                                            If provided, only bonds between these atoms are returned.
            bond_order (rdkit.Chem.BondType, optional): Filter bonds by bond order (e.g., SINGLE, DOUBLE, etc.).
            atom_types (list of str, optional): List of atomic symbols (e.g., ["C", "O"]) to filter bonds involving these atoms.
            aromatic (bool, optional): If True, return only aromatic bonds. If False, return only non-aromatic bonds.
            rule_fn (function, optional): A custom rule function that takes a bond object and returns a boolean to filter bonds.

        Returns:
            list: A list of bond indices that match the filtering criteria.
        """
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
        """
        Retrieve a bond by atom indices.

        Args:
            mol (rdkit.Chem.Mol): The input molecule.
            atom_indices (tuple): A tuple containing two atom indices.

        Returns:
            rdkit.Chem.Bond: The bond object if it exists, or None if no bond exists between the atoms.
        """
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
        """
        Add a bond between two atoms.

        Args:
            rwmol (rdkit.Chem.RWMol): The molecule object in editable form.
            atom1_idx (int): Index of the first atom.
            atom2_idx (int): Index of the second atom.
            bond_type (rdkit.Chem.BondType): The bond type to add.

        Raises:
            ValueError: If a bond already exists or the bond type is invalid.
        """
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
        """
        Remove a bond by its index and handle valence issues.

        Args:
            mol (rdkit.Chem.RWMol): The molecule object in editable form.
            bond_idx (int): The index of the bond to remove.

        Raises:
            ValueError: If the bond index is invalid or out of bounds.
            RuntimeError: If removal fails or if the resulting molecule is chemically invalid.
        """
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
        """Return the bond order of a bond type."""
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
        """
        A comprehensive rule function for evaluating bond modification feasibility,
        considering multiple chemical constraints and contextual factors.
        
        Args:
            bond (rdkit.Chem.Bond): The bond object to evaluate.
        
        Returns:
            bool: True if the bond meets all chemical modification rules.
        """
        atom1 = bond.GetBeginAtom()
        atom2 = bond.GetEndAtom()
        symbol1, symbol2 = atom1.GetSymbol(), atom2.GetSymbol()
        mol = bond.GetOwningMol()

        def _electronegativity_check():
            """
            Electronegativity-based bond evaluation.
            
            Returns:
                bool: True if the bond electronegativity is chemically reasonable.
            """
            def calculate_bond_polarity(en1, en2):
                """Calculate bond polarity based on electronegativity difference."""
                return abs(en1 - en2)
            
            # Retrieve electronegativities
            en1 = ELECTRONEGATIVITY.get(symbol1, None)
            en2 = ELECTRONEGATIVITY.get(symbol2, None)
            
            if en1 is None or en2 is None:
                return False  # Unknown element, conservative approach
            
            polarity = calculate_bond_polarity(en1, en2)
            
            return 0.5 <= polarity <= 2.0

        def _steric_hindrance_check():
            """
            Steric hindrance evaluation.
            
            Returns:
                bool: True if the bond modification avoids significant steric clash.
            """
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
            """
            Ring strain evaluation.
            
            Returns:
                bool: True if bond modification doesn't introduce severe ring strain.
            """
            if not bond.IsInRing():
                return True
            
            ring_info = mol.GetRingInfo()
            ring_size = max(len(ring) for ring in ring_info.AtomRings() 
                            if atom1.GetIdx() in ring and atom2.GetIdx() in ring)
            
            ring_strain_limits = {
                3: 60,   # Cyclopropane (extremely strained)
                4: 45,   # Cyclobutane (high strain)
                5: 15,   # Cyclopentane (moderate strain)
                6: 5,    # Cyclohexane (minimal strain)
                7: 10    # Cycloheptane (slight strain)
            }
            
            # Conservative approach for smaller rings
            return ring_size > 4 or ring_strain_limits.get(ring_size, 100) < 30

        def _valence_compliance():
            """
            Valence and bonding state evaluation.
            
            Returns:
                bool: True if bond modification maintains valence rules.
            """
            def check_atom_bond_counts(atom):
                valence = VALENCE_ELECTRON_COUNTS.get(atom.GetSymbol())
                return atom.GetDegree() <= valence
            
            return check_atom_bond_counts(atom1) and check_atom_bond_counts(atom2)

        def _bond_type_compatibility():
            """
            Evaluate bond type modification compatibility.
            
            Returns:
                bool: True if bond modification is chemically sensible.
            """
            current_bond_type = bond.GetBondType()
            
            if current_bond_type in [Chem.BondType.TRIPLE, Chem.BondType.AROMATIC]:
                return False
            
            # Specific element pair constraints
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

'''if __name__ == "__main__":
    smiles_list = ['N#CC1=C(C(F)(F)F)C([N+]([O-])=O)=C(C2=CC=CC([N+]([O-])=O)=C2)NC1=O',
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
                    'O=C(N(SC1=O)C2=C3C=CC=CC3=CC=C2)N1CC4=CC=CC=C4']
    
    # smiles_list = [
    #     'C1CCCCC1', 'C1=CC=C(C=C1)O', 'CC(=O)OC1=CC=CC=C1C(=O)O'
    # ]
    modified_mols = []
    modify_bond = ModifyBond(setup_logger())
    for smile in smiles_list:
        mol = Chem.MolFromSmiles(smile)
        
    # # atom_indices = modify_bond.get_optimal_bond_sites(mol)
    # # bond_indices = modify_bond.get_bond_indices(mol)
    # # print(atom_indices, bond_indices)
        modified_mol = modify_bond.optimize_bond(mol, 1, False)
        modified_mols.append(Chem.MolToSmiles(modified_mol))
        #print(modify_bond.get_bond_indices(mol))
    # # bond_by_idx = modify_bond._get_bond_by_indices(mol, (0,1))
    # # print(bond_by_idx)

    #modified_smiles = ['N#Cc1c[nH]c(-c2cccc([N+](=O)[O-])c2)c([N+](=O)[O-])c1C(F)(F)F.O', 'O=C(Nc1ccc(NC(=O)Nc2cccc3[nH]ccc23)cc1)Nc1cccc2[nH]ccc12', 'Br.CCNCc1nc2c(O)c(Cl)cc(Cl)c2c(=O)n1C', 'O=C(Nc1ccc(Cl)cc1O)c1cc(O)c2c(c1)C(=O)c1cccc(O)c1C2=O', 'Oc1ccc(Cc2ccc(O)c(Cc3ccc(O)c(Cc4ccc(O)cc4)c3)c2)cc1', 'CN1C(N)=N[C@](C)(c2cccc(NC(=O)c3ccc(F)cn3)c2)CS1(=O)=O.F', 'CC#Cc1cncc(-c2ccc3c(c2)C2(N=C(C)C(N)=N2)C2(CCC(OC)CC2)C3)c1', 'C[C@@]1(c2cc(NC(=O)c3ccc(C#N)cn3)ccc2F)C=CSC(N)=N1', 'C[CH][C@H]1CSC(N)=N[C@]1(CO)c1cc(NC(=O)c2cnc(C(F)F)cn2)ccc1F', 'O=C1NC(=O)C(c2cnc3ccccn23)=C1c1cn2c3c(cc(F)cc13)CN(C(=O)N1CCCCC1)CC2', 'COc1ccc(CSc2nnc(-c3ccc4ncsc4c3)o2)cc1C(F)(F)F', 'O=C(Nc1cc(-c2ccc(-c3ncon3)s2)ccn1)C1CC1', 'O.O=c1sn(-c2cccc3ccccc23)cn1Cc1ccccc1']
    print(modified_mols)
    from collections import Counter
    if Counter(smiles_list) == Counter(modified_mols):
        print("Lists are same")
    else:
        print("Lists are not same")'''
