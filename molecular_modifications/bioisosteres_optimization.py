import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from modification_imports import *
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
from logger import setup_molecule_logger

class ModifyBioisosteres:
    """
    A comprehensive module for performing bioisosteric and functional group modifications
    on molecules using RDKit. Implements chemically valid transformations commonly 
    used in drug discovery.
    """
    
    def __init__(self, logger, log: bool = False):
        super(ModifyBioisosteres, self).__init__()
        
        self.logger = logger
        self.log = log

        self.acidic_bioisosteres = {
            "carboxylic_acid": ["tetrazole", "phosphonic_acid"], # "sulfonic_acid"
            # "tetrazole": ["carboxylic_acid", "phosphonic_acid"],
            #"hydroxamic_acid": ["carboxylic_acid", "tetrazole"]
        }
        
        self.basic_bioisosteres = {
            "amine": ["pyridine"], # "imidazole", "guanidine"
            #"amidine": ["guanidine", "imidazole"],
            #"guanidine": ["amidine", "2-aminopyridine"],
            "amide": ["sulfonamide", "retroamide"], # "urea"
            #"ester": ["amide", "thioester", "ketone", "sulfonamide"],
            #"alkyl": ["cycloalkyl", "CF3", "tBu"],
            "phenyl": ["pyridyl", "thiophene", "furan", "pyrrole"],
            #"ketone": ["oxime", "hydrazone", "thiocarbonyl"]
        }
        
        self.fg_patterns = {
            "carboxylic_acid": "[CX3](=O)[OX2H1]",
            "tetrazole": "[nH]1nnnc1",
            "phosphonic_acid": "[PX4](=O)([OX2H1])[OX2H1]",
            "sulfonic_acid": "[SX4](=O)(=O)[OX2H1]",
            "amine": "[NX3;H2,H1;!$(NC=O)]",
            "amidine": "[NX3][CX3]=[NX2]",
            "guanidine": "[NX3][CX3](=[NX2])[NX3]",
            "hydroxamic_acid": "[CX3](=O)[NX3][OX2H1]",
            "amide": "[NX3][CX3](=[OX1])[#6]",
            "ester": "[OX2][CX3](=[OX1])[#6]",
            "sulfonamide": "[NX3][SX4](=[OX1])(=[OX1])[#6]",
            "ketone": "[CX3]=[OX1]",
            "phenyl": "c1ccccc1",
            "pyridyl": "n1ccccc1"
        }
        
    def identify_functional_groups(self, mol: Chem.Mol) -> Dict[str, List[int]]:
        """
        Identifies all functional groups in the molecule that could be candidates
        for bioisosteric replacement.
        
        Args:
            mol: RDKit molecule object
        
        Returns:
            Dictionary mapping functional group types to lists of matching atom indices
        """
        matches = {}
        for fg_name, pattern in self.fg_patterns.items():
            pattern_mol = Chem.MolFromSmarts(pattern)
            if pattern_mol is not None:
                matches[fg_name] = mol.GetSubstructMatches(pattern_mol)
        return matches

    def replace_carboxylic_acid_with_tetrazole(self, mol: Chem.Mol, match_atoms: Tuple[int]) -> Optional[Chem.Mol]:
        """
        Replaces a carboxylic acid group with a tetrazole ring.
        
        Args:
            mol: Input molecule
            match_atoms: Tuple of atom indices matching carboxylic acid pattern
        
        Returns:
            Modified molecule with tetrazole replacement, or None if failed
        """
        try:
            # Create editable molecule
            em = Chem.EditableMol(mol)
            
            # Remove carboxylic acid group
            for atom_idx in match_atoms[1:-1]:  # Keep the carbon
                em.RemoveAtom(atom_idx)
            
            # Add tetrazole atoms
            n1_idx = em.AddAtom(Chem.Atom(7))  # N
            n2_idx = em.AddAtom(Chem.Atom(7))  # N
            n3_idx = em.AddAtom(Chem.Atom(7))  # N
            n4_idx = em.AddAtom(Chem.Atom(7))  # N
            
            # Add bonds to form tetrazole
            em.AddBond(match_atoms[0], n1_idx, Chem.BondType.SINGLE)
            em.AddBond(n1_idx, n2_idx, Chem.BondType.SINGLE)
            em.AddBond(n2_idx, n3_idx, Chem.BondType.DOUBLE)
            em.AddBond(n3_idx, n4_idx, Chem.BondType.SINGLE)
            em.AddBond(n4_idx, match_atoms[0], Chem.BondType.DOUBLE)
            
            # Convert to molecule and sanitize
            new_mol = em.GetMol()
            try:
                Chem.SanitizeMol(new_mol)
                if self.log:
                    self.logger.info("Carboxylic Acid To Tetrazole Replacement Successful") 
                return Chem.MolToSmiles(Chem.Mol(new_mol))
            except Exception:
                if self.log:
                    self.logger.warning(f"Carboxylic Acid To Tetrazole Replacement Not Fully Sanitized")
                return Chem.MolToSmiles(mol)
            
        except Exception as e:
            if self.log:
                self.logger.error(f"Failed to replace carboxylic acid with tetrazole: {str(e)}")
            return Chem.MolToSmiles(mol)
        
    def replace_carboxylic_acid_with_phosphonic_acid(self, mol: Chem.Mol, match_atoms: Tuple[int]) -> Optional[Chem.Mol]:
        """
        Replaces a carboxylic acid or tetrazole with a phosphonic acid group.
        
        Args:
            mol: Input molecule
            match_atoms: Tuple of atom indices matching target pattern
        
        Returns:
            Modified molecule with phosphonic acid replacement, or None if failed
        """
        try:
            em = Chem.EditableMol(mol)
            
            # Remove existing group but keep attachment point
            attachment_idx = match_atoms[0]
            for atom_idx in match_atoms[1:-1]:
                em.RemoveAtom(atom_idx)
            
            # Add phosphonic acid atoms
            p_idx = em.AddAtom(Chem.Atom(15))  # P
            o1_idx = em.AddAtom(Chem.Atom(8))  # O (double bond)
            o2_idx = em.AddAtom(Chem.Atom(8))  # O (hydroxyl)
            o3_idx = em.AddAtom(Chem.Atom(8))  # O (hydroxyl)
            h1_idx = em.AddAtom(Chem.Atom(1))  # H
            h2_idx = em.AddAtom(Chem.Atom(1))  # H
            
            # Add bonds
            em.AddBond(attachment_idx, p_idx, Chem.BondType.SINGLE)
            em.AddBond(p_idx, o1_idx, Chem.BondType.DOUBLE)
            em.AddBond(p_idx, o2_idx, Chem.BondType.SINGLE)
            em.AddBond(p_idx, o3_idx, Chem.BondType.SINGLE)
            em.AddBond(o2_idx, h1_idx, Chem.BondType.SINGLE)
            em.AddBond(o3_idx, h2_idx, Chem.BondType.SINGLE)
            
            new_mol = em.GetMol()

            try:
                Chem.SanitizeMol(new_mol)
                if self.log:
                    self.logger.info("Carboxylic Acid To Phosphonic Acid Replacement Successful") 
                return Chem.MolToSmiles(Chem.Mol(new_mol))
            except Exception:
                if self.log:
                    self.logger.warning(f"Carboxylic Acid To Phosphonic Acid Replacement Not Fully Sanitized")
                return Chem.MolToSmiles(mol)
            
        except Exception as e:
            if self.log:
                self.logger.error(f"Failed to replace with phosphonic acid: {str(e)}")
            return Chem.MolToSmiles(mol)

    def replace_amine_with_pyridine(self, mol: Chem.Mol, match_atoms: Tuple[int]) -> Optional[Chem.Mol]:
        """
        Replaces a primary or secondary amine with a pyridine ring.
        
        Args:
            mol: Input molecule
            match_atoms: Tuple of atom indices matching amine pattern
        
        Returns:
            Modified molecule with pyridine replacement, or None if failed
        """
        try:
            em = Chem.EditableMol(mol)
            
            # Remove amine hydrogens
            amine_idx = match_atoms[0]
            for neighbor in mol.GetAtomWithIdx(amine_idx).GetNeighbors():
                if neighbor.GetAtomicNum() == 1:  # Hydrogen atom
                    em.RemoveAtom(neighbor.GetIdx())
            
            # Add pyridine carbons (aromatic ring with nitrogen)
            c2_idx = em.AddAtom(Chem.Atom(6))  # C
            c3_idx = em.AddAtom(Chem.Atom(6))  # C
            c4_idx = em.AddAtom(Chem.Atom(6))  # C
            c5_idx = em.AddAtom(Chem.Atom(6))  # C
            c6_idx = em.AddAtom(Chem.Atom(6))  # C
            n_idx = em.AddAtom(Chem.Atom(7))  # N (Nitrogen in pyridine)
            
            # Add bonds to form pyridine ring
            em.AddBond(amine_idx, c2_idx, Chem.BondType.SINGLE)  # N-C bond
            em.AddBond(c2_idx, c3_idx, Chem.BondType.DOUBLE)
            em.AddBond(c3_idx, c4_idx, Chem.BondType.SINGLE)
            em.AddBond(c4_idx, c5_idx, Chem.BondType.DOUBLE)
            em.AddBond(c5_idx, c6_idx, Chem.BondType.SINGLE)
            em.AddBond(c6_idx, c2_idx, Chem.BondType.DOUBLE)
            
            # Ensure nitrogen is part of the aromatic ring and not over-connected
            em.AddBond(c2_idx, n_idx, Chem.BondType.SINGLE)  # N as part of the ring
            
            # Convert to molecule and sanitize
            new_mol = em.GetMol()
            
            try:
                Chem.SanitizeMol(new_mol)
                new_mol.GetAtomWithIdx(n_idx).SetIsAromatic(True)         
                if self.log:
                    self.logger.info("Replaced amine with pyridine")
                return Chem.MolToSmiles(Chem.Mol(new_mol))
            
            except Exception:
                if self.log:
                    self.logger.warning(f"Amine Replacement Not Fully Sanitized. Replaced amine with pyridine.")
                return Chem.MolToSmiles(mol)

        except Exception as e:
            if self.log:
                self.logger.error(f"Failed to replace amine with pyridine: {str(e)}")
            return Chem.MolToSmiles(mol)

    def replace_phenyl_with_heterocycle(self, mol: Chem.Mol, match_atoms: Tuple[int], 
                                      replacement: str = "pyridyl") -> Optional[Chem.Mol]:
        """
        Replaces a phenyl ring with various heterocycles while maintaining connectivity.
        
        Args:
            mol: Input molecule
            match_atoms: Tuple of atom indices matching phenyl pattern
            het_type: Type of heterocycle to insert ("pyridyl", "thiophene", "furan", "pyrrole")
        """
        try:
            em = Chem.EditableMol(mol)

            # Map external connections before removing phenyl
            external_bonds = []
            atom_map = {}  # Track mapping of old to new indices
            for idx in match_atoms:
                atom = mol.GetAtomWithIdx(idx)
                for bond in atom.GetBonds():
                    if bond.GetOtherAtomIdx(idx) not in match_atoms:
                        external_bonds.append((idx, bond.GetOtherAtomIdx(idx)))

            # Remove phenyl ring
            for idx in sorted(match_atoms, reverse=True):
                em.RemoveAtom(idx)

            # Update external_bonds to reflect new indices
            atom_map = {old_idx: new_idx for new_idx, old_idx in enumerate(range(em.GetMol().GetNumAtoms()))}
            updated_external_bonds = [(atom_map.get(old, old), ext) for old, ext in external_bonds if old in atom_map]
            
            new_atoms = []
            # Add heterocycle
            if replacement == "pyridyl":
                new_atoms = [7, 6, 6, 6, 6, 6]  # N, C, C, C, C, C
            elif replacement == "thiophene":
                new_atoms = [16, 6, 6, 6, 6]    # S, C, C, C, C
            elif replacement == "furan":
                new_atoms = [8, 6, 6, 6, 6]     # O, C, C, C, C
            elif replacement == "pyrrole":
                new_atoms = [7, 6, 6, 6, 6]     # N, C, C, C, C
            
            # Add new atoms
            new_indices = []
            for atomic_num in new_atoms:
                new_idx = em.AddAtom(Chem.Atom(atomic_num))
                new_indices.append(new_idx)

            # Add bonds to form heterocycle
            for i in range(len(new_indices)):
                if not em.GetMol().GetBondBetweenAtoms(new_indices[i], new_indices[(i + 1) % len(new_indices)]):
                    if new_indices[i] != new_indices[(i + 1) % len(new_indices)]:
                        em.AddBond(new_indices[i], new_indices[(i + 1) % len(new_indices)], Chem.BondType.AROMATIC)
            
            # Reconnect external bonds
            for old_idx, ext_idx in updated_external_bonds:
                if old_idx in match_atoms:
                    attach_idx = new_indices[match_atoms.index(old_idx)]
                    if attach_idx < em.GetMol().GetNumAtoms() and ext_idx < em.GetMol().GetNumAtoms():
                        if not em.GetMol().GetBondBetweenAtoms(attach_idx, ext_idx):
                            em.AddBond(attach_idx, ext_idx, Chem.BondType.SINGLE)
            
            new_mol = em.GetMol()
            try:
                Chem.SanitizeMol(new_mol)
                if self.log:
                    self.logger.info(f"Phenyl Replacement Successful: Replaced phenyl with {replacement}")
                return Chem.MolToSmiles(Chem.Mol(new_mol))
            except Exception as e:
                if self.log:
                    self.logger.warning(f"Phenyl Replacement Not Fully Sanitized: Replaced phenyl with {replacement}")
                return Chem.MolToSmiles(mol)
            
        except Exception as e:
            if self.log:
                self.logger.error(f"Failed to replace phenyl with {replacement}: {str(e)}")
            return Chem.MolToSmiles(mol)
        
    def replace_amide_with_bioisostere(self, mol: Chem.Mol, match_atoms: Tuple[int], 
                                     replacement: str = "sulfonamide") -> Optional[Chem.Mol]:
        """
        Replaces an amide group with various bioisosteres.
        
        Args:
            mol: Input molecule
            match_atoms: Tuple of atom indices matching amide pattern
            replacement: Type of replacement ("sulfonamide", "retroamide", "urea")
        """
        try:
            em = Chem.EditableMol(mol)

            if replacement == "sulfonamide":
                # Remove amide group keeping N and C attachment points
                for idx in match_atoms[1:-1]:  # Keep N and C
                    em.RemoveAtom(idx)
                
                # Add sulfonamide atoms
                s_idx = em.AddAtom(Chem.Atom(16))  # S
                o1_idx = em.AddAtom(Chem.Atom(8))  # O
                o2_idx = em.AddAtom(Chem.Atom(8))  # O
                
                # Add bonds for sulfonamide group
                em.AddBond(match_atoms[0], s_idx, Chem.BondType.SINGLE)  # N-S
                em.AddBond(s_idx, match_atoms[-1], Chem.BondType.SINGLE)  # S-C
                em.AddBond(s_idx, o1_idx, Chem.BondType.DOUBLE)  # S=O
                em.AddBond(s_idx, o2_idx, Chem.BondType.DOUBLE)  # S=O

            elif replacement == "retroamide":
                # Reverse the amide connectivity
                n_idx = match_atoms[0]
                c_idx = match_atoms[1]
                o_idx = match_atoms[2]
                r_idx = match_atoms[3]
                
                em.RemoveBond(n_idx, c_idx)
                em.RemoveBond(c_idx, o_idx)
                em.RemoveBond(c_idx, r_idx)
                
                em.AddBond(r_idx, n_idx, Chem.BondType.SINGLE)
                em.AddBond(n_idx, c_idx, Chem.BondType.SINGLE)
                em.AddBond(c_idx, o_idx, Chem.BondType.DOUBLE)

            # Convert to molecule and sanitize
            new_mol = em.GetMol()

            # Attempt sanitization
            try:
                Chem.SanitizeMol(new_mol)
                if self.log:
                    self.logger.info(f"Amide Replacement Successful: Replaced amide with {replacement}")
                return Chem.MolToSmiles(Chem.Mol(new_mol))
            except Exception as e:
                if self.log:
                    self.logger.warning(f"Amide Replacement Not Fully Sanitized: Replaced amide with {replacement}")
                return Chem.MolToSmiles(mol)
        
        except Exception as e:
            if self.log:
                self.logger.error(f"Failed to replace amide: {str(e)}")
            return Chem.MolToSmiles(mol)

    def apply_modification(self, mol: Chem.Mol, chosen_fg_type: str, chosen_replacement: Optional[str] = None) -> Tuple[Optional[Chem.Mol], str]:
        """
        Applies a random valid bioisosteric replacement to the molecule.
        
        Args:
            mol: Input molecule
        
        Returns:
            Tuple of (modified molecule, description of modification applied)
            Returns (None, error message) if no valid modifications were possible
        """
        # Identify all possible modification sites
        fg_matches = self.identify_functional_groups(mol)
        
        # Filter to only groups with available replacements
        valid_modifications = []
        for fg, match_atoms in fg_matches.items():
            if fg in self.acidic_bioisosteres.keys() or fg in self.basic_bioisosteres.keys():
                for match in match_atoms:
                    valid_modifications.append((fg, match))
        
        if not valid_modifications:
            if self.log:    
                self.logger.error("No valid modifications found.")
                return Chem.MolToSmiles(mol)
        
        # Select random modification
        # BE ABLE TO CHOOSE WHICH FG_TYPE
        # BE ABLE TO CHOOSE WHICH REPLACEMENT FOR EACH FG_TYPE
        fg_type, match = None, None
        if chosen_fg_type:
            for fg, match_atoms in valid_modifications:
                if fg == chosen_fg_type:
                    fg_type, match = fg, match_atoms
                    break
        
        if not fg_type:
            if self.log:
                self.logger.error(f"No match in molecule for chosen bioisosteric replacement: {chosen_fg_type}.")
            return Chem.MolToSmiles(mol)
        
        #fg_type, match = random.choice(valid_modifications)
        
        # Apply corresponding replacement
        try:
            if fg_type == "carboxylic_acid":
                # replacement = random.choice(["tetrazole", "phosphonic_acid"])
                if chosen_replacement == "tetrazole":
                    new_mol = self.replace_carboxylic_acid_with_tetrazole(mol, match)
                    return Chem.MolToSmiles(Chem.Mol(new_mol))
                elif chosen_replacement == "phosphonic_acid":
                    new_mol = self.replace_carboxylic_acid_with_phosphonic_acid(mol, match)
                    return Chem.MolToSmiles(Chem.Mol(new_mol))
            elif fg_type == "amine":
                new_mol = self.replace_amine_with_pyridine(mol, match)
                return Chem.MolToSmiles(Chem.Mol(new_mol))
            elif fg_type == "amide":
               # replacement = random.choice(["sulfonamide", "retroamide"])
                new_mol = self.replace_amide_with_bioisostere(mol, match, chosen_replacement)
                return Chem.MolToSmiles(Chem.Mol(new_mol))     
            elif fg_type == "phenyl":
                # replacement = random.choice(["pyridyl", "thiophene", "furan", "pyrrole"])
                new_mol = self.replace_phenyl_with_heterocycle(mol, match, chosen_replacement)
                return Chem.MolToSmiles(Chem.Mol(new_mol))
            else:
                if self.log:
                    self.logger.error(f"Unknown functional group type: {fg_type}")
                return Chem.MolToSmiles(mol)
            
        except Exception as e:
            if self.log:
                self.logger.error(f"Failed to apply a modification. Error: {e}")
            return Chem.MolToSmiles(mol)

    def validate_modification(self, original_mol: Chem.Mol, modified_mol: Chem.Mol) -> bool:
        """
        Validates that a molecular modification maintains chemical validity
        and reasonable properties.
        
        Args:
            original_mol: Original molecule
            modified_mol: Modified molecule
        
        Returns:
            Boolean indicating if modification is valid
        """
        try:
            # Check basic chemistry validity
            Chem.SanitizeMol(modified_mol)
            
            # Ensure reasonable molecular weight change
            orig_mw = Chem.Descriptors.ExactMolWt(original_mol)
            mod_mw = Chem.Descriptors.ExactMolWt(modified_mol)
            if abs(mod_mw - orig_mw) > 200:  # Maximum allowed MW change
                return False
            
            # Check for maintained connectivity
            if Chem.rdmolops.GetFormalCharge(modified_mol) > 2:
                return False
                
            # More validation checks could be added here
            
            return True
            
        except Exception:
            if self.log:
                self.logger.error("Failed to validate modification")
            return False
        
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
    
    #smiles_list = ['CC(=O)OC1=CC=CC=C1C(=O)O']
    modify_bioisosteres = ModifyBioisosteres(setup_logger())
    all_matches = []
    new_mols = []
    for smile in smiles_list:
        mol = Chem.MolFromSmiles(smile)
        new_mol = modify_bioisosteres.apply_random_modification(mol)
        new_mols.append(Chem.MolToSmiles(new_mol))
        # fg_type, match = valid_modifications[0]
        # replacement = random.choice(["tetrazole", "phosphonic_acid"])
        # if replacement == "tetrazole":
        #     new_mol = modify_bioisosteres.replace_carboxylic_acid_with_tetrazole(mol, match)
        #     new_mols.append(Chem.MolToSmiles(new_mol))
        # elif replacement == "phosphonic_acid":
        #     new_mol = modify_bioisosteres.replace_carboxylic_acid_with_phosphonic_acid(mol, match)
        #     new_mols.append(Chem.MolToSmiles(new_mol))


        # fg_type, match = valid_modifications[0]
        # new_mol = modify_bioisosteres.replace_amine_with_pyridine(mol, match)
        # new_mols.append(Chem.MolToSmiles(new_mol))
        
        # fg_type, match = valid_modifications[3]
        # replacement = random.choice(["pyridyl", "thiophene", "furan", "pyrrole"])
        # new_mol = modify_bioisosteres.replace_phenyl_with_heterocycle(mol, match, 'thiophene')
        # new_mols.append(Chem.MolToSmiles(new_mol))        

        # fg_type, match = valid_modifications[0]
        # replacement = random.choice(["sulfonamide", "retroamide"])
        # new_mol = modify_bioisosteres.replace_amide_with_bioisostere(Chem.MolFromSmiles(smile), match, replacement)
        # new_mols.append(Chem.MolToSmiles(new_mol))

    print(new_mols)'''

