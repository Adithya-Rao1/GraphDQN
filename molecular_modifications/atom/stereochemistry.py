from typing import Optional, List, Tuple
from rdkit import Chem
from rdkit.Chem import AllChem

class OptimizeStereochemistry:
    def _perform_stereoaware_substitution(self, mol: Chem.RWMol, atom_idx: int, new_atom: str) -> Optional[Chem.Mol]:
        """
        Perform atom substitution with adjusted stereochemistry based on the new atom's environment.
        
        Args:
            mol (RWMol): Molecule to modify
            atom_idx (int): Atom index to substitute
            new_atom (str): Substitution atom symbol
        
        Returns:
            Modified molecule with adjusted stereochemistry or None if sanitization fails.
        """
        # Capture existing stereochemistry context
        existing_stereo = self._capture_stereochemistry(mol, atom_idx)
        
        # Perform atom substitution
        new_atomic_num = Chem.GetPeriodicTable().GetAtomicNumber(new_atom)
        mol.GetAtomWithIdx(atom_idx).SetAtomicNum(new_atomic_num)
        
        # Adjust stereochemistry based on new atom's properties
        self._adjust_stereochemistry(mol, atom_idx, existing_stereo)
        
        # Final sanitization and validity check
        try:
            Chem.SanitizeMol(mol)
            return mol
        except Chem.rdchem.MolSanitizeException:
            return None
    
    def _capture_stereochemistry(self, mol: Chem.RWMol, atom_idx: int) -> Optional[dict]:
        """
        Capture stereochemistry information (chiral centers and bonds) of the atom prior to substitution.
        
        Args:
            mol (RWMol): Molecule
            atom_idx (int): Index of the atom to capture stereochemistry from
        
        Returns:
            Dictionary containing stereochemical information or None if no stereochemistry exists.
        """
        atom = mol.GetAtomWithIdx(atom_idx)
        if atom.GetChiralTag() == Chem.rdchem.ChiralType.CHI_UNSPECIFIED:
            return None
        
        # Capture stereochemistry details
        wedged_bonds = self._get_wedged_bonds(mol, atom_idx)
        return {
            'chiral_tag': atom.GetChiralTag(),
            'wedged_bonds': wedged_bonds
        }
    
    def _adjust_stereochemistry(self, mol: Chem.RWMol, atom_idx: int, existing_stereo: Optional[dict]):
        """
        Adjust stereochemistry after atom substitution.
        
        Args:
            mol (RWMol): Molecule to modify
            atom_idx (int): Atom index that was substituted
            existing_stereo (dict): Existing stereochemistry information to be adjusted (if available)
        """
        # Adjusting chiral tag and stereochemical configuration
        atom = mol.GetAtomWithIdx(atom_idx)
        if existing_stereo:
            # If the atom was originally chiral, determine new stereochemistry based on surrounding atoms
            self._reassign_chirality(mol, atom_idx, existing_stereo)
        else:
            # If no stereochemistry was present, assign based on bonding
            self._assign_default_stereochemistry(mol, atom_idx)

        # Adjust bonds for correct stereochemical configuration
        self._adjust_bond_stereochemistry(mol, atom_idx, existing_stereo)
    
    def _reassign_chirality(self, mol: Chem.RWMol, atom_idx: int, existing_stereo: dict):
        """
        Reassign chirality after atom substitution, considering neighboring atoms and bonding.
        
        Args:
            mol (RWMol): Molecule to modify
            atom_idx (int): Atom index that was substituted
            existing_stereo (dict): Existing stereochemistry information to base adjustments on
        """
        atom = mol.GetAtomWithIdx(atom_idx)
        # Reassign chiral tag based on the atom's neighbors and bonding context
        atom.SetChiralTag(Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW if existing_stereo['chiral_tag'] == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW else Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW)

    def _assign_default_stereochemistry(self, mol: Chem.RWMol, atom_idx: int):
        """
        Assign stereochemistry to the modified atom if it was previously non-chiral.
        
        Args:
            mol (RWMol): Molecule to modify
            atom_idx (int): Atom index that was substituted
        """
        # Determine the appropriate stereochemistry based on the bonding context of the new atom
        atom = mol.GetAtomWithIdx(atom_idx)
        neighbors = [bond.GetOtherAtomIdx(atom_idx) for bond in mol.GetBonds() if bond.GetBeginAtomIdx() == atom_idx or bond.GetEndAtomIdx() == atom_idx]
        
        if len(neighbors) == 4:  # Tetrahedral center
            atom.SetChiralTag(Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW)  # Default to CCW, this can be adjusted further based on context
        
    def _get_wedged_bonds(self, mol: Chem.RWMol, atom_idx: int) -> List[Tuple[int, int]]:
        """
        Get wedged bonds associated with a specific atom to preserve chirality.
        
        Args:
            mol (RWMol): Molecule
            atom_idx (int): Atom index
            
        Returns:
            List of bond indices with stereo (wedged/dashed) bonds.
        """
        return [
            (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            for bond in mol.GetBonds()
            if (bond.GetBeginAtomIdx() == atom_idx or bond.GetEndAtomIdx() == atom_idx) and 
               bond.GetBondStereo() in [Chem.rdchem.BondStereo.STEREOANY, Chem.rdchem.BondStereo.STEREOZ]
        ]
    
    def _adjust_bond_stereochemistry(self, mol: Chem.RWMol, atom_idx: int, existing_stereo: Optional[dict]):
        """
        Adjust bonds around the substituted atom to maintain stereochemical consistency.
        
        Args:
            mol (RWMol): Molecule
            atom_idx (int): Atom index that was substituted
            existing_stereo (dict): Existing stereochemistry information to guide bond reconfiguration
        """
        if existing_stereo:
            # Reconfigure bonds to maintain stereochemistry around the atom
            for begin, end in existing_stereo['wedged_bonds']:
                bond = mol.GetBondBetweenAtoms(begin, end)
                if bond:
                    bond.SetBondStereo(Chem.rdchem.BondStereo.STEREOANY)  # Adjust this according to your specific stereochemical needs

