import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from molecular_modifications.modification_imports import *
#from stereochemistry import OptimizeStereochemistry
from molecular_modifications.logger import setup_molecule_logger

class ModifyAtom:
    def __init__(self, 
                 logger, 
                 substitution_atoms: Optional[List[str]] = None,
                 modification_strategy: str = 'balanced',
                 log:bool = False):
        """
        Initialize the ModifyAtom class with a list of substitution atoms and modification strategies.
        
        Args:
            substitution_atoms (List[str], optional): Atoms available for substitution
            modification_strategy (str): Modification approach ('balanced', 'drug-like', 'diversity')
        """
        self.logger = logger
        self.log = log

        # Prioritized substitution atoms based on drug discovery insights
        self.substitution_atoms = substitution_atoms or [
            "C", "N", "O", "S", "P",  
            "F", "Cl", "Br", "I",    
            "B", "Si"                 
        ]
        
        self.modification_strategy = modification_strategy

    def modify_atom(self, mol: Union[Chem.Mol, Chem.RWMol], action: int) -> Optional[Chem.Mol]:
        """
        Atom modification with stereochemistry and electronic considerations.
        
        Args:
            mol (Chem.Mol): Input molecule to modify
            action (int): Action determining atom substitution strategy
        
        Returns:
            Modified molecule or None
        """
        # Convert to RWMol
        if self.log:
            if not isinstance(mol, (Chem.Mol, Chem.RWMol)):
                self.logger.error("Input molecule must be a Chem.Mol or Chem.RWMol.")
        rwmol = Chem.RWMol(mol)

        # Identify potential modification sites
        modification_sites = self._identify_modification_sites(rwmol)
        
        if not modification_sites:
            if self.log:
                self.logger.error("No suitable modification sites found.")
            return Chem.MolToSmiles(mol)
        
        atom_idx = self._select_modification_site(modification_sites)
        substitution_atom = self._select_substitution_atom(rwmol, atom_idx, action)
        if substitution_atom is None:
            if self.log:
                self.logger.error("No suitable substitution atom found.")
            return Chem.MolToSmiles(mol)
        
        current_symbol = rwmol.GetAtomWithIdx(atom_idx).GetSymbol()
        
        try:
            rwmol.ReplaceAtom(atom_idx, Chem.Atom(substitution_atom))
            Chem.SanitizeMol(rwmol)
            Chem.AssignStereochemistry(rwmol, cleanIt=True, force=True)
            if self.log:
                self.logger.info(f"Successful Modify Action: Atom {atom_idx} modified from {current_symbol} to {substitution_atom}.")
            return Chem.MolToSmiles(Chem.Mol(rwmol))
        except Exception as e:
            if self.log:
                self.logger.error(f"Error modifying atom {current_symbol}: {str(e)}")
            return Chem.MolToSmiles(mol)
        
    def add_atom(self, mol: Union[Chem.Mol, Chem.RWMol], action: int) -> Optional[Chem.Mol]:
        """
        Add an atom to the molecule with stereochemistry and sanitization checks.
        
        Args:
            mol (Chem.Mol): Input molecule to modify.
            action (int): Action determining atom addition strategy.
            
        Returns:
            Modified molecule or None.
        """
        # Convert to RWMol
        if self.log:
            if not isinstance(mol, (Chem.Mol, Chem.RWMol)):
                self.logger.error("Input molecule must be a Chem.Mol or Chem.RWMol.")
        rwmol = Chem.RWMol(mol)
        
        # Select a random atom index for addition (or strategy-based selection)
        modification_sites = self._identify_modification_sites(rwmol)
        if not modification_sites:
            if self.log:
                self.logger.error("No suitable modification sites found.")
            return Chem.MolToSmiles(Chem.Mol(rwmol))
        
        atom_idx = self._select_modification_site(modification_sites)
        existing_atom = rwmol.GetAtomWithIdx(atom_idx)
        
        # Select an atom to add and define bond type
        substitution_atom = self._select_substitution_atom(rwmol, atom_idx, action)
        if substitution_atom is None:
            if self.log:
                self.logger.error("No suitable atom found to add.")
            return Chem.MolToSmiles(mol)
        
        bond_type = Chem.BondType.SINGLE  # Default bond type
        try:
            remaining_valence = VALENCE_ELECTRON_COUNTS.get(existing_atom.GetSymbol()) - existing_atom.GetTotalValence()
            if remaining_valence == 1:
                bond_type = Chem.BondType.SINGLE
            elif remaining_valence == 2:
                bond_type = Chem.BondType.DOUBLE
            elif remaining_valence == 3:
                bond_type = Chem.BondType.TRIPLE
            else:
                bond_type = Chem.BondType.SINGLE

            new_atom_idx = rwmol.AddAtom(Chem.Atom(substitution_atom))
            rwmol.AddBond(atom_idx, new_atom_idx, bond_type)
            Chem.SanitizeMol(rwmol)
            Chem.AssignStereochemistry(rwmol, cleanIt=True, force=True)
            
            if self.log:
                self.logger.info(f"Successful Add Action: Atom {substitution_atom} added to atom {existing_atom.GetSymbol()} (index {atom_idx}).")
            return Chem.MolToSmiles(Chem.Mol(rwmol))
        except Exception as e:
            if self.log:
                self.logger.error(f"Failed to add atom: {e}")
            return Chem.MolToSmiles(mol)
        
    def remove_atom(self, mol: Union[Chem.Mol, Chem.RWMol], action: int) -> Optional[Chem.Mol]:
        """
        Remove an atom from the molecule with stereochemistry and sanitization checks.
        
        Args:
            mol (Chem.Mol): Input molecule to modify.
            action (int): Action determining atom removal strategy.
            
        Returns:
            Modified molecule or None if the operation fails.
        """
        # Convert to RWMol
        if not isinstance(mol, (Chem.Mol, Chem.RWMol)):
            if self.log:
                self.logger.error("Input molecule must be a Chem.Mol or Chem.RWMol.")
            return None
        rwmol = Chem.RWMol(mol)
        
        # Identify potential removal sites
        modification_sites = self._identify_modification_sites(rwmol)
        if not modification_sites:
            if self.log:
                self.logger.error("No suitable modification sites found.")
            return Chem.MolToSmiles(mol)
        
        # Select an atom index for removal
        atom_idx = self._select_modification_site(modification_sites)
        atom_to_remove = rwmol.GetAtomWithIdx(atom_idx)

        # Avoid removing non-ring aromatic atoms
        if atom_to_remove.GetIsAromatic() and not atom_to_remove.IsInRing():
            if self.log:
                self.logger.error(f"Cannot remove non-ring aromatic atom: {atom_to_remove.GetSymbol()} (index {atom_idx}).")
            return Chem.MolToSmiles(mol)
        
        try:
            # Remove atom
            rwmol.RemoveAtom(atom_idx)
            
            Chem.SanitizeMol(rwmol)
            
            # Ensure stereochemistry is updated
            Chem.AssignStereochemistry(rwmol, cleanIt=True, force=True)

            if self.log:
                self.logger.info(f"Successful Atom Removal Action.")
            return Chem.MolToSmiles(Chem.Mol(rwmol))

        except Exception as e:
            if self.log:
                self.logger.error(f"Failed to remove atom: {e}")
            return Chem.MolToSmiles(mol)

    def _identify_modification_sites(self, mol: Chem.RWMol) -> List[Tuple[int, dict]]:
        """
        Identify potential atom modification sites for modification.
        
        Args:
            mol (RWMol): Molecule to analyze
        
        Returns:
            List of modification sites with their properties
        """
        modification_sites = []
        
        for atom_idx in range(mol.GetNumAtoms()):
            atom = mol.GetAtomWithIdx(atom_idx)
            
            # Comprehensive site assessment
            site_properties = {
                'is_in_ring': atom.IsInRing(),
                'formal_charge': atom.GetFormalCharge(),
                'valence': atom.GetTotalValence(),
                'num_explicit_hs': atom.GetNumExplicitHs(),
            }
            
            # Filtering criteria for modification sites
            if (not atom.IsInRing() or  # Allow modifications outside rings
                (atom.GetSymbol() != 'C' and atom.IsInRing())):  # Or specific non-carbon ring atoms
                modification_sites.append((atom_idx, site_properties))
        
        return modification_sites

    def _select_modification_site(self, sites: List[Tuple[int, dict]]) -> int:
        """
        Score and select a modification site.
        
        Args:
            sites (List): Potential modification sites        
        Returns:
            Selected atom index
        """
        # Score sites based on multiple criteria
        def score_site(site):
            _, props = site
            score = 0
            
            # Penalty for ring atoms
            score -= 10 if props['is_in_ring'] else 0
            
            # Favor sites with higher valence variability
            score += props['valence']
            
            # Penalize sites with existing formal charge
            score -= abs(props['formal_charge']) * 5
            
            # Bonus for atoms with fewer hydrogens (more substitution potential)
            score += 5 if props['num_explicit_hs'] < 2 else 0
            
            return max(score, 1)
        
        # Compute scores for all sites
        scored_sites = [(site[0], score_site(site)) for site in sites]
        
        # Extract indices and scores
        indices, scores = zip(*scored_sites)
        
        # Normalize scores to probabilities
        total_score = sum(scores)
        probabilities = [score / total_score for score in scores]
        
        # Use a weighted random choice to select the site
        selected_idx = random.choices(indices, weights=probabilities, k=1)[0]
        return selected_idx

    def _select_substitution_atom(self, rwmol: Chem.RWMol, atom_idx: int, action: int) -> Optional[str]:
        """
        Select substitution atom with electronic and structural considerations.
        
        Args:
            rwmol (RWMol): Editable molecule
            atom_idx (int): Atom to substitute
            action (int): Modification action index
        
        Returns:
            Selected substitution atom symbol
        """
        current_atom = rwmol.GetAtomWithIdx(atom_idx)
        current_symbol = current_atom.GetSymbol()
        
        # Filter substitution atoms based on strategy and current atom
        candidate_atoms = [
            atom for atom in self.substitution_atoms 
            if atom != current_symbol and 
            self._validate_substitution(rwmol, atom_idx, atom)
        ]
        
        if not candidate_atoms:
            return Chem.MolToSmiles(Chem.Mol(rwmol))
        
        # Strategy-based selection
        if self.modification_strategy == 'balanced':
            # Electronic and structural considerations
            def score_atom(atom):
                return (
                    abs(ELECTRONEGATIVITY[atom] - ELECTRONEGATIVITY[current_symbol]) +
                    abs(COVALENT_RADII[atom] - COVALENT_RADII[current_symbol])
                )
            
            return min(candidate_atoms, key=score_atom)
        
        elif self.modification_strategy == 'drug-like':
            # Favor atoms common in drug molecules
            drug_like_preference = ['C', 'N', 'O', 'S', 'P']
            preferred = [a for a in candidate_atoms if a in drug_like_preference]
            return preferred[action % len(preferred)] if preferred else candidate_atoms[action % len(candidate_atoms)]
        
        else:  # 'diversity'
            # Maximize structural diversity
            return candidate_atoms[action % len(candidate_atoms)]

    def _validate_substitution(self, rwmol: Chem.RWMol, atom_idx: int, new_atom: str) -> bool:
        """
        Comprehensive validation of atom substitution.
        
        Args:
            rwmol (RWMol): Editable molecule
            atom_idx (int): Atom index to substitute
            new_atom (str): Proposed substitution atom
        
        Returns:
            Boolean indicating if substitution is valid
        """
        atom = rwmol.GetAtomWithIdx(atom_idx)
        
        # Valence validation
        max_valence = VALENCE_ELECTRON_COUNTS.get(atom.GetSymbol())
        current_valence = atom.GetTotalValence()
        
        if current_valence > max_valence:
            return False
        
        # Create a copy of the molecule for testing
        test_mol = Chem.RWMol(rwmol)
        test_atom = test_mol.GetAtomWithIdx(atom_idx)
        test_atom.SetAtomicNum(Chem.GetPeriodicTable().GetAtomicNumber(new_atom))
        
        try:
            Chem.SanitizeMol(test_mol)
            return True
        except Chem.rdchem.MolSanitizeException:
            return False
        
    def explore_atom_modification_space(self, mol: Chem.Mol, modification_type: str = 'random', exploration_depth: int = 1) -> List[Chem.Mol]:
        """
        Systematically explore molecular modification space with comprehensive tracking.
        
        Args:
            mol (Mol): Initial molecule
            exploration_depth (int): Number of independent modification pathways
        
        Returns:
            List of modified molecular variants
        """
        modification_variants = []
        modifications = {
            'modify': self.modify_atom,
            'add': self.add_atom,
            'remove': self.remove_atom
        }
        
        # Generate multiple modification trajectories
        for _ in range(exploration_depth):
            # Create a deep copy of the molecule
            current_mol = Chem.RWMol(mol)
            
            # Random number of modifications
            num_mods = random.randint(1, 3)
            
            for _ in range(num_mods):
                action = random.randint(0, len(self.substitution_atoms) - 1)

                if modification_type == 'random':
                    modification_type = random.choice(['add', 'remove', 'modify'])
                
                current_mol = modifications[modification_type](current_mol, action)
                
                # Ensure modification was successful
                if current_mol is None:
                    break
            
            if current_mol is not None:
                modification_variants.append(current_mol)
        
        return list(set(modification_variants))


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
    
    # Create an instance of the modifier
    modifier = ModifyAtom(setup_logger())
    modified_mols = []

    for smile in smiles_list:
        mol = Chem.MolFromSmiles(smile)
        modified_mol = modifier.explore_atom_modification_space(mol, 'random', 3)
        modified_mols.extend(modified_mol)
    
    modified_mols = [Chem.MolToSmiles(mol) for mol in modified_mols]

    print(modified_mols)

    from collections import Counter
    if Counter(smiles_list) == Counter(modified_mols):
        print("Lists are same")
    else:
        print("Lists are not same")'''

