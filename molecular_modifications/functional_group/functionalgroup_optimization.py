from Model_Library.molecular_modifications.modification_imports import *
from functionalgroup_interconversion import ModifyFunctionalGroup

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModificationResult:
    """Stores the result of a molecular modification attempt."""
    success: bool
    modified_mol: Optional[Chem.Mol]
    modification_type: str
    description: str
    error: Optional[str] = None

class ModifyFunctionalGroup:
    """
    Initialize the ModifyFunctionalGroup class.
    """
    
    def __init__(self, logger):
        self.logger = logger
        self.functional_groups = {
            'carboxyl': '[CX3](=O)[OX2H1]',
            'hydroxyl': '[OX2H]',
            'amine': '[NX3;H2,H1;!$(NC=O)]',
            'amide': '[NX3][CX3](=[OX1])',
            'sulfonamide': '[NX3][SX4](=[OX1])(=[OX1])',
            'ester': '[#6][CX3](=O)[OX2H0][#6]',
            'ketone': '[#6][CX3](=O)[#6]',
            'aldehyde': '[CX3H1](=O)[#6]',
            'ether': '[OX2]([#6])[#6]',
            'thiol': '[SX2H]',
            'phosphate': '[PX4](=[OX1])([$([OX2H]),$([OX2][#6])])([$([OX2H]),$([OX2][#6])])[$([OX2H]),$([OX2][#6])]',
            'nitro': '[$([NX3](=O)=O),$([NX3+](=O)[O-])]'
        }
        
        self.bioisosteres = {
            'carboxyl': ['tetrazole', 'phosphonic_acid', 'sulfonic_acid'],
            'amine': ['pyridine', 'thiophene', 'furan'],
            'amide': ['sulfonamide', 'phosphonamide', 'sulfonylurea'],
            'ester': ['amide', 'thioester', 'phosphonate']
        }
        
    def identify_functional_groups(self, mol: Chem.Mol) -> Dict[str, List[int]]:
        """
        Identifies all functional groups present in the molecule.
        
        Args:
            mol (Chem.Mol): Input molecule
            
        Returns:
            Dictionary mapping functional group types to lists of atom indices
        """
        groups = {}
        for name, smarts in self.functional_groups.items():
            pattern = Chem.MolFromSmarts(smarts)
            if pattern is not None:
                matches = mol.GetSubstructMatches(pattern)
                if matches:
                    groups[name] = [list(match) for match in matches]
        return groups

    def validate_modification(self, mol: Chem.Mol) -> bool:
        """
        Validates chemical structure after modification.
        
        Args:
            mol (Chem.Mol): Modified molecule to validate
            
        Returns:
            Boolean indicating if molecule is valid
        """
        if mol is None:
            return False
            
        try:
            Chem.SanitizeMol(mol)
            
            if Descriptors.ExactMolWt(mol) > 500:
                return False
                
            if Descriptors.NumRotatableBonds(mol) > 20:
                return False
                
            if abs(Descriptors.MolLogP(mol)) > 6:
                return False
                
            return True
            
        except Exception as e:
            self.logger.warning(f"Validation failed: {str(e)}")
            return False

    def modify_functional_group(self, mol: Chem.Mol, group_type: str, 
                              position: List[int]) -> ModificationResult:
        """
        Modifies a specific functional group at the given position.
        
        Args:
            mol: Input molecule
            group_type: Type of functional group to modify
            position: Atom indices of the functional group
            
        Returns:
            ModificationResult containing the modified molecule and metadata
        """
        try:
            # Create editable mol object
            edit_mol = Chem.RWMol(mol)
            
            if group_type == 'carboxyl':
                # Convert carboxyl to ester
                edit_mol.RemoveAtom(position[2])  # Remove OH
                new_o = edit_mol.AddAtom(Chem.Atom(8))  # Add new O
                new_c = edit_mol.AddAtom(Chem.Atom(6))  # Add new C
                edit_mol.AddBond(position[0], new_o, Chem.BondType.SINGLE)
                edit_mol.AddBond(new_o, new_c, Chem.BondType.SINGLE)
                
            elif group_type == 'amine':
                # Convert primary amine to secondary amine
                new_c = edit_mol.AddAtom(Chem.Atom(6))  # Add new C
                edit_mol.AddBond(position[0], new_c, Chem.BondType.SINGLE)
                
            # Add more modification types here...
            
            new_mol = edit_mol.GetMol()
            
            # Cleanup and validate
            Chem.SanitizeMol(new_mol)
            AllChem.EmbedMolecule(new_mol, randomSeed=42)
            
            if self.validate_modification(new_mol):
                return ModificationResult(
                    success=True,
                    modified_mol=new_mol,
                    modification_type=f"modify_{group_type}",
                    description=f"Modified {group_type} at position {position}"
                )
            else:
                return ModificationResult(
                    success=False,
                    modified_mol=None,
                    modification_type=f"modify_{group_type}",
                    description="Failed validation",
                    error="Invalid chemical structure"
                )
                
        except Exception as e:
            return ModificationResult(
                success=False,
                modified_mol=None,
                modification_type=f"modify_{group_type}",
                description="Modification failed",
                error=str(e)
            )

    def add_functional_group(self, mol: Chem.Mol, group_type: str, 
                           position: int) -> ModificationResult:
        """
        Adds a new functional group to the molecule at specified position.
        
        Args:
            mol: Input molecule
            group_type: Type of functional group to add
            position: Atom index where to add the group
            
        Returns:
            ModificationResult containing the modified molecule and metadata
        """
        try:
            edit_mol = Chem.RWMol(mol)
            
            if group_type == 'hydroxyl':
                new_o = edit_mol.AddAtom(Chem.Atom(8))  # Add O
                edit_mol.AddBond(position, new_o, Chem.BondType.SINGLE)
                
            elif group_type == 'methyl':
                new_c = edit_mol.AddAtom(Chem.Atom(6))  # Add C
                edit_mol.AddBond(position, new_c, Chem.BondType.SINGLE)
                
            # Add more addition types...
            
            new_mol = edit_mol.GetMol()
            Chem.SanitizeMol(new_mol)
            
            if self.validate_modification(new_mol):
                return ModificationResult(
                    success=True,
                    modified_mol=new_mol,
                    modification_type=f"add_{group_type}",
                    description=f"Added {group_type} at position {position}"
                )
            else:
                return ModificationResult(
                    success=False,
                    modified_mol=None,
                    modification_type=f"add_{group_type}",
                    description="Failed validation",
                    error="Invalid chemical structure"
                )
                
        except Exception as e:
            return ModificationResult(
                success=False,
                modified_mol=None,
                modification_type=f"add_{group_type}",
                description="Addition failed",
                error=str(e)
            )

    def replace_with_bioisostere(self, mol: Chem.Mol, group_type: str,
                                position: List[int]) -> ModificationResult:
        """
        Replaces a functional group with a bioisostere.
        
        Args:
            mol: Input molecule
            group_type: Type of functional group to replace
            position: Atom indices of the functional group
            
        Returns:
            ModificationResult containing the modified molecule and metadata
        """
        try:
            if group_type not in self.bioisosteres:
                return ModificationResult(
                    success=False,
                    modified_mol=None,
                    modification_type="bioisostere",
                    description=f"No bioisosteres defined for {group_type}",
                    error="Invalid group type"
                )
                
            # Randomly select a bioisostere replacement
            replacement = np.random.choice(self.bioisosteres[group_type])
            
            # Create editable mol object
            edit_mol = Chem.RWMol(mol)
            
            # Remove original group
            for idx in sorted(position, reverse=True):
                edit_mol.RemoveAtom(idx)
                
            # Add bioisostere structure
            if replacement == 'tetrazole':
                # Implementation for tetrazole addition
                new_atoms = []
                for _ in range(5):  # Add 5 atoms for tetrazole ring
                    new_atoms.append(edit_mol.AddAtom(Chem.Atom(7)))  # N atoms
                
                # Add bonds to form tetrazole ring
                edit_mol.AddBond(new_atoms[0], new_atoms[1], Chem.BondType.SINGLE)
                edit_mol.AddBond(new_atoms[1], new_atoms[2], Chem.BondType.DOUBLE)
                edit_mol.AddBond(new_atoms[2], new_atoms[3], Chem.BondType.SINGLE)
                edit_mol.AddBond(new_atoms[3], new_atoms[4], Chem.BondType.DOUBLE)
                edit_mol.AddBond(new_atoms[4], new_atoms[0], Chem.BondType.SINGLE)
            
            # Add more bioisostere implementations...
            
            new_mol = edit_mol.GetMol()
            Chem.SanitizeMol(new_mol)
            
            if self.validate_modification(new_mol):
                return ModificationResult(
                    success=True,
                    modified_mol=new_mol,
                    modification_type="bioisostere",
                    description=f"Replaced {group_type} with {replacement}"
                )
            else:
                return ModificationResult(
                    success=False,
                    modified_mol=None,
                    modification_type="bioisostere",
                    description="Failed validation",
                    error="Invalid chemical structure"
                )
                
        except Exception as e:
            return ModificationResult(
                success=False,
                modified_mol=None,
                modification_type="bioisostere",
                description="Replacement failed",
                error=str(e)
            )

    def generate_modifications(self, mol: Chem.Mol, 
                             n_attempts: int = 10) -> List[ModificationResult]:
        """
        Generates multiple valid modifications of the input molecule.
        
        Args:
            mol: Input molecule
            n_attempts: Number of modification attempts to try
            
        Returns:
            List of successful ModificationResults
        """
        modifications = []
        groups = self.identify_functional_groups(mol)
        
        for _ in range(n_attempts):
            if not groups:
                continue
                
            # Randomly select modification type and group
            mod_type = np.random.choice(['modify', 'add', 'bioisostere'])
            group_type = np.random.choice(list(groups.keys()))
            
            if mod_type == 'modify':
                position = np.random.choice(groups[group_type])
                converter = ModifyFunctionalGroup()
                
                result = self.modify_functional_group(mol, group_type, position)
                
            elif mod_type == 'add':
                # Find suitable attachment points
                available_positions = [atom.GetIdx() for atom in mol.GetAtoms() 
                                    if atom.GetDegree() < atom.GetTotalValence()]
                if not available_positions:
                    continue
                    
                position = np.random.choice(available_positions)
                result = self.add_functional_group(mol, group_type, position)
                
            else:  # bioisostere
                if group_type not in self.bioisosteres:
                    continue
                    
                position = np.random.choice(groups[group_type])
                result = self.replace_with_bioisostere(mol, group_type, position)
                
            if result.success:
                modifications.append(result)
                
        return modifications