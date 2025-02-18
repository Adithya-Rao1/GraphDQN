from Model_Library.molecular_modifications.modification_imports import * 

class RingAtomModificationStrategy:
    @staticmethod
    def validate_modification(atom: Chem.Atom, modification_type: str) -> bool:
        """
        Validation of proposed atom modifications.
        
        Args:
            atom (Chem.Atom): RDKit atom to be modified
            modification_type (str): Type of modification being considered
        
        Returns:
            bool: Whether the modification is chemically sound
        """
        # Valence and bonding constraints
        max_valence_dict = {
            'charge': {'default': 3, 'exceptions': {'N': 4, 'P': 5, 'S': 6}},
            'hydrogen': {'default': 3, 'exceptions': {'C': 4, 'N': 3, 'O': 2, 'S': 2}},
            'radical': {'default': 3, 'exceptions': {'C': 3, 'N': 2, 'O': 2}}
        }
        
        atom_symbol = atom.GetSymbol()
        max_valence = max_valence_dict[modification_type].get(
            'exceptions', {}).get(atom_symbol, 
            max_valence_dict[modification_type]['default']
        )
        
        if modification_type == 'charge':
            return abs(atom.GetFormalCharge()) <= max_valence
        
        if modification_type == 'hydrogen':
            return (atom.GetNumExplicitHs() <= max_valence and 
                    atom.GetTotalNumHs() >= 0)
        
        if modification_type == 'radical':
            return atom.GetNumRadicalElectrons() <= max_valence
        
        return False
    
    @staticmethod
    def _chemistry_modifier(
        atom: Chem.Atom, 
        modification_type: str, 
        action: int
    ) -> Chem.Atom:
        """
        Atom modification with chemical constraints.
        
        Args:
            atom (Chem.Atom): RDKit atom to be modified
            modification_type (str): Type of modification
            action (int): Direction and magnitude of modification
        
        Returns:
            Chem.Atom: Modified atom
        """        
        atom_symbol = atom.GetSymbol()
        electronegativity = ELECTRONEGATIVITY.get(atom_symbol, 2.5)
        
        modification_ranges = {
            'charge': (-2, 2),
            'hydrogen': (0, 3),
            'radical': (0, 3)
        }
        
        # Stochastic quantum modification
        # Use electronegativity to bias the quantum noise and modification magnitude
        quantum_noise = np.random.normal(0, electronegativity / 10)
        modification_direction = (-1)**action
        
        min_val, max_val = modification_ranges[modification_type]
        
        if modification_type == 'charge':
            current_value = atom.GetFormalCharge()
            # Electronegativity scales the modification magnitude
            new_charge = current_value + modification_direction * (1 + quantum_noise * electronegativity)
            new_charge = max(min_val, min(new_charge, max_val))
            
            if RingAtomModificationStrategy.validate_modification(atom, 'charge'):
                atom.SetFormalCharge(int(round(new_charge)))
        
        elif modification_type == 'hydrogen':
            current_hydrogens = atom.GetNumExplicitHs()
            # More electronegative atoms are less likely to gain/lose hydrogens
            new_hydrogens = current_hydrogens + modification_direction * (1 + quantum_noise / electronegativity)
            new_hydrogens = max(min_val, min(new_hydrogens, max_val))
            
            if RingAtomModificationStrategy.validate_modification(atom, 'hydrogen'):
                atom.SetNumExplicitHs(int(round(new_hydrogens)))
        
        elif modification_type == 'radical':
            current_radicals = atom.GetNumRadicalElectrons()
            # Radical formation biased by electronegativity
            new_radicals = current_radicals + modification_direction * (1 + quantum_noise * (4 - electronegativity))
            new_radicals = max(min_val, min(new_radicals, max_val))
            
            if RingAtomModificationStrategy.validate_modification(atom, 'radical'):
                atom.SetNumRadicalElectrons(int(round(new_radicals)))
        
        return atom
    
    @staticmethod
    def advanced_chirality_modifier(
        atom: Chem.Atom, 
        action: int
    ) -> Chem.Atom:
        """
        Advanced chirality modification with stereochemical constraints.
        
        Args:
            atom (Chem.Atom): RDKit atom to be modified
            action (int): Direction of chirality modification
        
        Returns:
            Chem.Atom: Modified atom
        """
        # Comprehensive chirality type enumeration
        chirality_types = [
            Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL,
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
            Chem.rdchem.ChiralType.CHI_OTHER
        ]
        
        # Check for valid chirality modification
        if atom.GetHybridization() not in [
            Chem.rdchem.HybridizationType.SP3, 
            Chem.rdchem.HybridizationType.SP3D
        ]:
            return atom
        
        current_chirality = atom.GetChiralTag()
        current_index = chirality_types.index(current_chirality)
        
        # Stochastic chirality modification with bias
        noise_factor = np.random.normal(0, 0.3)
        new_index = int(current_index + (-1)**action * (1 + abs(noise_factor)))
        new_index = max(0, min(new_index, len(chirality_types) - 1))
        
        atom.SetChiralTag(chirality_types[new_index])
        return atom
    
    @staticmethod
    def generate_modification_strategies(
        probabilistic: bool = True
    ) -> List[Callable[[Chem.Atom, int], Chem.Atom]]:
        """
        Generate a list of probabilistic and quantum-chemistry-inspired modification strategies.
        
        Args:
            probabilistic (bool): Whether to use probabilistic modifications
        
        Returns:
            List of modification strategy functions
        """
        modification_strategies = [
            # Charge modification with electrochemical and quantum considerations
            lambda atom, action: (
                RingAtomModificationStrategy._chemistry_modifier(atom, 'charge', action)
            ),
            
            # Hydrogen manipulation with bond saturation and quantum insights
            lambda atom, action: (
                RingAtomModificationStrategy._chemistry_modifier(atom, 'hydrogen', action)
            ),
            
            # Radical electron modification with quantum chemistry perspective
            lambda atom, action: (
                RingAtomModificationStrategy._chemistry_modifier(atom, 'radical', action)
            ),
            
            # Advanced chirality tag modification
            lambda atom, action: (
                RingAtomModificationStrategy.advanced_chirality_modifier(atom, action)
            )
        ]
        
        # Optional probabilistic filtering
        if probabilistic:
            modification_strategies = [
                strategy for strategy in modification_strategies
                if np.random.random() > 0.3  # 70% chance of each strategy being applied
            ]
        
        return modification_strategies
    
    def _modify_ring_atom(mol: Union[Chem.Mol, Chem.RWMol], action: int) -> Optional[Chem.Mol]:
        """
        Advanced ring atom modification with quantum chemistry-inspired strategies.
        
        Args:
            mol (Mol): Input molecule to modify
            action (int): Action determining modification strategy
        
        Returns:
            Modified molecule or None
        """
        from atom_modifications import RingAtomModificationStrategy

        # Convert to RWMol if necessary
        rwmol = Chem.RWMol(mol) if isinstance(mol, Chem.Mol) else mol
        
        # Find all rings in the molecule
        rings = [list(ring) for ring in Chem.GetSymmSSSR(rwmol)]
            
        if not rings:
            print("No rings found in the molecule.")
            return None
        
        # Modification strategies mapped to their validation and execution
        modification_strategies = [
            {
                'name': 'charge',
                'strategy': lambda atom, action: RingAtomModificationStrategy._chemistry_modifier(
                    atom, 'charge', action
                ),
                'validation': lambda atom: RingAtomModificationStrategy.validate_modification(atom, 'charge')
            },
            {
                'name': 'hydrogen',
                'strategy': lambda atom, action: RingAtomModificationStrategy._chemistry_modifier(
                    atom, 'hydrogen', action
                ),
                'validation': lambda atom: RingAtomModificationStrategy.validate_modification(atom, 'hydrogen')
            },
            {
                'name': 'radical',
                'strategy': lambda atom, action: RingAtomModificationStrategy._chemistry_modifier(
                    atom, 'radical', action
                ),
                'validation': lambda atom: RingAtomModificationStrategy.validate_modification(atom, 'radical')
            },
            {
                'name': 'chirality',
                'strategy': lambda atom, action: RingAtomModificationStrategy.advanced_chirality_modifier(
                    atom, action
                ),
                'validation': lambda atom: atom.GetHybridization() in [
                    Chem.rdchem.HybridizationType.SP3, 
                    Chem.rdchem.HybridizationType.SP3D
                ]
            }
        ]
        
        # Select ring and atom
        ring = rings[action % len(rings)]
        atom_idx = ring[action % len(ring)]
        atom = rwmol.GetAtomWithIdx(atom_idx)
        
        # Attempt modifications with fallback logic
        successful_modification = False
        strategy_attempts = []
        
        for _ in range(len(modification_strategies)):
            # Randomly select a strategy not yet attempted
            available_strategies = [
                strategy for strategy in modification_strategies 
                if strategy not in strategy_attempts
            ]
            
            if not available_strategies:
                break
            
            current_strategy = np.random.choice(available_strategies)
            strategy_attempts.append(current_strategy)
            
            # Check if modification is applicable
            if current_strategy['validation'](atom):
                try:
                    # Apply the strategy
                    modified_atom = current_strategy['strategy'](atom, action)
                    
                    # Update the atom in the molecule with the modified properties
                    rwmol.GetAtomWithIdx(atom_idx).SetFormalCharge(modified_atom.GetFormalCharge())
                    rwmol.GetAtomWithIdx(atom_idx).SetNumExplicitHs(modified_atom.GetNumExplicitHs())
                    rwmol.GetAtomWithIdx(atom_idx).SetNumRadicalElectrons(modified_atom.GetNumRadicalElectrons())
                    rwmol.GetAtomWithIdx(atom_idx).SetChiralTag(modified_atom.GetChiralTag())

                    # Attempt to sanitize the molecule
                    try:
                        Chem.SanitizeMol(rwmol)
                        successful_modification = True
                        print(f"Successfully applied {current_strategy['name']} modification")
                        break
                    except Chem.rdchem.MolSanitizeException as sanitize_err:
                        # Revert any changes if sanitization fails
                        print(f"Sanitization failed for {current_strategy['name']}: {sanitize_err}")
                        # Optionally, restore the original atom state
                        rwmol.GetAtomWithIdx(atom_idx).SetFormalCharge(atom.GetFormalCharge())
                        rwmol.GetAtomWithIdx(atom_idx).SetNumExplicitHs(atom.GetNumExplicitHs())
                        rwmol.GetAtomWithIdx(atom_idx).SetNumRadicalElectrons(atom.GetNumRadicalElectrons())
                        rwmol.GetAtomWithIdx(atom_idx).SetChiralTag(atom.GetChiralTag())
                
                except Exception as strategy_err:
                    print(f"Error applying {current_strategy['name']} modification: {strategy_err}")
        
        if not successful_modification:
            print("No successful modification found.")
            return None
        
        return rwmol