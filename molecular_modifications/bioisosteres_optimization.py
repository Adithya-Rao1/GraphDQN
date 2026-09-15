from molecular_modifications.modification_imports import *
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
from molecular_modifications.logger import setup_molecule_logger

class ModifyBioisosteres:
    def __init__(self, logger, log: bool = False):
        super(ModifyBioisosteres, self).__init__()

        self.logger = logger
        self.log = log

        self.acidic_bioisosteres = {
            "carboxylic_acid": ["tetrazole", "phosphonic_acid", "sulfonic_acid"],
            "tetrazole": ["carboxylic_acid", "phosphonic_acid"],
            "hydroxamic_acid": ["carboxylic_acid", "tetrazole"]
        }

        self.basic_bioisosteres = {
            "amine": ["pyridine", "imidazole", "guanidine"],
            "amidine": ["guanidine", "imidazole"],
            "guanidine": ["amidine", "2-aminopyridine"],
            "amide": ["sulfonamide", "retroamide", "urea"],
            "ester": ["amide", "thioester", "ketone", "sulfonamide"],
            "alkyl": ["cycloalkyl", "CF3", "tBu"],
            "phenyl": ["pyridyl", "thiophene", "furan", "pyrrole"],
            "ketone": ["oxime", "hydrazone", "thiocarbonyl"]
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
        matches = {}
        for fg_name, pattern in self.fg_patterns.items():
            pattern_mol = Chem.MolFromSmarts(pattern)
            if pattern_mol is not None:
                matches[fg_name] = mol.GetSubstructMatches(pattern_mol)
        return matches

    @staticmethod
    def _remove_atoms_preserving_anchors(em: Chem.EditableMol, remove_indices, anchor_indices):
        """Remove remove_indices from an EditableMol and return each of
        anchor_indices' index after removal.

        RWMol/EditableMol.RemoveAtom shifts every atom with a higher index
        down by one; any index captured before a removal (e.g. a SMARTS match
        atom we intend to keep as an attachment point) is stale afterward
        unless corrected for how many removed atoms sat below it. Removing in
        descending order keeps not-yet-removed indices valid throughout.
        """
        remove_indices = sorted(set(remove_indices), reverse=True)
        for idx in remove_indices:
            em.RemoveAtom(idx)
        return {
            anchor: anchor - sum(1 for idx in remove_indices if idx < anchor)
            for anchor in anchor_indices
        }

    def replace_carboxylic_acid_with_tetrazole(self, mol: Chem.Mol, match_atoms: Tuple[int]) -> Optional[str]:
        try:
            em = Chem.EditableMol(mol)
            anchors = self._remove_atoms_preserving_anchors(em, match_atoms[1:], [match_atoms[0]])
            anchor_idx = anchors[match_atoms[0]]

            n1_idx = em.AddAtom(Chem.Atom(7))
            n2_idx = em.AddAtom(Chem.Atom(7))
            n3_idx = em.AddAtom(Chem.Atom(7))
            n4_idx = em.AddAtom(Chem.Atom(7))

            em.AddBond(anchor_idx, n1_idx, Chem.BondType.SINGLE)
            em.AddBond(n1_idx, n2_idx, Chem.BondType.SINGLE)
            em.AddBond(n2_idx, n3_idx, Chem.BondType.DOUBLE)
            em.AddBond(n3_idx, n4_idx, Chem.BondType.SINGLE)
            em.AddBond(n4_idx, anchor_idx, Chem.BondType.DOUBLE)

            new_mol = em.GetMol()
            try:
                Chem.SanitizeMol(new_mol)
                if self.log:
                    self.logger.info("Carboxylic Acid To Tetrazole Replacement Successful")
                return Chem.MolToSmiles(Chem.Mol(new_mol))
            except Exception:
                if self.log:
                    self.logger.warning(f"Carboxylic Acid To Tetrazole Replacement Not Fully Sanitized")
                return None

        except Exception as e:
            if self.log:
                self.logger.error(f"Failed to replace carboxylic acid with tetrazole: {str(e)}")
            return None

    def replace_carboxylic_acid_with_phosphonic_acid(self, mol: Chem.Mol, match_atoms: Tuple[int]) -> Optional[str]:
        try:
            em = Chem.EditableMol(mol)
            anchors = self._remove_atoms_preserving_anchors(em, match_atoms[1:], [match_atoms[0]])
            anchor_idx = anchors[match_atoms[0]]

            p_idx = em.AddAtom(Chem.Atom(15))
            o1_idx = em.AddAtom(Chem.Atom(8))  # P=O
            o2_idx = em.AddAtom(Chem.Atom(8))  # P-OH
            o3_idx = em.AddAtom(Chem.Atom(8))  # P-OH

            em.AddBond(anchor_idx, p_idx, Chem.BondType.SINGLE)
            em.AddBond(p_idx, o1_idx, Chem.BondType.DOUBLE)
            em.AddBond(p_idx, o2_idx, Chem.BondType.SINGLE)
            em.AddBond(p_idx, o3_idx, Chem.BondType.SINGLE)

            new_mol = em.GetMol()

            try:
                Chem.SanitizeMol(new_mol)
                if self.log:
                    self.logger.info("Carboxylic Acid To Phosphonic Acid Replacement Successful")
                return Chem.MolToSmiles(Chem.Mol(new_mol))
            except Exception:
                if self.log:
                    self.logger.warning(f"Carboxylic Acid To Phosphonic Acid Replacement Not Fully Sanitized")
                return None

        except Exception as e:
            if self.log:
                self.logger.error(f"Failed to replace with phosphonic acid: {str(e)}")
            return None

    def replace_amine_with_pyridine(self, mol: Chem.Mol, match_atoms: Tuple[int]) -> Optional[str]:
        try:
            amine_idx = match_atoms[0]
            heavy_neighbors = [n.GetIdx() for n in mol.GetAtomWithIdx(amine_idx).GetNeighbors()
                                if n.GetAtomicNum() != 1]
            if len(heavy_neighbors) != 1:
                if self.log:
                    self.logger.warning("Amine has other than one heavy-atom substituent; skipping pyridine replacement.")
                return None
            r_idx = heavy_neighbors[0]

            em = Chem.EditableMol(mol)
            anchors = self._remove_atoms_preserving_anchors(em, [amine_idx], [r_idx])
            r_idx = anchors[r_idx]

            c2_idx = em.AddAtom(Chem.Atom(6))
            c3_idx = em.AddAtom(Chem.Atom(6))
            c4_idx = em.AddAtom(Chem.Atom(6))
            c5_idx = em.AddAtom(Chem.Atom(6))
            c6_idx = em.AddAtom(Chem.Atom(6))
            n_idx = em.AddAtom(Chem.Atom(7))

            em.AddBond(r_idx, c2_idx, Chem.BondType.SINGLE)
            em.AddBond(c2_idx, c3_idx, Chem.BondType.DOUBLE)
            em.AddBond(c3_idx, c4_idx, Chem.BondType.SINGLE)
            em.AddBond(c4_idx, c5_idx, Chem.BondType.DOUBLE)
            em.AddBond(c5_idx, c6_idx, Chem.BondType.SINGLE)
            em.AddBond(c6_idx, n_idx, Chem.BondType.DOUBLE)
            em.AddBond(n_idx, c2_idx, Chem.BondType.SINGLE)

            new_mol = em.GetMol()

            try:
                Chem.SanitizeMol(new_mol)
                if self.log:
                    self.logger.info("Replaced amine with pyridine")
                return Chem.MolToSmiles(Chem.Mol(new_mol))

            except Exception:
                if self.log:
                    self.logger.warning(f"Amine Replacement Not Fully Sanitized. Replaced amine with pyridine.")
                return None

        except Exception as e:
            if self.log:
                self.logger.error(f"Failed to replace amine with pyridine: {str(e)}")
            return None

    _PHENYL_RING_SPECS = {
        "pyridyl": ([6, 6, 6, 6, 6, 7],
                    [Chem.BondType.DOUBLE, Chem.BondType.SINGLE, Chem.BondType.DOUBLE,
                     Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.SINGLE]),
        "thiophene": ([6, 6, 6, 6, 16],
                      [Chem.BondType.DOUBLE, Chem.BondType.SINGLE, Chem.BondType.DOUBLE,
                       Chem.BondType.SINGLE, Chem.BondType.SINGLE]),
        "furan": ([6, 6, 6, 6, 8],
                  [Chem.BondType.DOUBLE, Chem.BondType.SINGLE, Chem.BondType.DOUBLE,
                   Chem.BondType.SINGLE, Chem.BondType.SINGLE]),
        "pyrrole": ([6, 6, 6, 6, 7],
                    [Chem.BondType.DOUBLE, Chem.BondType.SINGLE, Chem.BondType.DOUBLE,
                     Chem.BondType.SINGLE, Chem.BondType.SINGLE]),
    }

    def replace_phenyl_with_heterocycle(self, mol: Chem.Mol, match_atoms: Tuple[int],
                                      replacement: str = "pyridyl") -> Optional[str]:
        try:
            if replacement not in self._PHENYL_RING_SPECS:
                if self.log:
                    self.logger.error(f"Unknown phenyl replacement: {replacement}")
                return None

            match_set = set(match_atoms)
            external_bonds = []
            for idx in match_atoms:
                atom = mol.GetAtomWithIdx(idx)
                for bond in atom.GetBonds():
                    other = bond.GetOtherAtomIdx(idx)
                    if other not in match_set:
                        external_bonds.append((idx, other))

            if len(external_bonds) != 1:
                if self.log:
                    self.logger.warning("Phenyl ring has other than one external attachment; skipping heterocycle replacement.")
                return None
            _, external_idx = external_bonds[0]

            em = Chem.EditableMol(mol)
            anchors = self._remove_atoms_preserving_anchors(em, match_atoms, [external_idx])
            external_idx = anchors[external_idx]

            atomic_nums, bond_orders = self._PHENYL_RING_SPECS[replacement]
            ring_indices = [em.AddAtom(Chem.Atom(an)) for an in atomic_nums]

            n = len(ring_indices)
            for i in range(n):
                em.AddBond(ring_indices[i], ring_indices[(i + 1) % n], bond_orders[i])

            em.AddBond(external_idx, ring_indices[0], Chem.BondType.SINGLE)

            new_mol = em.GetMol()
            try:
                Chem.SanitizeMol(new_mol)
                if self.log:
                    self.logger.info(f"Phenyl Replacement Successful: Replaced phenyl with {replacement}")
                return Chem.MolToSmiles(Chem.Mol(new_mol))
            except Exception:
                if self.log:
                    self.logger.warning(f"Phenyl Replacement Not Fully Sanitized: Replaced phenyl with {replacement}")
                return None

        except Exception as e:
            if self.log:
                self.logger.error(f"Failed to replace phenyl with {replacement}: {str(e)}")
            return None

    def replace_amide_with_bioisostere(self, mol: Chem.Mol, match_atoms: Tuple[int],
                                     replacement: str = "sulfonamide") -> Optional[str]:
        # match_atoms = (N, C(=O), O, R) from "[NX3][CX3](=[OX1])[#6]".
        try:
            if replacement == "sulfonamide":
                n_idx, c_idx, o_idx, r_idx = match_atoms

                em = Chem.EditableMol(mol)
                anchors = self._remove_atoms_preserving_anchors(em, [c_idx, o_idx], [n_idx, r_idx])
                n_idx, r_idx = anchors[n_idx], anchors[r_idx]

                s_idx = em.AddAtom(Chem.Atom(16))
                o1_idx = em.AddAtom(Chem.Atom(8))
                o2_idx = em.AddAtom(Chem.Atom(8))

                em.AddBond(n_idx, s_idx, Chem.BondType.SINGLE)   # N-S
                em.AddBond(s_idx, r_idx, Chem.BondType.SINGLE)   # S-R
                em.AddBond(s_idx, o1_idx, Chem.BondType.DOUBLE)  # S=O
                em.AddBond(s_idx, o2_idx, Chem.BondType.DOUBLE)  # S=O

            elif replacement == "retroamide":
                n_idx, c_idx, o_idx, r_idx = match_atoms
                n_atom = mol.GetAtomWithIdx(n_idx)
                n_external = [nb.GetIdx() for nb in n_atom.GetNeighbors()
                              if nb.GetIdx() != c_idx and nb.GetAtomicNum() != 1]
                if len(n_external) != 1:
                    if self.log:
                        self.logger.warning("Amide nitrogen has no single external substituent; skipping retroamide.")
                    return None
                n_ext_idx = n_external[0]

                em = Chem.EditableMol(mol)
                em.RemoveBond(n_idx, n_ext_idx)
                em.RemoveBond(c_idx, r_idx)
                em.AddBond(c_idx, n_ext_idx, Chem.BondType.SINGLE)
                em.AddBond(n_idx, r_idx, Chem.BondType.SINGLE)

            else:
                if self.log:
                    self.logger.error(f"Unknown amide replacement: {replacement}")
                return None

            new_mol = em.GetMol()
            try:
                Chem.SanitizeMol(new_mol)
                if self.log:
                    self.logger.info(f"Amide Replacement Successful: Replaced amide with {replacement}")
                return Chem.MolToSmiles(Chem.Mol(new_mol))
            except Exception:
                if self.log:
                    self.logger.warning(f"Amide Replacement Not Fully Sanitized: Replaced amide with {replacement}")
                return None

        except Exception as e:
            if self.log:
                self.logger.error(f"Failed to replace amide: {str(e)}")
            return None

    def apply_modification(self, mol: Chem.Mol, chosen_fg_type: str, chosen_replacement: Optional[str] = None) -> Optional[str]:
        fg_matches = self.identify_functional_groups(mol)

        valid_modifications = []
        for fg, match_atoms in fg_matches.items():
            if fg in self.acidic_bioisosteres.keys() or fg in self.basic_bioisosteres.keys():
                for match in match_atoms:
                    valid_modifications.append((fg, match))

        if not valid_modifications:
            if self.log:
                self.logger.error("No valid modifications found.")
            return None

        # TODO: BE ABLE TO CHOOSE WHICH FG_TYPE
        # TODO: BE ABLE TO CHOOSE WHICH REPLACEMENT FOR EACH FG_TYPE
        fg_type, match = None, None
        if chosen_fg_type:
            for fg, match_atoms in valid_modifications:
                if fg == chosen_fg_type:
                    fg_type, match = fg, match_atoms
                    break

        if not fg_type:
            if self.log:
                self.logger.error(f"No match in molecule for chosen bioisosteric replacement: {chosen_fg_type}.")
            return None

        try:
            if fg_type == "carboxylic_acid":
                if chosen_replacement == "tetrazole":
                    new_smiles = self.replace_carboxylic_acid_with_tetrazole(mol, match)
                elif chosen_replacement == "phosphonic_acid":
                    new_smiles = self.replace_carboxylic_acid_with_phosphonic_acid(mol, match)
                else:
                    new_smiles = None
            elif fg_type == "amine":
                new_smiles = self.replace_amine_with_pyridine(mol, match)
            elif fg_type == "amide":
                new_smiles = self.replace_amide_with_bioisostere(mol, match, chosen_replacement)
            elif fg_type == "phenyl":
                new_smiles = self.replace_phenyl_with_heterocycle(mol, match, chosen_replacement)
            else:
                if self.log:
                    self.logger.error(f"Unknown functional group type: {fg_type}")
                return None

            if new_smiles is None:
                return None

            new_mol = Chem.MolFromSmiles(new_smiles)
            if new_mol is None or not self.validate_modification(mol, new_mol):
                return None

            return new_smiles

        except Exception as e:
            if self.log:
                self.logger.error(f"Failed to apply a modification. Error: {e}")
            return None

    def validate_modification(self, original_mol: Chem.Mol, modified_mol: Chem.Mol) -> bool:
        try:
            Chem.SanitizeMol(modified_mol)
            orig_mw = Chem.Descriptors.ExactMolWt(original_mol)
            mod_mw = Chem.Descriptors.ExactMolWt(modified_mol)
            if abs(mod_mw - orig_mw) > 200:
                return False

            if Chem.rdmolops.GetFormalCharge(modified_mol) > 2:
                return False

            # TODO: Further validation checks

            return True

        except Exception:
            if self.log:
                self.logger.error("Failed to validate modification")
            return False

if __name__ == "__main__":
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
                    'O=C(N(SC1=O)C2=C3C=CC=CC3=CC=C2)N1CC4=CC=CC=C4',
                    'CC(=O)OC1=CC=CC=C1C(=O)O',]

    modify_bioisosteres = ModifyBioisosteres(setup_molecule_logger())
    new_mols = []
    for smile in smiles_list:
        mol = Chem.MolFromSmiles(smile)
        for fg_type, replacement in [
            ('carboxylic_acid', 'tetrazole'),
            ('carboxylic_acid', 'phosphonic_acid'),
            ('amine', None),
            ('amide', 'sulfonamide'),
            ('amide', 'retroamide'),
            ('phenyl', 'pyridyl'),
            ('phenyl', 'thiophene'),
            ('phenyl', 'furan'),
            ('phenyl', 'pyrrole'),
        ]:
            result = modify_bioisosteres.apply_modification(mol, fg_type, replacement)
            if result is not None:
                new_mols.append(result)

    print(new_mols)
