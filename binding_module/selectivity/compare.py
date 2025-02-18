def compare_affinities(target_affinity, off_target_affinity):
    """
    Compare affinities and rank SMILES based on KD ratios.
    
    Args:
        target_prot_seq (list): List of target protein sequences.
        prot_seq (list): List of off-target protein sequences.
        mol_smiles (list): List of SMILES strings for molecules.
        target_affinity (list): Affinity values for the target protein.
        off_target_affinity (list): Affinity values for the off-target protein.

    Returns:
        list: Ranked list of tuples (SMILES, KD_ratio) in descending order of KD_ratio.
    """
    kd_ratios = []
    
    if type(target_affinity) == list:
        # Compute KD ratio for each SMILES
        for target_aff, off_target_aff in zip(target_affinity, off_target_affinity):
            if target_aff == 0:  # Avoid division by zero
                continue
            kd_ratio = off_target_aff / target_aff
            kd_ratios.append(kd_ratio)
        
        return kd_ratios
    
    else:
        kd_ratio = off_target_affinity / target_affinity
        return kd_ratio

# mol_smiles = ["C1=CC=CC=C1", "C1=CC=CN=C1"]
# target_affinity = [1.0, 2.0]
# off_target_affinity = [5.0, 3.0]

# result = compare_affinities(mol_smiles, target_affinity, off_target_affinity)
# print(result)
