def provenance_kwargs(config) -> dict:
    return dict(
        config_id=config.id,
        config_name=config.name,
        starting_molecule_id=config.starting_molecule_id,
        starting_smiles=config.starting_molecule.canonical_smiles,
        target_protein_id=config.target_protein_id,
        target_protein_name=config.target_protein.name,
        off_target_protein_id=config.off_target_protein_id,
        off_target_protein_name=config.off_target_protein.name if config.off_target_protein else None,
    )
