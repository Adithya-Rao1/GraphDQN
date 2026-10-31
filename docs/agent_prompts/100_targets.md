# Task: targets-data (branch demo/targets). Owns studio/data/ (new). Network allowed ONLY for: rcsb.org / files.rcsb.org / data.rcsb.org, www.ebi.ac.uk (ChEMBL, UniProt/EBI APIs), rest.uniprot.org. Read-only public GETs, no credentials.

Objective: a small, verified repository of protein targets with structures, sequences and known drugs.

Build
1. `studio/data/targets/<id>.json` matching `schema.TargetInfo` for ~5 well-known drug targets with approved drugs. Candidates (verify every PDB id, chain, ligand and drug-target link against RCSB / ChEMBL before using; replace any that do not check out): EGFR kinase (erlotinib, PDB 1M17), HIV-1 protease (indinavir, 1HSG), ABL1 kinase (imatinib), and two more with an unambiguous, drug-like small-molecule ligand and a clean single-chain pocket (for example CDK2 or BRAF or carbonic anhydrase II). Prefer targets where the known drugs have 20-45 heavy atoms.
2. For each: protein-only cleaned PDB (chain `chain`, no waters/ligands/alt-locs) at `studio/data/structures/<id>.pdb`; the sequence of that chain from the PDB SEQRES or UniProt (record which); 3-5 known drugs with SMILES taken from ChEMBL/PubChem-style sources (record `source` and `chembl_id`; verify each parses in RDKit and matches the target in ChEMBL); pocket residue numbers (residues with any atom within 6 A of the co-crystal ligand, computed with RDKit/BioPython from the original complex).
3. `studio/data/targets/index.json`: list of ids and display names. `studio/data/README.md`: provenance table (what came from where, retrieval date), licence notes, and anything unverifiable.
4. `studio/data/dynamicbind_inputs/<id>/ligand.csv` in DynamicBind's input format (one column `ligand` of SMILES; check the DynamicBind repo README for the exact header) for each target's drugs, plus the protein PDB path.
5. A loader `studio/data/loader.py` (`list_targets()`, `get_target(id)`, `structure_path(id)`) and tests (every JSON validates against `TargetInfo`, every SMILES parses, every structure has the stated chain and sequence length within 5% of the sequence).

Acceptance: >= 4 targets fully verified; list any item you could not verify rather than guessing. Report retrieval details and file sizes.
