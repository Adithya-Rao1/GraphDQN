# studio/data: target repository

5 protein targets, each with a cleaned structure, sequence, known approved drugs and pocket residues.
Retrieved 2026-10-30 by read-only GETs to files.rcsb.org, data.rcsb.org, www.ebi.ac.uk (ChEMBL API), rest.uniprot.org. No credentials.
Use `studio.data.loader`: `list_targets()`, `get_target(id)` (validated `TargetInfo`), `structure_path(id)`, `dynamicbind_ligand_csv(id)`.

| id | PDB | chain | co-crystal ligand | resolution | ChEMBL target | UniProt | drugs (heavy atoms) |
|---|---|---|---|---|---|---|---|
| egfr | 1M17 | A | AQ4 = erlotinib | 2.60 A | CHEMBL203 | P00533 | erlotinib 29, gefitinib 31, lapatinib 40, afatinib 34 |
| hiv1_protease | 1HSG | A (+B) | MK1 = indinavir | 2.00 A | CHEMBL243 | P03367 | indinavir 45, saquinavir 49, ritonavir 50, nelfinavir 40, amprenavir 35 |
| abl1 | 2HYY | A | STI = imatinib | 2.40 A | CHEMBL1862 | P00519 | imatinib 37, nilotinib 39, dasatinib 33, bosutinib 36, ponatinib 39 |
| braf | 3OG7 | A | 032 = vemurafenib | 2.45 A | CHEMBL5145 | P15056 | vemurafenib 33, dabrafenib 35, encorafenib 36 |
| fxa | 2P16 | A | GG2 = apixaban | 2.30 A | CHEMBL244 | P00742 | apixaban 34, rivaroxaban 29, edoxaban 37, betrixaban 32 |

## Provenance
- **Structures** (`structures/<id>.pdb`): RCSB PDB entry, ATOM records of the stated chain(s) only; waters, ligands, ions, alt-locs other than blank/"A" removed. Original residue numbering kept. No hydrogens added, missing loops not rebuilt. PDB data is CC0.
- **Sequence**: PDB SEQRES of the stated chain, with expression tags trimmed to the span aligned to UniProt (EGFR "GSHMA", BRAF "MKKGHHHHHHGSRDAA" plus a trailing stretch). It is the crystallised construct, not full-length UniProt. BRAF differs from UniProt P15056 at ~11 positions (fusion construct / engineered), checked by alignment (247/258 match).
- **Drugs**: ChEMBL `canonical_smiles` (via API, ChEMBL is CC BY-SA 3.0). Each drug has >=1 ChEMBL activity with a pChEMBL value against the ChEMBL target. All SMILES parse in RDKit.
- **Co-crystal ligand check**: RCSB chem-comp SMILES vs the ChEMBL reference drug, first InChIKey block (connectivity) compared in RDKit: all 5 match.
- **Pocket residues**: residue numbers of the kept chain(s) with any atom within 6 A of any heavy atom of the co-crystal ligand, computed from the original complex (own script, no BioPython).
- **DynamicBind inputs** (`dynamicbind_inputs/<id>/ligand.csv`): header `ligand`, one SMILES per line (the drugs). Protein = `structures/<id>.pdb`.

## Could not verify / caveats
- ChEMBL `mechanism` API returned no rows for most pairs (table is sparse), so drug-target links were verified through bioactivity records instead, not mechanism-of-action. Indinavir (CHEMBL115) has no pChEMBL activity under that ID; the activities are filed under its hydrate CHEMBL5483011 (parent = CHEMBL115). Medians for imatinib/ABL1 (pChEMBL 6.7) include cell assays.
- The DynamicBind CSV format could not be checked against its GitHub README (github.com is outside the allowed hosts). Header `ligand` follows the task brief; confirm on the Lambda box.
- HIV-1 protease is a homodimer; the pocket needs both chains, so the structure keeps A and B. `sequence` is chain A (99 aa); pocket numbers are identical for both chains.
- 1M17, 2HYY etc. are single crystal structures; ligand-bound (holo) conformations. DFG-out (imatinib/ABL1) pose may not suit dasatinib/bosutinib.
- Resolved residues are 95-100% of `sequence`; unresolved loops are not modelled.
- ABL1 uses human 2HYY, not mouse 1IEP; CDK2/CA II were rejected (no approved CDK2 drug; CA II drugs are <20 heavy atoms).
