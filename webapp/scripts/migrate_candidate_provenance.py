import sqlite3

from webapp.backend.config import DATABASE_PATH

NEW_COLUMNS = {
    "training_run_id": "INTEGER",
    "config_id": "INTEGER",
    "config_name": "VARCHAR",
    "starting_molecule_id": "INTEGER",
    "starting_smiles": "VARCHAR",
    "target_protein_id": "INTEGER",
    "target_protein_name": "VARCHAR",
    "off_target_protein_id": "INTEGER",
    "off_target_protein_name": "VARCHAR",
}


def main():
    print(f"Migrating {DATABASE_PATH} ...")
    conn = sqlite3.connect(str(DATABASE_PATH))
    cur = conn.cursor()

    tables = {row[0] for row in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "generated_candidates" not in tables:
        print("No generated_candidates table yet -- nothing to migrate (it will be created fresh).")
        conn.close()
        return

    existing_cols = {row[1] for row in cur.execute("PRAGMA table_info(generated_candidates)")}
    added = []
    for col, coltype in NEW_COLUMNS.items():
        if col not in existing_cols:
            cur.execute(f"ALTER TABLE generated_candidates ADD COLUMN {col} {coltype}")
            added.append(col)
    conn.commit()
    print(f"Added columns: {added}" if added else "All columns already present.")

    cur.execute("""
        SELECT gc.id, gb.training_run_id, tr.config_id, oc.name,
               oc.starting_molecule_id, sm.canonical_smiles,
               oc.target_protein_id, tp.name,
               oc.off_target_protein_id, otp.name
        FROM generated_candidates gc
        JOIN generation_batches gb ON gc.generation_batch_id = gb.id
        JOIN training_runs tr ON gb.training_run_id = tr.id
        JOIN optimization_configs oc ON tr.config_id = oc.id
        JOIN starting_molecules sm ON oc.starting_molecule_id = sm.id
        JOIN proteins tp ON oc.target_protein_id = tp.id
        LEFT JOIN proteins otp ON oc.off_target_protein_id = otp.id
        WHERE gc.training_run_id IS NULL
    """)
    rows = cur.fetchall()
    for (cid, run_id, config_id, config_name, mol_id, mol_smiles,
         prot_id, prot_name, off_id, off_name) in rows:
        cur.execute("""
            UPDATE generated_candidates
            SET training_run_id=?, config_id=?, config_name=?,
                starting_molecule_id=?, starting_smiles=?,
                target_protein_id=?, target_protein_name=?,
                off_target_protein_id=?, off_target_protein_name=?
            WHERE id=?
        """, (run_id, config_id, config_name, mol_id, mol_smiles,
              prot_id, prot_name, off_id, off_name, cid))
    conn.commit()
    print(f"Backfilled {len(rows)} existing candidate row(s).")
    conn.close()
    print("Migration complete.")


if __name__ == "__main__":
    main()
