import sqlite3

from webapp.backend.config import DATABASE_PATH

NEW_COLUMNS = {
    "generated_by_pareto_sweep_id": "INTEGER",
}


def main():
    print(f"Migrating {DATABASE_PATH} ...")
    conn = sqlite3.connect(str(DATABASE_PATH))
    cur = conn.cursor()

    tables = {row[0] for row in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "optimization_configs" not in tables:
        print("No optimization_configs table yet -- nothing to migrate (it will be created fresh).")
        conn.close()
        return

    existing_cols = {row[1] for row in cur.execute("PRAGMA table_info(optimization_configs)")}
    added = []
    for col, coltype in NEW_COLUMNS.items():
        if col not in existing_cols:
            cur.execute(f"ALTER TABLE optimization_configs ADD COLUMN {col} {coltype}")
            added.append(col)
    conn.commit()
    print(f"Added columns: {added}" if added else "All columns already present.")
    
    conn.close()
    print("Migration complete.")


if __name__ == "__main__":
    main()
