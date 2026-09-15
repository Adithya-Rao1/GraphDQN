import sqlite3

from webapp.backend.config import DATABASE_PATH


def main():
    print(f"Migrating {DATABASE_PATH} ...")
    conn = sqlite3.connect(str(DATABASE_PATH))
    cur = conn.cursor()

    tables = {row[0] for row in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "generation_batches" not in tables:
        print("No generation_batches table yet -- nothing to migrate (it will be created fresh).")
        conn.close()
        return

    existing_cols = {row[1] for row in cur.execute("PRAGMA table_info(generation_batches)")}
    if "trajectories" in existing_cols:
        print("Column already present.")
    else:
        cur.execute("ALTER TABLE generation_batches ADD COLUMN trajectories JSON")
        conn.commit()
        print("Added column: trajectories")

    conn.close()
    print("Migration complete.")


if __name__ == "__main__":
    main()
