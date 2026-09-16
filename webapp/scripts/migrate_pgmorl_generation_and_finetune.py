import re
import sqlite3

from webapp.backend.config import DATABASE_PATH

NEW_GENERATION_BATCH_COLUMNS = {
    "pareto_sweep_id": "INTEGER",
    "population_member_config_id": "INTEGER",
}
NEW_SWEEP_COLUMNS = {
    "llm_finetune_interval_steps": "INTEGER",
    "cumulative_steps_since_finetune": "INTEGER NOT NULL DEFAULT 0",
}
NEW_EDIT_OUTCOME_LOG_COLUMNS = {
    "pareto_sweep_id": "INTEGER",
    "population_member_config_id": "INTEGER",
}


def _add_columns(conn, table, columns):
    cur = conn.cursor()
    existing = {row[1] for row in cur.execute(f"PRAGMA table_info({table})")}
    added = []
    for col, coltype in columns.items():
        if col not in existing:
            cur.execute(f"ALTER TABLE {table} ADD COLUMN {col} {coltype}")
            added.append(col)
    return added


def _column_is_not_null(conn, table, column) -> bool:
    for row in conn.execute(f"PRAGMA table_info({table})"):
        if row[1] == column:
            return bool(row[3])
    raise ValueError(f"column {column!r} not found on table {table!r}")


def _relax_not_null(conn, table, column) -> bool:
    if not _column_is_not_null(conn, table, column):
        return False

    cur = conn.cursor()
    row = cur.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    if row is None or row[0] is None:
        raise RuntimeError(f"could not read CREATE TABLE SQL for {table!r}")
    create_sql = row[0]

    tmp_table = f"{table}__migrate_tmp"
    new_sql = re.sub(rf'CREATE TABLE ["\']?{table}["\']?', f'CREATE TABLE "{tmp_table}"', create_sql, count=1)

    col_pattern = re.compile(rf'((?:"{column}"|{column})\s+\w+(?:\(\d+\))?)\s+NOT NULL', re.IGNORECASE)
    new_sql, n = col_pattern.subn(r'\1', new_sql, count=1)
    if n != 1:
        raise RuntimeError(
            f"could not locate a NOT NULL constraint for {table}.{column} in its CREATE TABLE SQL "
            f"-- migration script's regex needs updating for this schema. SQL was:\n{create_sql}"
        )

    indexes = cur.execute(
        "SELECT name, sql FROM sqlite_master WHERE type='index' AND tbl_name=? AND sql IS NOT NULL", (table,)
    ).fetchall()

    cur.execute(new_sql)
    columns = [r[1] for r in cur.execute(f"PRAGMA table_info({table})")]
    col_list = ", ".join(columns)
    cur.execute(f"INSERT INTO {tmp_table} ({col_list}) SELECT {col_list} FROM {table}")
    cur.execute(f"DROP TABLE {table}")
    cur.execute(f"ALTER TABLE {tmp_table} RENAME TO {table}")

    for index_name, index_sql in indexes:
        cur.execute(f"DROP INDEX IF EXISTS {index_name}")
        cur.execute(index_sql)

    return True


def main():
    print(f"Migrating {DATABASE_PATH} ...")
    conn = sqlite3.connect(str(DATABASE_PATH))
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}

    if "generation_batches" in tables:
        added = _add_columns(conn, "generation_batches", NEW_GENERATION_BATCH_COLUMNS)
        conn.commit()
        print(f"generation_batches: added columns {added}" if added
              else "generation_batches: new columns already present.")
        relaxed = _relax_not_null(conn, "generation_batches", "training_run_id")
        conn.commit()
        print("generation_batches.training_run_id: relaxed to nullable." if relaxed
              else "generation_batches.training_run_id: already nullable.")
    else:
        print("No generation_batches table yet -- nothing to migrate (it will be created fresh).")

    if "generated_candidates" in tables:
        relaxed = _relax_not_null(conn, "generated_candidates", "training_run_id")
        conn.commit()
        print("generated_candidates.training_run_id: relaxed to nullable." if relaxed
              else "generated_candidates.training_run_id: already nullable.")
    else:
        print("No generated_candidates table yet -- nothing to migrate (it will be created fresh).")

    if "pareto_sweep_runs" in tables:
        added = _add_columns(conn, "pareto_sweep_runs", NEW_SWEEP_COLUMNS)
        conn.commit()
        print(f"pareto_sweep_runs: added columns {added}" if added
              else "pareto_sweep_runs: new columns already present.")
    else:
        print("No pareto_sweep_runs table yet -- nothing to migrate (it will be created fresh).")

    if "edit_outcome_logs" in tables:
        added = _add_columns(conn, "edit_outcome_logs", NEW_EDIT_OUTCOME_LOG_COLUMNS)
        conn.commit()
        print(f"edit_outcome_logs: added columns {added}" if added
              else "edit_outcome_logs: new columns already present.")
        relaxed = _relax_not_null(conn, "edit_outcome_logs", "training_run_id")
        conn.commit()
        print("edit_outcome_logs.training_run_id: relaxed to nullable." if relaxed
              else "edit_outcome_logs.training_run_id: already nullable.")
    else:
        print("No edit_outcome_logs table yet -- nothing to migrate (it will be created fresh).")

    conn.close()
    print("Migration complete.")


if __name__ == "__main__":
    main()
