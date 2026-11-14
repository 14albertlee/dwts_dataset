import os

import duckdb

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATABASE_DIR = os.path.join(REPO_ROOT, "data", "database")
MODERN_DB_PATH = os.path.join(DATABASE_DIR, "dwts_dataset.duckdb")
COMPAT_DB_PATH = os.path.join(DATABASE_DIR, "dwts_dataset_compat.duckdb")


def _sql_path(path):
    return os.path.abspath(path).replace("\\", "/").replace("'", "''")


# Connect to a fresh engine instance
con = duckdb.connect()

# Attach your modern DB (Python 2.0+ can read it)
con.execute(f"ATTACH '{_sql_path(MODERN_DB_PATH)}' AS modern_db;")

# Create the retro-compatible v1.5 file
con.execute(f"ATTACH '{_sql_path(COMPAT_DB_PATH)}' AS old_db (STORAGE_VERSION 'v1.5.0');")

# Copy the database over
con.execute("COPY FROM DATABASE modern_db TO old_db;")
print("Success! Open 'dwts_dataset_compat.duckdb' in VS Code.")
