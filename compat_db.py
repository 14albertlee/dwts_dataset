import duckdb

# Connect to a fresh engine instance
con = duckdb.connect()

# Attach your modern DB (Python 2.0+ can read it)
con.execute("ATTACH 'dwts_dataset.duckdb' AS modern_db;")

# Create the retro-compatible v1.5 file
con.execute("ATTACH 'dwts_dataset_compat.duckdb' AS old_db (STORAGE_VERSION 'v1.5.0');")

# Copy the database over
con.execute("COPY FROM DATABASE modern_db TO old_db;")
print("Success! Open 'dwts_dataset_compat.duckdb' in VS Code.")
