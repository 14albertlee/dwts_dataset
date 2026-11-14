-- COPY (
    SELECT 
        database_name, 
        schema_name, 
        table_name, 
        column_name, 
        data_type,
        is_nullable
    FROM duckdb_columns()
    WHERE NOT internal
    ORDER BY database_name, schema_name, table_name, column_index
-- ) TO 'output.csv' (FORMAT CSV);
