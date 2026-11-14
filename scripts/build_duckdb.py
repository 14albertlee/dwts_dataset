"""Build a queryable DuckDB database from the csv files in data/raw/.

Run from the repository root (after regenerating the csvs with
`scripts/parse_scores_json_to_tables.py`):

    python scripts/build_duckdb.py

This (re)creates `data/database/dwts_dataset.duckdb`, with one table per
csv, primary keys on every `<table>_id` column, foreign keys wired up
between the fact tables (`couples`, `performances`, `scores`) and their
dimension tables, and unique constraints on each dimension table's natural
key (e.g. `celebrities.celebrity`, `couples.(celebrity_id, professional_id)`),
so loading it also double-checks that the csvs have no duplicate dimension
rows, and you can query the dataset directly with SQL, e.g.:

    duckdb data/database/dwts_dataset.duckdb -c "
        SELECT c.celebrity, p.professional, avg(s.judge_score) AS avg_score
        FROM scores s
        JOIN performances perf ON perf.performance_id = s.performance_id
        JOIN couples co ON co.couple_id = perf.couple_id
        JOIN celebrities c ON c.celebrity_id = co.celebrity_id
        JOIN professionals p ON p.professional_id = co.professional_id
        GROUP BY 1, 2
        ORDER BY avg_score DESC
        LIMIT 10;
    "
"""

import os

import duckdb

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(REPO_ROOT, "data", "raw")
DB_PATH = os.path.join(REPO_ROOT, "data", "database", "dwts_dataset.duckdb")

# Each entry is (table_name, create_table_sql). Order matters: a table can
# only reference primary keys of tables created earlier in this list.
TABLES = [
    (
        "celebrities",
        """
        CREATE TABLE celebrities (
            celebrity_id INTEGER PRIMARY KEY,
            celebrity VARCHAR UNIQUE,
            notability VARCHAR
        )
        """,
    ),
    (
        "professionals",
        """
        CREATE TABLE professionals (
            professional_id INTEGER PRIMARY KEY,
            professional VARCHAR UNIQUE
        )
        """,
    ),
    (
        "judges",
        """
        CREATE TABLE judges (
            judge_id INTEGER PRIMARY KEY,
            judge VARCHAR UNIQUE
        )
        """,
    ),
    (
        "primary_dance_styles",
        """
        CREATE TABLE primary_dance_styles (
            primary_dance_style_id INTEGER PRIMARY KEY,
            primary_dance_style VARCHAR UNIQUE
        )
        """,
    ),
    (
        "primary_song_artists",
        """
        CREATE TABLE primary_song_artists (
            primary_song_artist_id INTEGER PRIMARY KEY,
            primary_song_artist VARCHAR UNIQUE
        )
        """,
    ),
    (
        "primary_songs",
        """
        CREATE TABLE primary_songs (
            primary_song_id INTEGER PRIMARY KEY,
            primary_song VARCHAR,
            primary_song_artist_id INTEGER REFERENCES primary_song_artists,
            UNIQUE (primary_song, primary_song_artist_id)
        )
        """,
    ),
    (
        "week_themes",
        """
        CREATE TABLE week_themes (
            week_theme_id INTEGER PRIMARY KEY,
            week_theme VARCHAR UNIQUE
        )
        """,
    ),
    (
        "couples",
        """
        CREATE TABLE couples (
            couple_id INTEGER PRIMARY KEY,
            couple VARCHAR,
            celebrity_id INTEGER REFERENCES celebrities,
            professional_id INTEGER REFERENCES professionals,
            UNIQUE (celebrity_id, professional_id)
        )
        """,
    ),
    (
        "performances",
        """
        CREATE TABLE performances (
            performance_id INTEGER PRIMARY KEY,
            season INTEGER,
            week INTEGER,
            week_theme_id INTEGER REFERENCES week_themes,
            couple_id INTEGER REFERENCES couples,
            primary_dance_style_id INTEGER REFERENCES primary_dance_styles,
            primary_song_id INTEGER REFERENCES primary_songs,
            eliminated BOOLEAN
        )
        """,
    ),
    (
        "scores",
        """
        CREATE TABLE scores (
            score_id INTEGER PRIMARY KEY,
            performance_id INTEGER REFERENCES performances,
            judge_id INTEGER REFERENCES judges,
            judge_score DOUBLE
        )
        """,
    ),
]


def main():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    con = duckdb.connect(DB_PATH)
    for table_name, create_sql in TABLES:
        csv_path = os.path.join(DATASET_DIR, f"{table_name}.csv")
        con.execute(create_sql)
        con.execute(
            f"INSERT INTO {table_name} SELECT * FROM read_csv_auto(?)",
            [csv_path],
        )
        count = con.execute(f"SELECT count(*) FROM {table_name}").fetchone()[0]
        print(f"loaded {table_name}: {count} rows")
    con.close()
    print(f"wrote {DB_PATH}")


if __name__ == "__main__":
    main()
