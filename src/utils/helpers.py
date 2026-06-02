import os
import sqlite3
import pandas as pd
from src.utils.logger import get_logger
from src.utils.config import DATA_PATH, DB_PATH

logger = get_logger("helpers")


def build_sqlite_db(force_rebuild: bool = False):
    """
    Load CSVs into a SQLite database.
    Only runs if DB doesn't exist or force_rebuild=True.
    """
    if os.path.exists(DB_PATH) and not force_rebuild:
        logger.info(f"DB already exists at {DB_PATH} — skipping rebuild")
        return

    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    logger.info(f"Building SQLite DB at {DB_PATH}...")

    conn = sqlite3.connect(DB_PATH)

    tables = {
        "application_train": "application_train.csv",
        "bureau":            "bureau.csv",
        "previous_application": "previous_application.csv",
    }

    for table_name, filename in tables.items():
        path = os.path.join(DATA_PATH, filename)
        if os.path.exists(path):
            logger.info(f"Loading {filename} → table '{table_name}'...")
            # Load in chunks to avoid memory issues
            chunks = pd.read_csv(path, chunksize=50000)
            for i, chunk in enumerate(chunks):
                chunk.to_sql(
                    table_name, conn,
                    if_exists='replace' if i == 0 else 'append',
                    index=False
                )
            logger.info(f"  ✓ {table_name} loaded")
        else:
            logger.warning(f"  ✗ {filename} not found — skipping")

    conn.close()
    logger.info("SQLite DB build complete")


def get_db_schema() -> str:
    """Return schema string for all tables — injected into LLM prompt."""
    if not os.path.exists(DB_PATH):
        build_sqlite_db()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    schema_parts = []
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]

    for table in tables:
        cursor.execute(f"PRAGMA table_info({table})")
        columns = cursor.fetchall()
        col_str = ", ".join([f"{col[1]} ({col[2]})" for col in columns[:30]])
        schema_parts.append(f"Table: {table}\nColumns: {col_str}")

    conn.close()
    return "\n\n".join(schema_parts)


def run_sql_query(sql: str) -> tuple:
    """
    Execute SQL query safely. Returns (dataframe, error_message).
    """
    if not os.path.exists(DB_PATH):
        build_sqlite_db()

    # Basic safety check — only allow SELECT
    sql_clean = sql.strip().upper()
    if not sql_clean.startswith("SELECT"):
        return None, "Only SELECT queries are allowed."

    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(sql, conn)
        conn.close()
        return df, None
    except Exception as e:
        return None, str(e)