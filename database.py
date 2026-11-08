"""
database.py — SQLite database management for QueryMind AI.

Responsibilities:
- Initialize and connect to the SQLite database via SQLAlchemy.
- Upload CSV / Excel files as tables.
- Inspect schema (tables, columns, types, keys).
- Execute queries safely with timeout and row limits.
- Duplicate detection and removal.
"""

import os
import re
import sqlite3
from pathlib import Path
from contextlib import contextmanager

import pandas as pd
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────
DB_PATH = Path(os.getenv("DB_PATH", "data/database.db"))
MAX_ROWS = int(os.getenv("MAX_RESULT_ROWS", 1000))
TIMEOUT  = int(os.getenv("QUERY_TIMEOUT_SECONDS", 30))

# Blocked DDL keywords (never allowed)
BLOCKED_KEYWORDS = {"drop", "alter", "truncate", "create", "rename", "replace"}


# ── Engine factory ─────────────────────────────────────────────────────────────

def get_engine():
    """Return (and cache) a SQLAlchemy engine for the SQLite database."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(
        f"sqlite:///{DB_PATH}",
        connect_args={"timeout": TIMEOUT, "check_same_thread": False},
    )


def init_db():
    """Ensure the database file and directory exist."""
    engine = get_engine()
    # Touch the database to create the file
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    return engine


# ── Upload helpers ─────────────────────────────────────────────────────────────

def _safe_table_name(name: str) -> str:
    """Convert a file name to a safe SQLite table name."""
    stem = Path(name).stem
    safe = re.sub(r"[^a-zA-Z0-9_]", "_", stem).strip("_").lower()
    return safe or "uploaded_table"


def upload_dataframe(df: pd.DataFrame, table_name: str, if_exists: str = "replace") -> str:
    """
    Write a DataFrame to SQLite.

    Returns the actual table name used.
    """
    engine = get_engine()
    table_name = _safe_table_name(table_name)
    df.to_sql(table_name, engine, if_exists=if_exists, index=False)
    return table_name


def upload_csv(file, table_name: str = None) -> tuple[str, pd.DataFrame]:
    """Read a CSV file object and store it as a SQLite table."""
    df = pd.read_csv(file)
    tname = _safe_table_name(table_name or getattr(file, "name", "table"))
    actual = upload_dataframe(df, tname)
    return actual, df


def upload_excel(file, table_name: str = None) -> tuple[str, pd.DataFrame]:
    """Read an Excel file object and store it as a SQLite table."""
    df = pd.read_excel(file, engine="openpyxl")
    tname = _safe_table_name(table_name or getattr(file, "name", "table"))
    actual = upload_dataframe(df, tname)
    return actual, df


# ── Schema inspection ──────────────────────────────────────────────────────────

def get_tables() -> list[str]:
    """Return all user table names in the database."""
    engine = get_engine()
    insp = inspect(engine)
    return insp.get_table_names()


def get_schema(table_name: str) -> dict:
    """
    Return schema info for one table:
    {columns: [{name, type, nullable, primary_key}], foreign_keys: [...]}
    """
    engine = get_engine()
    insp = inspect(engine)
    columns = [
        {
            "name": col["name"],
            "type": str(col["type"]),
            "nullable": col.get("nullable", True),
            "primary_key": col.get("primary_key", False),
        }
        for col in insp.get_columns(table_name)
    ]
    try:
        fks = insp.get_foreign_keys(table_name)
    except Exception:
        fks = []
    return {"table": table_name, "columns": columns, "foreign_keys": fks}


def get_full_schema() -> dict[str, dict]:
    """Return schema for all tables as {table_name: schema_dict}."""
    return {t: get_schema(t) for t in get_tables()}


def schema_as_text(schema_dict: dict) -> str:
    """Format a schema dict as a readable SQL-like description string."""
    lines = [f"Table: {schema_dict['table']}"]
    for col in schema_dict["columns"]:
        pk = " PRIMARY KEY" if col["primary_key"] else ""
        nn = " NOT NULL" if not col["nullable"] else ""
        lines.append(f"  {col['name']} {col['type']}{pk}{nn}")
    for fk in schema_dict.get("foreign_keys", []):
        lines.append(
            f"  FOREIGN KEY ({', '.join(fk['constrained_columns'])}) "
            f"REFERENCES {fk['referred_table']} ({', '.join(fk['referred_columns'])})"
        )
    return "\n".join(lines)


def get_full_schema_text() -> str:
    """All tables schema concatenated as text (for LLM context)."""
    schemas = get_full_schema()
    if not schemas:
        return "No tables found in the database."
    return "\n\n".join(schema_as_text(s) for s in schemas.values())


# ── Query execution ────────────────────────────────────────────────────────────

def _has_blocked_keyword(sql: str) -> bool:
    tokens = re.findall(r"\b\w+\b", sql.lower())
    return any(t in BLOCKED_KEYWORDS for t in tokens)


def execute_query(sql: str, params: dict = None) -> pd.DataFrame:
    """
    Execute a SQL query and return results as a DataFrame.

    Raises:
        ValueError: for blocked or empty queries.
        SQLAlchemyError: for database errors.
    """
    sql = sql.strip()
    if not sql:
        raise ValueError("Empty query.")
    if _has_blocked_keyword(sql):
        raise ValueError(
            "Blocked query: DROP, ALTER, TRUNCATE, CREATE and RENAME are not allowed."
        )

    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(text(sql), params or {})
        if result.returns_rows:
            rows = result.fetchmany(MAX_ROWS)
            df = pd.DataFrame(rows, columns=list(result.keys()))
        else:
            conn.commit()
            df = pd.DataFrame([{"affected_rows": result.rowcount}])
    return df


def execute_select(sql: str) -> pd.DataFrame:
    """Read-only wrapper — rejects anything that is not a SELECT."""
    if not sql.strip().lower().startswith("select"):
        raise ValueError("Only SELECT queries are allowed in read-only mode.")
    return execute_query(sql)


# ── Duplicate handling ─────────────────────────────────────────────────────────

def detect_duplicates(table_name: str) -> pd.DataFrame:
    """Return rows that appear more than once (all columns match)."""
    df = execute_query(f'SELECT * FROM "{table_name}"')
    return df[df.duplicated(keep=False)]


def remove_duplicates(table_name: str) -> int:
    """
    Remove duplicate rows, keeping the first occurrence.
    Returns number of rows removed.
    """
    df = execute_query(f'SELECT * FROM "{table_name}"')
    before = len(df)
    deduped = df.drop_duplicates()
    removed = before - len(deduped)
    if removed > 0:
        upload_dataframe(deduped, table_name, if_exists="replace")
    return removed


# ── Row count helper ───────────────────────────────────────────────────────────

def get_row_count(table_name: str) -> int:
    """Return number of rows in a table."""
    df = execute_query(f'SELECT COUNT(*) as cnt FROM "{table_name}"')
    return int(df.iloc[0]["cnt"])


def get_preview(table_name: str, n: int = 5) -> pd.DataFrame:
    """Return first n rows of a table."""
    return execute_query(f'SELECT * FROM "{table_name}" LIMIT {n}')
