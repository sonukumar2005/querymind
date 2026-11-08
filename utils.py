"""
utils.py — Shared utilities for QueryMind AI.

Responsibilities:
- SQL validation via SQLGlot.
- Query classification (SELECT / UPDATE / DELETE / DESTRUCTIVE).
- Confirmation requirement check.
- Plotly chart suggestion and rendering.
- Summary statistics.
- DataFrame export helpers.
"""

import io
import re
from typing import Literal

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sqlglot
from sqlglot.errors import ParseError


# ── SQL Validation ─────────────────────────────────────────────────────────────

DESTRUCTIVE_KEYWORDS = {"drop", "alter", "truncate", "create", "rename", "replace"}
CONFIRMATION_KEYWORDS = {"update", "delete", "insert"}


def validate_sql(sql: str) -> tuple[bool, str]:
    """
    Parse SQL with SQLGlot.

    Returns (is_valid: bool, error_message: str).
    """
    sql = sql.strip()
    if not sql:
        return False, "Empty query."
    try:
        statements = sqlglot.parse(sql, dialect="sqlite")
        if not statements or statements[0] is None:
            return False, "Could not parse the SQL statement."
        return True, ""
    except ParseError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Unexpected parse error: {e}"


def classify_query(sql: str) -> Literal["SELECT", "INSERT", "UPDATE", "DELETE", "DESTRUCTIVE", "UNKNOWN"]:
    """Return the top-level operation type of a SQL statement."""
    first_token = sql.strip().split()[0].lower() if sql.strip() else ""
    if first_token in DESTRUCTIVE_KEYWORDS:
        return "DESTRUCTIVE"
    mapping = {
        "select": "SELECT",
        "insert": "INSERT",
        "update": "UPDATE",
        "delete": "DELETE",
    }
    return mapping.get(first_token, "UNKNOWN")


def is_destructive(sql: str) -> bool:
    """True if the query contains any schema-altering keywords."""
    tokens = set(re.findall(r"\b\w+\b", sql.lower()))
    return bool(tokens & DESTRUCTIVE_KEYWORDS)


def requires_confirmation(sql: str) -> bool:
    """True for UPDATE, DELETE, INSERT — requires explicit user confirmation."""
    first = sql.strip().split()[0].lower() if sql.strip() else ""
    return first in CONFIRMATION_KEYWORDS


def clean_sql(raw: str) -> str:
    """
    Strip markdown code fences and extra whitespace from LLM output.
    LLMs often return ```sql ... ``` blocks.
    """
    # Remove ```sql ... ``` fences
    raw = re.sub(r"```(?:sql)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"```", "", raw)
    return raw.strip()


# ── Chart suggestion ───────────────────────────────────────────────────────────

def suggest_chart(df: pd.DataFrame) -> str | None:
    """
    Heuristic chart type suggestion based on DataFrame shape.

    Returns: 'bar' | 'line' | 'pie' | 'scatter' | None
    """
    if df is None or df.empty or len(df.columns) < 2:
        return None

    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()

    # Detect datetime columns
    dt_cols = []
    for col in cat_cols:
        try:
            parsed = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
            if parsed.notna().mean() > 0.7:
                dt_cols.append(col)
        except Exception:
            pass

    if dt_cols and num_cols:
        return "line"
    if cat_cols and num_cols:
        unique_ratio = df[cat_cols[0]].nunique() / max(len(df), 1)
        if unique_ratio < 0.15:  # few unique categories → pie
            return "pie"
        return "bar"
    if len(num_cols) >= 2:
        return "scatter"
    return None


def render_chart(df: pd.DataFrame, chart_type: str, x_col: str = None, y_col: str = None) -> go.Figure | None:
    """
    Build and return a Plotly figure.

    Auto-detects x/y columns if not provided.
    """
    if df is None or df.empty:
        return None

    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()

    # Auto-detect columns
    if x_col is None:
        x_col = cat_cols[0] if cat_cols else (num_cols[0] if num_cols else df.columns[0])
    if y_col is None:
        y_col = num_cols[0] if num_cols else (df.columns[1] if len(df.columns) > 1 else df.columns[0])

    # Dark theme template
    template = "plotly_dark"
    color_seq = px.colors.sequential.Plasma

    try:
        if chart_type == "bar":
            fig = px.bar(df, x=x_col, y=y_col, template=template, color=y_col,
                         color_continuous_scale=color_seq)
        elif chart_type == "line":
            fig = px.line(df, x=x_col, y=y_col, template=template,
                          markers=True, line_shape="spline")
        elif chart_type == "pie":
            fig = px.pie(df, names=x_col, values=y_col, template=template,
                         color_discrete_sequence=px.colors.sequential.Plasma_r)
        elif chart_type == "scatter":
            fig = px.scatter(df, x=x_col, y=y_col, template=template,
                             color=y_col, color_continuous_scale=color_seq)
        else:
            return None

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e0e0e0"),
            margin=dict(l=20, r=20, t=40, b=20),
        )
        return fig
    except Exception:
        return None


# ── Summary statistics ─────────────────────────────────────────────────────────

def compute_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a summary statistics DataFrame with extended info:
    type, non-null, null_count, unique, mean/min/max (numeric).
    """
    if df is None or df.empty:
        return pd.DataFrame()

    rows = []
    for col in df.columns:
        series = df[col]
        row = {
            "Column": col,
            "Type": str(series.dtype),
            "Non-Null": int(series.notna().sum()),
            "Null Count": int(series.isna().sum()),
            "Unique Values": int(series.nunique()),
        }
        if pd.api.types.is_numeric_dtype(series):
            row["Mean"] = round(series.mean(), 4) if series.notna().any() else None
            row["Min"] = series.min()
            row["Max"] = series.max()
        else:
            row["Mean"] = None
            row["Min"] = None
            row["Max"] = None
        rows.append(row)
    return pd.DataFrame(rows)


# ── Export helpers ─────────────────────────────────────────────────────────────

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Return DataFrame as UTF-8 CSV bytes."""
    return df.to_csv(index=False).encode("utf-8")


def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    """Return DataFrame as Excel bytes (in-memory)."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Results")
    return buf.getvalue()
