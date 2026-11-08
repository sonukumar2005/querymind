"""
data/clean_dataset.py — Clean and transform the UCI Online Retail dataset.

Cleaning decisions are documented inline.  The cleaned Parquet file is
the single source of truth for the database loader.

Run from the project root:
    python data/clean_dataset.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
RAW_EXCEL = PROJECT_ROOT / "data" / "raw" / "Online_Retail_raw.xlsx"
CLEAN_PARQUET = PROJECT_ROOT / "data" / "processed" / "online_retail_clean.parquet"
CLEAN_CSV = PROJECT_ROOT / "data" / "processed" / "online_retail_clean.csv"


# ─────────────────────────────────────────────────────────────────────────────
# Business Definitions (used consistently throughout the project)
# ─────────────────────────────────────────────────────────────────────────────
# revenue       = Quantity * UnitPrice  (only for non-cancelled rows, Qty > 0)
# cancellation  = InvoiceNo starts with 'C'  (Quantity is negative on these)
# return_amount = abs(Quantity) * UnitPrice  (for cancellation rows)
# net_quantity  = sum(Quantity) after cancellations are signed negative
# ─────────────────────────────────────────────────────────────────────────────


def load_raw(path: Path) -> pd.DataFrame:
    print(f"Loading raw Excel: {path} …")
    df = pd.read_excel(path, engine="openpyxl", dtype={"CustomerID": str})
    print(f"  Raw shape: {df.shape}")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    original_count = len(df)
    log: list[str] = []

    # ── 1. Normalise column names ─────────────────────────────────────────────
    # Strip whitespace from column names to guard against invisible characters.
    df.columns = df.columns.str.strip()

    # ── 2. Parse InvoiceDate as datetime ─────────────────────────────────────
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    n_bad_dates = df["InvoiceDate"].isnull().sum()
    if n_bad_dates:
        log.append(f"Dropped {n_bad_dates} rows with unparseable InvoiceDate.")
        df = df[df["InvoiceDate"].notna()]

    # ── 3. Flag cancellations BEFORE filtering ────────────────────────────────
    # Cancellations have InvoiceNo starting with 'C'.  We keep them because
    # they are analytically important (returns, net revenue).
    df["IsCancellation"] = df["InvoiceNo"].astype(str).str.startswith("C")

    # ── 4. Drop rows with missing Description ────────────────────────────────
    n_missing_desc = df["Description"].isnull().sum()
    df = df[df["Description"].notna()]
    log.append(f"Dropped {n_missing_desc} rows with missing Description.")

    # ── 5. Strip whitespace from string columns ───────────────────────────────
    for col in ["InvoiceNo", "StockCode", "Description", "Country"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # ── 6. Drop rows where UnitPrice <= 0 on NON-cancellation rows ───────────
    # A zero or negative price on a regular sale is bad data (no value).
    # Cancellation rows carry negative Quantity; their UnitPrice should still
    # be > 0 (it represents the original sale price being reversed).
    bad_price_mask = (~df["IsCancellation"]) & (df["UnitPrice"] <= 0)
    n_bad_price = bad_price_mask.sum()
    df = df[~bad_price_mask]
    log.append(f"Dropped {n_bad_price} non-cancellation rows with UnitPrice <= 0.")

    # ── 7. Drop rows with Quantity == 0 ──────────────────────────────────────
    # Zero-quantity rows carry no analytical meaning.
    n_zero_qty = (df["Quantity"] == 0).sum()
    df = df[df["Quantity"] != 0]
    log.append(f"Dropped {n_zero_qty} rows with Quantity == 0.")

    # ── 8. CustomerID: keep NaN as explicit NULL ──────────────────────────────
    # ~25% of rows have no CustomerID (guest checkouts).  We do NOT invent IDs.
    # They remain NULL in the database.  Customer-level analysis is scoped to
    # rows where CustomerID is not null.
    n_missing_cid = df["CustomerID"].isnull().sum()
    log.append(
        f"Retained {n_missing_cid} rows with missing CustomerID as NULL "
        f"(guest checkouts — not fabricated)."
    )

    # ── 9. Compute Revenue and ReturnAmount ───────────────────────────────────
    # Revenue applies only to sales rows (Quantity > 0).
    # ReturnAmount applies only to cancellation rows (Quantity < 0).
    df["LineRevenue"] = df.apply(
        lambda r: r["Quantity"] * r["UnitPrice"] if not r["IsCancellation"] else 0.0,
        axis=1,
    )
    df["LineReturnAmount"] = df.apply(
        lambda r: abs(r["Quantity"]) * r["UnitPrice"] if r["IsCancellation"] else 0.0,
        axis=1,
    )

    # ── 10. Add derived date fields ───────────────────────────────────────────
    df["InvoiceYear"] = df["InvoiceDate"].dt.year
    df["InvoiceMonth"] = df["InvoiceDate"].dt.month
    df["InvoiceYearMonth"] = df["InvoiceDate"].dt.to_period("M").astype(str)

    # ── 11. Remove duplicates ─────────────────────────────────────────────────
    n_dup = df.duplicated().sum()
    df = df.drop_duplicates()
    log.append(f"Dropped {n_dup} exact duplicate rows.")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n══ Cleaning log ═══════════════════════════════════════════════════")
    for entry in log:
        print(f"  • {entry}")
    removed = original_count - len(df)
    print(f"\n  Original rows : {original_count:,}")
    print(f"  Rows removed  : {removed:,}  ({removed/original_count*100:.1f}%)")
    print(f"  Clean rows    : {len(df):,}")
    print(f"  Cancellations : {df['IsCancellation'].sum():,}")
    print(f"  Sales rows    : {(~df['IsCancellation']).sum():,}")
    print(f"  Total revenue : £{df['LineRevenue'].sum():,.2f}")
    print(f"  Total returns : £{df['LineReturnAmount'].sum():,.2f}")

    return df.reset_index(drop=True)


def save(df: pd.DataFrame) -> None:
    CLEAN_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CLEAN_PARQUET, index=False)
    df.to_csv(CLEAN_CSV, index=False)
    print(f"\nSaved Parquet : {CLEAN_PARQUET}")
    print(f"Saved CSV     : {CLEAN_CSV}")


def main() -> None:
    if not RAW_EXCEL.exists():
        print(f"ERROR: {RAW_EXCEL} not found. Run:  python data/download_dataset.py")
        sys.exit(1)
    df = load_raw(RAW_EXCEL)
    df = clean(df)
    save(df)
    print("\n✓ Cleaning complete. Next step: python database/load_data.py")


if __name__ == "__main__":
    main()
