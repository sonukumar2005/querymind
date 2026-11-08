"""
data/inspect_dataset.py — Explore the raw UCI Online Retail dataset.

Prints a concise data-quality report so every cleaning decision in
clean_dataset.py is made with evidence, not guesswork.

Run from the project root:
    python data/inspect_dataset.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
RAW_EXCEL = PROJECT_ROOT / "data" / "raw" / "Online_Retail_raw.xlsx"


def inspect(path: Path) -> None:
    print(f"Loading: {path}")
    df = pd.read_excel(path, engine="openpyxl")

    print("\n═══ 1. SHAPE ════════════════════════════════════════════════════")
    print(f"  Rows: {len(df):,}   Columns: {df.shape[1]}")

    print("\n═══ 2. COLUMN NAMES & DTYPES ════════════════════════════════════")
    print(df.dtypes.to_string())

    print("\n═══ 3. FIRST 3 ROWS ═════════════════════════════════════════════")
    print(df.head(3).to_string())

    print("\n═══ 4. MISSING VALUES ═══════════════════════════════════════════")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    print(pd.DataFrame({"missing": missing, "pct": missing_pct}).to_string())

    print("\n═══ 5. BASIC STATISTICS (numeric cols) ══════════════════════════")
    print(df.describe().to_string())

    print("\n═══ 6. CANCELLATIONS (InvoiceNo starts with 'C') ════════════════")
    cancellations = df[df["InvoiceNo"].astype(str).str.startswith("C")]
    print(f"  Cancellation rows: {len(cancellations):,}")

    print("\n═══ 7. NEGATIVE QUANTITY ════════════════════════════════════════")
    neg_qty = df[df["Quantity"] < 0]
    print(f"  Rows with Quantity < 0: {len(neg_qty):,}")

    print("\n═══ 8. ZERO / NEGATIVE UNIT PRICE ══════════════════════════════")
    zero_price = df[df["UnitPrice"] <= 0]
    print(f"  Rows with UnitPrice <= 0: {len(zero_price):,}")

    print("\n═══ 9. MISSING CustomerID ═══════════════════════════════════════")
    missing_cid = df["CustomerID"].isnull().sum()
    print(f"  Rows missing CustomerID: {missing_cid:,}  ({missing_cid/len(df)*100:.1f}%)")

    print("\n═══ 10. DATE RANGE ══════════════════════════════════════════════")
    print(f"  Min InvoiceDate: {df['InvoiceDate'].min()}")
    print(f"  Max InvoiceDate: {df['InvoiceDate'].max()}")

    print("\n═══ 11. UNIQUE COUNTS ═══════════════════════════════════════════")
    for col in df.columns:
        print(f"  {col}: {df[col].nunique():,} unique values")

    print("\n═══ 12. TOP 5 COUNTRIES ═════════════════════════════════════════")
    print(df["Country"].value_counts().head(5).to_string())

    print("\n═══ 13. TOP 5 STOCK CODES ═══════════════════════════════════════")
    print(df["StockCode"].value_counts().head(5).to_string())

    print("\n✓ Inspection complete.")


if __name__ == "__main__":
    if not RAW_EXCEL.exists():
        print(f"ERROR: Raw Excel not found at {RAW_EXCEL}")
        print("Run first:  python data/download_dataset.py")
        sys.exit(1)
    inspect(RAW_EXCEL)
