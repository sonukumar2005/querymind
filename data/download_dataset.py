"""
data/download_dataset.py — Reproducible dataset acquisition.

The UCI Online Retail dataset (ID 352) is an Excel file.  This script
downloads it once and saves it to data/raw/ so the rest of the pipeline
has a stable, immutable input to work from.

UCI dataset page: https://archive.ics.uci.edu/dataset/352/online+retail

Run from the project root:
    python data/download_dataset.py
"""

from __future__ import annotations

import hashlib
import sys
import zipfile
from pathlib import Path

import requests

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# UCI provides a direct download link for the zip archive
DATASET_ZIP_URL = (
    "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
)
ZIP_PATH = RAW_DIR / "online_retail.zip"
EXCEL_FILENAME = "Online Retail.xlsx"          # name inside the zip
RAW_EXCEL_PATH = RAW_DIR / "Online_Retail_raw.xlsx"

# SHA-256 of the official zip (update if UCI ever republishes the file)
EXPECTED_SHA256 = None   # Set to verified hash after first successful download


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def download_zip(url: str, dest: Path, chunk_size: int = 8192) -> None:
    print(f"Downloading dataset from:\n  {url}")
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                print(f"\r  {pct:.1f}%  ({downloaded:,} / {total:,} bytes)", end="", flush=True)
    print()  # newline after progress


def extract_excel(zip_path: Path, excel_filename: str, dest: Path) -> None:
    print(f"Extracting '{excel_filename}' from zip …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        print(f"  Files in zip: {names}")
        # Find the actual filename (may differ slightly from expected)
        match = next((n for n in names if n.lower().endswith(".xlsx")), None)
        if not match:
            raise FileNotFoundError(
                f"No .xlsx file found inside {zip_path}.\n"
                f"Zip contents: {names}"
            )
        print(f"  Extracting: {match}")
        data = zf.read(match)
    with open(dest, "wb") as f:
        f.write(data)
    print(f"  Saved raw Excel to: {dest}")


def main() -> None:
    # ── Step 1: Download zip ──────────────────────────────────────────────────
    if ZIP_PATH.exists():
        print(f"Zip already exists at {ZIP_PATH} — skipping download.")
    else:
        download_zip(DATASET_ZIP_URL, ZIP_PATH)
        actual_hash = _sha256(ZIP_PATH)
        print(f"  SHA-256: {actual_hash}")
        if EXPECTED_SHA256 and actual_hash != EXPECTED_SHA256:
            print(
                f"WARNING: SHA-256 mismatch!\n"
                f"  Expected: {EXPECTED_SHA256}\n"
                f"  Got:      {actual_hash}\n"
                f"The file may have changed on the UCI server. Verify manually."
            )

    # ── Step 2: Extract Excel ─────────────────────────────────────────────────
    if RAW_EXCEL_PATH.exists():
        print(f"Raw Excel already exists at {RAW_EXCEL_PATH} — skipping extraction.")
    else:
        extract_excel(ZIP_PATH, EXCEL_FILENAME, RAW_EXCEL_PATH)

    print("\n✓ Dataset ready.")
    print(f"  Raw Excel: {RAW_EXCEL_PATH}")
    print("  Next step: run  python data/inspect_dataset.py")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
