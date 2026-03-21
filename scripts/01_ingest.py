"""
01_ingest.py — Data ingestion and validation

Reads public_labels.csv, extracts reports from one or more Avast
archives, validates the SHA-256 join, and logs missingness.

Output:
  - data/processed/metadata.parquet (labels + dates + report paths)
  - artifacts/ingestion_log.json (counts, missing, date range)
"""

import json
import io
import os
import zipfile
from pathlib import Path
from datetime import datetime

import pandas as pd

# ── Configuration ──────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR = RAW_DIR / "reports"

LABELS_FILE = RAW_DIR / "public_labels.csv"
DEFAULT_ARCHIVE_NAMES = [
    "avast_ctu_reduced.zip",
    "public_small_reports.zip",
    "public_reports.zip",
    "full_reports.zip",
    "1.zip",
    "2.zip",
]

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def discover_archives():
    """Find supported archive inputs in a predictable order."""
    archives = []
    seen = set()

    for name in DEFAULT_ARCHIVE_NAMES:
        path = RAW_DIR / name
        if path.exists():
            archives.append(path)
            seen.add(path.resolve())

    for path in sorted(RAW_DIR.glob("*.zip")):
        resolved = path.resolve()
        if resolved not in seen:
            archives.append(path)
            seen.add(resolved)

    valid_archives = []
    for path in archives:
        if not zipfile.is_zipfile(path):
            print(f"  WARNING: Skipping invalid ZIP file: {path.name}")
            continue
        valid_archives.append(path)

    return valid_archives


def extract_reports_from_zip(zf, archive_label):
    """Extract report JSONs from a ZIP file, including nested dataset bundles."""
    valid_reports = set()

    for name in zf.namelist():
        basename = os.path.basename(name)

        if basename.startswith("._"):
            continue

        if basename.endswith(".zip"):
            print(f"    Entering nested archive: {basename}")
            try:
                nested_bytes = io.BytesIO(zf.read(name))
                with zipfile.ZipFile(nested_bytes, "r") as nested_zf:
                    valid_reports.update(
                        extract_reports_from_zip(nested_zf, f"{archive_label}:{basename}")
                    )
            except zipfile.BadZipFile:
                print(f"    WARNING: Skipping invalid nested ZIP: {basename}")
            continue

        if not basename.endswith(".json"):
            continue

        sha = basename.replace(".json", "").lower().strip()
        if len(sha) != 64:
            continue
        if not all(c in "0123456789abcdef" for c in sha):
            continue

        target = REPORTS_DIR / basename.lower()
        if not target.exists():
            data = zf.read(name)
            target.write_bytes(data)
        valid_reports.add(sha)

    return valid_reports


def ensure_labels_available(archive_paths):
    """Extract public_labels.csv from a dataset bundle when needed."""
    if LABELS_FILE.exists():
        return

    for archive_path in archive_paths:
        with zipfile.ZipFile(archive_path, "r") as zf:
            for name in zf.namelist():
                basename = os.path.basename(name)
                if basename.lower() == "public_labels.csv":
                    print(f"  Extracting labels from {archive_path.name}...")
                    LABELS_FILE.write_bytes(zf.read(name))
                    return


def extract_archives(archive_paths):
    """Extract report JSONs from all discovered archive files."""
    valid_reports = set()

    if not archive_paths:
        print("  WARNING: No report archives found in data/raw/.")
        print("  Expected the official Avast archive or split files like 1.zip/2.zip.")
        return valid_reports

    for archive_path in archive_paths:
        print(f"  Extracting {archive_path.name}...")
        with zipfile.ZipFile(archive_path, "r") as zf:
            valid_reports.update(extract_reports_from_zip(zf, archive_path.name))
    return valid_reports


def load_labels():
    """Load and normalise the label file."""
    df = pd.read_csv(LABELS_FILE)
    # Normalise column names
    df.columns = [c.strip().lower() for c in df.columns]
    # Normalise SHA-256 to lowercase
    sha_col = [c for c in df.columns if "sha" in c][0]
    df = df.rename(columns={sha_col: "sha256"})
    df["sha256"] = df["sha256"].str.lower().str.strip()
    # Parse date
    date_col = [c for c in df.columns if "date" in c][0]
    df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    # Normalise family
    fam_col = [c for c in df.columns if "family" in c][0]
    df = df.rename(columns={fam_col: "family"})
    df["family"] = df["family"].str.lower().str.strip()
    return df


def validate_join(labels_df, extracted_shas):
    """Check 1:1 correspondence between labels and reports."""
    label_shas = set(labels_df["sha256"].values)
    matched = label_shas & extracted_shas
    missing_reports = label_shas - extracted_shas
    extra_reports = extracted_shas - label_shas
    return matched, missing_reports, extra_reports


def main():
    print("=" * 60)
    print("MPhil Data Ingestion and Validation")
    print("=" * 60)

    # Step 1: Extract archives
    print("\n[1/4] Extracting archives...")
    archives = discover_archives()
    if archives:
        print("  Archive inputs:")
        for archive in archives:
            print(f"    - {archive.name}")
    ensure_labels_available(archives)
    extracted = extract_archives(archives)
    print(f"  Extracted {len(extracted)} valid report files.")

    # Step 2: Load labels
    print("\n[2/4] Loading labels...")
    labels = load_labels()
    print(f"  Loaded {len(labels)} labelled samples.")
    print(f"  Families: {sorted(labels['family'].unique())}")
    print(f"  Date range: {labels['date'].min()} to {labels['date'].max()}")

    # Step 3: Validate join
    print("\n[3/4] Validating SHA-256 join...")
    matched, missing, extra = validate_join(labels, extracted)
    print(f"  Matched: {len(matched)} / {len(labels)}")
    print(f"  Missing reports: {len(missing)}")
    print(f"  Extra reports (no label): {len(extra)}")

    if len(missing) > 0:
        print(f"  WARNING: {len(missing)} labelled samples have no report!")
        for s in sorted(missing)[:10]:
            print(f"    {s}")

    # Step 4: Save processed metadata
    print("\n[4/4] Saving processed metadata...")
    labels["report_path"] = labels["sha256"].apply(
        lambda s: str(REPORTS_DIR / f"{s}.json")
    )
    labels["has_report"] = labels["sha256"].isin(matched)
    labels.to_parquet(PROCESSED_DIR / "metadata.parquet", index=False)

    # Save ingestion log
    log = {
        "timestamp": datetime.now().isoformat(),
        "archives_used": [p.name for p in archives],
        "n_labels": len(labels),
        "n_extracted_reports": len(extracted),
        "n_matched": len(matched),
        "n_missing_reports": len(missing),
        "n_extra_reports": len(extra),
        "date_min": str(labels["date"].min()),
        "date_max": str(labels["date"].max()),
        "families": sorted(labels["family"].unique().tolist()),
        "family_counts": labels["family"].value_counts().to_dict(),
    }
    with open(ARTIFACTS_DIR / "ingestion_log.json", "w") as f:
        json.dump(log, f, indent=2)

    print(f"\n  Metadata saved to: {PROCESSED_DIR / 'metadata.parquet'}")
    print(f"  Ingestion log saved to: {ARTIFACTS_DIR / 'ingestion_log.json'}")
    print("\nIngestion complete.")

    if len(matched) == len(labels) and len(missing) == 0:
        print("✓ PASS: Perfect 1:1 join (48,976/48,976).")
    else:
        print("✗ WARN: Join is incomplete. Check missing reports.")


if __name__ == "__main__":
    main()
