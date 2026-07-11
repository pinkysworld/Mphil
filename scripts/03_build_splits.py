"""
03_build_splits.py — Generate all evaluation splits

Creates and stores the three split protocols:
  1. Random stratified 80/20
  2. Global chronological 80/20
  3. Per-family chronological 80/20

Output:
  - data/splits/random_stratified.json
  - data/splits/global_chronological.json
  - data/splits/per_family_chronological.json
  - artifacts/splits/split_summary.json
"""

from pathlib import Path
from datetime import datetime

import pandas as pd
from sklearn.model_selection import train_test_split

from reproducibility import (
    file_record,
    ordered_rows_sha256,
    sha256_bytes,
    write_deterministic_gzip,
    write_json,
)

# ── Configuration ──────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "splits"
REPRO_DIR = PROJECT_ROOT / "artifacts" / "reproducibility" / "splits"
SPLITS_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
TRAIN_FRAC = 0.80
VAL_FRAC_OF_TRAIN = 0.10  # for fusion/calibration tuning


def write_split_manifests(meta: pd.DataFrame, splits: dict) -> Path:
    """Write exact sample-hash assignments for every defended split."""
    REPRO_DIR.mkdir(parents=True, exist_ok=True)

    catalog = meta[["sha256", "family", "date"]].copy()
    catalog.insert(0, "index", catalog.index.astype(int))
    catalog["date"] = pd.to_datetime(catalog["date"]).dt.strftime("%Y-%m-%d")
    catalog_payload = catalog.to_csv(index=False, lineterminator="\n").encode("utf-8")
    catalog_path = REPRO_DIR / "sample_catalog.csv.gz"
    write_deterministic_gzip(catalog_path, catalog_payload)

    assignment_rows = []
    split_records = {}
    for split_name, split in splits.items():
        role_indices = {role: list(split.get(role, [])) for role in ["train", "val", "test"]}
        flattened = [index for indices in role_indices.values() for index in indices]
        if len(flattened) != len(set(flattened)):
            raise ValueError(f"Split {split_name} contains overlapping assignments.")
        if set(flattened) != set(range(len(meta))):
            raise ValueError(f"Split {split_name} does not cover the complete sample set.")

        membership_rows = []
        for role in ["train", "val", "test"]:
            for index in role_indices[role]:
                row = catalog.loc[index]
                record = {
                    "split": split_name,
                    "role": role,
                    "index": int(index),
                    "sha256": row["sha256"],
                    "family": row["family"],
                    "date": row["date"],
                }
                assignment_rows.append(record)
                membership_rows.append(
                    f"{role},{index},{row['sha256']},{row['family']},{row['date']}"
                )

        split_records[split_name] = {
            "n_train": len(role_indices["train"]),
            "n_val": len(role_indices["val"]),
            "n_test": len(role_indices["test"]),
            "ordered_membership_sha256": ordered_rows_sha256(membership_rows),
            "split_json": file_record(SPLITS_DIR / f"{split_name}.json", PROJECT_ROOT),
        }

    assignments = pd.DataFrame(assignment_rows)
    assignments_payload = assignments.to_csv(index=False, lineterminator="\n").encode("utf-8")
    assignments_path = REPRO_DIR / "split_assignments.csv.gz"
    write_deterministic_gzip(assignments_path, assignments_payload)

    manifest = {
        "schema_version": "1.0",
        "index_semantics": (
            "Zero-based row index after filtering metadata to has_report == True "
            "and resetting the index."
        ),
        "sample_catalog": {
            **file_record(catalog_path, PROJECT_ROOT),
            "uncompressed_sha256": sha256_bytes(catalog_payload),
        },
        "split_assignments": {
            **file_record(assignments_path, PROJECT_ROOT),
            "uncompressed_sha256": sha256_bytes(assignments_payload),
        },
        "splits": split_records,
    }
    manifest_path = REPRO_DIR / "split_manifest.json"
    write_json(manifest_path, manifest)
    return manifest_path


def build_random_stratified(meta: pd.DataFrame) -> dict:
    """Stratified random 80/20 split."""
    train_idx, test_idx = train_test_split(
        meta.index, test_size=1 - TRAIN_FRAC,
        stratify=meta["family"], random_state=SEED
    )
    # Carve validation from training
    train_sub, val_idx = train_test_split(
        train_idx, test_size=VAL_FRAC_OF_TRAIN,
        stratify=meta.loc[train_idx, "family"], random_state=SEED
    )
    return {
        "name": "random_stratified",
        "train": sorted(train_sub.tolist()),
        "val": sorted(val_idx.tolist()),
        "test": sorted(test_idx.tolist()),
        "seed": SEED,
    }


def build_global_chronological(meta: pd.DataFrame) -> dict:
    """Global chronological 80/20 split by detection date."""
    sorted_meta = meta.sort_values("date").reset_index(drop=True)
    n = len(sorted_meta)
    split_idx = int(n * TRAIN_FRAC)
    boundary_date = sorted_meta.iloc[split_idx]["date"]

    # Use original index
    sorted_by_date = meta.sort_values("date")
    train_all = sorted_by_date.index[:split_idx].tolist()
    test_idx = sorted_by_date.index[split_idx:].tolist()

    # Validation: last VAL_FRAC_OF_TRAIN of training period
    val_size = int(len(train_all) * VAL_FRAC_OF_TRAIN)
    train_idx = train_all[:-val_size]
    val_idx = train_all[-val_size:]

    return {
        "name": "global_chronological",
        "train": sorted(train_idx),
        "val": sorted(val_idx),
        "test": sorted(test_idx),
        "boundary_date": str(boundary_date),
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "seed": SEED,
    }


def build_per_family_chronological(meta: pd.DataFrame) -> dict:
    """Per-family chronological 80/20 split."""
    train_idx, val_idx, test_idx = [], [], []

    for fam in sorted(meta["family"].unique()):
        fam_mask = meta["family"] == fam
        fam_df = meta[fam_mask].sort_values("date")
        n = len(fam_df)
        split_point = int(n * TRAIN_FRAC)

        fam_train_all = fam_df.index[:split_point].tolist()
        fam_test = fam_df.index[split_point:].tolist()

        # Validation from end of training
        val_size = max(1, int(len(fam_train_all) * VAL_FRAC_OF_TRAIN))
        fam_train = fam_train_all[:-val_size]
        fam_val = fam_train_all[-val_size:]

        train_idx.extend(fam_train)
        val_idx.extend(fam_val)
        test_idx.extend(fam_test)

    return {
        "name": "per_family_chronological",
        "train": sorted(train_idx),
        "val": sorted(val_idx),
        "test": sorted(test_idx),
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "seed": SEED,
    }


def main():
    print("=" * 60)
    print("MPhil Split Generation")
    print("=" * 60)

    meta = pd.read_parquet(PROCESSED_DIR / "metadata.parquet")
    meta = meta[meta["has_report"]].reset_index(drop=True)
    print(f"Building splits for {len(meta)} samples.")

    splits = {}
    summary = {"timestamp": datetime.now().isoformat(), "n_total": len(meta)}

    # 1. Random stratified
    print("\n[1/3] Random stratified split...")
    s = build_random_stratified(meta)
    splits["random_stratified"] = s
    print(f"  Train: {len(s['train'])}, Val: {len(s['val'])}, Test: {len(s['test'])}")

    # 2. Global chronological
    print("\n[2/3] Global chronological split...")
    s = build_global_chronological(meta)
    splits["global_chronological"] = s
    print(f"  Train: {s['n_train']}, Val: {s['n_val']}, Test: {s['n_test']}")
    print(f"  Boundary date: {s['boundary_date']}")

    # 3. Per-family chronological
    print("\n[3/3] Per-family chronological split...")
    s = build_per_family_chronological(meta)
    splits["per_family_chronological"] = s
    print(f"  Train: {s['n_train']}, Val: {s['n_val']}, Test: {s['n_test']}")

    # Save each split
    for name, split_data in splits.items():
        path = SPLITS_DIR / f"{name}.json"
        write_json(path, split_data)
        print(f"\n  Saved: {path}")
        summary[name] = {
            "n_train": len(split_data["train"]),
            "n_val": len(split_data["val"]),
            "n_test": len(split_data["test"]),
        }

    # Save summary
    write_json(ARTIFACTS_DIR / "split_summary.json", summary)

    manifest_path = write_split_manifests(meta, splits)

    print(f"\nSplit summary saved to: {ARTIFACTS_DIR / 'split_summary.json'}")
    print(f"Split/hash manifest saved to: {manifest_path}")
    print("\n✓ All splits written.")


if __name__ == "__main__":
    main()
