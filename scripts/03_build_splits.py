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

import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ── Configuration ──────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "splits"
SPLITS_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
TRAIN_FRAC = 0.80
VAL_FRAC_OF_TRAIN = 0.10  # for fusion/calibration tuning


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
    meta = meta[meta["has_report"] == True].reset_index(drop=True)
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
        with open(path, "w") as f:
            json.dump(split_data, f)
        print(f"\n  Saved: {path}")
        summary[name] = {
            "n_train": len(split_data["train"]),
            "n_val": len(split_data["val"]),
            "n_test": len(split_data["test"]),
        }

    # Save summary
    with open(ARTIFACTS_DIR / "split_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSplit summary saved to: {ARTIFACTS_DIR / 'split_summary.json'}")
    print("\n✓ All splits written.")


if __name__ == "__main__":
    main()
