"""
06_walk_forward.py — Walk-Forward (Rolling-Origin) Evaluation

Trains on a rolling window and evaluates on the next month.
Produces a time-series of macro-F1 per view and fusion.

Output:
  - artifacts/walk_forward/walk_forward_results.csv
  - artifacts/walk_forward/walk_forward_plot.png
"""

import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUT_DIR = PROJECT_ROOT / "artifacts" / "walk_forward"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
DELTA_DAYS = 30  # test window size
MIN_TRAIN_SAMPLES = 500
MIN_TEST_SAMPLES = 50


def extract_api_text(report_path):
    """Quick API token extraction for a single report."""
    try:
        with open(report_path, "r", encoding="utf-8", errors="replace") as f:
            report = json.load(f)
        apis = report.get("behavior", {}).get("summary", {}).get("resolved_apis", [])
        return " ".join(str(a).lower() for a in apis if isinstance(a, str))
    except Exception:
        return ""


def main():
    print("=" * 60)
    print("Walk-Forward Evaluation")
    print("=" * 60)

    meta = pd.read_parquet(PROCESSED_DIR / "metadata.parquet")
    meta = meta[meta["has_report"] == True].reset_index(drop=True)
    meta["date"] = pd.to_datetime(meta["date"])
    meta = meta.sort_values("date").reset_index(drop=True)

    # Focus on dense period: 2018-01 to 2020-01
    start_date = pd.Timestamp("2018-06-01")
    end_date = meta["date"].max()

    # Pre-extract all API texts (expensive but done once)
    print("Extracting API tokens for all samples...")
    api_texts = []
    for _, row in meta.iterrows():
        api_texts.append(extract_api_text(row["report_path"]))
    meta["api_text"] = api_texts

    # Walk-forward loop
    results = []
    boundary = start_date

    while boundary + pd.Timedelta(days=DELTA_DAYS) <= end_date:
        test_end = boundary + pd.Timedelta(days=DELTA_DAYS)

        train_mask = meta["date"] < boundary
        test_mask = (meta["date"] >= boundary) & (meta["date"] < test_end)

        n_train = train_mask.sum()
        n_test = test_mask.sum()

        if n_train < MIN_TRAIN_SAMPLES or n_test < MIN_TEST_SAMPLES:
            boundary += pd.Timedelta(days=DELTA_DAYS)
            continue

        train_df = meta[train_mask]
        test_df = meta[test_mask]
        families_present = test_df["family"].nunique()

        print(f"\n  Window: train<{boundary.date()} | "
              f"test=[{boundary.date()}, {test_end.date()}) | "
              f"n_train={n_train}, n_test={n_test}, "
              f"families={families_present}")

        # Vectorise (hashing for speed)
        hasher = HashingVectorizer(
            n_features=131072, ngram_range=(1, 2), alternate_sign=False
        )
        X_train = hasher.transform(train_df["api_text"])
        X_test = hasher.transform(test_df["api_text"])
        y_train = train_df["family"].values
        y_test = test_df["family"].values

        # Train and evaluate
        clf = SGDClassifier(
            loss="log_loss", class_weight="balanced",
            max_iter=1000, random_state=SEED
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        macro_f1 = f1_score(y_test, y_pred, average="macro")

        print(f"    API macro-F1: {macro_f1:.4f}")

        results.append({
            "boundary": str(boundary.date()),
            "test_start": str(boundary.date()),
            "test_end": str(test_end.date()),
            "n_train": int(n_train),
            "n_test": int(n_test),
            "families_present": int(families_present),
            "api_macro_f1": float(macro_f1),
        })

        boundary += pd.Timedelta(days=DELTA_DAYS)

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUT_DIR / "walk_forward_results.csv", index=False)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    ax.plot(pd.to_datetime(results_df["boundary"]),
            results_df["api_macro_f1"],
            marker="o", markersize=4, label="API tokens")
    ax.set_xlabel("Training Boundary Date")
    ax.set_ylabel("macro-F1")
    ax.set_title("Walk-Forward macro-F1 (API-only, monthly windows)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "walk_forward_plot.png", dpi=200)

    print(f"\n✓ Results saved to {OUT_DIR}")
    print(f"  {len(results)} windows evaluated.")


if __name__ == "__main__":
    main()
