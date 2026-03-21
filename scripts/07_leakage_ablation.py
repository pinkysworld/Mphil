"""
07_leakage_ablation.py — Leakage Ablation Study (RQ2)

Compares leakage-controlled vs leakage-permitted models
under both random and chronological splits.

Output:
  - artifacts/leakage_ablation/ablation_results.json
  - artifacts/leakage_ablation/ablation_table.csv
"""

import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
OUT_DIR = PROJECT_ROOT / "artifacts" / "leakage_ablation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
FAMILY_NAMES = {
    "emotet", "swisyn", "qakbot", "trickbot", "lokibot",
    "njrat", "zeus", "ursnif", "adload", "harhar"
}


def extract_api_tokens(report_path, filter_families=True):
    """Extract API tokens, optionally filtering family names."""
    try:
        with open(report_path, "r", encoding="utf-8", errors="replace") as f:
            report = json.load(f)
        behavior = report.get("behavior", {})
        summary = behavior.get("summary", {})
        apis = summary.get("resolved_apis", [])
        tokens = []
        for api in apis:
            if isinstance(api, str):
                t = api.lower().strip()
                if filter_families:
                    for fam in FAMILY_NAMES:
                        t = t.replace(fam, "")
                if t.strip():
                    tokens.append(t.strip())

        # Also include artifact strings that might contain family names
        for field in ["mutexes", "files", "write_files"]:
            for item in summary.get(field, []):
                if isinstance(item, str):
                    t = item.lower()
                    if filter_families:
                        for fam in FAMILY_NAMES:
                            t = t.replace(fam, "")
                    if t.strip():
                        tokens.append(f"ART:{t.strip()[:80]}")

        return " ".join(tokens)
    except Exception:
        return ""


def run_experiment(meta, split_name, filter_families, label):
    """Train and evaluate a single configuration."""
    split_path = SPLITS_DIR / f"{split_name}.json"
    with open(split_path) as f:
        split = json.load(f)

    train_idx = split["train"] + split.get("val", [])
    test_idx = split["test"]

    # Extract texts
    texts = []
    for _, row in meta.iterrows():
        texts.append(extract_api_tokens(row["report_path"], filter_families))

    hasher = HashingVectorizer(
        n_features=262144, ngram_range=(1, 2), alternate_sign=False
    )
    X = hasher.transform(texts)
    y = meta["family"].values

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    clf = SGDClassifier(
        loss="log_loss", class_weight="balanced",
        max_iter=1000, random_state=SEED
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")

    return {
        "label": label,
        "split": split_name,
        "filter_families": filter_families,
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "n_train": len(train_idx),
        "n_test": len(test_idx),
    }


def main():
    print("=" * 60)
    print("Leakage Ablation Study")
    print("=" * 60)

    meta = pd.read_parquet(PROCESSED_DIR / "metadata.parquet")
    meta = meta[meta["has_report"] == True].reset_index(drop=True)

    results = []

    configs = [
        ("random_stratified", True, "Controlled + Random"),
        ("random_stratified", False, "Permitted + Random"),
        ("global_chronological", True, "Controlled + Chrono"),
        ("global_chronological", False, "Permitted + Chrono"),
    ]

    for split_name, filter_fam, label in configs:
        print(f"\n  Running: {label}...")
        r = run_experiment(meta, split_name, filter_fam, label)
        results.append(r)
        print(f"    Acc={r['accuracy']}, macro-F1={r['macro_f1']}, "
              f"weighted-F1={r['weighted_f1']}")

    # Compute deltas
    print("\n--- Ablation Summary ---")
    for split in ["random_stratified", "global_chronological"]:
        ctrl = [r for r in results if r["split"] == split and r["filter_families"]][0]
        perm = [r for r in results if r["split"] == split and not r["filter_families"]][0]
        delta = round(perm["macro_f1"] - ctrl["macro_f1"], 4)
        print(f"  {split}: macro-F1 delta (permitted - controlled) = {delta:+.4f}")

    # Save
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUT_DIR / "ablation_table.csv", index=False)

    output = {
        "timestamp": datetime.now().isoformat(),
        "results": results,
    }
    with open(OUT_DIR / "ablation_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ Ablation results saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
