"""
08_calibration_full.py — Full-Dataset Calibration and Selective Prediction

Extends the 20K-subset calibration analysis to the full dataset.
Includes per-family ECE and margin-based selective prediction.

Output:
  - artifacts/calibration/full_calibration_metrics.json
  - artifacts/calibration/reliability_diagram.png
  - artifacts/calibration/per_family_ece.csv
  - artifacts/calibration/selective_prediction.csv
  - artifacts/calibration/risk_coverage_comparison.png
"""

import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    f1_score, accuracy_score, brier_score_loss
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
OUT_DIR = PROJECT_ROOT / "artifacts" / "calibration"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
SPLIT_NAME = "global_chronological"


def compute_ece(y_true, y_prob, n_bins=10):
    """Expected Calibration Error on max-probability."""
    max_probs = y_prob.max(axis=1)
    preds = y_prob.argmax(axis=1)
    correct = (preds == y_true).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (max_probs > bin_boundaries[i]) & (max_probs <= bin_boundaries[i + 1])
        if mask.sum() > 0:
            avg_conf = max_probs[mask].mean()
            avg_acc = correct[mask].mean()
            ece += mask.sum() / len(y_true) * abs(avg_acc - avg_conf)
    return float(ece)


def compute_per_family_ece(y_true, y_prob, classes, n_bins=10):
    """Per-family ECE: for each family, ECE on predictions assigned to it."""
    preds = y_prob.argmax(axis=1)
    max_probs = y_prob.max(axis=1)
    rows = []
    for i, fam in enumerate(classes):
        mask = preds == i
        if mask.sum() == 0:
            continue
        correct = (y_true[mask] == i).astype(float)
        # Simple ECE for this family
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for b in range(n_bins):
            bmask = (max_probs[mask] > bin_boundaries[b]) & \
                    (max_probs[mask] <= bin_boundaries[b + 1])
            if bmask.sum() > 0:
                avg_conf = max_probs[mask][bmask].mean()
                avg_acc = correct[bmask].mean()
                ece += bmask.sum() / mask.sum() * abs(avg_acc - avg_conf)
        rows.append({
            "family": fam,
            "n_predictions": int(mask.sum()),
            "ece": round(float(ece), 4),
            "mean_confidence": round(float(max_probs[mask].mean()), 4),
            "empirical_accuracy": round(float(correct.mean()), 4),
        })
    return pd.DataFrame(rows)


def selective_prediction_sweep(y_true, y_prob, method="max_prob"):
    """Sweep thresholds and compute selective accuracy and macro-F1."""
    preds = y_prob.argmax(axis=1)
    max_probs = y_prob.max(axis=1)

    if method == "margin":
        sorted_probs = np.sort(y_prob, axis=1)
        scores = sorted_probs[:, -1] - sorted_probs[:, -2]
    else:
        scores = max_probs

    thresholds = [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    rows = []
    for tau in thresholds:
        mask = scores >= tau
        coverage = mask.sum() / len(y_true)
        if mask.sum() == 0:
            continue
        sel_acc = accuracy_score(y_true[mask], preds[mask])
        sel_f1 = f1_score(y_true[mask], preds[mask], average="macro")
        rows.append({
            "method": method,
            "threshold": tau,
            "coverage": round(float(coverage), 4),
            "selective_accuracy": round(float(sel_acc), 4),
            "selective_macro_f1": round(float(sel_f1), 4),
            "n_covered": int(mask.sum()),
        })
    return pd.DataFrame(rows)


def reliability_diagram(y_true, y_prob, title, path, n_bins=10):
    """Plot reliability diagram."""
    max_probs = y_prob.max(axis=1)
    preds = y_prob.argmax(axis=1)
    correct = (preds == y_true).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_accs = []
    bin_counts = []

    for i in range(n_bins):
        mask = (max_probs > bin_boundaries[i]) & (max_probs <= bin_boundaries[i + 1])
        if mask.sum() > 0:
            bin_centers.append((bin_boundaries[i] + bin_boundaries[i + 1]) / 2)
            bin_accs.append(correct[mask].mean())
            bin_counts.append(mask.sum())

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8),
                                    gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    ax1.bar(bin_centers, bin_accs, width=0.08, alpha=0.7, label="Model")
    ax1.set_ylabel("Empirical Accuracy")
    ax1.set_title(title)
    ax1.legend()
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)

    ax2.bar(bin_centers, bin_counts, width=0.08, alpha=0.7, color="gray")
    ax2.set_xlabel("Confidence (max probability)")
    ax2.set_ylabel("Count")
    ax2.set_xlim(0, 1)

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    print("=" * 60)
    print("Full-Dataset Calibration Analysis")
    print("=" * 60)

    # Load data
    with open(SPLITS_DIR / f"{SPLIT_NAME}.json") as f:
        split = json.load(f)
    labels = pd.read_parquet(CACHE_DIR / "labels.parquet")

    train_idx = split["train"]
    val_idx = split["val"]
    test_idx = split["test"]

    X = sp.load_npz(CACHE_DIR / f"api_tfidf_{SPLIT_NAME}.npz")
    y_all = labels["family"].values
    classes = sorted(labels["family"].unique())
    label_map = {c: i for i, c in enumerate(classes)}
    y_numeric = np.array([label_map[f] for f in y_all])

    X_train, y_train = X[train_idx], y_numeric[train_idx]
    X_val, y_val = X[val_idx], y_numeric[val_idx]
    X_test, y_test = X[test_idx], y_numeric[test_idx]

    # 1. Uncalibrated model
    print("\n[1/3] Training uncalibrated model...")
    clf = SGDClassifier(
        loss="log_loss", class_weight="balanced",
        max_iter=1000, random_state=SEED
    )
    clf.fit(X_train, y_train)
    prob_uncal = clf.predict_proba(X_test)
    pred_uncal = clf.predict(X_test)

    acc_uncal = accuracy_score(y_test, pred_uncal)
    f1_uncal = f1_score(y_test, pred_uncal, average="macro")
    ece_uncal = compute_ece(y_test, prob_uncal)
    print(f"  Uncalibrated: acc={acc_uncal:.4f}, macro-F1={f1_uncal:.4f}, "
          f"ECE={ece_uncal:.4f}")

    # 2. Sigmoid-calibrated
    print("\n[2/3] Training sigmoid-calibrated model...")
    cal_clf = CalibratedClassifierCV(clf, method="sigmoid", cv="prefit")
    cal_clf.fit(X_val, y_val)
    prob_cal = cal_clf.predict_proba(X_test)
    pred_cal = np.argmax(prob_cal, axis=1)

    acc_cal = accuracy_score(y_test, pred_cal)
    f1_cal = f1_score(y_test, pred_cal, average="macro")
    ece_cal = compute_ece(y_test, prob_cal)
    print(f"  Calibrated: acc={acc_cal:.4f}, macro-F1={f1_cal:.4f}, "
          f"ECE={ece_cal:.4f}")

    # 3. Per-family ECE
    print("\n[3/3] Computing per-family ECE...")
    pf_ece = compute_per_family_ece(y_test, prob_uncal, classes)
    pf_ece.to_csv(OUT_DIR / "per_family_ece.csv", index=False)
    print(pf_ece.to_string(index=False))

    # Selective prediction (both methods)
    sp_max = selective_prediction_sweep(y_test, prob_uncal, "max_prob")
    sp_margin = selective_prediction_sweep(y_test, prob_uncal, "margin")
    sp_all = pd.concat([sp_max, sp_margin])
    sp_all.to_csv(OUT_DIR / "selective_prediction.csv", index=False)

    # Reliability diagrams
    reliability_diagram(
        y_test, prob_uncal,
        "Reliability: Uncalibrated (full dataset, global chrono)",
        OUT_DIR / "reliability_uncalibrated.png"
    )
    reliability_diagram(
        y_test, prob_cal,
        "Reliability: Sigmoid-calibrated (full dataset, global chrono)",
        OUT_DIR / "reliability_calibrated.png"
    )

    # Risk-coverage comparison plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for method, label in [("max_prob", "Max probability"),
                           ("margin", "Prediction margin")]:
        df = sp_all[sp_all["method"] == method]
        ax.plot(df["coverage"], df["selective_accuracy"],
                marker="o", label=label)
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Selective Accuracy")
    ax.set_title("Risk–Coverage: Max Probability vs Margin")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "risk_coverage_comparison.png", dpi=200)

    # Save summary
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "uncalibrated": {"accuracy": acc_uncal, "macro_f1": f1_uncal, "ece": ece_uncal},
        "sigmoid_calibrated": {"accuracy": acc_cal, "macro_f1": f1_cal, "ece": ece_cal},
    }
    with open(OUT_DIR / "full_calibration_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✓ Full calibration analysis saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
