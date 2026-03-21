"""
05_invariance_discriminability.py — Invariance–Discriminability Trade-Off

Computes discriminability (within-window macro-F1 via CV, mean |coef|)
and invariance (JSD between train/test token distributions, Jaccard
stability of top features) for each view.

Produces the trade-off scatter plot (Figure 5.X1).

Output:
  - artifacts/invariance/trade_off_metrics.json
  - artifacts/invariance/trade_off_scatter.png
"""

import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial.distance import jensenshannon
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Configuration ──────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
OUT_DIR = PROJECT_ROOT / "artifacts" / "invariance"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPLIT_NAME = "global_chronological"
SEED = 42
CV_FOLDS = 5


def load_split():
    with open(SPLITS_DIR / f"{SPLIT_NAME}.json") as f:
        return json.load(f)


def load_labels():
    return pd.read_parquet(CACHE_DIR / "labels.parquet")


def compute_discriminability(X_train, y_train, name):
    """Within-window macro-F1 (CV) and mean |coefficient|."""
    print(f"  [{name}] Computing discriminability...")
    clf = SGDClassifier(
        loss="log_loss", class_weight="balanced",
        max_iter=1000, random_state=SEED
    )
    scores = cross_val_score(
        clf, X_train, y_train, cv=CV_FOLDS,
        scoring="f1_macro", n_jobs=-1
    )
    cv_f1 = float(np.mean(scores))

    # Fit full model for coefficient magnitudes
    clf.fit(X_train, y_train)
    mean_abs_coef = float(np.mean(np.abs(clf.coef_)))

    print(f"    CV macro-F1: {cv_f1:.4f}, Mean |coef|: {mean_abs_coef:.6f}")
    return cv_f1, mean_abs_coef


def compute_jsd_sparse(X_train, X_test):
    """JSD between train and test token-presence frequency vectors."""
    # Binary presence then column means = presence rate
    train_freq = np.asarray((X_train > 0).mean(axis=0)).flatten()
    test_freq = np.asarray((X_test > 0).mean(axis=0)).flatten()
    # Add small epsilon to avoid zero division
    eps = 1e-10
    train_freq = train_freq + eps
    test_freq = test_freq + eps
    # Normalise to probability distributions
    train_freq = train_freq / train_freq.sum()
    test_freq = test_freq / test_freq.sum()
    return float(jensenshannon(train_freq, test_freq))


def compute_jsd_dense(X_train, X_test):
    """JSD for dense numeric features (counts, PE)."""
    from scipy.stats import ks_2samp
    # Average KS statistic across features as a proxy
    n_features = X_train.shape[1]
    ks_stats = []
    for j in range(n_features):
        stat, _ = ks_2samp(X_train[:, j], X_test[:, j])
        ks_stats.append(stat)
    return float(np.mean(ks_stats))


def compute_top_k_jaccard(X_train, y_train, X_test, y_test, k=50):
    """Jaccard similarity of top-k features between train and test models."""
    clf_train = SGDClassifier(
        loss="log_loss", class_weight="balanced",
        max_iter=1000, random_state=SEED
    )
    clf_train.fit(X_train, y_train)
    top_train = set(np.argsort(np.abs(clf_train.coef_).sum(axis=0))[-k:])

    # Retrain on (train + a portion of test) as a proxy for next window
    # Alternative: just use the same top-k from the test-period perspective
    clf_test = SGDClassifier(
        loss="log_loss", class_weight="balanced",
        max_iter=1000, random_state=SEED
    )
    clf_test.fit(X_test, y_test)
    top_test = set(np.argsort(np.abs(clf_test.coef_).sum(axis=0))[-k:])

    jaccard = len(top_train & top_test) / len(top_train | top_test)
    return float(jaccard)


def main():
    print("=" * 60)
    print("Invariance–Discriminability Analysis")
    print("=" * 60)

    split = load_split()
    labels = load_labels()
    train_idx = split["train"]
    val_idx = split.get("val", [])
    test_idx = split["test"]
    train_all = train_idx + val_idx  # use full train for this analysis

    y_train = labels.iloc[train_all]["family"].values
    y_test = labels.iloc[test_idx]["family"].values

    results = {}

    # --- View A: API tokens (TF-IDF) ---
    print("\n[1/4] API tokens (TF-IDF)...")
    X_api = sp.load_npz(CACHE_DIR / f"api_tfidf_{SPLIT_NAME}.npz")
    X_api_train = X_api[train_all]
    X_api_test = X_api[test_idx]
    cv_f1, mean_coef = compute_discriminability(X_api_train, y_train, "API")
    jsd = compute_jsd_sparse(X_api_train, X_api_test)
    jaccard = compute_top_k_jaccard(X_api_train, y_train, X_api_test, y_test)
    results["API tokens"] = {
        "cv_f1": cv_f1, "mean_abs_coef": mean_coef,
        "jsd": jsd, "jaccard_top50": jaccard,
    }

    # --- View B: Artifact tokens (TF-IDF) ---
    print("\n[2/4] Artifact tokens (TF-IDF)...")
    X_art = sp.load_npz(CACHE_DIR / f"art_tfidf_{SPLIT_NAME}.npz")
    X_art_train = X_art[train_all]
    X_art_test = X_art[test_idx]
    cv_f1, mean_coef = compute_discriminability(X_art_train, y_train, "Artifacts")
    jsd = compute_jsd_sparse(X_art_train, X_art_test)
    jaccard = compute_top_k_jaccard(X_art_train, y_train, X_art_test, y_test)
    results["Artifact tokens"] = {
        "cv_f1": cv_f1, "mean_abs_coef": mean_coef,
        "jsd": jsd, "jaccard_top50": jaccard,
    }

    # --- View C: Behavioral counts ---
    print("\n[3/4] Behavioral counts...")
    X_counts = np.load(CACHE_DIR / f"counts_scaled_{SPLIT_NAME}.npy")
    X_c_train = X_counts[train_all]
    X_c_test = X_counts[test_idx]
    cv_f1, mean_coef = compute_discriminability(X_c_train, y_train, "Counts")
    jsd = compute_jsd_dense(X_c_train, X_c_test)
    results["Behavioral counts"] = {
        "cv_f1": cv_f1, "mean_abs_coef": mean_coef,
        "jsd": jsd, "jaccard_top50": None,  # too few features for top-50
    }

    # --- View D: Static PE ---
    print("\n[4/4] Static PE...")
    X_pe = np.load(CACHE_DIR / f"pe_scaled_{SPLIT_NAME}.npy")
    X_pe_train = X_pe[train_all]
    X_pe_test = X_pe[test_idx]
    cv_f1, mean_coef = compute_discriminability(X_pe_train, y_train, "PE")
    jsd = compute_jsd_dense(X_pe_train, X_pe_test)
    results["Static PE"] = {
        "cv_f1": cv_f1, "mean_abs_coef": mean_coef,
        "jsd": jsd, "jaccard_top50": None,
    }

    # --- Scatter plot ---
    print("\nGenerating scatter plot...")
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    for name, m in results.items():
        invariance = 1.0 - m["jsd"]  # higher = more stable
        discriminability = m["cv_f1"]
        ax.scatter(invariance, discriminability, s=120, zorder=5)
        ax.annotate(name, (invariance, discriminability),
                    textcoords="offset points", xytext=(8, 5), fontsize=10)

    ax.set_xlabel("Temporal Invariance (1 − JSD)", fontsize=12)
    ax.set_ylabel("Discriminability (within-window macro-F1)", fontsize=12)
    ax.set_title("Invariance–Discriminability Trade-Off by Feature View",
                 fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "trade_off_scatter.png", dpi=200)
    print(f"  Saved: {OUT_DIR / 'trade_off_scatter.png'}")

    # Save metrics
    output = {"timestamp": datetime.now().isoformat(), "views": results}
    with open(OUT_DIR / "trade_off_metrics.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved: {OUT_DIR / 'trade_off_metrics.json'}")
    print("\n✓ Invariance–discriminability analysis complete.")


if __name__ == "__main__":
    main()
