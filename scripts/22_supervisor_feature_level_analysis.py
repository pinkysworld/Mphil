"""Feature-level invariance-discriminability analysis requested by supervisor.

This analysis replaces the four-view-only inferential claim with feature-level
statistics. Raw shift scales remain view-appropriate; pooled inference uses
within-view percentile ranks.

Outputs:
  artifacts/supervisor_revision/feature_level_metrics.csv.gz
  artifacts/supervisor_revision/feature_level_summary.json
  artifacts/supervisor_revision/feature_level_tradeoff.png
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import ks_2samp, spearmanr
from sklearn.feature_selection import chi2

from run_experiment import PROJECT_ROOT

CACHE_DIR = PROJECT_ROOT / "data" / "cache"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
OUT_DIR = PROJECT_ROOT / "artifacts" / "supervisor_revision"
SPLIT_NAME = "global_chronological"

MIN_TRAIN_SUPPORT = 20
RNG_SEED = 20260829
N_BOOTSTRAP = 1000
N_PERMUTATIONS = 1000
PLOT_MAX_PER_VIEW = 5000


def load_split() -> dict:
    return json.loads(
        (SPLITS_DIR / f"{SPLIT_NAME}.json").read_text(encoding="utf-8")
    )


def load_labels() -> pd.DataFrame:
    frame = pd.read_parquet(CACHE_DIR / "labels.parquet")
    frame["family"] = frame["family"].astype(str)
    return frame


def bernoulli_jsd(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Jensen-Shannon divergence in bits for Bernoulli feature prevalence."""
    eps = np.finfo(float).eps
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    q = np.clip(np.asarray(q, dtype=float), eps, 1.0 - eps)
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return (
            a * np.log2(a / b)
            + (1.0 - a) * np.log2((1.0 - a) / (1.0 - b))
        )

    return 0.5 * (kl(p, m) + kl(q, m))


def token_feature_frame(
    matrix_path: Path,
    vectorizer_path: Path,
    view: str,
    train_idx: list[int],
    test_idx: list[int],
    y_train: np.ndarray,
) -> pd.DataFrame:
    matrix = sp.load_npz(matrix_path).tocsr()
    binary_train = (matrix[train_idx] > 0).astype(np.uint8).tocsr()
    binary_test = (matrix[test_idx] > 0).astype(np.uint8).tocsr()

    support = np.asarray(binary_train.sum(axis=0)).ravel().astype(int)
    keep = support >= MIN_TRAIN_SUPPORT
    if not np.any(keep):
        raise RuntimeError(f"No {view} features meet support threshold.")

    binary_train = binary_train[:, keep]
    binary_test = binary_test[:, keep]

    # Cramer's V for the K x 2 family/presence table. Because min(K-1, 2-1)=1,
    # V = sqrt(chi-square / n).
    chi2_stat, chi2_p = chi2(binary_train, y_train)
    cramers_v = np.sqrt(np.maximum(chi2_stat, 0.0) / binary_train.shape[0])
    cramers_v = np.clip(cramers_v, 0.0, 1.0)

    p_train = np.asarray(binary_train.mean(axis=0)).ravel()
    p_test = np.asarray(binary_test.mean(axis=0)).ravel()
    shift = bernoulli_jsd(p_train, p_test)

    vectorizer = joblib.load(vectorizer_path)
    names = np.asarray(vectorizer.get_feature_names_out(), dtype=object)[keep]

    return pd.DataFrame(
        {
            "view": view,
            "feature": names.astype(str),
            "feature_type": "token",
            "train_support": support[keep],
            "discriminability_effect": cramers_v,
            "discriminability_p": chi2_p,
            "temporal_shift": shift,
            "train_prevalence": p_train,
            "test_prevalence": p_test,
        }
    )


def eta_squared(values: np.ndarray, labels: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    grand = float(np.mean(values))
    total = float(np.sum((values - grand) ** 2))
    if not np.isfinite(total) or total <= 0:
        return 0.0

    between = 0.0
    for family in np.unique(labels):
        group = values[labels == family]
        if group.size:
            between += group.size * (float(np.mean(group)) - grand) ** 2
    return float(np.clip(between / total, 0.0, 1.0))


def dense_feature_frame(
    frame_path: Path,
    view: str,
    train_idx: list[int],
    test_idx: list[int],
    y_train: np.ndarray,
) -> pd.DataFrame:
    data = pd.read_parquet(frame_path)
    numeric = data.apply(pd.to_numeric, errors="coerce")

    rows: list[dict[str, object]] = []
    for column in numeric.columns:
        train = numeric.iloc[train_idx][column].to_numpy(dtype=float)
        test = numeric.iloc[test_idx][column].to_numpy(dtype=float)

        finite_train = train[np.isfinite(train)]
        median = float(np.median(finite_train)) if finite_train.size else 0.0
        train = np.where(np.isfinite(train), train, median)
        test = np.where(np.isfinite(test), test, median)

        effect = eta_squared(train, y_train)
        ks = ks_2samp(train, test, alternative="two-sided", mode="auto")

        rows.append(
            {
                "view": view,
                "feature": str(column),
                "feature_type": "numeric",
                "train_support": int(len(train)),
                "discriminability_effect": effect,
                "discriminability_p": np.nan,
                "temporal_shift": float(ks.statistic),
                "shift_p": float(ks.pvalue),
            }
        )
    return pd.DataFrame(rows)


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    out = np.full_like(p, np.nan)
    finite = np.isfinite(p)
    if not finite.any():
        return out

    values = p[finite]
    order = np.argsort(values)
    ranked = values[order]
    m = len(ranked)
    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    out[finite] = restored
    return out


def add_within_view_ranks(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["discriminability_rank"] = frame.groupby("view")[
        "discriminability_effect"
    ].rank(method="average", pct=True)
    frame["shift_rank"] = frame.groupby("view")["temporal_shift"].rank(
        method="average", pct=True
    )
    frame["invariance_rank"] = 1.0 - frame["shift_rank"]
    return frame


def stratified_resample_rho(
    frame: pd.DataFrame,
    rng: np.random.Generator,
) -> float:
    parts = []
    for _, group in frame.groupby("view", sort=True):
        positions = rng.integers(0, len(group), size=len(group))
        parts.append(group.iloc[positions])
    sampled = pd.concat(parts, ignore_index=True)
    return float(
        spearmanr(
            sampled["discriminability_rank"],
            sampled["shift_rank"],
        ).statistic
    )


def stratified_permutation_rho(
    frame: pd.DataFrame,
    rng: np.random.Generator,
) -> float:
    x = frame["discriminability_rank"].to_numpy(dtype=float)
    y = frame["shift_rank"].to_numpy(dtype=float).copy()
    for _, indices in frame.groupby("view", sort=True).groups.items():
        idx = np.asarray(list(indices), dtype=int)
        y[idx] = rng.permutation(y[idx])
    return float(spearmanr(x, y).statistic)


def association_summary(frame: pd.DataFrame) -> dict:
    per_view = []
    for view, group in frame.groupby("view", sort=True):
        result = spearmanr(
            group["discriminability_effect"],
            group["temporal_shift"],
        )
        per_view.append(
            {
                "view": view,
                "n_features": int(len(group)),
                "spearman_rho_discriminability_vs_shift": float(result.statistic),
                "p_value": float(result.pvalue),
            }
        )

    q_values = bh_adjust(np.array([row["p_value"] for row in per_view]))
    for row, q_value in zip(per_view, q_values):
        row["bh_fdr_q_value"] = float(q_value)

    pooled = spearmanr(
        frame["discriminability_rank"],
        frame["shift_rank"],
    )
    observed = float(pooled.statistic)

    rng = np.random.default_rng(RNG_SEED)
    bootstrap = np.array(
        [stratified_resample_rho(frame, rng) for _ in range(N_BOOTSTRAP)],
        dtype=float,
    )
    ci_low, ci_high = np.quantile(bootstrap, [0.025, 0.975])

    permutation = np.array(
        [stratified_permutation_rho(frame, rng) for _ in range(N_PERMUTATIONS)],
        dtype=float,
    )
    # One-sided test matches the mechanism prediction: greater discriminability
    # is associated with greater future shift.
    p_perm = (1.0 + np.sum(permutation >= observed)) / (N_PERMUTATIONS + 1.0)

    return {
        "per_view": per_view,
        "pooled_stratified": {
            "n_features": int(len(frame)),
            "spearman_rho_discriminability_vs_shift": observed,
            "equivalent_direction_for_discriminability_vs_invariance": "negative",
            "bootstrap_95_ci": [float(ci_low), float(ci_high)],
            "stratified_permutation_one_sided_p": float(p_perm),
            "bootstrap_replicates": N_BOOTSTRAP,
            "permutation_replicates": N_PERMUTATIONS,
            "rng_seed": RNG_SEED,
        },
    }


def plot_tradeoff(frame: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(RNG_SEED)
    fig, ax = plt.subplots(figsize=(9, 6))
    for view, group in frame.groupby("view", sort=True):
        if len(group) > PLOT_MAX_PER_VIEW:
            chosen = rng.choice(len(group), PLOT_MAX_PER_VIEW, replace=False)
            group = group.iloc[chosen]
        ax.scatter(
            group["shift_rank"],
            group["discriminability_rank"],
            s=8,
            alpha=0.25,
            label=view,
        )

    ax.set_xlabel("Temporal shift percentile within view (higher = less invariant)")
    ax.set_ylabel("Discriminability percentile within view")
    ax.set_title("Feature-level invariance-discriminability re-analysis")
    ax.legend()
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    split = load_split()
    labels = load_labels()
    train_idx = list(split["train"]) + list(split.get("val", []))
    test_idx = list(split["test"])
    y_train = labels.iloc[train_idx]["family"].to_numpy()

    frames = [
        token_feature_frame(
            CACHE_DIR / f"api_tfidf_{SPLIT_NAME}.npz",
            CACHE_DIR / f"api_tfidf_vectorizer_{SPLIT_NAME}.pkl",
            "API tokens",
            train_idx,
            test_idx,
            y_train,
        ),
        token_feature_frame(
            CACHE_DIR / f"art_tfidf_{SPLIT_NAME}.npz",
            CACHE_DIR / f"art_tfidf_vectorizer_{SPLIT_NAME}.pkl",
            "Artifact tokens",
            train_idx,
            test_idx,
            y_train,
        ),
        dense_feature_frame(
            CACHE_DIR / "counts_raw.parquet",
            "Behavioral counts",
            train_idx,
            test_idx,
            y_train,
        ),
        dense_feature_frame(
            CACHE_DIR / "pe_raw.parquet",
            "Static PE",
            train_idx,
            test_idx,
            y_train,
        ),
    ]
    metrics = pd.concat(frames, ignore_index=True)
    metrics = metrics.replace([np.inf, -np.inf], np.nan)
    metrics = metrics.dropna(
        subset=["discriminability_effect", "temporal_shift"]
    ).reset_index(drop=True)
    metrics = add_within_view_ranks(metrics)

    summary = {
        "schema_version": "1.0",
        "split": SPLIT_NAME,
        "train_scope": "train_plus_val",
        "test_scope": "fixed_future_test",
        "minimum_token_train_support": MIN_TRAIN_SUPPORT,
        "metric_policy": {
            "token_discriminability": "Cramer's V from family x token-presence chi-square",
            "token_shift": "Bernoulli Jensen-Shannon divergence, base 2",
            "numeric_discriminability": "one-way eta-squared across families",
            "numeric_shift": "two-sample Kolmogorov-Smirnov statistic",
            "pooled_scale": "within-view percentile ranks",
            "multiplicity": "Benjamini-Hochberg FDR for view-level Spearman tests",
        },
        "association": association_summary(metrics),
        "falsification_note": (
            "The mechanism predicts a positive association between discriminability "
            "and future shift, equivalently a negative association between "
            "discriminability and temporal invariance. Null, reversed, or strongly "
            "view-specific results challenge the general trade-off claim."
        ),
    }

    metrics.to_csv(
        OUT_DIR / "feature_level_metrics.csv.gz",
        index=False,
        compression="gzip",
    )
    (OUT_DIR / "feature_level_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    plot_tradeoff(metrics, OUT_DIR / "feature_level_tradeoff.png")

    print(json.dumps(summary["association"], indent=2))
    print(f"Wrote supervisor revision outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
