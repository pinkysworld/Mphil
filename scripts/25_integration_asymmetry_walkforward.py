"""Route-B exploratory test of integration asymmetry under temporal drift.

The analysis operationalises H5 as a lagged, window-level hypothesis:
when the predictive contribution of the four feature views is more concentrated
in one or a few views at time t, the fused classifier is expected to suffer a
larger loss of Macro-F1 in the next eligible time window.

This is an exploratory theory test, not a replacement for the primary H1-H4
experiments. It uses leakage-controlled raw feature checkpoints produced by
04_extract_features.py and a fixed, fit-free hashing representation for token
views so that every temporal window uses the same representational geometry.

For each rolling-origin window the script fits:
  * full four-view fusion;
  * four leave-one-view-out fusion models.

For view i, positive marginal contribution is
  max(0, F1_full - F1_without_i).
These positive contributions are normalised to sum to one. Integration
asymmetry is the normalised Herfindahl concentration relative to a uniform
four-view reference. Negative marginal contributions are retained separately
rather than silently clipped away.

H5 predicts a positive association between asymmetry at window t and
next-window performance decay F1_t - F1_(t+1). Because adjacent windows are
temporally dependent, uncertainty uses a moving-block bootstrap and a circular
shift permutation test rather than treating windows as IID observations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import MaxAbsScaler, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_FEATURE_DIR = PROJECT_ROOT / "data" / "cache" / "raw_feature_chunks"
OUT_DIR = PROJECT_ROOT / "artifacts" / "route_b_integration_asymmetry"

VIEWS = ("api", "artifact", "counts", "pe")
DEFAULT_SEED = 42
DEFAULT_START_DATE = "2018-06-01"
DEFAULT_DELTA_DAYS = 30
DEFAULT_MIN_TRAIN = 500
DEFAULT_MIN_TEST = 50
DEFAULT_HASH_DIM = 32768


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start-date", default=DEFAULT_START_DATE)
    p.add_argument("--delta-days", type=int, default=DEFAULT_DELTA_DAYS)
    p.add_argument("--min-train-samples", type=int, default=DEFAULT_MIN_TRAIN)
    p.add_argument("--min-test-samples", type=int, default=DEFAULT_MIN_TEST)
    p.add_argument("--hash-dim", type=int, default=DEFAULT_HASH_DIM)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--permutations", type=int, default=2000)
    return p.parse_args()


def load_raw_features(meta: pd.DataFrame):
    chunks = sorted(RAW_FEATURE_DIR.glob("features_*.parquet"))
    if not chunks:
        raise FileNotFoundError(
            "No raw feature checkpoints found. Run scripts/04_extract_features.py --extract-only first."
        )
    frame = pd.concat([pd.read_parquet(p) for p in chunks], ignore_index=True)
    if len(frame) != len(meta):
        raise ValueError(f"Raw-feature row count {len(frame)} != metadata row count {len(meta)}")
    if frame["sha256"].astype(str).tolist() != meta["sha256"].astype(str).tolist():
        raise ValueError("Raw-feature checkpoints are not aligned with metadata order.")
    api_docs = frame["api_doc"].fillna("").astype(str).tolist()
    art_docs = frame["art_doc"].fillna("").astype(str).tolist()
    counts = pd.DataFrame([json.loads(v) for v in frame["counts_json"]]).fillna(0.0)
    pe = pd.DataFrame([json.loads(v) for v in frame["pe_json"]])
    return api_docs, art_docs, counts, pe


def build_token_matrices(api_docs, art_docs, dim: int):
    common = dict(
        analyzer="word",
        lowercase=True,
        token_pattern=r"(?u)\b\w\w+\b",
        n_features=dim,
        alternate_sign=False,
        norm="l2",
    )
    api = HashingVectorizer(ngram_range=(1, 2), **common).transform(api_docs).tocsr()
    art = HashingVectorizer(ngram_range=(1, 1), **common).transform(art_docs).tocsr()
    return api, art


def scale_numeric(train_idx, all_counts, all_pe):
    counts_scaler = MaxAbsScaler()
    counts_scaler.fit(all_counts.iloc[train_idx])
    counts_scaled = counts_scaler.transform(all_counts)

    train_medians = all_pe.iloc[train_idx].median(numeric_only=True)
    pe_filled = all_pe.fillna(train_medians).fillna(0.0)
    pe_scaler = StandardScaler()
    pe_scaler.fit(pe_filled.iloc[train_idx])
    pe_scaled = pe_scaler.transform(pe_filled)
    return sp.csr_matrix(counts_scaled), sp.csr_matrix(pe_scaled)


def fit_f1(parts, active_views, train_idx, test_idx, y, seed):
    matrices = [parts[v] for v in active_views]
    x = matrices[0] if len(matrices) == 1 else sp.hstack(matrices, format="csr")
    clf = SGDClassifier(
        loss="log_loss",
        class_weight="balanced",
        max_iter=1000,
        random_state=seed,
        tol=1e-3,
    )
    clf.fit(x[train_idx], y[train_idx])
    pred = clf.predict(x[test_idx])
    return float(f1_score(y[test_idx], pred, average="macro"))


def normalised_hhi(weights: np.ndarray) -> float:
    k = len(weights)
    raw = float(np.sum(np.square(weights)))
    uniform = 1.0 / k
    return float((raw - uniform) / (1.0 - uniform))


def contribution_metrics(full_f1: float, loo: dict[str, float]):
    signed = {v: float(full_f1 - loo[v]) for v in VIEWS}
    positive = np.array([max(0.0, signed[v]) for v in VIEWS], dtype=float)
    negative_burden = float(sum(max(0.0, -signed[v]) for v in VIEWS))
    if positive.sum() <= 0:
        weights = np.full(len(VIEWS), 1.0 / len(VIEWS))
        asymmetry = 0.0
        no_positive = True
    else:
        weights = positive / positive.sum()
        asymmetry = normalised_hhi(weights)
        no_positive = False
    return signed, weights, asymmetry, negative_burden, no_positive


def eligible_windows(meta: pd.DataFrame, args):
    boundary = pd.Timestamp(args.start_date)
    delta = pd.Timedelta(days=args.delta_days)
    latest = meta["date"].max()
    while boundary + delta <= latest:
        end = boundary + delta
        tr = np.flatnonzero((meta["date"] < boundary).to_numpy())
        te = np.flatnonzero(((meta["date"] >= boundary) & (meta["date"] < end)).to_numpy())
        if len(tr) >= args.min_train_samples and len(te) >= args.min_test_samples:
            yield boundary, end, tr, te
        boundary += delta


def moving_block_bootstrap(x, y, reps, rng, block=3):
    n = len(x)
    vals = []
    starts = np.arange(n)
    for _ in range(reps):
        idx = []
        while len(idx) < n:
            s = int(rng.choice(starts))
            idx.extend([(s + j) % n for j in range(block)])
        idx = np.asarray(idx[:n], dtype=int)
        rho = spearmanr(x[idx], y[idx]).statistic
        if np.isfinite(rho):
            vals.append(float(rho))
    if not vals:
        return [None, None]
    return [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]


def circular_shift_p(x, y, observed, reps, rng):
    n = len(y)
    if n < 4:
        return None
    vals = []
    legal = np.arange(1, n)
    for _ in range(reps):
        shift = int(rng.choice(legal))
        rho = spearmanr(x, np.roll(y, shift)).statistic
        if np.isfinite(rho):
            vals.append(float(rho))
    return float((1 + sum(v >= observed for v in vals)) / (len(vals) + 1))


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    meta = pd.read_parquet(PROCESSED_DIR / "metadata.parquet")
    meta = meta[meta["has_report"]].reset_index(drop=True)
    meta["date"] = pd.to_datetime(meta["date"], errors="raise")
    y = meta["family"].astype(str).to_numpy()

    api_docs, art_docs, counts, pe = load_raw_features(meta)
    api_x, art_x = build_token_matrices(api_docs, art_docs, args.hash_dim)

    rows = []
    for boundary, end, train_idx, test_idx in eligible_windows(meta, args):
        print(f"Window {boundary.date()} -> {end.date()} train={len(train_idx)} test={len(test_idx)}")
        counts_x, pe_x = scale_numeric(train_idx, counts, pe)
        parts = {"api": api_x, "artifact": art_x, "counts": counts_x, "pe": pe_x}

        full = fit_f1(parts, VIEWS, train_idx, test_idx, y, args.seed)
        loo = {}
        for omitted in VIEWS:
            active = tuple(v for v in VIEWS if v != omitted)
            loo[omitted] = fit_f1(parts, active, train_idx, test_idx, y, args.seed)

        signed, weights, asym, neg, no_pos = contribution_metrics(full, loo)
        row = {
            "boundary": str(boundary.date()),
            "test_end": str(end.date()),
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "fusion_macro_f1": full,
            "integration_asymmetry": asym,
            "negative_contribution_burden": neg,
            "no_positive_marginal_contribution": no_pos,
        }
        for i, v in enumerate(VIEWS):
            row[f"loo_without_{v}_macro_f1"] = loo[v]
            row[f"marginal_contribution_{v}"] = signed[v]
            row[f"positive_contribution_weight_{v}"] = float(weights[i])
        rows.append(row)

    df = pd.DataFrame(rows)
    if len(df) < 5:
        raise RuntimeError(f"Only {len(df)} eligible windows; H5 requires at least five.")
    df["next_window_fusion_macro_f1"] = df["fusion_macro_f1"].shift(-1)
    df["next_window_decay"] = df["fusion_macro_f1"] - df["next_window_fusion_macro_f1"]
    paired = df.iloc[:-1].copy()

    x = paired["integration_asymmetry"].to_numpy(float)
    y_decay = paired["next_window_decay"].to_numpy(float)
    rho = float(spearmanr(x, y_decay).statistic)
    rng = np.random.default_rng(20260830)
    ci = moving_block_bootstrap(x, y_decay, args.bootstrap, rng)
    p = circular_shift_p(x, y_decay, rho, args.permutations, rng)

    summary = {
        "schema_version": "1.0",
        "status": "exploratory_route_b_test",
        "hypothesis_h5": (
            "Greater concentration of positive marginal view contributions at window t "
            "is associated with larger fused-model Macro-F1 decay in the next eligible window."
        ),
        "n_windows": int(len(df)),
        "n_lagged_pairs": int(len(paired)),
        "association": {
            "spearman_rho_asymmetry_vs_next_window_decay": rho,
            "moving_block_bootstrap_95_ci": ci,
            "circular_shift_one_sided_p": p,
            "bootstrap_replicates": args.bootstrap,
            "permutation_replicates": args.permutations,
            "block_length": 3,
        },
        "asymmetry_definition": (
            "Normalised Herfindahl concentration of non-negative leave-one-view-out "
            "marginal Macro-F1 contributions, relative to uniform four-view contribution."
        ),
        "negative_contribution_policy": (
            "Negative marginal contributions are excluded from the normalised positive weights "
            "and reported separately as negative_contribution_burden."
        ),
        "interpretation_boundary": (
            "This is a lagged exploratory test using overlapping expanding training windows. "
            "It does not establish causality and should be replicated cross-corpus before a "
            "general theory claim. A null or negative association weakens H5."
        ),
        "configuration": vars(args),
    }

    df.to_csv(OUT_DIR / "integration_asymmetry_windows.csv", index=False)
    (OUT_DIR / "integration_asymmetry_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
