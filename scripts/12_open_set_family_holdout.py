"""
12_open_set_family_holdout.py — Held-out-family open-set evaluation.

For each malware family, hold that family out of training entirely, train on the
remaining families with a chronological split, and evaluate:

  - closed-set performance on future known-family samples
  - open-set unknown detection on future held-out-family samples
  - thresholded open-set performance using max probability and margin scores

Outputs:
  - artifacts/open_set/heldout_family_summary.csv
  - artifacts/open_set/heldout_family_thresholds.csv
  - artifacts/open_set/heldout_family_summary.json
  - artifacts/open_set/heldout_family_plot.png
  - artifacts/open_set/README.md
"""

import argparse
import importlib.util
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler

from run_experiment import PROJECT_ROOT, build_model, combine_views, resolve_views


PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
OUT_DIR = PROJECT_ROOT / "artifacts" / "open_set"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FRAC = 0.80
VAL_FRAC_OF_TRAIN = 0.10
DEFAULT_VIEW_REQUESTS = ("api_tfidf", "fusion")
TOKEN_VIEWS = {"api_tfidf", "art_tfidf"}
NUMERIC_VIEWS = {"counts", "pe"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a held-out-family open-set evaluation on future samples."
    )
    parser.add_argument(
        "--views",
        nargs="+",
        default=list(DEFAULT_VIEW_REQUESTS),
        help="View requests to evaluate, e.g. api_tfidf fusion.",
    )
    parser.add_argument(
        "--model",
        default="sgd",
        choices=["sgd", "logistic"],
        help="Linear baseline to train for the held-out-family protocol.",
    )
    parser.add_argument(
        "--target-known-retention",
        type=float,
        default=0.95,
        help="Validation retention target used to set the open-set threshold.",
    )
    parser.add_argument(
        "--min-unknown-test",
        type=int,
        default=25,
        help="Skip a held-out family if too few post-boundary unknown samples remain.",
    )
    return parser.parse_args()


def load_extract_module():
    script_path = PROJECT_ROOT / "scripts" / "04_extract_features.py"
    spec = importlib.util.spec_from_file_location("extract_features", script_path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ImportError(f"Unable to load feature extractor module from {script_path}")
    spec.loader.exec_module(module)
    return module


def load_metadata():
    meta = pd.read_parquet(PROCESSED_DIR / "metadata.parquet")
    meta = meta[meta["has_report"] == True].reset_index(drop=True)
    meta["date"] = pd.to_datetime(meta["date"])
    return meta


def load_static_frames():
    counts_df = pd.read_parquet(CACHE_DIR / "counts_raw.parquet").fillna(0)
    pe_df = pd.read_parquet(CACHE_DIR / "pe_raw.parquet")
    return counts_df, pe_df


def extract_token_docs(meta, need_api, need_art):
    extractor_module = load_extract_module()
    api_docs = None
    art_docs = None
    if not (need_api or need_art):
        return api_docs, art_docs

    api_docs = [] if need_api else None
    art_docs = [] if need_art else None

    report_dir = PROJECT_ROOT / "data" / "raw" / "reports"

    for idx, row in meta.iterrows():
        report_path = Path(row["report_path"])
        if not report_path.exists():
            report_path = report_dir / f"{row['sha256']}.json"
        try:
            with open(report_path, "r", encoding="utf-8", errors="replace") as handle:
                report = json.load(handle)
        except (OSError, json.JSONDecodeError):
            report = {}
        if need_api:
            api_docs.append(extractor_module.extract_api_tokens(report))
        if need_art:
            art_docs.append(extractor_module.extract_artifact_tokens(report))
        if (idx + 1) % 5000 == 0:
            print(f"  extracted token docs for {idx + 1} / {len(meta)} reports")

    return api_docs, art_docs


def build_known_split(meta, heldout_family):
    known_meta = meta[meta["family"] != heldout_family].sort_values("date")
    split_idx = int(len(known_meta) * TRAIN_FRAC)
    boundary_date = known_meta.iloc[split_idx]["date"]

    known_train_all = known_meta.index[:split_idx].tolist()
    known_test = known_meta.index[split_idx:].tolist()
    val_size = max(1, int(len(known_train_all) * VAL_FRAC_OF_TRAIN))
    known_train = known_train_all[:-val_size]
    known_val = known_train_all[-val_size:]

    unknown_test = meta[
        (meta["family"] == heldout_family) & (meta["date"] >= boundary_date)
    ].index.tolist()

    return {
        "boundary_date": boundary_date,
        "train": known_train,
        "val": known_val,
        "known_test": known_test,
        "unknown_test": unknown_test,
    }


def fit_component_matrix(view_name, train_idx, api_docs, art_docs, counts_df, pe_df):
    if view_name == "api_tfidf":
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=50000,
            min_df=2,
            sublinear_tf=True,
        )
        vectorizer.fit([api_docs[i] for i in train_idx])
        return vectorizer.transform(api_docs)

    if view_name == "art_tfidf":
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 1),
            max_features=50000,
            min_df=2,
            sublinear_tf=True,
        )
        vectorizer.fit([art_docs[i] for i in train_idx])
        return vectorizer.transform(art_docs)

    if view_name == "counts":
        filled = counts_df.fillna(0)
        scaler = StandardScaler()
        scaler.fit(filled.iloc[train_idx])
        return scaler.transform(filled)

    if view_name == "pe":
        filled = pe_df.fillna(pe_df.iloc[train_idx].median())
        scaler = StandardScaler()
        scaler.fit(filled.iloc[train_idx])
        return scaler.transform(filled)

    raise ValueError(f"Unsupported component view: {view_name}")


def derive_scores(probabilities, method):
    if method == "max_prob":
        return probabilities.max(axis=1)
    sorted_probs = np.sort(probabilities, axis=1)
    if sorted_probs.shape[1] == 1:
        return sorted_probs[:, -1]
    return sorted_probs[:, -1] - sorted_probs[:, -2]


def quantile_threshold(scores, target_known_retention):
    quantile = max(0.0, min(1.0, 1.0 - target_known_retention))
    return float(np.quantile(scores, quantile))


def safe_auc(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan")
    return float(roc_auc_score(y_true, y_score)), float(average_precision_score(y_true, y_score))


def write_readme(path, view_requests, model_name, summary_df, target_known_retention):
    lines = [
        "# Held-out-family open-set evaluation",
        "",
        f"Model: `{model_name}`",
        f"Views: `{', '.join(view_requests)}`",
        f"Validation known-retention target: `{target_known_retention}`",
        "",
        "Files:",
        "",
        "- `heldout_family_summary.csv`",
        "- `heldout_family_thresholds.csv`",
        "- `heldout_family_summary.json`",
        "- `heldout_family_plot.png`",
        "",
    ]

    if not summary_df.empty:
        max_prob_df = summary_df[summary_df["score_method"] == "max_prob"]
        if not max_prob_df.empty:
            means = (
                max_prob_df.groupby("view_request")[
                    ["open_set_macro_f1", "unknown_rejection_rate", "known_acceptance_rate"]
                ]
                .mean()
                .round(4)
            )
            lines.append("Mean results for max-probability thresholding:")
            lines.append("")
            for view_request, row in means.iterrows():
                lines.append(
                    f"- `{view_request}`: open-set macro-F1={row['open_set_macro_f1']}, "
                    f"unknown rejection={row['unknown_rejection_rate']}, "
                    f"known acceptance={row['known_acceptance_rate']}"
                )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    args = parse_args()
    meta = load_metadata()
    counts_df, pe_df = load_static_frames()

    view_requests = args.views
    component_views = []
    for view_request in view_requests:
        for component in resolve_views(view_request):
            if component not in component_views:
                component_views.append(component)

    need_api = "api_tfidf" in component_views
    need_art = "art_tfidf" in component_views
    api_docs, art_docs = extract_token_docs(meta, need_api, need_art)

    families = sorted(meta["family"].unique())
    summary_rows = []
    threshold_rows = []

    for heldout_family in families:
        print("=" * 60)
        print(f"Held-out family: {heldout_family}")
        split = build_known_split(meta, heldout_family)
        if len(split["unknown_test"]) < args.min_unknown_test:
            print(
                f"  skip: only {len(split['unknown_test'])} post-boundary unknown samples"
            )
            continue

        component_cache = {}
        for component in component_views:
            print(f"  fitting component view: {component}")
            component_cache[component] = fit_component_matrix(
                component,
                split["train"],
                api_docs,
                art_docs,
                counts_df,
                pe_df,
            )

        for view_request in view_requests:
            views = resolve_views(view_request)
            X = combine_views([component_cache[view] for view in views])
            known_families = sorted(f for f in families if f != heldout_family)

            label_encoder = LabelEncoder()
            label_encoder.fit(known_families)

            X_train = X[split["train"]]
            X_val = X[split["val"]]
            X_known_test = X[split["known_test"]]
            X_unknown_test = X[split["unknown_test"]]

            y_train = label_encoder.transform(meta.loc[split["train"], "family"].values)
            y_val = label_encoder.transform(meta.loc[split["val"], "family"].values)
            y_known_test = meta.loc[split["known_test"], "family"].values

            model = build_model(args.model)
            model.fit(X_train, y_train)

            val_prob = model.predict_proba(X_val)
            known_test_prob = model.predict_proba(X_known_test)
            unknown_test_prob = model.predict_proba(X_unknown_test)

            known_test_pred = label_encoder.inverse_transform(model.predict(X_known_test))
            closed_known_macro_f1 = f1_score(
                y_known_test,
                known_test_pred,
                average="macro",
                zero_division=0,
            )
            closed_known_accuracy = accuracy_score(y_known_test, known_test_pred)

            for score_method in ("max_prob", "margin"):
                val_scores = derive_scores(val_prob, score_method)
                threshold = quantile_threshold(
                    val_scores, args.target_known_retention
                )
                threshold_rows.append(
                    {
                        "heldout_family": heldout_family,
                        "view_request": view_request,
                        "score_method": score_method,
                        "threshold": round(threshold, 6),
                        "val_mean_score": round(float(np.mean(val_scores)), 6),
                        "val_min_score": round(float(np.min(val_scores)), 6),
                        "val_max_score": round(float(np.max(val_scores)), 6),
                        "target_known_retention": args.target_known_retention,
                    }
                )

                known_test_scores = derive_scores(known_test_prob, score_method)
                unknown_test_scores = derive_scores(unknown_test_prob, score_method)
                unknown_test_pred = label_encoder.inverse_transform(
                    model.predict(X_unknown_test)
                )

                known_open_pred = np.where(
                    known_test_scores < threshold, "unknown", known_test_pred
                )
                unknown_open_pred = np.where(
                    unknown_test_scores < threshold, "unknown", unknown_test_pred
                )

                y_open_true = np.concatenate(
                    [
                        y_known_test,
                        np.repeat("unknown", len(split["unknown_test"])),
                    ]
                )
                y_open_pred = np.concatenate([known_open_pred, unknown_open_pred])
                open_labels = known_families + ["unknown"]

                unknown_indicator = np.concatenate(
                    [
                        np.zeros(len(split["known_test"]), dtype=int),
                        np.ones(len(split["unknown_test"]), dtype=int),
                    ]
                )
                unknown_scores = np.concatenate(
                    [-known_test_scores, -unknown_test_scores]
                )
                auroc, auprc = safe_auc(unknown_indicator, unknown_scores)

                summary_rows.append(
                    {
                        "heldout_family": heldout_family,
                        "view_request": view_request,
                        "model": args.model,
                        "score_method": score_method,
                        "boundary_date": split["boundary_date"].strftime("%Y-%m-%d"),
                        "n_train": len(split["train"]),
                        "n_val": len(split["val"]),
                        "n_known_test": len(split["known_test"]),
                        "n_unknown_test": len(split["unknown_test"]),
                        "closed_known_accuracy": round(float(closed_known_accuracy), 4),
                        "closed_known_macro_f1": round(float(closed_known_macro_f1), 4),
                        "open_set_accuracy": round(
                            float(accuracy_score(y_open_true, y_open_pred)), 4
                        ),
                        "open_set_macro_f1": round(
                            float(
                                f1_score(
                                    y_open_true,
                                    y_open_pred,
                                    labels=open_labels,
                                    average="macro",
                                    zero_division=0,
                                )
                            ),
                            4,
                        ),
                        "known_acceptance_rate": round(
                            float(np.mean(known_open_pred != "unknown")), 4
                        ),
                        "unknown_rejection_rate": round(
                            float(np.mean(unknown_open_pred == "unknown")), 4
                        ),
                        "unknown_detection_auroc": round(auroc, 4),
                        "unknown_detection_auprc": round(auprc, 4),
                    }
                )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["view_request", "score_method", "heldout_family"]
    )
    thresholds_df = pd.DataFrame(threshold_rows).sort_values(
        ["view_request", "score_method", "heldout_family"]
    )

    summary_df.to_csv(OUT_DIR / "heldout_family_summary.csv", index=False)
    thresholds_df.to_csv(OUT_DIR / "heldout_family_thresholds.csv", index=False)

    aggregate = []
    if not summary_df.empty:
        grouped = summary_df.groupby(["view_request", "score_method"])
        for (view_request, score_method), group in grouped:
            aggregate.append(
                {
                    "view_request": view_request,
                    "score_method": score_method,
                    "mean_open_set_macro_f1": round(
                        float(group["open_set_macro_f1"].mean()), 4
                    ),
                    "mean_unknown_rejection_rate": round(
                        float(group["unknown_rejection_rate"].mean()), 4
                    ),
                    "mean_known_acceptance_rate": round(
                        float(group["known_acceptance_rate"].mean()), 4
                    ),
                    "mean_unknown_detection_auroc": round(
                        float(group["unknown_detection_auroc"].mean()), 4
                    ),
                }
            )

    payload = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "view_requests": view_requests,
        "target_known_retention": args.target_known_retention,
        "n_families_evaluated": int(summary_df["heldout_family"].nunique())
        if not summary_df.empty
        else 0,
        "aggregate": aggregate,
    }
    with open(OUT_DIR / "heldout_family_summary.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    plot_df = summary_df[summary_df["score_method"] == "max_prob"].copy()
    if not plot_df.empty:
        fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
        for view_request in plot_df["view_request"].unique():
            subset = plot_df[plot_df["view_request"] == view_request]
            axes[0].plot(
                subset["heldout_family"],
                subset["unknown_rejection_rate"],
                marker="o",
                label=view_request,
            )
            axes[1].plot(
                subset["heldout_family"],
                subset["open_set_macro_f1"],
                marker="o",
                label=view_request,
            )
        axes[0].set_ylabel("Unknown Rejection")
        axes[0].set_title("Held-out family open-set summary (max probability)")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        axes[0].set_ylim(0, 1.05)
        axes[1].set_ylabel("Open-set Macro-F1")
        axes[1].set_xlabel("Held-out family")
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0, 1.05)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "heldout_family_plot.png", dpi=200)
        plt.close(fig)

    write_readme(
        OUT_DIR / "README.md",
        view_requests,
        args.model,
        summary_df,
        args.target_known_retention,
    )
    print(f"[ok] held-out-family summary written to {OUT_DIR}")


if __name__ == "__main__":
    main()
