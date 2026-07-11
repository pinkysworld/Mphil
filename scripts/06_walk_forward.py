"""Leakage-controlled expanding-window evaluation with exact manifests.

The model is retrained from scratch on all reports before each monthly
boundary and evaluated on the immediately following window. API documents are
created by the same shared extractor used by ``04_extract_features.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score

from feature_extraction import extract_api_tokens, extractor_policy
from reproducibility import (
    environment_manifest,
    file_record,
    hashing_vectorizer_manifest,
    ordered_rows_sha256,
    sha256_json,
    write_deterministic_gzip,
    write_json,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_OUT_DIR = PROJECT_ROOT / "artifacts" / "walk_forward"

DEFAULT_SEED = 42
DEFAULT_DELTA_DAYS = 30
DEFAULT_MIN_TRAIN_SAMPLES = 500
DEFAULT_MIN_TEST_SAMPLES = 50
DEFAULT_N_FEATURES = 131072
DEFAULT_WORD_TOKEN_PATTERN = r"(?u)\b\w\w+\b"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=PROCESSED_DIR / "metadata.parquet",
        help="Processed metadata parquet produced by scripts/01_ingest.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory for metrics, predictions, plot, and manifests.",
    )
    parser.add_argument("--start-date", default="2018-06-01")
    parser.add_argument(
        "--end-date",
        default=None,
        help="Optional inclusive corpus end date. Defaults to the latest sample date.",
    )
    parser.add_argument("--delta-days", type=int, default=DEFAULT_DELTA_DAYS)
    parser.add_argument(
        "--min-train-samples", type=int, default=DEFAULT_MIN_TRAIN_SAMPLES
    )
    parser.add_argument(
        "--min-test-samples", type=int, default=DEFAULT_MIN_TEST_SAMPLES
    )
    parser.add_argument("--n-features", type=int, default=DEFAULT_N_FEATURES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def build_hasher(n_features: int) -> HashingVectorizer:
    """Build the explicit historical hashing configuration."""
    return HashingVectorizer(
        analyzer="word",
        lowercase=True,
        token_pattern=DEFAULT_WORD_TOKEN_PATTERN,
        n_features=n_features,
        ngram_range=(1, 2),
        alternate_sign=False,
        norm="l2",
    )


def load_api_document(report_path) -> str:
    """Load one reduced report and apply the shared defended extractor."""
    try:
        with Path(report_path).open("r", encoding="utf-8", errors="replace") as handle:
            report = json.load(handle)
    except (OSError, json.JSONDecodeError, TypeError):
        return ""
    return extract_api_tokens(report)


def prepare_metadata(path: Path) -> pd.DataFrame:
    meta = pd.read_parquet(path)
    meta = meta[meta["has_report"]].reset_index(drop=True)
    required = {"sha256", "family", "date", "report_path"}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(f"Metadata is missing required columns: {sorted(missing)}")
    if not meta["sha256"].astype(str).str.fullmatch(r"[0-9a-fA-F]{64}").all():
        raise ValueError("Metadata contains an invalid SHA-256 sample identifier.")
    meta["sha256"] = meta["sha256"].str.lower()
    meta["date"] = pd.to_datetime(meta["date"], errors="raise")
    # Explicit quicksort preserves the ordering behaviour of the archived run
    # while the environment lock fixes the pandas implementation used.
    return meta.sort_values("date", kind="quicksort").reset_index(drop=True)


def frame_order_hash(frame: pd.DataFrame) -> str:
    return ordered_rows_sha256(
        f"{row.sha256},{row.family},{row.date.isoformat()}"
        for row in frame.itertuples(index=False)
    )


def evaluate(meta: pd.DataFrame, args) -> tuple[pd.DataFrame, pd.DataFrame, list, dict]:
    print("Extracting API documents through the shared leakage-controlled extractor...")
    api_documents = []
    for position, report_path in enumerate(meta["report_path"], start=1):
        api_documents.append(load_api_document(report_path))
        if position % 5000 == 0:
            print(f"  Extracted {position} / {len(meta)}")
    meta = meta.copy()
    meta["api_document"] = api_documents

    start_date = pd.Timestamp(args.start_date)
    latest_date = meta["date"].max()
    end_date = pd.Timestamp(args.end_date) if args.end_date else latest_date
    if end_date > latest_date:
        raise ValueError("The requested end date is after the latest corpus sample.")

    hasher = build_hasher(args.n_features)
    model_configuration = {
        "class": "sklearn.linear_model.SGDClassifier",
        "loss": "log_loss",
        "class_weight": "balanced",
        "max_iter": 1000,
        "random_state": args.seed,
    }

    results = []
    predictions = []
    window_manifests = []
    boundary = start_date
    delta = pd.Timedelta(days=args.delta_days)

    while boundary + delta <= end_date:
        test_end = boundary + delta
        train_mask = meta["date"] < boundary
        test_mask = (meta["date"] >= boundary) & (meta["date"] < test_end)
        train_df = meta[train_mask]
        test_df = meta[test_mask]

        if len(train_df) < args.min_train_samples or len(test_df) < args.min_test_samples:
            boundary += delta
            continue

        print(
            f"  Window {boundary.date()} to {test_end.date()}: "
            f"train={len(train_df)}, test={len(test_df)}, "
            f"families={test_df['family'].nunique()}"
        )

        x_train = hasher.transform(train_df["api_document"])
        x_test = hasher.transform(test_df["api_document"])
        y_train = train_df["family"].to_numpy()
        y_test = test_df["family"].to_numpy()

        classifier = SGDClassifier(
            loss="log_loss",
            class_weight="balanced",
            max_iter=1000,
            random_state=args.seed,
        )
        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)
        macro_f1 = f1_score(y_test, y_pred, average="macro")
        print(f"    API macro-F1: {macro_f1:.4f}")

        train_hash = frame_order_hash(train_df)
        test_hash = frame_order_hash(test_df)
        result = {
            "boundary": str(boundary.date()),
            "test_start": str(boundary.date()),
            "test_end": str(test_end.date()),
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            "families_present": int(test_df["family"].nunique()),
            "api_macro_f1": float(macro_f1),
            "train_order_sha256": train_hash,
            "test_order_sha256": test_hash,
        }
        results.append(result)
        window_manifests.append(
            {
                **{key: result[key] for key in ["boundary", "test_start", "test_end", "n_train", "n_test"]},
                "train_order_sha256": train_hash,
                "test_order_sha256": test_hash,
            }
        )

        for sample, predicted in zip(test_df.itertuples(index=False), y_pred):
            predictions.append(
                {
                    "boundary": str(boundary.date()),
                    "sha256": sample.sha256,
                    "date": sample.date.strftime("%Y-%m-%d"),
                    "true_family": sample.family,
                    "predicted_family": str(predicted),
                }
            )

        boundary += delta

    if not results:
        raise RuntimeError("No walk-forward windows met the configured size thresholds.")

    extraction_state = {
        "ordered_api_documents_sha256": ordered_rows_sha256(meta["api_document"]),
        "ordered_corpus_sha256": frame_order_hash(meta),
    }
    return (
        pd.DataFrame(results),
        pd.DataFrame(predictions),
        window_manifests,
        {
            "hasher": hashing_vectorizer_manifest(hasher),
            "model": model_configuration,
            "model_configuration_sha256": sha256_json(model_configuration),
            "extraction": extraction_state,
        },
    )


def save_outputs(results, predictions, windows, run_state, meta, args) -> Path:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "walk_forward_results.csv"
    results.to_csv(results_path, index=False, lineterminator="\n")

    predictions_payload = predictions.to_csv(index=False, lineterminator="\n").encode("utf-8")
    predictions_path = output_dir / "walk_forward_predictions.csv.gz"
    write_deterministic_gzip(predictions_path, predictions_payload)

    catalog = meta[["sha256", "family", "date"]].copy()
    catalog.insert(0, "sorted_index", range(len(catalog)))
    catalog["date"] = catalog["date"].dt.strftime("%Y-%m-%d")
    catalog_payload = catalog.to_csv(index=False, lineterminator="\n").encode("utf-8")
    catalog_path = output_dir / "walk_forward_sample_catalog.csv.gz"
    write_deterministic_gzip(catalog_path, catalog_payload)

    plot_path = output_dir / "walk_forward_plot.png"
    figure, axis = plt.subplots(1, 1, figsize=(12, 5))
    axis.plot(
        pd.to_datetime(results["boundary"]),
        results["api_macro_f1"],
        marker="o",
        markersize=4,
        label="API tokens",
    )
    axis.set_xlabel("Training boundary date")
    axis.set_ylabel("Macro-F1")
    axis.set_title("Walk-forward macro-F1 (API-only, monthly windows)")
    axis.legend()
    axis.grid(True, alpha=0.3)
    axis.set_ylim(0, 1.05)
    figure.tight_layout()
    figure.savefig(plot_path, dpi=200)
    plt.close(figure)

    manifest = {
        "schema_version": "1.0",
        "protocol": {
            "type": "expanding_window_rolling_origin",
            "start_date": args.start_date,
            "end_date": args.end_date or str(meta["date"].max().date()),
            "delta_days": args.delta_days,
            "min_train_samples": args.min_train_samples,
            "min_test_samples": args.min_test_samples,
        },
        "extractor": extractor_policy(),
        "run_state": run_state,
        "windows": windows,
        "environment": environment_manifest(PROJECT_ROOT),
        "inputs": {
            "metadata": file_record(args.metadata, PROJECT_ROOT),
        },
        "outputs": [
            file_record(results_path, PROJECT_ROOT),
            file_record(predictions_path, PROJECT_ROOT),
            file_record(catalog_path, PROJECT_ROOT),
            file_record(plot_path, PROJECT_ROOT),
        ],
    }
    manifest_path = output_dir / "walk_forward_manifest.json"
    write_json(manifest_path, manifest)
    return manifest_path


def main():
    args = parse_args()
    print("=" * 60)
    print("Leakage-Controlled Walk-Forward Evaluation")
    print("=" * 60)
    meta = prepare_metadata(args.metadata)
    results, predictions, windows, run_state = evaluate(meta, args)
    manifest_path = save_outputs(results, predictions, windows, run_state, meta, args)
    print(f"\nResults saved to {args.output_dir}")
    print(f"Windows evaluated: {len(results)}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
