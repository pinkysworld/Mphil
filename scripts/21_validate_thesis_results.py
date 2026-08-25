"""Independently validate the prediction archives used by the thesis.

The script recomputes classification metrics from sample-level predictions,
checks split and label alignment against the public reproducibility catalogue,
verifies the tracked confusion matrices and paired error transitions, and
derives the forecast-horizon table from an archived global API-hashing run.

Run without flags for a read-only validation. Use ``--write-derived`` only
when intentionally refreshing the deterministic CSV/JSON validation outputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = PROJECT_ROOT / "results" / "2026-07-11"
PREDICTION_ROOT = PROJECT_ROOT / "artifacts" / "predictions"
CATALOGUE_PATH = RESULT_ROOT / "reproducibility" / "sample_catalog.csv.gz"
ASSIGNMENTS_PATH = RESULT_ROOT / "reproducibility" / "split_assignments.csv.gz"

PREDICTION_MANIFEST_PATH = PREDICTION_ROOT / "manifest.json"
VIEW_SUMMARY_PATH = RESULT_ROOT / "metrics" / "view_ablation_replay_summary.csv"
FORECAST_PATH = RESULT_ROOT / "forecast_horizon" / "api_hash_global_horizons.csv"
VALIDATION_REPORT_PATH = (
    RESULT_ROOT / "reproducibility" / "deep_validation_report.json"
)


RUNS = {
    "baseline_random_api_hash_sgd": {
        "prediction": "baseline_random_api_hash_sgd.npz",
        "split": "random_stratified",
        "view": "api_hash",
        "model": "sgd",
        "accuracy": 0.9089,
        "macro_f1": 0.8718,
        "weighted_f1": 0.9076,
        "n_test": 9796,
        "confusion": "baseline_random_api_hash_sgd_confusion.csv",
    },
    "table_5_8": {
        "prediction": "table_5_8.npz",
        "split": "global_chronological",
        "view": "api_tfidf",
        "model": "sgd",
        "accuracy": 0.9573,
        "macro_f1": 0.9024,
        "weighted_f1": 0.9602,
        "n_test": 9796,
        "confusion": "table_5_8_confusion.csv",
    },
    "global_art_tfidf_sgd": {
        "prediction": "global_art_tfidf_sgd.npz",
        "split": "global_chronological",
        "view": "art_tfidf",
        "model": "sgd",
        "accuracy": 0.9408,
        "macro_f1": 0.8872,
        "weighted_f1": 0.9503,
        "n_test": 9796,
    },
    "global_counts_logistic": {
        "prediction": "global_counts_logistic.npz",
        "split": "global_chronological",
        "view": "counts",
        "model": "logistic",
        "accuracy": 0.4600,
        "macro_f1": 0.3964,
        "weighted_f1": 0.5418,
        "n_test": 9796,
    },
    "global_pe_logistic": {
        "prediction": "global_pe_logistic.npz",
        "split": "global_chronological",
        "view": "pe",
        "model": "logistic",
        "accuracy": 0.4135,
        "macro_f1": 0.2017,
        "weighted_f1": 0.3849,
        "n_test": 9796,
    },
    "table_5_11_fusion_global_sgd": {
        "prediction": "table_5_11_fusion_global_sgd.npz",
        "split": "global_chronological",
        "view": "fusion",
        "model": "sgd",
        "accuracy": 0.9606,
        "macro_f1": 0.9317,
        "weighted_f1": 0.9630,
        "n_test": 9796,
        "confusion": "table_5_11_fusion_global_sgd_confusion.csv",
    },
    "per_family_api_tfidf_sgd": {
        "prediction": "per_family_api_tfidf_sgd.npz",
        "split": "per_family_chronological",
        "view": "api_tfidf",
        "model": "sgd",
        "accuracy": 0.9550,
        "macro_f1": 0.9536,
        "weighted_f1": 0.9555,
        "n_test": 9799,
    },
    "per_family_art_tfidf_sgd": {
        "prediction": "per_family_art_tfidf_sgd.npz",
        "split": "per_family_chronological",
        "view": "art_tfidf",
        "model": "sgd",
        "accuracy": 0.9454,
        "macro_f1": 0.9348,
        "weighted_f1": 0.9475,
        "n_test": 9799,
    },
    "per_family_counts_logistic": {
        "prediction": "per_family_counts_logistic.npz",
        "split": "per_family_chronological",
        "view": "counts",
        "model": "logistic",
        "accuracy": 0.5081,
        "macro_f1": 0.4476,
        "weighted_f1": 0.5716,
        "n_test": 9799,
    },
    "per_family_pe_logistic": {
        "prediction": "per_family_pe_logistic.npz",
        "split": "per_family_chronological",
        "view": "pe",
        "model": "logistic",
        "accuracy": 0.3895,
        "macro_f1": 0.2699,
        "weighted_f1": 0.3509,
        "n_test": 9799,
    },
    "table_5_12_fusion_per_family_sgd": {
        "prediction": "table_5_12_fusion_per_family_sgd.npz",
        "split": "per_family_chronological",
        "view": "fusion",
        "model": "sgd",
        "accuracy": 0.9665,
        "macro_f1": 0.9607,
        "weighted_f1": 0.9667,
        "n_test": 9799,
        "confusion": "table_5_12_fusion_per_family_sgd_confusion.csv",
    },
    "table_6_1_fusion_global_lightgbm": {
        "prediction": "table_6_1_fusion_global_lightgbm.npz",
        "split": "global_chronological",
        "view": "fusion",
        "model": "lightgbm",
        "accuracy": 0.9459,
        "macro_f1": 0.9118,
        "weighted_f1": 0.9496,
        "n_test": 9796,
        "confusion": "table_6_1_fusion_global_lightgbm_confusion.csv",
    },
    "global_api_hash_sgd": {
        "prediction": "global_api_hash_sgd.npz",
        "split": "global_chronological",
        "view": "api_hash",
        "model": "sgd",
        "accuracy": 0.8997,
        "macro_f1": 0.7977,
        "weighted_f1": 0.8976,
        "n_test": 9796,
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rounded_equal(actual: float, expected: float) -> bool:
    return f"{actual:.4f}" == f"{expected:.4f}"


def exact_mcnemar_p(b: int, c: int) -> float:
    discordant = b + c
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(0, min(b, c) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def load_prediction(path: Path) -> dict[str, np.ndarray]:
    # The archives use an object-dtype string array for label_classes. They are
    # repository-controlled files whose SHA-256 values are checked below.
    with np.load(path, allow_pickle=True) as bundle:
        required = {"y_true", "y_pred", "test_indices", "label_classes"}
        missing = required.difference(bundle.files)
        if missing:
            raise ValueError(f"{path}: missing arrays {sorted(missing)}")
        return {name: np.asarray(bundle[name]) for name in required}


def compute_metrics(prediction: dict[str, np.ndarray]) -> dict[str, float]:
    y_true = prediction["y_true"]
    y_pred = prediction["y_pred"]
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
    }


def validate_runs(
    catalogue: pd.DataFrame, assignments: pd.DataFrame
) -> tuple[dict[str, dict], dict[str, dict[str, np.ndarray]], list[str]]:
    errors: list[str] = []
    results: dict[str, dict] = {}
    predictions: dict[str, dict[str, np.ndarray]] = {}

    catalogue_by_index = catalogue.set_index("index")
    test_membership = {
        split: set(group["index"].astype(int).tolist())
        for split, group in assignments[assignments["role"] == "test"].groupby("split")
    }

    for run_name, specification in RUNS.items():
        path = PREDICTION_ROOT / specification["prediction"]
        if not path.is_file():
            errors.append(f"missing prediction archive: {path.relative_to(PROJECT_ROOT)}")
            continue

        prediction = load_prediction(path)
        predictions[run_name] = prediction
        n_test = len(prediction["y_true"])
        if not (
            len(prediction["y_pred"])
            == len(prediction["test_indices"])
            == n_test
            == specification["n_test"]
        ):
            errors.append(f"length mismatch: {run_name}")

        indices = prediction["test_indices"].astype(int)
        if len(set(indices.tolist())) != n_test:
            errors.append(f"duplicate test indices: {run_name}")

        expected_membership = test_membership[specification["split"]]
        if set(indices.tolist()) != expected_membership:
            errors.append(f"split membership mismatch: {run_name}")

        classes = np.asarray(prediction["label_classes"], dtype=str)
        encoded_families = classes[prediction["y_true"].astype(int)]
        catalogue_families = (
            catalogue_by_index.loc[indices, "family"].astype(str).str.lower().to_numpy()
        )
        if not np.array_equal(encoded_families, catalogue_families):
            errors.append(f"reference-label mismatch: {run_name}")

        metrics = compute_metrics(prediction)
        metric_matches = {
            key: rounded_equal(metrics[key], float(specification[key]))
            for key in ("accuracy", "macro_f1", "weighted_f1")
        }
        if not all(metric_matches.values()):
            errors.append(f"metric mismatch: {run_name}")

        confusion_relative = specification.get("confusion")
        confusion_match = None
        if confusion_relative:
            confusion_path = RESULT_ROOT / "metrics" / str(confusion_relative)
            archived = pd.read_csv(confusion_path, index_col=0).to_numpy(dtype=int)
            recomputed = confusion_matrix(
                prediction["y_true"],
                prediction["y_pred"],
                labels=np.arange(len(classes)),
            )
            confusion_match = bool(np.array_equal(archived, recomputed))
            if not confusion_match:
                errors.append(f"confusion-matrix mismatch: {run_name}")

        results[run_name] = {
            "prediction_path": str(path.relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(path),
            "split": specification["split"],
            "view": specification["view"],
            "model": specification["model"],
            "n_test": n_test,
            "metrics": metrics,
            "expected_at_4dp": {
                key: specification[key]
                for key in ("accuracy", "macro_f1", "weighted_f1")
            },
            "metric_matches_at_4dp": metric_matches,
            "split_membership_match": set(indices.tolist()) == expected_membership,
            "reference_labels_match": bool(
                np.array_equal(encoded_families, catalogue_families)
            ),
            "confusion_matrix_match": confusion_match,
        }

    return results, predictions, errors


def validate_paired_transitions(
    predictions: dict[str, dict[str, np.ndarray]]
) -> tuple[dict, list[str]]:
    errors: list[str] = []
    comparisons = {
        "API-only SGD to fused SGD": (
            "table_5_8",
            "table_5_11_fusion_global_sgd",
        ),
        "fused SGD to fused LightGBM": (
            "table_5_11_fusion_global_sgd",
            "table_6_1_fusion_global_lightgbm",
        ),
    }
    archived = pd.read_csv(
        PROJECT_ROOT / "artifacts" / "thesis_expansion" / "paired_error_transitions.csv"
    )
    report: dict[str, dict] = {}

    for label, (run_a, run_b) in comparisons.items():
        a = predictions[run_a]
        b = predictions[run_b]
        for field in ("test_indices", "y_true", "label_classes"):
            if not np.array_equal(a[field], b[field]):
                errors.append(f"paired alignment mismatch ({field}): {label}")

        a_correct = a["y_pred"] == a["y_true"]
        b_correct = b["y_pred"] == b["y_true"]
        a_wrong_b_correct = int(np.sum(~a_correct & b_correct))
        a_correct_b_wrong = int(np.sum(a_correct & ~b_correct))
        classes = np.asarray(a["label_classes"], dtype=str)

        family_rows = []
        for class_index, family in enumerate(classes):
            mask = a["y_true"] == class_index
            family_rows.append(
                {
                    "family": family,
                    "support": int(mask.sum()),
                    "a_wrong_b_correct": int(np.sum(mask & ~a_correct & b_correct)),
                    "a_correct_b_wrong": int(np.sum(mask & a_correct & ~b_correct)),
                }
            )

        archived_subset = archived[archived["comparison"] == label].copy()
        for row in family_rows:
            match = archived_subset[archived_subset["family"] == row["family"]]
            if len(match) != 1:
                errors.append(f"missing archived family transition: {label}/{row['family']}")
                continue
            archived_row = match.iloc[0]
            if not (
                int(archived_row["support"]) == row["support"]
                and int(archived_row["a_wrong_b_correct"])
                == row["a_wrong_b_correct"]
                and int(archived_row["a_correct_b_wrong"])
                == row["a_correct_b_wrong"]
            ):
                errors.append(f"family transition mismatch: {label}/{row['family']}")

        report[label] = {
            "n_samples": int(len(a["y_true"])),
            "a_wrong_b_correct": a_wrong_b_correct,
            "a_correct_b_wrong": a_correct_b_wrong,
            "net_correct_change_b_minus_a": a_wrong_b_correct - a_correct_b_wrong,
            "mcnemar_two_sided_exact_p": exact_mcnemar_p(
                a_wrong_b_correct, a_correct_b_wrong
            ),
            "family_rows_match_archive": not any(
                error.startswith("family transition mismatch")
                or error.startswith("missing archived family transition")
                for error in errors
            ),
        }

    return report, errors


def derive_forecast_horizons(
    catalogue: pd.DataFrame,
    assignments: pd.DataFrame,
    prediction: dict[str, np.ndarray],
) -> tuple[list[dict], dict]:
    global_roles = assignments[
        assignments["split"] == "global_chronological"
    ].copy()
    development = global_roles[
        global_roles["role"].isin(["train", "val", "validation"])
    ]
    cut_off = pd.to_datetime(development["date"]).max()

    catalogue_by_index = catalogue.set_index("index")
    test_dates = pd.to_datetime(
        catalogue_by_index.loc[prediction["test_indices"].astype(int), "date"]
    ).reset_index(drop=True)
    horizons = (test_dates - cut_off).dt.days.to_numpy()
    y_true = prediction["y_true"]
    y_pred = prediction["y_pred"]

    bins = [
        ("0-30d", (horizons >= 0) & (horizons <= 30)),
        ("31-60d", (horizons > 30) & (horizons <= 60)),
        ("61-90d", (horizons > 60) & (horizons <= 90)),
        (">90d", horizons > 90),
    ]
    rows: list[dict] = []
    for label, mask in bins:
        if not mask.any():
            raise ValueError(f"forecast-horizon bin is empty: {label}")
        observed_classes = np.unique(y_true[mask])
        rows.append(
            {
                "horizon_bin": label,
                "n": int(mask.sum()),
                "families_present": int(len(observed_classes)),
                "accuracy": float(accuracy_score(y_true[mask], y_pred[mask])),
                "macro_f1_present_families": float(
                    f1_score(
                        y_true[mask],
                        y_pred[mask],
                        labels=observed_classes,
                        average="macro",
                        zero_division=0,
                    )
                ),
            }
        )

    tie_mask = horizons == 0
    tie_excluded_first = (horizons > 0) & (horizons <= 30)
    if sum(row["n"] for row in rows) != len(y_true):
        raise ValueError("forecast-horizon bins do not cover the complete test set")

    sensitivity = {
        "development_end_date": cut_off.date().isoformat(),
        "boundary_tie_cases_in_test": int(tie_mask.sum()),
        "first_bin_includes_horizon_zero": True,
        "tie_excluded_first_bin_n": int(tie_excluded_first.sum()),
        "tie_excluded_first_bin_macro_f1_present_families": float(
            f1_score(
                y_true[tie_excluded_first],
                y_pred[tie_excluded_first],
                labels=np.unique(y_true[tie_excluded_first]),
                average="macro",
                zero_division=0,
            )
        ),
    }
    return rows, sensitivity


def prediction_manifest(run_results: dict[str, dict]) -> dict:
    return {
        "schema_version": "1.0",
        "description": (
            "Sample-level prediction archives used to validate full-dataset thesis "
            "metrics, paired comparisons, and the forecast-horizon diagnostic."
        ),
        "outputs": [
            {
                "run": run_name,
                "path": result["prediction_path"],
                "sha256": result["sha256"],
                "split": result["split"],
                "view": result["view"],
                "model": result["model"],
                "n_test": result["n_test"],
            }
            for run_name, result in run_results.items()
        ],
    }


def view_summary_rows(run_results: dict[str, dict]) -> list[dict]:
    return [
        {
            "run": run_name,
            "view": result["view"],
            "split": result["split"],
            "model": result["model"],
            "n_test": result["n_test"],
            "accuracy": f"{result['metrics']['accuracy']:.4f}",
            "macro_f1": f"{result['metrics']['macro_f1']:.4f}",
            "weighted_f1": f"{result['metrics']['weighted_f1']:.4f}",
            "prediction_path": result["prediction_path"],
            "prediction_sha256": result["sha256"],
        }
        for run_name, result in run_results.items()
    ]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def validate_derived_file(path: Path, expected_text: str, errors: list[str]) -> None:
    if not path.is_file():
        errors.append(f"missing derived output: {path.relative_to(PROJECT_ROOT)}")
        return
    if path.read_text(encoding="utf-8") != expected_text:
        errors.append(f"stale derived output: {path.relative_to(PROJECT_ROOT)}")


def serialise_csv(rows: list[dict]) -> str:
    from io import StringIO

    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0].keys()), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-derived",
        action="store_true",
        help="Refresh the deterministic manifest, summary, horizon, and report files.",
    )
    args = parser.parse_args()

    catalogue = pd.read_csv(CATALOGUE_PATH)
    assignments = pd.read_csv(ASSIGNMENTS_PATH)
    run_results, predictions, errors = validate_runs(catalogue, assignments)
    if len(run_results) != len(RUNS):
        errors.append("not all configured runs were validated")

    paired_results, paired_errors = validate_paired_transitions(predictions)
    errors.extend(paired_errors)
    forecast_rows, forecast_sensitivity = derive_forecast_horizons(
        catalogue, assignments, predictions["global_api_hash_sgd"]
    )

    manifest = prediction_manifest(run_results)
    summary_rows = view_summary_rows(run_results)
    report = {
        "schema_version": "1.0",
        "status": "ok" if not errors else "failed",
        "scope": (
            "Independent recomputation from archived predictions; full raw-data model "
            "training remains governed by the locked environment and input manifests."
        ),
        "runs_validated": len(run_results),
        "confusion_matrices_validated": sum(
            result["confusion_matrix_match"] is True for result in run_results.values()
        ),
        "run_results": run_results,
        "paired_comparisons": paired_results,
        "forecast_horizon": {
            "rows": forecast_rows,
            "boundary_sensitivity": forecast_sensitivity,
        },
        "errors": errors,
    }

    manifest_text = json.dumps(manifest, indent=2) + "\n"
    summary_text = serialise_csv(summary_rows)
    forecast_text = serialise_csv(
        [
            {
                **{key: row[key] for key in ("horizon_bin", "n", "families_present")},
                "accuracy": f"{row['accuracy']:.4f}",
                "macro_f1_present_families": (
                    f"{row['macro_f1_present_families']:.4f}"
                ),
            }
            for row in forecast_rows
        ]
    )
    report_text = json.dumps(report, indent=2) + "\n"

    if args.write_derived:
        PREDICTION_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        PREDICTION_MANIFEST_PATH.write_text(manifest_text, encoding="utf-8")
        VIEW_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
        VIEW_SUMMARY_PATH.write_text(summary_text, encoding="utf-8")
        FORECAST_PATH.parent.mkdir(parents=True, exist_ok=True)
        FORECAST_PATH.write_text(forecast_text, encoding="utf-8")
        VALIDATION_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        VALIDATION_REPORT_PATH.write_text(report_text, encoding="utf-8")
    else:
        validate_derived_file(PREDICTION_MANIFEST_PATH, manifest_text, errors)
        validate_derived_file(VIEW_SUMMARY_PATH, summary_text, errors)
        validate_derived_file(FORECAST_PATH, forecast_text, errors)
        validate_derived_file(VALIDATION_REPORT_PATH, report_text, errors)

    final_status = "ok" if not errors else "failed"
    print(
        json.dumps(
            {
                "status": final_status,
                "runs_validated": len(run_results),
                "confusion_matrices_validated": report[
                    "confusion_matrices_validated"
                ],
                "paired_comparisons": paired_results,
                "forecast_horizon": forecast_rows,
                "errors": errors,
            },
            indent=2,
        )
    )
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
