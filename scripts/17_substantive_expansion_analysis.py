"""Derive auditable analyses used in the substantive thesis expansion.

The script deliberately reuses archived, sample-level predictions. It does not
retrain a model or alter any published experiment. Its outputs support three
questions that cannot be answered by marginal summary metrics alone:

1. Which families account for the paired error changes between models?
2. Which families deteriorate during the rolling-origin evaluation?
3. How sensitive is the illustrative retraining policy to its thresholds?

Outputs are written to ``artifacts/thesis_expansion``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import precision_recall_fscore_support

from run_experiment import PROJECT_ROOT


OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "thesis_expansion"
PREDICTION_DIR = PROJECT_ROOT / "artifacts" / "predictions"
RESULTS_DIR = PROJECT_ROOT / "results" / "2026-07-11"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_path(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def load_predictions(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as archive:
        required = {"y_true", "y_pred", "test_indices", "label_classes"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
        return {name: archive[name] for name in required}


def validate_alignment(
    reference: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    name: str,
) -> None:
    for key in ("y_true", "test_indices", "label_classes"):
        if not np.array_equal(reference[key], candidate[key]):
            raise ValueError(f"Prediction archive {name!r} is not aligned on {key!r}.")


def exact_mcnemar(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
) -> dict[str, float | int]:
    correct_a = pred_a == y_true
    correct_b = pred_b == y_true
    a_correct_b_wrong = int(np.sum(correct_a & ~correct_b))
    a_wrong_b_correct = int(np.sum(~correct_a & correct_b))
    discordant = a_correct_b_wrong + a_wrong_b_correct
    p_value = (
        float(
            binomtest(
                a_correct_b_wrong,
                discordant,
                0.5,
                alternative="two-sided",
            ).pvalue
        )
        if discordant
        else 1.0
    )
    return {
        "a_correct_b_wrong": a_correct_b_wrong,
        "a_wrong_b_correct": a_wrong_b_correct,
        "discordant_pairs": discordant,
        "two_sided_exact_p": p_value,
    }


def paired_error_rows(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    classes: np.ndarray,
    comparison: str,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    correct_a = pred_a == y_true
    correct_b = pred_b == y_true
    for label_id, family in enumerate(classes):
        mask = y_true == label_id
        a_to_b_gain = int(np.sum(mask & ~correct_a & correct_b))
        a_to_b_loss = int(np.sum(mask & correct_a & ~correct_b))
        rows.append(
            {
                "comparison": comparison,
                "family": str(family),
                "support": int(mask.sum()),
                "model_a_correct": int(np.sum(mask & correct_a)),
                "model_b_correct": int(np.sum(mask & correct_b)),
                "a_wrong_b_correct": a_to_b_gain,
                "a_correct_b_wrong": a_to_b_loss,
                "net_correct_change_b_minus_a": a_to_b_gain - a_to_b_loss,
            }
        )
    return pd.DataFrame(rows)


def build_paired_analyses() -> tuple[pd.DataFrame, dict[str, object]]:
    paths = {
        "api_sgd": PREDICTION_DIR / "table_5_8.npz",
        "fusion_sgd": PREDICTION_DIR / "table_5_11_fusion_global_sgd.npz",
        "fusion_lightgbm": PREDICTION_DIR / "table_6_1_fusion_global_lightgbm.npz",
    }
    archives = {name: load_predictions(path) for name, path in paths.items()}
    reference = archives["api_sgd"]
    for name, archive in archives.items():
        validate_alignment(reference, archive, name)

    y_true = reference["y_true"]
    classes = reference["label_classes"]
    comparisons = [
        ("api_sgd", "fusion_sgd", "API-only SGD to fused SGD"),
        ("fusion_sgd", "fusion_lightgbm", "fused SGD to fused LightGBM"),
    ]
    frames = []
    comparison_summaries: dict[str, object] = {}
    for model_a, model_b, label in comparisons:
        frames.append(
            paired_error_rows(
                y_true,
                archives[model_a]["y_pred"],
                archives[model_b]["y_pred"],
                classes,
                label,
            )
        )
        comparison_summaries[label] = {
            "model_a": model_a,
            "model_b": model_b,
            **exact_mcnemar(
                y_true,
                archives[model_a]["y_pred"],
                archives[model_b]["y_pred"],
            ),
        }

    summary: dict[str, object] = {
        "alignment": {
            "n_samples": int(len(y_true)),
            "test_indices_equal": True,
            "reference_labels_equal": True,
            "label_classes_equal": True,
            "label_classes": [str(value) for value in classes],
        },
        "comparisons": comparison_summaries,
        "sources": {
            name: {
                "path": relative_path(path),
                "sha256": sha256_file(path),
            }
            for name, path in paths.items()
        },
    }
    return pd.concat(frames, ignore_index=True), summary


def family_metric_frame(
    y_true: pd.Series,
    y_pred: pd.Series,
    families: list[str],
) -> pd.DataFrame:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=families,
        zero_division=0,
    )
    return pd.DataFrame(
        {
            "family": families,
            "support": support.astype(int),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )


def build_walk_forward_analyses() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions_path = RESULTS_DIR / "walk_forward" / "walk_forward_predictions.csv.gz"
    results_path = RESULTS_DIR / "walk_forward" / "walk_forward_results.csv"
    predictions = pd.read_csv(predictions_path, dtype=str)
    results = pd.read_csv(results_path)
    required = {"boundary", "true_family", "predicted_family"}
    if missing := required.difference(predictions.columns):
        raise ValueError(f"Walk-forward predictions are missing columns: {sorted(missing)}")

    with np.load(PREDICTION_DIR / "table_5_8.npz", allow_pickle=True) as archive:
        families = [str(value) for value in archive["label_classes"]]

    observed_boundaries = predictions["boundary"].drop_duplicates().tolist()
    expected_boundaries = results["boundary"].astype(str).tolist()
    if observed_boundaries != expected_boundaries:
        raise ValueError("Walk-forward prediction and result boundaries differ in order.")
    observed_counts = predictions.groupby("boundary", sort=False).size().to_numpy()
    if not np.array_equal(observed_counts, results["n_test"].to_numpy()):
        raise ValueError("Walk-forward prediction counts do not match n_test.")

    pooled = family_metric_frame(
        predictions["true_family"],
        predictions["predicted_family"],
        families,
    )

    window_frames = []
    for boundary, group in predictions.groupby("boundary", sort=False):
        metrics = family_metric_frame(
            group["true_family"],
            group["predicted_family"],
            families,
        )
        metrics.insert(0, "boundary", boundary)
        window_frames.append(metrics)
    by_window = pd.concat(window_frames, ignore_index=True)

    result_lookup = results.set_index("boundary")
    error_rows: list[dict[str, float | int | str]] = []
    for boundary, group in predictions.groupby("boundary", sort=False):
        result = result_lookup.loc[boundary]
        if float(result["api_macro_f1"]) >= 0.75:
            continue
        errors = group.loc[
            group["true_family"] != group["predicted_family"],
            ["true_family", "predicted_family"],
        ]
        counts = (
            errors.value_counts(sort=False)
            .rename("count")
            .reset_index()
            .sort_values(
                ["count", "true_family", "predicted_family"],
                ascending=[False, True, True],
                kind="mergesort",
            )
            .head(4)
            .reset_index(drop=True)
        )
        n_errors = int(len(errors))
        for rank, row in counts.iterrows():
            count = int(row["count"])
            error_rows.append(
                {
                    "boundary": boundary,
                    "n_test": int(result["n_test"]),
                    "n_errors": n_errors,
                    "api_macro_f1": float(result["api_macro_f1"]),
                    "rank": rank + 1,
                    "true_family": row["true_family"],
                    "predicted_family": row["predicted_family"],
                    "count": count,
                    "share_of_window_errors": count / n_errors,
                }
            )
    return pooled, by_window, pd.DataFrame(error_rows)


def episode_starts(flags: pd.Series, boundaries: pd.Series) -> list[str]:
    starts = flags & ~flags.shift(fill_value=False)
    return boundaries.loc[starts].astype(str).tolist()


def build_trigger_sensitivity() -> pd.DataFrame:
    results_path = RESULTS_DIR / "walk_forward" / "walk_forward_results.csv"
    results = pd.read_csv(results_path)
    scores = results["api_macro_f1"].astype(float)
    trailing_mean = scores.shift(1).rolling(window=3, min_periods=3).mean()
    rows = []
    for floor in (0.70, 0.75, 0.80):
        for drop_margin in (0.05, 0.10, 0.15):
            floor_trigger = scores < floor
            drop_trigger = (trailing_mean - scores) >= drop_margin
            flags = floor_trigger | drop_trigger.fillna(False)
            starts = episode_starts(flags, results["boundary"])
            rows.append(
                {
                    "floor": floor,
                    "drop_margin": drop_margin,
                    "trigger_windows": int(flags.sum()),
                    "episodes": len(starts),
                    "recommended_review_boundaries": ";".join(starts),
                }
            )
    return pd.DataFrame(rows)


def write_outputs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paired_rows, paired_summary = build_paired_analyses()
    pooled, by_window, top_errors = build_walk_forward_analyses()
    sensitivity = build_trigger_sensitivity()

    paired_rows.to_csv(OUTPUT_DIR / "paired_error_transitions.csv", index=False)
    (OUTPUT_DIR / "paired_model_summary.json").write_text(
        json.dumps(paired_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    pooled.to_csv(
        OUTPUT_DIR / "walk_forward_per_family.csv",
        index=False,
        float_format="%.10g",
    )
    by_window.to_csv(
        OUTPUT_DIR / "walk_forward_family_window.csv",
        index=False,
        float_format="%.10g",
    )
    top_errors.to_csv(
        OUTPUT_DIR / "walk_forward_top_error_pairs.csv",
        index=False,
        float_format="%.10g",
    )
    sensitivity.to_csv(
        OUTPUT_DIR / "retraining_trigger_sensitivity.csv",
        index=False,
        float_format="%.10g",
    )


if __name__ == "__main__":
    write_outputs()
