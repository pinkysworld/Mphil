"""Time-ordered calibration and selective-prediction policy experiment.

The historical global split contains one training role, one validation role,
and one later test role.  This experiment subdivides the validation role at a
date boundary so that calibration fitting and policy-threshold selection are
separate.  Samples tied to the original train/validation and validation/test
boundary dates are excluded from the two validation sub-roles.  The resulting
order is therefore:

    model training < calibration fitting < threshold selection < testing

The confidence threshold is selected without test labels by maximising
validation coverage subject to a pre-specified 1% selective-error target.  The
frozen threshold is then evaluated once on the later test set.  Both
uncalibrated and sigmoid-calibrated probabilities are retained so calibration,
ranking, threshold selection, and future-period policy performance remain
distinguishable.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import sklearn
from sklearn.calibration import CalibratedClassifierCV, _SigmoidCalibration
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
SPLIT_PATH = PROJECT_ROOT / "data" / "splits" / "global_chronological.json"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "calibrated_selective_policy"
REPLAY_INPUT_PATH = OUTPUT_DIR / "replay_inputs.csv.gz"
REPLAY_METADATA_PATH = OUTPUT_DIR / "replay_inputs_metadata.json"
BASE_EVIDENCE_COMMIT = "8174d6bacd675f28a2c3fe3f7cde9f186001d841"
SEED = 42
TARGET_SELECTIVE_ERROR = 0.01
MIN_SELECTION_CASES = 100


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_deterministic_gzip(path: Path, text: str) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            zipped.write(text.encode("utf-8"))


def expected_calibration_error(
    y_true: np.ndarray, probabilities: np.ndarray, classes: np.ndarray, bins: int = 10
) -> float:
    confidence = probabilities.max(axis=1)
    prediction = classes[probabilities.argmax(axis=1)]
    correct = (prediction == y_true).astype(float)
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for index in range(bins):
        if index == 0:
            mask = (confidence >= boundaries[index]) & (
                confidence <= boundaries[index + 1]
            )
        else:
            mask = (confidence > boundaries[index]) & (
                confidence <= boundaries[index + 1]
            )
        if mask.any():
            value += float(mask.mean()) * abs(
                float(correct[mask].mean()) - float(confidence[mask].mean())
            )
    return float(value)


def multiclass_brier(
    y_true: np.ndarray, probabilities: np.ndarray, classes: np.ndarray
) -> float:
    class_index = {label: index for index, label in enumerate(classes)}
    one_hot = np.zeros_like(probabilities)
    for row, label in enumerate(y_true):
        one_hot[row, class_index[label]] = 1.0
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def reliability_metrics(
    y_true: np.ndarray, probabilities: np.ndarray, classes: np.ndarray
) -> dict[str, float]:
    prediction = classes[probabilities.argmax(axis=1)]
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "macro_f1": float(
            f1_score(y_true, prediction, labels=classes, average="macro", zero_division=0)
        ),
        "ece_10_equal_width": expected_calibration_error(
            y_true, probabilities, classes
        ),
        "multiclass_brier": multiclass_brier(y_true, probabilities, classes),
        "log_loss": float(log_loss(y_true, probabilities, labels=classes)),
    }


def select_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
    target_error: float = TARGET_SELECTIVE_ERROR,
    min_cases: int = MIN_SELECTION_CASES,
) -> dict[str, float | int]:
    confidence = probabilities.max(axis=1)
    prediction = classes[probabilities.argmax(axis=1)]
    candidates: list[dict[str, float | int]] = []
    for threshold in np.unique(confidence):
        selected = confidence >= threshold
        count = int(selected.sum())
        if count < min_cases:
            continue
        error = float(np.mean(prediction[selected] != y_true[selected]))
        if error <= target_error:
            candidates.append(
                {
                    "threshold": float(threshold),
                    "n_selected": count,
                    "coverage": float(count / len(y_true)),
                    "selective_error": error,
                    "selective_accuracy": float(1.0 - error),
                }
            )
    if not candidates:
        raise RuntimeError(
            "No validation threshold met the selective-error target and minimum count."
        )
    return max(candidates, key=lambda row: (row["coverage"], -row["threshold"]))


def evaluate_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    confidence = probabilities.max(axis=1)
    prediction = classes[probabilities.argmax(axis=1)]
    selected = confidence >= threshold
    count = int(selected.sum())
    if count == 0:
        return {
            "n_selected": 0,
            "n_deferred": int(len(y_true)),
            "coverage": 0.0,
            "selective_error": 0.0,
            "selective_accuracy": 0.0,
            "selective_macro_f1": 0.0,
        }
    error = float(np.mean(prediction[selected] != y_true[selected]))
    return {
        "n_selected": count,
        "n_deferred": int(len(y_true) - count),
        "coverage": float(count / len(y_true)),
        "selective_error": error,
        "selective_accuracy": float(1.0 - error),
        "selective_macro_f1": float(
            f1_score(
                y_true[selected],
                prediction[selected],
                labels=classes,
                average="macro",
                zero_division=0,
            )
        ),
    }


def strict_validation_roles(
    labels: pd.DataFrame, split: dict[str, object]
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    train = labels.iloc[split["train"]]
    validation = labels.iloc[split["val"]].copy()
    test = labels.iloc[split["test"]]
    train_end = train["date"].max()
    test_start = test["date"].min()

    validation["row_index"] = validation.index
    eligible = validation[
        (validation["date"] > train_end) & (validation["date"] < test_start)
    ].sort_values(["date", "sha256"], kind="mergesort")
    if eligible.empty:
        raise RuntimeError("No strictly ordered validation samples remain.")

    counts = eligible.groupby("date", sort=True).size()
    midpoint = len(eligible) / 2
    cumulative = counts.cumsum()
    policy_start = cumulative.index[int(np.argmax(cumulative.to_numpy() >= midpoint))]
    calibration = eligible[eligible["date"] < policy_start]
    policy = eligible[eligible["date"] >= policy_start]
    if calibration.empty or policy.empty:
        raise RuntimeError("Could not split validation at a strict date boundary.")

    metadata = {
        "train_end": str(train_end.date()),
        "calibration_start": str(calibration["date"].min().date()),
        "calibration_end": str(calibration["date"].max().date()),
        "policy_selection_start": str(policy["date"].min().date()),
        "policy_selection_end": str(policy["date"].max().date()),
        "test_start": str(test_start.date()),
        "excluded_validation_boundary_ties": int(len(validation) - len(eligible)),
        "n_calibration": int(len(calibration)),
        "n_policy_selection": int(len(policy)),
    }
    return (
        calibration["row_index"].to_numpy(dtype=int),
        policy["row_index"].to_numpy(dtype=int),
        metadata,
    )


def fit_sigmoid_probabilities(
    calibration_scores: np.ndarray,
    calibration_truth: np.ndarray,
    target_scores: np.ndarray,
    classes: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, float | str]]]:
    """Fit the same one-vs-rest Platt mapping used by sklearn 1.4.2.

    The public replay bundle stores the base model's decision scores. Keeping
    calibration fitting in this script means the published experiment still
    separates calibration fitting from policy selection and future testing;
    it does not merely replay already calibrated probabilities.
    """

    calibration_scores = np.asarray(calibration_scores, dtype=float)
    target_scores = np.asarray(target_scores, dtype=float)
    if calibration_scores.ndim != 2 or target_scores.ndim != 2:
        raise ValueError("Expected two-dimensional multiclass decision scores.")
    if calibration_scores.shape[1] != len(classes):
        raise ValueError("Calibration score columns do not match the class order.")
    if target_scores.shape[1] != len(classes):
        raise ValueError("Target score columns do not match the class order.")

    probabilities = np.zeros_like(target_scores, dtype=float)
    parameters: list[dict[str, float | str]] = []
    for class_index, class_name in enumerate(classes):
        calibrator = _SigmoidCalibration()
        binary_truth = (calibration_truth == class_name).astype(int)
        calibrator.fit(calibration_scores[:, class_index], binary_truth)
        probabilities[:, class_index] = calibrator.predict(
            target_scores[:, class_index]
        )
        parameters.append(
            {
                "class": str(class_name),
                "a": float(calibrator.a_),
                "b": float(calibrator.b_),
            }
        )

    denominator = probabilities.sum(axis=1, keepdims=True)
    uniform = np.full_like(probabilities, 1.0 / len(classes))
    probabilities = np.divide(
        probabilities,
        denominator,
        out=uniform,
        where=denominator != 0,
    )
    probabilities[(1.0 < probabilities) & (probabilities <= 1.0 + 1e-5)] = 1.0
    return probabilities, parameters


def score_columns(prefix: str, classes: np.ndarray) -> list[str]:
    return [f"{prefix}__{class_name}" for class_name in classes]


def source_cache_paths() -> tuple[Path, Path, Path]:
    return (
        SPLIT_PATH,
        CACHE_DIR / "labels.parquet",
        CACHE_DIR / "api_tfidf_global_chronological.npz",
    )


def build_public_replay_bundle() -> None:
    """Build the compact, Git-trackable input bundle from the defended cache."""

    split_path, labels_path, features_path = source_cache_paths()
    missing = [path for path in (split_path, labels_path, features_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Source-cache mode requires the defended local inputs: "
            + ", ".join(str(path) for path in missing)
        )

    labels_path = CACHE_DIR / "labels.parquet"
    features_path = CACHE_DIR / "api_tfidf_global_chronological.npz"
    labels = pd.read_parquet(labels_path)
    labels["date"] = pd.to_datetime(labels["date"], errors="raise")
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    features = sp.load_npz(features_path)

    calibration_idx, policy_idx, role_metadata = strict_validation_roles(labels, split)
    train_idx = np.asarray(split["train"], dtype=int)
    test_idx = np.asarray(split["test"], dtype=int)
    classes = np.sort(labels["family"].unique())

    calibration_families = set(labels.iloc[calibration_idx]["family"])
    missing_calibration = sorted(set(classes) - calibration_families)
    if missing_calibration:
        raise RuntimeError(
            f"Calibration role lacks classes required by the fitted model: {missing_calibration}"
        )

    model = SGDClassifier(
        loss="log_loss",
        class_weight="balanced",
        max_iter=1000,
        random_state=SEED,
    )
    model.fit(features[train_idx], labels.iloc[train_idx]["family"].to_numpy())
    calibrated = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
    calibrated.fit(
        features[calibration_idx], labels.iloc[calibration_idx]["family"].to_numpy()
    )

    if not np.array_equal(model.classes_, calibrated.classes_):
        raise RuntimeError("Class order differs between base and calibrated models.")
    classes = model.classes_
    role_indices = {
        "calibration": calibration_idx,
        "policy_selection": policy_idx,
        "future_test": test_idx,
    }
    frames: list[pd.DataFrame] = []
    decision_columns = score_columns("decision", classes)
    probability_columns = score_columns("probability", classes)
    for role, indices in role_indices.items():
        frame = labels.iloc[indices][["sha256", "date", "family"]].copy()
        frame.insert(0, "source_row_index", np.asarray(indices, dtype=int))
        frame.insert(1, "role", role)
        frame = frame.rename(columns={"family": "true_family"})
        frame["date"] = frame["date"].dt.strftime("%Y-%m-%d")
        decisions = np.asarray(model.decision_function(features[indices]), dtype=float)
        uncalibrated = np.asarray(model.predict_proba(features[indices]), dtype=float)
        if decisions.shape != uncalibrated.shape:
            raise RuntimeError("Decision-score and probability shapes differ.")
        for column_index, column in enumerate(decision_columns):
            frame[column] = decisions[:, column_index]
        for column_index, column in enumerate(probability_columns):
            frame[column] = uncalibrated[:, column_index]
        frames.append(frame)

    bundle = pd.concat(frames, ignore_index=True)
    write_deterministic_gzip(
        REPLAY_INPUT_PATH,
        bundle.to_csv(index=False, lineterminator="\n", float_format="%.17g"),
    )

    calibration_frame = bundle[bundle["role"] == "calibration"]
    calibration_scores = calibration_frame[decision_columns].to_numpy(dtype=float)
    calibration_truth = calibration_frame["true_family"].to_numpy()
    maximum_difference = 0.0
    sigmoid_parameters: list[dict[str, float | str]] | None = None
    for role, indices in {
        "policy_selection": policy_idx,
        "future_test": test_idx,
    }.items():
        role_frame = bundle[bundle["role"] == role]
        replay_probabilities, parameters = fit_sigmoid_probabilities(
            calibration_scores,
            calibration_truth,
            role_frame[decision_columns].to_numpy(dtype=float),
            classes,
        )
        reference_probabilities = calibrated.predict_proba(features[indices])
        maximum_difference = max(
            maximum_difference,
            float(np.max(np.abs(replay_probabilities - reference_probabilities))),
        )
        if sigmoid_parameters is None:
            sigmoid_parameters = parameters
        elif sigmoid_parameters != parameters:
            raise RuntimeError("Sigmoid parameters changed between replay roles.")
    if maximum_difference > 1e-12:
        raise RuntimeError(
            "Replay calibration does not reproduce CalibratedClassifierCV: "
            f"maximum absolute difference {maximum_difference:.3g}"
        )

    role_metadata = {
        **role_metadata,
        "n_model_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
    }
    metadata = {
        "schema_version": "1.0",
        "base_evidence_commit": BASE_EVIDENCE_COMMIT,
        "classes": [str(value) for value in classes],
        "roles": role_metadata,
        "model": {
            "type": "sklearn.linear_model.SGDClassifier",
            "loss": "log_loss",
            "class_weight": "balanced",
            "max_iter": 1000,
            "random_state": SEED,
            "sklearn_version": sklearn.__version__,
        },
        "calibration": {
            "method": "one-vs-rest sigmoid",
            "parameters": sigmoid_parameters,
            "reference_maximum_absolute_difference": maximum_difference,
        },
        "source_cache": [
            {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "sha256": sha256_file(path),
            }
            for path in (split_path, labels_path, features_path)
        ],
        "bundle": {
            "path": str(REPLAY_INPUT_PATH.relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(REPLAY_INPUT_PATH),
            "rows": int(len(bundle)),
            "decision_score_columns": decision_columns,
            "uncalibrated_probability_columns": probability_columns,
        },
    }
    write_json(REPLAY_METADATA_PATH, metadata)


def load_public_replay_bundle() -> tuple[
    pd.DataFrame, dict[str, object], np.ndarray, list[str], list[str]
]:
    if not REPLAY_INPUT_PATH.is_file() or not REPLAY_METADATA_PATH.is_file():
        raise FileNotFoundError(
            "The public replay bundle is missing. Run source-cache mode in the "
            "defended environment or retrieve the tracked artifact files."
        )
    metadata = json.loads(REPLAY_METADATA_PATH.read_text(encoding="utf-8"))
    expected_hash = metadata["bundle"]["sha256"]
    actual_hash = sha256_file(REPLAY_INPUT_PATH)
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"Replay-bundle checksum mismatch: expected {expected_hash}, got {actual_hash}"
        )
    bundle = pd.read_csv(
        REPLAY_INPUT_PATH,
        compression="gzip",
        dtype={"sha256": "string", "role": "string", "true_family": "string"},
        float_precision="round_trip",
    )
    classes = np.asarray(metadata["classes"], dtype=str)
    decision_columns = list(metadata["bundle"]["decision_score_columns"])
    probability_columns = list(
        metadata["bundle"]["uncalibrated_probability_columns"]
    )
    required_columns = {
        "source_row_index",
        "role",
        "sha256",
        "date",
        "true_family",
        *decision_columns,
        *probability_columns,
    }
    missing_columns = sorted(required_columns.difference(bundle.columns))
    if missing_columns:
        raise RuntimeError(f"Replay bundle lacks columns: {missing_columns}")
    expected_roles = {
        "calibration": int(metadata["roles"]["n_calibration"]),
        "policy_selection": int(metadata["roles"]["n_policy_selection"]),
        "future_test": int(metadata["roles"]["n_test"]),
    }
    observed_roles = bundle["role"].value_counts().to_dict()
    if observed_roles != expected_roles:
        raise RuntimeError(
            f"Replay role counts differ: expected {expected_roles}, got {observed_roles}"
        )
    return bundle, metadata, classes, decision_columns, probability_columns


def run_public_replay() -> None:
    bundle, metadata, classes, decision_columns, probability_columns = (
        load_public_replay_bundle()
    )
    calibration_frame = bundle[bundle["role"] == "calibration"].copy()
    policy_frame = bundle[bundle["role"] == "policy_selection"].copy()
    test_frame = bundle[bundle["role"] == "future_test"].copy()

    calibration_scores = calibration_frame[decision_columns].to_numpy(dtype=float)
    calibration_truth = calibration_frame["true_family"].to_numpy(dtype=str)
    policy_scores = policy_frame[decision_columns].to_numpy(dtype=float)
    test_scores = test_frame[decision_columns].to_numpy(dtype=float)
    calibrated_policy, fitted_parameters = fit_sigmoid_probabilities(
        calibration_scores, calibration_truth, policy_scores, classes
    )
    calibrated_test, repeated_parameters = fit_sigmoid_probabilities(
        calibration_scores, calibration_truth, test_scores, classes
    )
    if fitted_parameters != repeated_parameters:
        raise RuntimeError("Sigmoid parameters changed between replay targets.")
    recorded_parameters = metadata["calibration"]["parameters"]
    for fitted, recorded in zip(fitted_parameters, recorded_parameters):
        if fitted["class"] != recorded["class"]:
            raise RuntimeError("Recorded and fitted sigmoid class orders differ.")
        if abs(float(fitted["a"]) - float(recorded["a"])) > 1e-12:
            raise RuntimeError("Recorded and fitted sigmoid slopes differ.")
        if abs(float(fitted["b"]) - float(recorded["b"])) > 1e-12:
            raise RuntimeError("Recorded and fitted sigmoid intercepts differ.")

    y_policy = policy_frame["true_family"].to_numpy(dtype=str)
    y_test = test_frame["true_family"].to_numpy(dtype=str)
    probabilities = {
        "uncalibrated": {
            "policy": policy_frame[probability_columns].to_numpy(dtype=float),
            "test": test_frame[probability_columns].to_numpy(dtype=float),
        },
        "sigmoid_calibrated": {
            "policy": calibrated_policy,
            "test": calibrated_test,
        },
    }

    threshold_rows = []
    metrics: dict[str, object] = {
        "schema_version": "1.0",
        "seed": SEED,
        "target_selective_error": TARGET_SELECTIVE_ERROR,
        "minimum_policy_selection_cases": MIN_SELECTION_CASES,
        "roles": metadata["roles"],
        "methods": {},
        "claim_boundary": {
            "threshold_selected_without_test_labels": True,
            "strict_date_order_after_boundary_tie_exclusion": True,
            "single_corpus": True,
            "prospective_external_validation": False,
            "analyst_outcome_validation": False,
        },
    }

    prediction_frame = test_frame[["sha256", "date", "true_family"]].copy()

    for method, role_probabilities in probabilities.items():
        policy_probability = role_probabilities["policy"]
        test_probability = role_probabilities["test"]
        selected = select_threshold(y_policy, policy_probability, classes)
        test_policy = evaluate_threshold(
            y_test, test_probability, classes, float(selected["threshold"])
        )
        policy_reliability = reliability_metrics(y_policy, policy_probability, classes)
        test_reliability = reliability_metrics(y_test, test_probability, classes)
        metrics["methods"][method] = {
            "policy_selection_reliability": policy_reliability,
            "test_reliability": test_reliability,
            "selected_threshold": selected,
            "future_test_policy": test_policy,
        }
        threshold_rows.append(
            {
                "method": method,
                "threshold": selected["threshold"],
                "selection_n": selected["n_selected"],
                "selection_coverage": selected["coverage"],
                "selection_error": selected["selective_error"],
                "test_n": test_policy["n_selected"],
                "test_deferred": test_policy["n_deferred"],
                "test_coverage": test_policy["coverage"],
                "test_error": test_policy["selective_error"],
                "test_accuracy": test_policy["selective_accuracy"],
                "test_macro_f1": test_policy["selective_macro_f1"],
                "policy_ece": policy_reliability["ece_10_equal_width"],
                "test_ece": test_reliability["ece_10_equal_width"],
                "test_brier": test_reliability["multiclass_brier"],
                "test_log_loss": test_reliability["log_loss"],
            }
        )

        confidence = test_probability.max(axis=1)
        prediction = classes[test_probability.argmax(axis=1)]
        prediction_frame[f"{method}_prediction"] = prediction
        prediction_frame[f"{method}_confidence"] = confidence
        prediction_frame[f"{method}_selected"] = confidence >= float(
            selected["threshold"]
        )

    metrics_path = OUTPUT_DIR / "metrics.json"
    thresholds_path = OUTPUT_DIR / "policy_thresholds.csv"
    predictions_path = OUTPUT_DIR / "test_policy_predictions.csv.gz"
    write_json(metrics_path, metrics)
    pd.DataFrame(threshold_rows).to_csv(
        thresholds_path, index=False, lineterminator="\n", float_format="%.10f"
    )
    write_deterministic_gzip(
        predictions_path,
        prediction_frame.to_csv(index=False, lineterminator="\n", float_format="%.10f"),
    )

    manifest = {
        "schema_version": "1.0",
        "inputs": [
            {
                "path": str(REPLAY_INPUT_PATH.relative_to(PROJECT_ROOT)),
                "sha256": sha256_file(REPLAY_INPUT_PATH),
            },
            {
                "path": str(REPLAY_METADATA_PATH.relative_to(PROJECT_ROOT)),
                "sha256": sha256_file(REPLAY_METADATA_PATH),
            },
            {
                "path": str(Path(__file__).resolve().relative_to(PROJECT_ROOT)),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
        ],
        "outputs": [
            {"path": str(metrics_path.relative_to(PROJECT_ROOT)), "sha256": sha256_file(metrics_path)},
            {"path": str(thresholds_path.relative_to(PROJECT_ROOT)), "sha256": sha256_file(thresholds_path)},
            {"path": str(predictions_path.relative_to(PROJECT_ROOT)), "sha256": sha256_file(predictions_path)},
        ],
    }
    manifest_path = OUTPUT_DIR / "manifest.json"
    write_json(manifest_path, manifest)

    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"Outputs: {OUTPUT_DIR}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-mode",
        choices=("auto", "source-cache", "replay-bundle"),
        default="auto",
        help=(
            "Use defended local feature caches, the tracked replay bundle, or "
            "auto-select source caches when all are present."
        ),
    )
    return parser.parse_args()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    arguments = parse_args()
    source_available = all(path.is_file() for path in source_cache_paths())
    if arguments.input_mode == "source-cache" or (
        arguments.input_mode == "auto" and source_available
    ):
        build_public_replay_bundle()
    elif arguments.input_mode == "auto" and not REPLAY_INPUT_PATH.is_file():
        raise FileNotFoundError(
            "Neither the defended source caches nor the public replay bundle are available."
        )
    run_public_replay()


if __name__ == "__main__":
    main()
