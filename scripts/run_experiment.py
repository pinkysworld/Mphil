"""
run_experiment.py — Train and evaluate a cached-feature baseline.

Loads one or more cached feature views, fits a baseline classifier on the
requested split, and writes:
  - metrics JSON
  - classification report CSV
  - confusion matrix CSV
  - confusion matrix PNG

Examples:
  .venv/bin/python scripts/run_experiment.py \
    --view api_tfidf \
    --split global_chronological \
    --model sgd \
    --output artifacts/metrics/table_5_8.json

  .venv/bin/python scripts/run_experiment.py \
    --view fusion \
    --split per_family_chronological \
    --model logistic \
    --train-scope train_plus_val \
    --output artifacts/metrics/table_5_12.json
"""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import LabelEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

SEED = 42
FUSION_VIEWS = ["api_tfidf", "art_tfidf", "counts", "pe"]
VIEW_ALIASES = {
    "api": "api_tfidf",
    "artifacts": "art_tfidf",
    "artifacts_tfidf": "art_tfidf",
    "counts_scaled": "counts",
    "pe_scaled": "pe",
}
VALID_VIEWS = {
    "api_tfidf",
    "api_hash",
    "art_tfidf",
    "counts",
    "pe",
    "fusion",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate a baseline on cached features."
    )
    parser.add_argument(
        "--view",
        required=True,
        help=(
            "Feature view name or a comma/plus-separated combination. "
            "Examples: api_tfidf, counts, api_tfidf+counts, fusion"
        ),
    )
    parser.add_argument(
        "--split",
        required=True,
        help="Split name, e.g. random_stratified or global_chronological.",
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["sgd", "logistic", "lightgbm"],
        help="Baseline model to fit.",
    )
    parser.add_argument(
        "--train-scope",
        default="train_plus_val",
        choices=["train_only", "train_plus_val"],
        help="Whether to fit on train only or train plus validation.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the metrics JSON output file.",
    )
    parser.add_argument(
        "--model-output",
        help="Optional path for a saved model bundle (.pkl).",
    )
    return parser.parse_args()


def resolve_views(raw_view):
    parts = [p.strip().lower() for p in re.split(r"[,+]", raw_view) if p.strip()]
    if not parts:
        raise ValueError("No valid view names were provided.")

    resolved = []
    for part in parts:
        canonical = VIEW_ALIASES.get(part, part)
        if canonical == "fusion":
            resolved.extend(FUSION_VIEWS)
            continue
        if canonical not in VALID_VIEWS:
            raise ValueError(
                f"Unsupported view '{part}'. Supported values: "
                f"{', '.join(sorted(VALID_VIEWS))}"
            )
        resolved.append(canonical)

    # Preserve order while removing duplicates.
    ordered = []
    seen = set()
    for view in resolved:
        if view not in seen:
            ordered.append(view)
            seen.add(view)
    return ordered


def load_labels():
    labels_path = CACHE_DIR / "labels.parquet"
    if labels_path.exists():
        return pd.read_parquet(labels_path)

    metadata_path = PROCESSED_DIR / "metadata.parquet"
    if not metadata_path.exists():
        raise FileNotFoundError(
            "Missing labels.parquet and metadata.parquet. "
            "Run scripts/01_ingest.py first."
        )

    meta = pd.read_parquet(metadata_path)
    meta = meta[meta["has_report"] == True].reset_index(drop=True)
    return meta[["sha256", "family", "date"]]


def load_split(split_name):
    path = SPLITS_DIR / f"{split_name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Split file not found: {path}. Run scripts/03_build_splits.py first."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_view_matrix(view_name, split_name):
    if view_name == "api_tfidf":
        path = CACHE_DIR / f"api_tfidf_{split_name}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing cache file: {path}")
        return sp.load_npz(path)

    if view_name == "api_hash":
        path = CACHE_DIR / f"api_hash_{split_name}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing cache file: {path}")
        return sp.load_npz(path)

    if view_name == "art_tfidf":
        path = CACHE_DIR / f"art_tfidf_{split_name}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing cache file: {path}")
        return sp.load_npz(path)

    if view_name == "counts":
        path = CACHE_DIR / f"counts_scaled_{split_name}.npy"
        if not path.exists():
            raise FileNotFoundError(f"Missing cache file: {path}")
        return np.load(path)

    if view_name == "pe":
        path = CACHE_DIR / f"pe_scaled_{split_name}.npy"
        if not path.exists():
            raise FileNotFoundError(f"Missing cache file: {path}")
        return np.load(path)

    raise ValueError(f"Unsupported view name: {view_name}")


def combine_views(matrices):
    if len(matrices) == 1:
        return matrices[0]

    any_sparse = any(sp.issparse(m) for m in matrices)
    if any_sparse:
        sparse_parts = []
        for matrix in matrices:
            if sp.issparse(matrix):
                sparse_parts.append(matrix.tocsr())
            else:
                sparse_parts.append(sp.csr_matrix(np.asarray(matrix)))
        return sp.hstack(sparse_parts).tocsr()

    return np.hstack([np.asarray(m) for m in matrices])


def build_model(model_name):
    if model_name == "sgd":
        return SGDClassifier(
            loss="log_loss",
            class_weight="balanced",
            max_iter=1000,
            random_state=SEED,
        )

    if model_name == "logistic":
        return LogisticRegression(
            solver="saga",
            class_weight="balanced",
            max_iter=2000,
            random_state=SEED,
        )

    if model_name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise ImportError(
                "lightgbm is not installed in the current environment."
            ) from exc
        return LGBMClassifier(
            objective="multiclass",
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
            verbose=-1,
        )

    raise ValueError(f"Unsupported model: {model_name}")


def ensure_parent(path):
    path.parent.mkdir(parents=True, exist_ok=True)


def derive_side_paths(output_path):
    stem = output_path.stem
    return {
        "report_csv": output_path.with_name(f"{stem}_report.csv"),
        "confusion_csv": output_path.with_name(f"{stem}_confusion.csv"),
        "confusion_png": output_path.with_name(f"{stem}_confusion.png"),
    }


def plot_confusion(cm, labels, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    threshold = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    views = resolve_views(args.view)
    split = load_split(args.split)
    labels_df = load_labels()

    matrices = [load_view_matrix(view_name, args.split) for view_name in views]
    X = combine_views(matrices)
    y = labels_df["family"].values

    if X.shape[0] != len(y):
        raise ValueError(
            f"Feature row count ({X.shape[0]}) does not match labels ({len(y)})."
        )

    train_idx = split["train"]
    val_idx = split.get("val", [])
    test_idx = split["test"]
    fit_idx = train_idx if args.train_scope == "train_only" else train_idx + val_idx

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train = X[fit_idx]
    X_test = X[test_idx]
    y_train = y_encoded[fit_idx]
    y_test = y_encoded[test_idx]

    model = build_model(args.model)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")

    report_dict = classification_report(
        y_test,
        y_pred,
        labels=np.arange(len(label_encoder.classes_)),
        target_names=label_encoder.classes_.tolist(),
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).transpose()

    cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(label_encoder.classes_)))
    cm_df = pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_)

    output_path = Path(args.output)
    ensure_parent(output_path)
    side_paths = derive_side_paths(output_path)
    for side_path in side_paths.values():
        ensure_parent(side_path)

    report_df.to_csv(side_paths["report_csv"])
    cm_df.to_csv(side_paths["confusion_csv"])

    # Archive per-sample predictions for paired bootstrap and McNemar tests.
    # See scripts/10_paired_bootstrap.py and results/<date>/bootstrap/README.md.
    predictions_dir = PROJECT_ROOT / "artifacts" / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = predictions_dir / (output_path.stem + ".npz")
    np.savez_compressed(
        prediction_path,
        y_true=y_test,
        y_pred=y_pred,
        test_indices=np.asarray(test_idx),
        label_classes=label_encoder.classes_,
    )

    plot_confusion(
        cm,
        label_encoder.classes_.tolist(),
        side_paths["confusion_png"],
        title=f"{args.model} on {', '.join(views)} ({args.split})",
    )

    metrics = {
        "timestamp": datetime.now().isoformat(),
        "view_request": args.view,
        "views": views,
        "split": args.split,
        "model": args.model,
        "train_scope": args.train_scope,
        "n_features": int(X.shape[1]),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_fit": int(len(fit_idx)),
        "n_test": int(len(test_idx)),
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "report_csv": str(side_paths["report_csv"]),
        "confusion_csv": str(side_paths["confusion_csv"]),
        "confusion_png": str(side_paths["confusion_png"]),
        "predictions_npz": str(prediction_path),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    if args.model_output:
        model_output_path = Path(args.model_output)
        ensure_parent(model_output_path)
        joblib.dump(
            {
                "model": model,
                "label_encoder": label_encoder,
                "split": args.split,
                "views": views,
                "model_name": args.model,
                "train_scope": args.train_scope,
            },
            model_output_path,
        )

    print("=" * 60)
    print("Experiment complete")
    print("=" * 60)
    print(f"Views: {', '.join(views)}")
    print(f"Split: {args.split}")
    print(f"Model: {args.model}")
    print(f"Train scope: {args.train_scope}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print(f"Weighted-F1: {weighted_f1:.4f}")
    print(f"Metrics JSON: {output_path}")
    print(f"Classification report: {side_paths['report_csv']}")
    print(f"Confusion matrix CSV: {side_paths['confusion_csv']}")
    print(f"Confusion matrix PNG: {side_paths['confusion_png']}")
    if args.model_output:
        print(f"Model bundle: {args.model_output}")


if __name__ == "__main__":
    main()
