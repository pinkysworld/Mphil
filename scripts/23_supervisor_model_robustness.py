"""Repeated-seed / model-class robustness requested by supervisor.

Re-runs API-only and four-view fusion on the fixed global chronological split
without hyperparameter search. Outputs quantify whether conclusions depend on
one seed or one model class.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

from run_experiment import (
    PROJECT_ROOT,
    combine_views,
    load_labels,
    load_split,
    load_view_matrix,
)

OUT_DIR = PROJECT_ROOT / "artifacts" / "supervisor_revision"
SPLIT_NAME = "global_chronological"
DEFAULT_SEEDS = [17, 29, 42, 71, 101]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["sgd", "lightgbm"],
        default=["sgd", "lightgbm"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    return parser.parse_args()


def build_model(name: str, seed: int):
    if name == "sgd":
        from sklearn.linear_model import SGDClassifier

        return SGDClassifier(
            loss="log_loss",
            class_weight="balanced",
            max_iter=1000,
            random_state=seed,
        )

    if name == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            objective="multiclass",
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )

    raise ValueError(name)


def percentile_ci(values: np.ndarray) -> list[float]:
    values = np.asarray(values, dtype=float)
    return [float(v) for v in np.quantile(values, [0.025, 0.975])]


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    split = load_split(SPLIT_NAME)
    labels = load_labels()
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels["family"].astype(str).to_numpy())

    fit_idx = list(split["train"]) + list(split.get("val", []))
    test_idx = list(split["test"])

    api = load_view_matrix("api_tfidf", SPLIT_NAME)
    fusion = combine_views(
        [
            api,
            load_view_matrix("art_tfidf", SPLIT_NAME),
            load_view_matrix("counts", SPLIT_NAME),
            load_view_matrix("pe", SPLIT_NAME),
        ]
    )
    matrices = {"api_only": api, "four_view_fusion": fusion}

    rows: list[dict[str, object]] = []
    for model_name in args.models:
        for seed in args.seeds:
            for view_name, matrix in matrices.items():
                model = build_model(model_name, seed)
                model.fit(matrix[fit_idx], y[fit_idx])
                prediction = model.predict(matrix[test_idx])
                rows.append(
                    {
                        "model": model_name,
                        "seed": int(seed),
                        "view": view_name,
                        "macro_f1": float(
                            f1_score(y[test_idx], prediction, average="macro")
                        ),
                    }
                )
                print(
                    f"{model_name:9s} seed={seed:3d} {view_name:16s} "
                    f"macro-F1={rows[-1]['macro_f1']:.6f}"
                )

    results = pd.DataFrame(rows)
    results.to_csv(OUT_DIR / "model_robustness_runs.csv", index=False)

    summaries = []
    paired = []
    for model_name, model_rows in results.groupby("model", sort=True):
        for view_name, group in model_rows.groupby("view", sort=True):
            values = group["macro_f1"].to_numpy(dtype=float)
            summaries.append(
                {
                    "model": model_name,
                    "view": view_name,
                    "n_seeds": int(len(values)),
                    "mean_macro_f1": float(np.mean(values)),
                    "std_macro_f1": float(np.std(values, ddof=1))
                    if len(values) > 1
                    else 0.0,
                    "seed_percentile_95_interval": percentile_ci(values),
                    "min_macro_f1": float(np.min(values)),
                    "max_macro_f1": float(np.max(values)),
                }
            )

        pivot = model_rows.pivot(index="seed", columns="view", values="macro_f1")
        pivot = pivot.dropna(subset=["api_only", "four_view_fusion"])
        delta = (
            pivot["four_view_fusion"].to_numpy()
            - pivot["api_only"].to_numpy()
        )
        paired.append(
            {
                "model": model_name,
                "n_paired_seeds": int(len(delta)),
                "mean_fusion_minus_api_macro_f1": float(np.mean(delta)),
                "std_fusion_minus_api_macro_f1": float(np.std(delta, ddof=1))
                if len(delta) > 1
                else 0.0,
                "seed_percentile_95_interval": percentile_ci(delta),
                "all_seed_deltas_positive": bool(np.all(delta > 0)),
                "seed_deltas": [float(value) for value in delta],
            }
        )

    payload = {
        "schema_version": "1.0",
        "split": SPLIT_NAME,
        "train_scope": "train_plus_val",
        "seeds": [int(seed) for seed in args.seeds],
        "models": args.models,
        "summaries": summaries,
        "paired_fusion_minus_api": paired,
        "interpretation_boundary": (
            "Seed intervals summarize sensitivity over the prespecified seed set; "
            "they are not population confidence intervals. The experiment tests "
            "ordering robustness, not hyperparameter optimality."
        ),
    }
    (OUT_DIR / "model_robustness_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["paired_fusion_minus_api"], indent=2))


if __name__ == "__main__":
    main()
