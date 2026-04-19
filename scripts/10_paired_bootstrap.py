"""
scripts/10_paired_bootstrap.py — Sample-level paired bootstrap and McNemar test.

Requires that two experiment runs have archived per-sample predictions on the
same test partition as NumPy .npz files (see run_experiment.py patch).
Each .npz must contain arrays y_true, y_pred aligned to the same test indices.

Outputs:
  - macro_f1 point estimate and 95% CI for each run (paired bootstrap on the
    shared test indices);
  - Delta_macro_f1 = run_a - run_b with 95% CI (paired);
  - McNemar's exact test on per-sample correctness (b+c table).

Usage:
  python scripts/10_paired_bootstrap.py \\
    --run-a artifacts/predictions/table_5_11.npz \\
    --run-b artifacts/predictions/table_5_8.npz \\
    --output results/2026-03-21/bootstrap/paired_fusion_vs_api.json \\
    --replicates 10000 --seed 2026
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
from sklearn.metrics import f1_score
from scipy.stats import binomtest


def paired_bootstrap(y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray,
                     B: int, seed: int):
    """Resample the shared sample indices, recompute macro-F1 for both models
    on each replicate, and return the distributions and the paired delta."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    mf1_a = np.zeros(B)
    mf1_b = np.zeros(B)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        mf1_a[b] = f1_score(y_true[idx], y_pred_a[idx], average="macro",
                            zero_division=0)
        mf1_b[b] = f1_score(y_true[idx], y_pred_b[idx], average="macro",
                            zero_division=0)
    delta = mf1_a - mf1_b
    return mf1_a, mf1_b, delta


def mcnemar_exact(y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray):
    """Exact McNemar on per-sample correctness. b = A correct, B wrong;
    c = A wrong, B correct."""
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "p_value": 1.0, "note": "no discordant pairs"}
    result = binomtest(b, n, p=0.5, alternative="two-sided")
    return {"b": b, "c": c, "n_discordant": n, "p_value": float(result.pvalue)}


def ci(arr, lo=2.5, hi=97.5):
    return float(np.percentile(arr, lo)), float(np.percentile(arr, hi))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-a", type=Path, required=True,
                        help="NPZ with y_true, y_pred for run A.")
    parser.add_argument("--run-b", type=Path, required=True,
                        help="NPZ with y_true, y_pred for run B.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    a = np.load(args.run_a)
    b_ = np.load(args.run_b)
    y_true_a = a["y_true"]
    y_true_b = b_["y_true"]
    if len(y_true_a) != len(y_true_b) or not np.array_equal(y_true_a, y_true_b):
        raise SystemExit("y_true differs between runs; paired bootstrap "
                         "requires identical test partitions.")
    y_true = y_true_a
    y_pred_a = a["y_pred"]
    y_pred_b = b_["y_pred"]

    mf1_a, mf1_b, delta = paired_bootstrap(
        y_true, y_pred_a, y_pred_b, args.replicates, args.seed
    )
    a_lo, a_hi = ci(mf1_a)
    b_lo, b_hi = ci(mf1_b)
    d_lo, d_hi = ci(delta)
    mc = mcnemar_exact(y_true, y_pred_a, y_pred_b)

    a_point = float(f1_score(y_true, y_pred_a, average="macro", zero_division=0))
    b_point = float(f1_score(y_true, y_pred_b, average="macro", zero_division=0))

    result = {
        "timestamp": datetime.now().isoformat(),
        "run_a": str(args.run_a),
        "run_b": str(args.run_b),
        "n_test": int(len(y_true)),
        "replicates": args.replicates,
        "seed": args.seed,
        "run_a_macro_f1_point": a_point,
        "run_a_macro_f1_ci_95": [a_lo, a_hi],
        "run_b_macro_f1_point": b_point,
        "run_b_macro_f1_ci_95": [b_lo, b_hi],
        "delta_macro_f1_point": a_point - b_point,
        "delta_macro_f1_ci_95": [d_lo, d_hi],
        "delta_ci_excludes_zero": bool(d_lo > 0 or d_hi < 0),
        "mcnemar": mc,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nRun A: macro-F1 = {a_point:.4f}  [95% CI {a_lo:.4f}, {a_hi:.4f}]")
    print(f"Run B: macro-F1 = {b_point:.4f}  [95% CI {b_lo:.4f}, {b_hi:.4f}]")
    print(f"Delta (A - B) = {a_point - b_point:+.4f}  [95% CI {d_lo:+.4f}, {d_hi:+.4f}]")
    print(f"  CI excludes zero: {result['delta_ci_excludes_zero']}")
    print(f"McNemar: b={mc['b']}  c={mc['c']}  n_discordant={mc.get('n_discordant', 0)}  "
          f"p={mc['p_value']:.4g}")
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
