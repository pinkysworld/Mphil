"""
scripts/09_bootstrap_ci.py — Confusion-matrix bootstrap for macro-F1 CIs.

Reads archived confusion matrices from results/<date>/metrics/*_confusion.csv
and computes 95 percent percentile bootstrap intervals via multinomial
resampling of each row of the confusion matrix. Outputs a JSON bundle plus
a flat CSV for inclusion in the thesis artifact package.

Method: for each bootstrap replicate b in [1..B], each row k of the confusion
matrix is resampled as cm_b[k, :] ~ Multinomial(row_total[k], row_probs[k]),
where row_probs[k] is the empirical row distribution of the observed CM.
Macro-F1, per-class precision/recall/F1 are recomputed on cm_b. Percentile
intervals are taken over the B replicates.

This estimates finite-per-class-support uncertainty under an in-row
independence assumption. For paired-comparison claims, use sample-level
paired bootstrap (requires per-sample predictions; see scripts/10_paired_bootstrap.py).

Usage:
  python scripts/09_bootstrap_ci.py \\
    --results-dir results/2026-03-21/metrics \\
    --output-dir results/2026-03-21/bootstrap \\
    --replicates 10000 --seed 2026
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd


def per_class_prf(cm: np.ndarray):
    K = cm.shape[0]
    precision = np.zeros(K)
    recall = np.zeros(K)
    f1 = np.zeros(K)
    for k in range(K):
        tp = cm[k, k]
        fn = cm[k, :].sum() - tp
        fp = cm[:, k].sum() - tp
        if tp + fp > 0:
            precision[k] = tp / (tp + fp)
        if tp + fn > 0:
            recall[k] = tp / (tp + fn)
        if precision[k] + recall[k] > 0:
            f1[k] = 2 * precision[k] * recall[k] / (precision[k] + recall[k])
    return precision, recall, f1, f1.mean()


def bootstrap_cm(cm: np.ndarray, B: int, seed: int):
    rng = np.random.default_rng(seed)
    K = cm.shape[0]
    row_totals = cm.sum(axis=1)
    row_probs = np.where(
        row_totals[:, None] > 0,
        cm / np.maximum(row_totals[:, None], 1),
        0.0,
    )
    boot_macro_f1 = np.zeros(B)
    boot_precision = np.zeros((B, K))
    boot_recall = np.zeros((B, K))
    boot_f1 = np.zeros((B, K))
    for b in range(B):
        cm_b = np.zeros((K, K), dtype=np.int64)
        for k in range(K):
            if row_totals[k] > 0:
                cm_b[k, :] = rng.multinomial(int(row_totals[k]), row_probs[k])
        prec, rec, f1, mf1 = per_class_prf(cm_b)
        boot_macro_f1[b] = mf1
        boot_precision[b] = prec
        boot_recall[b] = rec
        boot_f1[b] = f1
    return boot_macro_f1, boot_precision, boot_recall, boot_f1


def ci(arr, lo=2.5, hi=97.5):
    return float(np.percentile(arr, lo)), float(np.percentile(arr, hi))


def analyze_confusion_file(path: Path, B: int, seed: int):
    df = pd.read_csv(path, index_col=0)
    families = df.index.tolist()
    cm = df.values
    prec_pt, rec_pt, f1_pt, mf1_pt = per_class_prf(cm)
    boot_mf1, boot_prec, boot_rec, boot_f1 = bootstrap_cm(cm, B, seed)
    mf1_lo, mf1_hi = ci(boot_mf1)
    result = {
        "confusion_csv": str(path),
        "n_test": int(cm.sum()),
        "B": B,
        "seed": seed,
        "macro_f1_point": float(mf1_pt),
        "macro_f1_ci_95_lo": mf1_lo,
        "macro_f1_ci_95_hi": mf1_hi,
        "macro_f1_ci_width": mf1_hi - mf1_lo,
        "per_family": {},
    }
    for i, fam in enumerate(families):
        r_lo, r_hi = ci(boot_rec[:, i])
        p_lo, p_hi = ci(boot_prec[:, i])
        f_lo, f_hi = ci(boot_f1[:, i])
        result["per_family"][fam] = {
            "support": int(cm[i, :].sum()),
            "recall_point": float(rec_pt[i]),
            "recall_ci_95": [r_lo, r_hi],
            "precision_point": float(prec_pt[i]),
            "precision_ci_95": [p_lo, p_hi],
            "f1_point": float(f1_pt[i]),
            "f1_ci_95": [f_lo, f_hi],
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True,
                        help="Directory containing *_confusion.csv files.")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Where to write bootstrap_ci_results.json and .csv.")
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    confusion_files = sorted(args.results_dir.glob("*_confusion.csv"))
    if not confusion_files:
        raise SystemExit(f"No *_confusion.csv under {args.results_dir}")

    bundle = {
        "timestamp": datetime.now().isoformat(),
        "method": "confusion_matrix_multinomial_bootstrap",
        "replicates": args.replicates,
        "seed": args.seed,
        "runs": [],
    }
    rows_for_flat = []
    for cf in confusion_files:
        print(f"Bootstrapping {cf.name} ...")
        result = analyze_confusion_file(cf, args.replicates, args.seed)
        bundle["runs"].append({"name": cf.stem.replace("_confusion", ""), **result})
        rows_for_flat.append({
            "run": cf.stem.replace("_confusion", ""),
            "macro_f1_point": result["macro_f1_point"],
            "macro_f1_ci_lo": result["macro_f1_ci_95_lo"],
            "macro_f1_ci_hi": result["macro_f1_ci_95_hi"],
            "ci_width": result["macro_f1_ci_width"],
            "n_test": result["n_test"],
        })

    json_path = args.output_dir / "bootstrap_ci_results.json"
    with open(json_path, "w") as f:
        json.dump(bundle, f, indent=2)
    csv_path = args.output_dir / "bootstrap_ci_summary.csv"
    pd.DataFrame(rows_for_flat).to_csv(csv_path, index=False)
    print(f"\nWrote {json_path}")
    print(f"Wrote {csv_path}")

    # Print a compact summary
    print("\n=== SUMMARY ===")
    for r in rows_for_flat:
        print(f"  {r['run']:60s} mF1={r['macro_f1_point']:.4f} "
              f"[{r['macro_f1_ci_lo']:.4f}, {r['macro_f1_ci_hi']:.4f}] "
              f"width={r['ci_width']:.4f}  n={r['n_test']}")


if __name__ == "__main__":
    main()
