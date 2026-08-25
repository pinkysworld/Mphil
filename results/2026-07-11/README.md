# Defended Result Bundle - 11 July 2026

This folder preserves the reported full-run outputs and adds the final
leakage-controlled walk-forward rerun and exact reproducibility evidence.

Contents:

- `summary_metrics.csv`: quick comparison of the main baseline runs
- `metrics/`: JSON, CSV, and confusion-matrix outputs for the baseline models
- `metrics/view_ablation_replay_summary.csv`: independently recomputed metrics
  for all archived full-dataset prediction runs
- `forecast_horizon/`: the corrected, leakage-controlled API-hashing horizon
  diagnostic derived from aligned sample-level predictions
- `bootstrap/`: confidence intervals and paired comparison checks
- `walk_forward/`: shared-extractor rolling-origin results, all sample-level
  predictions, plot, sample catalog, and manifest
- `retraining_trigger/`: the trigger analysis regenerated from the defended
  walk-forward series
- `invariance/`: invariance/discriminability metrics and scatter plot
- `leakage_ablation/`: ablation table and JSON summary
- `calibration/`: calibration metrics, plots, and per-family breakdown
- `leakage_audit/`: exact-segment full-corpus audit outputs
- `splits/`: split summary used for the experiment set
- `reproducibility/`: exact split assignments, fitted vocabulary and IDF
  hashes, hashing-configuration hashes, locked environment record, report
  content checksums, code and input checksums, the deep validation report, and
  `SHA256SUMS`

The defended walk-forward run uses the same shared family-segment filter as
the feature pipeline. Its sample counts and all 19 macro-F1 values are exactly
identical to the archived series. The locked run used CPython 3.11.2,
scikit-learn 1.4.2, and NumPy 1.24.0.

Use the manifest paths and checksums in `reproducibility/` to verify the scope
of an exact rerun. Earlier auxiliary results retain the level of traceability
provided by their archived artifacts.

Run `scripts/21_validate_thesis_results.py` from the repository root to
recompute the archived metrics, paired transitions, and forecast-horizon table
without retraining the models.
