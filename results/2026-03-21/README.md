# Result Bundle — 21 March 2026

This folder contains a tracked copy of the main outputs from the first full
pipeline run.

Contents:

- `summary_metrics.csv` — quick comparison of the main baseline runs
- `metrics/` — JSON, CSV, and confusion-matrix outputs for the baseline models
- `walk_forward/` — rolling-origin results and plot
- `invariance/` — invariance/discriminability metrics and scatter plot
- `leakage_ablation/` — ablation table and JSON summary
- `calibration/` — calibration metrics, plots, and per-family breakdown
- `leakage_audit/` — leakage-audit summary files
- `splits/` — split summary used for the experiment set

The original generated files also remain available locally in `artifacts/`,
but this copy is the version intended for GitHub and later reference.
