# Thesis Wiring Checklist

Status checked on 2026-03-31 against the live working tree and the frozen
result bundles under `results/`.

## Current status snapshot

- `Months 07-12` is complete in the live repo. The expected-output check passes
  against `experiments/months_07_12/EXPECTED_OUTPUTS.txt`.
- `Months 13-18` outputs are also already present in the live repo, even though
  they sit beyond the month-12 milestone.
- `Months 03-06` is complete except for one local provenance file:
  `artifacts/ingestion_log.json`.
- `results/2026-03-22_verified/` is the safest thesis-reference bundle.
  It passed checksum verification and contains the full metrics, robustness,
  calibration, leakage-audit, split, and explainability exports.
- `results/2026-03-22_research/` is also verified, but it additionally preserves
  dirty-tree patch and untracked-file state for audit purposes.

## Recommended citation bundle

Use `results/2026-03-22_verified/` as the primary frozen bundle when you cite
file-backed results in the thesis. Keep `results/2026-03-22_research/` for the
appendix or supervisor-facing reproducibility notes when you want the extra Git
state capture.

## Chapter-by-chapter wiring

### Chapter 3: Data, preprocessing, and study design

- Wire in the dataset and split provenance from `data/splits/` and
  `results/2026-03-22_verified/split_definitions/`.
- Use `artifacts/splits/split_summary.json` or
  `results/2026-03-22_verified/splits/split_summary.json` to report split sizes
  and chronological boundaries.
- Keep the leakage discussion anchored in
  `results/2026-03-22_verified/leakage_audit/audit_results.json` and
  `results/2026-03-22_verified/leakage_audit/flagged_samples.csv`.
- Note one housekeeping gap: `artifacts/ingestion_log.json` is still missing
  locally, so the ingestion narrative should rely on the processed data and
  split outputs unless that log is regenerated.

### Chapter 4: Experimental setup and evaluation policy

- Use `results/2026-03-22_verified/metrics/summary_metrics.csv` as the master
  matrix for runs, splits, models, and headline metrics.
- State explicitly that `macro_f1` is the main comparison metric.
- Keep the random split diagnostic-only and the LightGBM run sensitivity-only.
- Keep the thesis baseline family centered on the linear models:
  `SGDClassifier` and multinomial logistic regression.

### Chapter 5: Core baseline results

- Keep `table_5_8` as the first strong chronological baseline:
  `results/2026-03-22_verified/metrics/table_5_8.json`.
- Add the full single-view month-7-to-12 matrix now that it exists:
  `global_art_tfidf_sgd`, `global_counts_logistic`, `global_pe_logistic`,
  `per_family_api_tfidf_sgd`, `per_family_art_tfidf_sgd`,
  `per_family_counts_logistic`, and `per_family_pe_logistic`.
- Keep the two fused headline baselines central:
  `table_5_11_fusion_global_sgd` and `table_5_12_fusion_per_family_sgd`.
- Use the corresponding `*_report.csv` and `*_confusion.png` files for the
  polished result tables and confusion-matrix figures.

### Chapter 6: Robustness, calibration, and model interpretation

- Keep `table_6_1_fusion_global_lightgbm.json` as a model-class sensitivity
  check rather than a replacement headline model.
- Add walk-forward evidence from
  `results/2026-03-22_verified/walk_forward/walk_forward_results.csv` and
  `walk_forward_plot.png` to support the temporal-stability discussion.
- Add leakage-ablation evidence from
  `results/2026-03-22_verified/leakage_ablation/ablation_table.csv`.
  The exported table currently shows no material controlled-vs-permitted gap.
- Add calibration and operating-policy material from
  `results/2026-03-22_verified/calibration/full_calibration_metrics.json`,
  `per_family_ece.csv`, and `risk_coverage_comparison.png`.
- Add explainability outputs for the linear baselines from
  `results/2026-03-22_verified/explainability/`.
  The most thesis-ready folders are `table_5_11_fusion_global_sgd/`,
  `table_5_12_fusion_per_family_sgd/`, and `table_5_8/`.

### Chapter 7: Discussion, limitations, and conclusion

- Emphasize that the fused per-family chronological SGD baseline is the
  strongest completed reference result in the current bundle.
- Contrast the strong API and artifact views with the much weaker counts-only
  and PE-only baselines.
- Use the walk-forward variation to support a careful drift discussion rather
  than a single aggregate stability claim.
- Use the calibration outputs to argue for reliability and deployment policy,
  even where calibrated accuracy shifts slightly.

### Appendices and reproducibility

- Put the complete run matrix in an appendix using
  `results/2026-03-22_verified/metrics/summary_metrics.csv`.
- Add selected explainability tables from
  `global_top_features.csv`, `top_confusions.csv`, and
  `case_feature_contributions.csv`.
- Add reproducibility evidence from `bundle_manifest.json`,
  `file_checksums.csv`, `environment_freeze.txt`, and `git_commit.txt`.
- If you want a concise reproducibility appendix paragraph, point to the
  workflow snapshot preserved under `results/2026-03-22_verified/workflow/`.

## Short list of next writing actions

1. Refresh the thesis tables and figure captions from
   `results/2026-03-22_verified/metrics/summary_metrics.csv`.
2. Pull one walk-forward figure, one calibration figure, and one leakage
   ablation table into the main narrative if Chapter 6 has room.
3. Add one explainability subsection based on the fused global or fused
   per-family linear baseline.
4. Regenerate `artifacts/ingestion_log.json` only if you want the early-phase
   expected-output checklist to be fully green.
