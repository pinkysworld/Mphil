# Bootstrap CI artifacts

This directory archives bootstrap uncertainty estimates for the tracked
full-dataset runs reported in the MPhil thesis (Chapter 5 and §6.1.4).

## What is computed

Two bootstrap variants are supported, chosen per run by what the experiment
bundle preserves:

**Confusion-matrix bootstrap** (used for all current tracked runs). For each
of B = 10,000 replicates, each row k of the confusion matrix is resampled
as cm_b[k, :] ~ Multinomial(row_total[k], row_probs[k]), where row_probs[k]
is the empirical row distribution of the observed confusion matrix. Macro-F1,
per-class precision, recall, and F1 are recomputed on cm_b. 95 percent
percentile intervals are taken over the B replicates. This estimates
uncertainty due to finite per-class test support under an in-row
independence assumption.

**Sample-level paired bootstrap with McNemar's test** (used for runs with
per-sample predictions archived to `../../artifacts/predictions/*.npz`).
This is the correct test for paired-model claims and is run via
`scripts/10_paired_bootstrap.py`.

## When the intervals overlap

If two runs' marginal CIs do not overlap, the difference is reliably
detectable. If they overlap, the difference is indeterminate from
confusion-matrix bootstrap alone and must be resolved with sample-level
paired bootstrap (requires per-sample predictions). The thesis reports
overlap/non-overlap conclusions transparently and flags the API-only vs
fusion comparison as indeterminate at the marginal-CI level.

## How to regenerate

From the repository root:

```
python scripts/09_bootstrap_ci.py \
  --results-dir results/2026-03-21/metrics \
  --output-dir results/2026-03-21/bootstrap \
  --replicates 10000 --seed 2026
```

For sample-level paired bootstrap (requires the `run_experiment.py` patch
that archives predictions):

```
python scripts/10_paired_bootstrap.py \
  --run-a artifacts/predictions/table_5_11_fusion_global_sgd.npz \
  --run-b artifacts/predictions/table_5_8.npz \
  --output results/2026-03-21/bootstrap/paired_fusion_vs_api.json
```

## Files

- `bootstrap_ci_results.json` — headline CIs for all four tracked runs and
  the three pairwise overlap comparisons used in the thesis.
- `bootstrap_ci_summary.csv` — flat CSV for convenience.
- (after sample-level re-runs) `paired_*.json` — true paired bootstrap
  results with McNemar p-values.
