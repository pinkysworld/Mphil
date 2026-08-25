# Leakage-controlled, time-aware malware family classification

This repository contains the executable research supplement for the MPhil
thesis *Explainable and Time-Aware Malware Family Classification from CAPEv2
Sandbox Reports Using Leakage-Controlled Multi-View Features*. It provides the
defended pipeline, the final archived results, and the bounded decision-support
methodology.

## Authoritative release contents

- `results/2026-07-11/` — defended result bundle and checksum manifest
- `artifacts/decision_support/` — four-state policy contract and replay outputs
- `artifacts/calibrated_selective_policy/` — strictly date-ordered calibration
  and threshold-transfer sensitivity experiment
- `artifacts/thesis_expansion/` — derived statistical analyses used in the
  expanded thesis
- `artifacts/predictions/` — sample-level outputs and SHA-256 manifest for the
  full-dataset result reconciliation
- `artifacts/open_set/`, `artifacts/explainability_case_studies/`, and
  `artifacts/retraining_trigger/` — bounded supporting analyses
- `scripts/` — ingestion, feature, modelling, validation, and policy code
- `tests/` — executable regression and policy-contract tests
- `site/` — public project summary for GitHub Pages

The earlier duplicate result snapshot has been removed. Some files inside
`results/2026-07-11/reproducibility/` intentionally duplicate files elsewhere
in the same dated bundle: they form a self-contained, checksum-verifiable
archive. Bootstrap regeneration commands and metadata now resolve only to the
canonical dated bundle. See
[`docs/release_inventory.md`](docs/release_inventory.md).

## Evidence boundaries

- Four-view fused SGD reaches macro-F1 0.9317 on the aligned global
  chronological test; the API-only baseline reaches 0.9024.
- The API-only rolling-origin proxy ranges from 0.6690 to 0.9289 across 19
  monthly windows (mean 0.7908). One split is therefore not treated as a
  deployment estimate.
- The matched exact-segment leakage ablation is null within the evaluated
  allow-listed fields. It does not exclude every indirect shortcut.
- In the strictly ordered sensitivity experiment, sigmoid calibration reduces
  ECE from 0.1031 to 0.0318 but does not improve validation-selected selective
  routing: future error is 2.56% calibrated versus 0.34% uncalibrated.
- The four-state demonstrator passes all eight deterministic contract cases,
  but no archived replay reaches State A. It is not evidence of analyst
  benefit, production readiness, or transport beyond the archived
  Avast-CTU/CAPEv2 corpus.

## Environment

Python 3.11.2 and all direct and transitive packages are pinned:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --require-hashes -r requirements-lock.txt
```

The readable dependency input is `requirements.txt`; the defended lock is
`requirements-lock.txt`. The exact lock and native LightGBM run were recorded
on Linux x86-64; other host architectures need compatible native libraries.

## Data boundary

Raw third-party malware archives, extracted reports, and generated feature
caches are intentionally excluded. Place the Avast-CTU public labels and
reduced CAPEv2 archive under `data/raw/` before running the full pipeline. The
ingestion script documents the accepted filenames.

## Defended pipeline

```bash
.venv/bin/python scripts/01_ingest.py
.venv/bin/python scripts/02_leakage_audit.py
.venv/bin/python scripts/03_build_splits.py
.venv/bin/python scripts/04_extract_features.py
.venv/bin/python scripts/06_walk_forward.py
```

The split, feature, environment, report-content, and walk-forward manifests are
archived under `results/2026-07-11/reproducibility/`. Follow
[`docs/reproducibility.md`](docs/reproducibility.md) for the exact acceptance
checks.

## Decision-support experiments

```bash
.venv/bin/python scripts/19_decision_support_experiments.py
.venv/bin/python scripts/20_calibrated_selective_policy.py --input-mode replay-bundle
.venv/bin/python scripts/21_validate_thesis_results.py
.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
.venv/bin/python tests/test_leakage_filters.py
.venv/bin/python tests/test_reproducibility.py
```

The calibration script uses the tracked 13,650-row replay bundle, so its
four-way date separation can be reproduced without the larger local feature
caches. The explanation-case archive requires the original local explanation
exports and therefore accepts an explicit `--source-dir`; its public curated
outputs remain independently hash-verifiable.

## Release verification

```bash
.venv/bin/python scripts/verify_release.py --require-clean
```

This checks the dated release inventory, forbidden local paths, defended
`SHA256SUMS`, policy and prediction manifests, and explanation-archive hashes.
A passing check confirms repository integrity; it does not extend the
empirical claims above.
