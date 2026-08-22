# Malware Family Classification

Working repository for the experimental part of the MPhil thesis. It contains
the defended ingestion, leakage audit, split, feature, modelling, walk-forward,
and reproducibility pipelines for the Avast-CTU CAPEv2 reports.

## Repository Layout

- `scripts/` pipeline scripts
- `configs/` project settings
- `data/raw/` local dataset files
- `data/processed/`, `data/cache/`, `data/splits/` derived data
- `artifacts/` logs, figures, and result files
- `docs/` working notes
- `site/` public project website for GitHub Pages

## Local Setup

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --require-hashes -r requirements-lock.txt
```

The uncompiled `requirements.txt` remains the human-readable input. The lock
file and `.python-version` are the defended environment specification.

## Data

Place the Avast-CTU files in `data/raw/`. The ingestion script accepts:

- `public_labels.csv`
- `public_small_reports.zip`
- `avast_ctu_reduced.zip`
- `1.zip` and `2.zip` if using a locally split archive

Running `scripts/01_ingest.py` extracts report JSON files to
`data/raw/reports/`.

## First Run

```bash
.venv/bin/python scripts/01_ingest.py
.venv/bin/python scripts/02_leakage_audit.py
.venv/bin/python scripts/03_build_splits.py
.venv/bin/python scripts/04_extract_features.py
.venv/bin/python scripts/06_walk_forward.py
```

`03_build_splits.py` writes exact sample-hash split manifests.
`04_extract_features.py` writes fitted vocabulary, IDF, hashing-configuration,
and cache checksums. `06_walk_forward.py` uses the same shared leakage-filtered
API extractor and writes per-window membership hashes and predictions. See
[`docs/reproducibility.md`](docs/reproducibility.md) for the submission bundle
and checksum procedure.

## Baseline Run

```bash
.venv/bin/python scripts/run_experiment.py \
  --view api_tfidf \
  --split global_chronological \
  --model sgd \
  --output artifacts/metrics/table_5_8.json
```

## Decision-support demonstrator

The four-state analyst-facing policy is machine-readable in
`configs/decision_support_policy.json` and implemented by
`scripts/18_decision_support_demonstrator.py`. Its bounded contract,
confidence-gate, temporal-gate, and open-set capability experiments are
reproduced with:

```bash
.venv/bin/python scripts/19_decision_support_experiments.py
.venv/bin/python -m unittest tests/test_decision_support_policy.py
```

See `artifacts/decision_support/README.md` for the results and their claim
boundary. The demonstrator is a research artifact, not a production control or
an analyst-usability validation.

## Time-ordered calibration and selective policy

The calibration-policy sensitivity experiment separates model fitting,
calibration fitting, threshold selection, and future testing at strict date
boundaries. It freezes a confidence threshold on the later validation segment
before evaluating the subsequent test period:

```bash
.venv/bin/python scripts/20_calibrated_selective_policy.py --input-mode replay-bundle
.venv/bin/python -m unittest tests/test_calibrated_selective_policy.py
```

Outputs and their checksums are stored in
`artifacts/calibrated_selective_policy/`. The experiment distinguishes an
improvement in calibration error from an improvement in selective-routing
safety; the two are not treated as equivalent. A tracked 13,650-row replay
bundle contains the exact base-model decision scores and uncalibrated
probabilities needed for calibration fitting, threshold selection, and future
testing. Its metadata records the hashes of the larger defended source caches.
Use `--input-mode source-cache` only when those local cache files are available
and the public replay bundle needs to be regenerated.

Run all policy tests with:

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_*policy.py' -v
```
