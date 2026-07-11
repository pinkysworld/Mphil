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
