# Malware Family Classification

Working repository for the experimental part of the MPhil thesis. The current
focus is on ingesting the Avast-CTU CAPEv2 reports, checking label/report
alignment, building evaluation splits, and extracting the baseline feature
views used in the first round of experiments.

## Repository Layout

- `scripts/` pipeline scripts
- `configs/` project settings
- `data/raw/` local dataset files
- `data/processed/`, `data/cache/`, `data/splits/` derived data
- `artifacts/` logs, figures, and result files
- `docs/` working notes

## Local Setup

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

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
```

Additional notes are in `docs/data_notes.md` and `docs/work_plan.md`.
