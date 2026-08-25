# Exact reproducibility procedure

The repository separates deterministic rules from data-dependent evidence.
The scripts are versioned here; the generated manifests must be archived with
the defended result bundle after a full-corpus run.

## Locked environment

- `.python-version` fixes Python 3.11.2.
- `requirements-lock.txt` fixes direct and transitive Python dependencies and
  includes package hashes. It was resolved for Linux x86-64 with the latest
  packages available by 2026-05-06 while retaining the thesis-recorded NumPy,
  pandas, and scikit-learn versions.
- Each run also records the actual interpreter, platform, package versions,
  git revision, and dirty-worktree state in its JSON manifest.

Create the environment with:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --require-hashes -r requirements-lock.txt
```

## Full defended run

Place the official reduced CAPEv2 archive in `data/raw/`, then run:

```bash
.venv/bin/python scripts/01_ingest.py
.venv/bin/python scripts/02_leakage_audit.py
.venv/bin/python scripts/03_build_splits.py
.venv/bin/python scripts/04_extract_features.py
.venv/bin/python scripts/06_walk_forward.py
```

The split script writes an exact catalog containing sample SHA-256, family,
date, zero-based row index, split name, and train/validation/test role. The
feature script writes hashes for the learned TF-IDF vocabularies and fitted IDF
arrays, together with checksums for each cached matrix and fitted transformer.

HashingVectorizer has no fitted vocabulary. For hashing runs, the manifest
therefore records the complete vectorizer configuration and a canonical hash
of that configuration. The walk-forward run additionally records ordered
train/test membership hashes for every monthly window and archives every
sample-level prediction.

## Prediction-archive reconciliation

The compact sample-level prediction archives support an independent check of
the thesis's full-dataset tables without requiring the raw malware corpus:

```bash
.venv/bin/python scripts/21_validate_thesis_results.py
```

This command recomputes accuracy, macro-F1, and weighted-F1 for 13 archived
runs; checks exact test membership and reference labels; verifies five tracked
confusion matrices and both paired family-transition analyses; and validates
the corrected forecast-horizon table. It also checks every prediction archive
against `artifacts/predictions/manifest.json`.

This is a result reconciliation, not a substitute for model retraining. An
exact full rerun still requires the official input archive and the locked
Linux x86-64 environment described above.

## Submission bundle

After the full run, create the final self-checking bundle:

```bash
.venv/bin/python scripts/15_build_reproducibility_bundle.py \
  --run-dir results/2026-07-11
```

The command checksums every reduced JSON report, relevant input archive,
environment file, script, split catalog, vocabulary manifest, prediction file,
and result artefact. It writes `SHA256SUMS` at the root of the reproducibility
folder. Verify it from the dated result directory with:

```bash
sha256sum -c reproducibility/SHA256SUMS
```

Do not describe a result as exactly reproduced unless its generated manifests
and checksums are present in the dated public bundle and the recorded git state
is clean.
