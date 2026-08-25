# Prediction archives

This directory contains the sample-level outputs used to verify the thesis's
full-dataset result tables. Every NPZ file stores `y_true`, `y_pred`,
`test_indices`, and `label_classes`. The index semantics and sample identities
are defined by `results/2026-07-11/reproducibility/sample_catalog.csv.gz` and
`split_assignments.csv.gz`.

Run the independent reconciliation check from the repository root:

```bash
.venv/bin/python scripts/21_validate_thesis_results.py
```

The script recomputes metrics, checks split membership and reference labels,
compares the five tracked confusion matrices, verifies paired family-level
transitions, and rebuilds the forecast-horizon values. `manifest.json` records
the SHA-256 digest and experimental role of every prediction archive.

The compact NPZ files support result verification without republishing the
licensed/raw corpus. Re-training still requires the official reduced CAPEv2
archive and the locked Linux x86-64 environment described in
`results/2026-07-11/reproducibility/EXACT_REPRODUCIBILITY.md`.
