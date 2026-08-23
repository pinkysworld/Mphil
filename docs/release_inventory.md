# Public release inventory

This document distinguishes the authoritative evidence from intentional
archive copies and local-only inputs.

## Authoritative evidence

`results/2026-07-11/` is the only dated result bundle in the release. Its
`reproducibility/SHA256SUMS` file covers the reported metrics, figures, split
summaries, walk-forward outputs, and reproducibility records. The duplicate
March snapshot was removed because it repeated superseded copies of these
files.

The current methodological supplements are:

- `artifacts/decision_support/`
- `artifacts/calibrated_selective_policy/`
- `artifacts/thesis_expansion/`
- `artifacts/open_set/`
- `artifacts/explainability_case_studies/`
- `artifacts/retraining_trigger/`

Each policy bundle has a manifest. The explanation archive now has hashes for
its curated public exports. Run `scripts/verify_release.py` to verify all of
them together.

## Intentional duplication

The reproducibility directory contains copies of split, feature, environment,
and walk-forward records so that the dated bundle can be verified in isolation.
The walk-forward files also appear in the bundle's `walk_forward/` directory,
and the retraining analysis is both a derived artefact and part of the dated
result record. These are deliberate archival copies, not competing versions.

Several immutable bootstrap records inside the dated bundle retain their
original March input path as execution provenance. No March result directory
is present or used by current code.

## Local-only inputs

Raw malware archives, extracted JSON reports, feature matrices, and model
caches are not distributed. They are large third-party or derived data and are
excluded by `.gitignore`. The tracked calibration replay bundle contains only
the scores and probabilities needed to rerun the four-way date-ordered policy
sensitivity analysis.

The token explanation casebook was curated from local explanation exports. The
source matrices are not distributed; the public CSV and Markdown exports are
preserved with hashes and a bounded provenance statement.

## Claim boundary

Repository integrity makes the archived computations auditable. It does not
show that the classifier is production-ready, that selective routing is stable
under future drift, or that analysts benefit from the four-state policy. Those
claims require prospective data and an analyst-centred evaluation.
