# Thesis expansion analyses

These tables are deterministic derivations from the archived predictions in
`artifacts/predictions` and `results/2026-07-11/walk_forward`. No model is
retrained by this script.

- `paired_error_transitions.csv`: family-level direction of paired error changes.
- `paired_model_summary.json`: alignment checks, source hashes, and exact McNemar tests.
- `walk_forward_per_family.csv`: pooled family metrics over all rolling-origin windows.
- `walk_forward_family_window.csv`: family metrics within each rolling-origin window.
- `walk_forward_top_error_pairs.csv`: four most frequent directed errors per low-performing window.
- `retraining_trigger_sensitivity.csv`: threshold grid for the illustrative review policy.

Run with `python scripts/17_substantive_expansion_analysis.py` from the project root.
