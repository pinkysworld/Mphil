# Supervisor reanalysis result bundle

This directory archives the outputs of GitHub Actions run `33269449501`,
produced from the `supervisor-revision-2026-08-29` branch in response to
Dr R.M.M. Pradeep's synopsis feedback.

The bundle contains the feature-level invariance-discriminability analysis,
repeated-seed/model-class robustness analysis, label-provenance audit,
temporal label-composition summaries, the feature-level plot, the full
compressed feature-level metric table, and dataset checksum evidence.

`SHA256SUMS` is generated after copying the workflow artifact into this
repository. The original GitHub Actions artifact is retained separately by
GitHub for its configured retention period; this directory is the durable
repository copy.

Important interpretation boundary: the feature-level H1 result is strong in
the pooled analysis and in API/artifact token views, while the repeated-seed
robustness experiment does not support a general claim that four-view fusion
always outperforms API-only. The latter result is therefore retained as a
falsification/qualification result rather than hidden or overwritten.
