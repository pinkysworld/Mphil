# Leakage audit and family-name filtering patch notes

This review patch changes leakage matching and family-name filtering from substring matching to exact alphanumeric-segment matching.

Why this matters:

- `CreateRemoteThread` contains the letter sequence `emotet`, but it is a normal API name and should not be flagged as an Emotet leakage token.
- `CryptVerifySignatureW` contains `signature`, but it is a normal API name and should not be counted as a detection-indicator hit.
- Direct family strings such as `Global\TrickBot` are still removed/flagged because `trickbot` appears as an exact segment.

Scripts touched:

- `scripts/02_leakage_audit.py`
- `scripts/04_extract_features.py`
- `scripts/07_leakage_ablation.py`
- `tests/test_leakage_filters.py`

Rerun status:

- Full pipeline rerun completed on 2026-05-06 from the patched repository.
- Headline metrics reproduced the archived thesis values: API-only global SGD macro-F1 0.9024, fused global SGD macro-F1 0.9317, fused per-family SGD macro-F1 0.9607, and fused global LightGBM macro-F1 0.9118.
- The patched exact-segment leakage audit is archived under `results/2026-03-21/leakage_audit/`.
- Per-sample prediction archives are available under `artifacts/predictions/` for paired bootstrap checks.
