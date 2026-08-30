# Supervisor reanalysis interpretation — 30 August 2026

This note records the interpretation of the completed supervisor reanalysis before Annexure A is revised. It is deliberately separated from the original July defended result bundle and from the prospective experiment-plan note.

## 1. Feature-level invariance–discriminability result

The feature-level analysis strongly supports the proposed trade-off at the pooled level. Across 16,416 feature-level observations, the stratified Spearman association between discriminability and future shift is `rho = 0.75886`, with a 95% stratified-bootstrap interval `[0.75099, 0.76648]` and one-sided stratified permutation `p = 0.000999`. Expressed as discriminability versus invariance, the direction is equivalently negative.

The result is not uniform across all four feature views. API tokens show `rho = 0.76477` over 14,543 features and artifact tokens show `rho = 0.71944` over 1,856 features, both surviving BH-FDR adjustment. Behavioral counts (`n = 13`) and static PE (`n = 4`) do not show statistically persuasive within-view associations. The thesis must therefore describe the pooled result as strong but heterogeneous, rather than claiming identical behaviour in every representation class.

## 2. Repeated-seed/model-class robustness qualifies H3

The robustness experiment does **not** support a general statement that four-view fusion always outperforms API-only.

For SGD, API-only is stable across seeds 17/42/101 at approximately `0.9020–0.9030` Macro-F1. Four-view fusion is much more seed-sensitive: `0.84086`, `0.93169`, and `0.91970`. The mean paired fusion-minus-API delta is `-0.00509`, with both positive and negative seed-level deltas.

For LightGBM, API-only reaches `0.91663` and four-view fusion `0.91179`; the paired fusion-minus-API delta is approximately `-0.00484` for each prespecified seed in this implementation.

Therefore the previously archived seed-42 fused-SGD result (`Macro-F1 = 0.9317`) remains a valid result for that specified run, but it is not evidence of a model-independent or seed-independent fusion advantage. The revised proposal and thesis should treat H3 as a genuinely falsifiable hypothesis and report this robustness qualification explicitly. The result is scientifically useful because it prevents the thesis from overstating fusion as a universal improvement.

## 3. Label-provenance boundary

All 48,976 samples have a detection date and a supplied malware-family target. The target originates from the Avast-CTU public label file / Avast backend classification records according to the official dataset documentation.

The public corpus does not provide a separate per-sample labelling date, engine version used for labelling, or cross-vendor agreement field. Consequently, this corpus alone cannot identify how much observed temporal change is malware-behaviour drift versus vendor labelling-policy drift. Temporal composition summaries are diagnostic only. The family label remains the supervised target `y`; leakage control removes prohibited family/detection signals from `X`.

## 4. Route B selected

Route B is selected as the additional theoretical extension for the MPhil revision, with a deliberately bounded H5 analysis.

**H5 (integration-asymmetry hypothesis):** Greater concentration of positive marginal contributions across the four feature views in one rolling-origin window is associated with greater fused-model Macro-F1 decay in the next eligible window.

The implementation in `scripts/25_integration_asymmetry_walkforward.py` estimates per-window marginal contributions using full fusion versus four leave-one-view-out models. Positive contributions are normalised and their concentration is measured by a normalised Herfindahl index relative to equal four-view contribution. Negative marginal contributions are retained separately rather than being hidden. H5 is tested lagged against next-window performance decay, with a moving-block bootstrap interval and circular-shift permutation test to reduce the IID assumption for adjacent temporal windows.

This H5 test is exploratory. A null, negative, or unstable association weakens Route B and will be reported as such. A positive result would motivate, but not by itself establish, a more general integration-asymmetry theory. Cross-corpus replication remains a later PhD-level extension rather than an MPhil requirement.

## 5. Consequences for the Annexure A revision

The revised proposal should: promote the measurement chain to the organising theoretical framework; define feature-level analysis as the primary statistical unit for H1; add effect sizes, confidence intervals, permutation testing, repeated seeds and multiplicity handling; add explicit falsification conditions; instrument label provenance without removing target labels; include Route B/H5 as a bounded exploratory theoretical extension; and avoid claiming that fusion is universally superior.

The durable supervisor-reanalysis outputs are archived under `results/2026-08-29-supervisor-reanalysis/` with checksums. Route B outputs will be archived separately under `results/2026-08-30-route-b-integration-asymmetry/` by the dedicated GitHub Actions workflow when that run completes.
