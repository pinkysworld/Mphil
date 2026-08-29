# Supervisor revision plan — 29 August 2026

This note translates Dr R.M.M. Pradeep's advisory note into executable changes while preserving the defended July 2026 result bundle unchanged.

## What changes scientifically

1. **Promote the measurement chain from a caveat to the organising framework.** The empirical claim is no longer only that some feature views drift more than others. The revised claim treats temporal invariance as a property of a constructed observation chain: phenomenon / sandbox observation -> retained report schema -> feature representation -> supplied family label -> model output -> analyst decision.
2. **State and test a mechanism for the invariance–discriminability relationship.** Coarser features closer to stable attacker objectives are expected to be more temporally stable but less family-separating; implementation-proximal features may separate families more strongly but churn with tooling and implementation changes. Because the repository already contains prior results, any prediction recorded now is a *prospective re-analysis plan*, not a claim that the complete study was preregistered before observing data.
3. **Move the statistical unit from four views to features.** View-level points remain descriptive only. The confirmatory analysis is performed at feature level, with within-view metrics and a stratified pooled analysis so heterogeneous feature types are not forced onto one raw shift scale.
4. **Add uncertainty and multiplicity handling.** Report effect sizes, bootstrap confidence intervals, stratified permutation tests, repeated model seeds, and Benjamini-Hochberg FDR-adjusted q-values for the family of view-level association tests.
5. **Add falsification conditions.** The framework is challenged if no negative invariance–discriminability relationship is observed at feature level, if an apparent relationship is confined to one view/model, or if it disappears under label-provenance sensitivity analysis.
6. **Instrument label provenance now.** Preserve the family label as the prediction target. Record where the target came from and which provenance fields are actually available rather than silently treating the label as direct malware truth.

## Important clarification: labels are not removed from the target

The experiments still require a malware-family target `y`. Leakage control means that family names, vendor detections, signatures, and analyst/detection strings must not be allowed to enter the **input feature matrix `X`**. The existing controlled/permitted leakage ablation intentionally compares feature sets with and without those input-side filters while keeping the same target labels.

The supervisor's new concern is different: the **target labels themselves were assigned from vendor records**. A measured temporal change could therefore contain both real malware change and changes in labelling practice. The response is to record and test label provenance where possible, not to train a classifier with no labels.

## Public-dataset provenance boundary

The official Avast-CTU dataset documentation exposes, per sample, SHA-256, malware family, malware type, and detection date. It states that samples were identified/classified in Avast backend systems. It does not publish a separate per-sample labelling date, antivirus-engine version, or cross-vendor agreement field. The revised pipeline therefore records these fields explicitly as `not_available_in_public_dataset` unless an additional defensible source is added later. Missing provenance is reported as a limitation, never inferred.

## Executable experiment changes

### A. Feature-level invariance–discriminability analysis

New script: `scripts/22_supervisor_feature_level_analysis.py`

- Token views (API and artifact TF-IDF):
  - discriminability effect size = weighted dispersion of class-conditional token prevalence around overall training prevalence;
  - temporal shift = Bernoulli Jensen-Shannon divergence between earlier and future prevalence for each feature.
- Dense views (behavioural counts and static PE):
  - discriminability effect size = one-way eta-squared across malware families in the training window;
  - temporal shift = two-sample Kolmogorov-Smirnov statistic per feature.
- Each view is analysed separately.
- For the pooled test, discriminability and shift are converted to within-view percentile ranks. Spearman association is then estimated across feature-level observations while preserving view strata.
- 95% confidence intervals are obtained by stratified bootstrap resampling of features.
- A stratified permutation test breaks the discriminability/shift pairing within each view to estimate the null distribution.
- View-level p-values are adjusted with Benjamini-Hochberg FDR.

Interpretation is expressed as `discriminability vs shift`: the proposed trade-off predicts a **positive** association. Equivalently, `discriminability vs invariance` predicts a negative association.

### B. Repeated-seed and model-class robustness

New script: `scripts/23_supervisor_model_robustness.py`

- Re-runs API-only and four-view fusion on the fixed global chronological split.
- Uses repeated seeds for SGD and LightGBM.
- Reports per-model/per-view Macro-F1 mean, standard deviation, percentile confidence interval, and the paired fusion-minus-API effect for each seed.
- The purpose is not hyperparameter search. It tests whether the ordering is an artefact of one random seed or one model class.

### C. Label-provenance audit

New script: `scripts/24_label_provenance_audit.py`

- reads the public metadata already ingested by the pipeline;
- writes a machine-readable provenance manifest;
- records that family classification originates from the Avast-CTU public label file / Avast backend records;
- records detection date as available;
- records separate labelling date, engine version, and cross-vendor agreement as unavailable in the public dataset unless explicitly supplied later;
- summarizes family/date coverage so any temporal label-composition change is visible.

This audit does **not** claim to separate malware drift from vendor policy drift on its own. It makes the identifiability boundary explicit and creates the hooks needed for a later independent-label or cross-corpus sensitivity analysis.

### D. Existing leakage ablation is retained

`scripts/07_leakage_ablation.py` remains useful for RQ2 because it changes input-side family-name filtering while holding the target labels constant. Its current null result should be reported as a bounded result: exact family-name filtering did not change performance in the evaluated allow-listed fields; it does not prove that all indirect leakage is absent.

## Route A / Route B decision

The experiments above are required regardless of theoretical anchor. If Route B (integration-asymmetry) is selected, add H5 and a linkage-contribution analysis using existing ablations. That should be a separate, explicitly theory-driven analysis. It is intentionally not silently added here before the theoretical route is chosen.

## Falsification table to mirror in Annexure A

| Claim | Result that challenges/refutes it |
|---|---|
| Feature-level invariance–discriminability relationship | pooled stratified association is near zero with a CI spanning practically meaningful positive and negative effects, and no consistent within-view evidence |
| Mechanism is general rather than one-view artefact | relationship appears only in one feature view or reverses across views without an explained moderator |
| Robust to model choice | API/fusion ordering or temporal conclusions reverse across model classes/seeds |
| Leakage is controlled | blocked label/detection tokens materially improve performance when permitted, or appear among influential controlled features |
| Temporal drift is malware-related rather than only labelling-related | the apparent effect disappears when a defensible independent/provenance-controlled label set is used |

The final row may remain unresolved if the public dataset lacks the required provenance. In that case the thesis must state the identifiability limitation and pursue a credible route to independent-label or cross-corpus replication rather than manufacture unavailable metadata.
