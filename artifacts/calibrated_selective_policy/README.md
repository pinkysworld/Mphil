# Time-ordered calibration and selective-policy experiment

This directory contains the complete public replay package for experiment CP1.
It separates model training, calibration fitting, threshold selection, and
future testing at strict date boundaries. The 64 validation samples tied to an
outer split boundary date are excluded before the validation role is divided.

## Reproduce from a fresh clone

Create the pinned environment as described in the repository root, then run:

```bash
.venv/bin/python scripts/20_calibrated_selective_policy.py --input-mode replay-bundle
.venv/bin/python -m unittest tests/test_calibrated_selective_policy.py
```

`replay_inputs.csv.gz` is a deterministic, 13,650-row input bundle. It contains
the base SGD model's decision scores and uncalibrated probabilities for the
calibration, policy-selection, and future-test roles. The earlier model-training
role is represented by its fitted outputs; no test labels are used to fit the
model, calibrator, or confidence thresholds. `replay_inputs_metadata.json`
records the model configuration, role dates and counts, sigmoid parameters,
source-cache hashes, class order, and bundle checksum.

When the larger defended feature caches are available locally, the bundle can
be regenerated and checked against scikit-learn's `CalibratedClassifierCV`:

```bash
.venv/bin/python scripts/20_calibrated_selective_policy.py --input-mode source-cache
```

The replay bundle exists because the 130 MB sparse feature cache exceeds
GitHub's normal single-file limit. Publishing the small score bundle preserves
the calibration and policy experiment while keeping the raw dataset and large
derived caches outside Git.

## Time-ordered roles

- Model training: 35,262 samples, ending 22 June 2019.
- Calibration fitting: 1,878 samples, 23 June to 25 July 2019.
- Threshold selection: 1,976 samples, 26 July to 8 September 2019.
- Future test: 9,796 samples, beginning 9 September 2019.

## Results and boundary

Sigmoid calibration reduces future-test ECE from 0.1031 to 0.0318, but worsens
multiclass Brier score and log loss. The validation-selected calibrated policy
records 2.56% selective error on the future test and therefore misses its 1%
target. The corresponding uncalibrated policy records 0.34% error at 90.31%
coverage. These results reject the inference that a lower ECE automatically
produces safer future routing. They remain a single-corpus retrospective
sensitivity experiment, not prospective external or analyst-outcome
validation.
