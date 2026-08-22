# Four-state decision-support demonstrator

This artifact implements the thesis policy as a deterministic, auditable
research demonstrator for a malware triage analyst. It is not an autonomous
classifier, a production control, a usability result, or a prospective
deployment validation.

## Reproduce

```bash
.venv/bin/python scripts/19_decision_support_experiments.py
.venv/bin/python -m unittest tests/test_decision_support_policy.py
```

The standalone JSONL interface is `scripts/18_decision_support_demonstrator.py`;
the machine-readable gates are in `configs/decision_support_policy.json`.

## Results

- Contract experiment: 8/8 cases passed,
  exercising all four states and gate precedence.
- Confidence replay: 8840 archived test
  samples with uncalibrated max-probability scores at or above the 0.60
  deferral gate remain in state B; 956 below 0.60 are deferred to state C.
  Even the 7,625 scores at or above 0.80 cannot reach state A because
  their provenance is uncalibrated.
- Temporal replay: 7 of
  19 rolling-origin windows activate state C; the remaining
  12 remain state B because per-case
  calibrated confidence is unavailable in this evidence lane.
- Open-set capability replay: 7 of
  9 held-out-family experiments meet the illustrative 0.80
  unknown-rejection gate and exercise state D;
  2 fail the capability gate and are
  deferred. This is an aggregate family-experiment replay, not a claim that each
  unknown sample was identified.

## Interpretation boundary

The strongest state A appears only in a synthetic contract case proving that
the code path works. The retained empirical archive lacks a jointly aligned,
per-sample record containing calibrated confidence, temporal status, novelty
status, explanation availability, and analyst outcome. Prospective replay and
a separate analyst study are therefore required before operational use. The
policy's methodological lineage is limited to Pradeep and Morris's (2022)
computing-research distinction between what is observed, what can be known,
and what action is valued. No unrelated application-domain claim is used.
