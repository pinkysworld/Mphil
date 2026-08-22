"""Reproduce the bounded experiments for the four-state demonstrator.

No model is retrained. The script replays the policy against archived,
aggregate evidence from the defended 11 July 2026 result bundle and runs
synthetic contract cases. Aggregate results are never represented as
sample-level operational validation.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from decision_support_policy import decide, load_policy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = PROJECT_ROOT / "configs" / "decision_support_policy.json"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "decision_support"
CALIBRATION_PATH = (
    PROJECT_ROOT / "results" / "2026-07-11" / "calibration" / "selective_prediction.csv"
)
TEMPORAL_PATH = (
    PROJECT_ROOT
    / "results"
    / "2026-07-11"
    / "retraining_trigger"
    / "retraining_trigger_windows.csv"
)
OPEN_SET_PATH = PROJECT_ROOT / "artifacts" / "open_set" / "heldout_family_summary.csv"
BASE_EVIDENCE_COMMIT = "8174d6bacd675f28a2c3fe3f7cde9f186001d841"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty experiment output: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def base_case(case_id: str) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "sample_hash": None,
        "extractor_version": "archive-2026-07-11.1",
        "model_version": "archived-summary-only",
        "calibrator_version": None,
        "predicted_family": "illustrative_family",
        "top_alternative": "illustrative_alternative",
        "evidence": [],
        "confidence": None,
        "confidence_provenance": "not_available",
        "schema_supported": True,
        "taxonomy_supported": True,
        "novelty_flag": False,
        "open_set_monitor_pass": True,
        "data_quality_ok": True,
        "evidence_available": True,
        "abstain": False,
        "temporal_status": "stable",
        "threshold_validated": False,
        "explanation_available": True,
    }


def contract_experiment(policy: dict[str, Any]) -> list[dict[str, Any]]:
    specifications = [
        (
            "contract_a_calibrated_support",
            "A",
            {
                "confidence": 0.92,
                "confidence_provenance": "calibrated",
                "threshold_validated": True,
            },
        ),
        (
            "contract_b_uncalibrated_high",
            "B",
            {"confidence": 0.92, "confidence_provenance": "uncalibrated"},
        ),
        (
            "contract_b_calibrated_mid",
            "B",
            {
                "confidence": 0.72,
                "confidence_provenance": "calibrated",
                "threshold_validated": True,
            },
        ),
        (
            "contract_c_low_confidence",
            "C",
            {"confidence": 0.40, "confidence_provenance": "uncalibrated"},
        ),
        ("contract_c_temporal_trigger", "C", {"temporal_status": "triggered"}),
        ("contract_c_missing_evidence", "C", {"evidence_available": False}),
        ("contract_d_novelty", "D", {"novelty_flag": True}),
        ("contract_d_unsupported_schema", "D", {"schema_supported": False}),
    ]
    rows = []
    for case_id, expected, updates in specifications:
        case = base_case(case_id)
        case.update(updates)
        result = decide(case, policy)
        rows.append(
            {
                "case_id": case_id,
                "expected_state": expected,
                "observed_state": result["state"],
                "passed": result["state"] == expected,
                "reason_codes": "|".join(result["reason_codes"]),
            }
        )
    if not all(row["passed"] for row in rows):
        raise AssertionError("One or more decision-policy contract cases failed.")
    return rows


def confidence_experiment(policy: dict[str, Any]) -> list[dict[str, Any]]:
    rows = read_csv(CALIBRATION_PATH)
    support_threshold = float(policy["thresholds"]["supported_confidence"])
    defer_threshold = float(policy["thresholds"]["defer_below_confidence"])
    full = next(
        row
        for row in rows
        if row["method"] == "max_prob" and float(row["threshold"]) == 0.0
    )
    selected = next(
        row
        for row in rows
        if row["method"] == "max_prob"
        and float(row["threshold"]) == support_threshold
    )
    retained_for_assistance = next(
        row
        for row in rows
        if row["method"] == "max_prob"
        and float(row["threshold"]) == defer_threshold
    )
    total = int(full["n_covered"])
    high = int(selected["n_covered"])
    at_least_defer = int(retained_for_assistance["n_covered"])
    middle = at_least_defer - high
    deferred = total - at_least_defer

    high_case = base_case("confidence_covered_uncalibrated")
    high_case.update(
        {
            "confidence": support_threshold,
            "confidence_provenance": "uncalibrated",
        }
    )
    low_case = base_case("confidence_below_deferral_gate")
    low_case.update(
        {
            "confidence": defer_threshold - 0.01,
            "confidence_provenance": "uncalibrated",
        }
    )
    middle_case = base_case("confidence_between_deferral_and_support_gates")
    middle_case.update(
        {
            "confidence": (defer_threshold + support_threshold) / 2,
            "confidence_provenance": "uncalibrated",
        }
    )
    high_decision = decide(high_case, policy)
    middle_decision = decide(middle_case, policy)
    low_decision = decide(low_case, policy)
    return [
        {
            "evidence_group": "score_at_or_above_0.80",
            "n_samples": high,
            "fraction": round(high / total, 6),
            "archived_selective_accuracy": selected["selective_accuracy"],
            "confidence_provenance": "uncalibrated",
            "policy_state": high_decision["state"],
            "reason_codes": "|".join(high_decision["reason_codes"]),
        },
        {
            "evidence_group": "score_at_or_above_0.60_and_below_0.80",
            "n_samples": middle,
            "fraction": round(middle / total, 6),
            "archived_selective_accuracy": "not_reported_for_band",
            "confidence_provenance": "uncalibrated",
            "policy_state": middle_decision["state"],
            "reason_codes": "|".join(middle_decision["reason_codes"]),
        },
        {
            "evidence_group": "score_below_0.60",
            "n_samples": deferred,
            "fraction": round(deferred / total, 6),
            "archived_selective_accuracy": "not_reported_for_band",
            "confidence_provenance": "uncalibrated",
            "policy_state": low_decision["state"],
            "reason_codes": "|".join(low_decision["reason_codes"]),
        },
    ]


def temporal_experiment(policy: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for source in read_csv(TEMPORAL_PATH):
        triggered = source["trigger"].strip().lower() == "true"
        case = base_case(f"temporal_{source['boundary']}")
        case["temporal_status"] = "triggered" if triggered else "stable"
        decision = decide(case, policy)
        rows.append(
            {
                "boundary": source["boundary"],
                "n_test": int(source["n_test"]),
                "api_macro_f1": source["api_macro_f1"],
                "archived_trigger": triggered,
                "trigger_reason": source["trigger_reason"],
                "policy_state": decision["state"],
                "reason_codes": "|".join(decision["reason_codes"]),
            }
        )
    return rows


def open_set_experiment(policy: dict[str, Any]) -> list[dict[str, Any]]:
    minimum = float(policy["thresholds"]["minimum_unknown_rejection_rate"])
    sources = [
        row
        for row in read_csv(OPEN_SET_PATH)
        if row["view_request"] == "fusion" and row["score_method"] == "max_prob"
    ]
    rows = []
    for source in sources:
        rejection = float(source["unknown_rejection_rate"])
        case = base_case(f"open_set_{source['heldout_family']}")
        if rejection >= minimum:
            case["novelty_flag"] = True
        else:
            case["open_set_monitor_pass"] = False
        decision = decide(case, policy)
        rows.append(
            {
                "heldout_family": source["heldout_family"],
                "unknown_test_n": int(source["n_unknown_test"]),
                "unknown_rejection_rate": source["unknown_rejection_rate"],
                "minimum_required_rate": minimum,
                "capability_gate_passed": rejection >= minimum,
                "policy_state": decision["state"],
                "reason_codes": "|".join(decision["reason_codes"]),
            }
        )
    if len(rows) != 9:
        raise ValueError(f"Expected nine held-out-family rows, found {len(rows)}.")
    return rows


def counts(rows: Iterable[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row[key]) for row in rows).items()))


def weighted_counts(
    rows: Iterable[dict[str, Any]], key: str, weight: str
) -> dict[str, int]:
    totals: Counter[str] = Counter()
    for row in rows:
        totals[str(row[key])] += int(row[weight])
    return dict(sorted(totals.items()))


def build_readme_template(summary: dict[str, Any]) -> str:
    contract = summary["experiments"]["contract"]
    confidence = summary["experiments"]["confidence_gate"]
    temporal = summary["experiments"]["temporal_gate"]
    open_set = summary["experiments"]["open_set_gate"]
    return f"""# Four-state decision-support demonstrator

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

- Contract experiment: {contract['passed']}/{contract['total']} cases passed,
  exercising all four states and gate precedence.
- Confidence replay: {confidence['state_counts'].get('B', 0)} archived test
  samples with uncalibrated max-probability scores at or above 0.80 remain in
  state B; {confidence['state_counts'].get('C', 0)} below the illustrative gate
  are deferred to state C. No archived case is promoted to state A.
- Temporal replay: {temporal['state_counts'].get('C', 0)} of
  {temporal['windows']} rolling-origin windows activate state C; the remaining
  {temporal['state_counts'].get('B', 0)} remain state B because per-case
  calibrated confidence is unavailable in this evidence lane.
- Open-set capability replay: {open_set['state_counts'].get('D', 0)} of
  {open_set['families']} held-out-family experiments meet the illustrative 0.80
  unknown-rejection gate and exercise state D;
  {open_set['state_counts'].get('C', 0)} fail the capability gate and are
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
"""


def build_readme(summary: dict[str, Any]) -> str:
    text = build_readme_template(summary)
    confidence = summary["experiments"]["confidence_gate"]
    old = (
        "samples with uncalibrated max-probability scores at or above 0.80 remain in\n"
        f"  state B; {confidence['state_counts'].get('C', 0)} below the illustrative gate\n"
        "  are deferred to state C. No archived case is promoted to state A."
    )
    new = (
        "samples with uncalibrated max-probability scores at or above the 0.60\n"
        f"  deferral gate remain in state B; {confidence['state_counts'].get('C', 0)} "
        "below 0.60 are deferred to state C.\n"
        "  Even the 7,625 scores at or above 0.80 cannot reach state A because\n"
        "  their provenance is uncalibrated."
    )
    if old not in text:
        raise AssertionError("README confidence paragraph template did not match.")
    return text.replace(old, new)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    policy = load_policy(POLICY_PATH)
    contract_rows = contract_experiment(policy)
    confidence_rows = confidence_experiment(policy)
    temporal_rows = temporal_experiment(policy)
    open_set_rows = open_set_experiment(policy)

    outputs = {
        "contract_cases.csv": contract_rows,
        "confidence_gate_replay.csv": confidence_rows,
        "temporal_gate_replay.csv": temporal_rows,
        "open_set_gate_replay.csv": open_set_rows,
    }
    for name, rows in outputs.items():
        write_csv(OUTPUT_DIR / name, rows)

    summary = {
        "schema_version": "1.0",
        "policy_version": policy["policy_version"],
        "base_evidence_commit": BASE_EVIDENCE_COMMIT,
        "experiments": {
            "contract": {
                "total": len(contract_rows),
                "passed": sum(bool(row["passed"]) for row in contract_rows),
                "state_counts": counts(contract_rows, "observed_state"),
            },
            "confidence_gate": {
                "samples": sum(int(row["n_samples"]) for row in confidence_rows),
                "state_counts": weighted_counts(
                    confidence_rows, "policy_state", "n_samples"
                ),
                "interpretation": "aggregate replay of archived uncalibrated selective-prediction evidence",
            },
            "temporal_gate": {
                "windows": len(temporal_rows),
                "state_counts": counts(temporal_rows, "policy_state"),
                "interpretation": "window-level retrospective replay; trigger uses labelled outcomes",
            },
            "open_set_gate": {
                "families": len(open_set_rows),
                "state_counts": counts(open_set_rows, "policy_state"),
                "interpretation": "family-experiment capability replay, not sample-level novelty decisions",
            },
        },
        "claim_boundary": {
            "state_a_empirically_observed": False,
            "prospective_validation": False,
            "analyst_usability_study": False,
            "production_readiness": False,
        },
    }
    summary_path = OUTPUT_DIR / "experiment_summary.json"
    write_json(summary_path, summary)
    readme_path = OUTPUT_DIR / "README.md"
    readme_path.write_text(build_readme(summary), encoding="utf-8")

    source_paths = [
        POLICY_PATH,
        CALIBRATION_PATH,
        TEMPORAL_PATH,
        OPEN_SET_PATH,
        PROJECT_ROOT / "scripts" / "18_decision_support_demonstrator.py",
        Path(__file__).resolve(),
        PROJECT_ROOT / "scripts" / "decision_support_policy.py",
        PROJECT_ROOT / "tests" / "test_decision_support_policy.py",
    ]
    output_paths = [
        *(OUTPUT_DIR / name for name in outputs),
        summary_path,
        readme_path,
    ]
    manifest = {
        "schema_version": "1.0",
        "base_evidence_commit": BASE_EVIDENCE_COMMIT,
        "sources": [
            {"path": relative(path), "sha256": sha256_file(path)} for path in source_paths
        ],
        "outputs": [
            {"path": relative(path), "sha256": sha256_file(path)} for path in output_paths
        ],
    }
    write_json(OUTPUT_DIR / "manifest.json", manifest)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
