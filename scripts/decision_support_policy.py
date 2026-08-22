"""Deterministic four-state decision-support policy for the thesis demonstrator.

The policy is deliberately conservative. It separates a model score from an
analyst-facing decision and prevents an uncalibrated score from reaching the
strongest support state. The module has no third-party dependencies so that
the policy contract can be tested independently of model training.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REQUIRED_BOOLEAN_FIELDS = (
    "schema_supported",
    "taxonomy_supported",
    "novelty_flag",
    "open_set_monitor_pass",
    "data_quality_ok",
    "evidence_available",
    "abstain",
    "threshold_validated",
    "explanation_available",
)


def load_policy(path: Path) -> dict[str, Any]:
    """Load and validate the machine-readable policy configuration."""

    with path.open(encoding="utf-8") as handle:
        policy = json.load(handle)
    thresholds = policy.get("thresholds", {})
    support = thresholds.get("supported_confidence")
    defer = thresholds.get("defer_below_confidence")
    unknown = thresholds.get("minimum_unknown_rejection_rate")
    if not all(isinstance(value, (int, float)) for value in (support, defer, unknown)):
        raise ValueError("All policy thresholds must be numeric.")
    if not 0 <= defer < support <= 1:
        raise ValueError("Confidence thresholds must satisfy 0 <= defer < support <= 1.")
    if not 0 <= unknown <= 1:
        raise ValueError("Unknown-rejection threshold must be within [0, 1].")
    if set(policy.get("states", {})) != {"A", "B", "C", "D"}:
        raise ValueError("Policy must define exactly states A, B, C, and D.")
    return policy


def validate_case(case: dict[str, Any]) -> None:
    """Validate the fields that act as policy safety gates."""

    missing = [field for field in REQUIRED_BOOLEAN_FIELDS if field not in case]
    if "temporal_status" not in case:
        missing.append("temporal_status")
    if "confidence" not in case:
        missing.append("confidence")
    if "confidence_provenance" not in case:
        missing.append("confidence_provenance")
    if missing:
        raise ValueError(f"Case is missing required fields: {', '.join(sorted(missing))}")
    for field in REQUIRED_BOOLEAN_FIELDS:
        if not isinstance(case[field], bool):
            raise ValueError(f"{field} must be boolean.")
    confidence = case["confidence"]
    if confidence is not None:
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
            raise ValueError("confidence must be numeric or null.")
        if not 0 <= float(confidence) <= 1:
            raise ValueError("confidence must be within [0, 1].")
    if case["confidence_provenance"] not in {
        "calibrated",
        "uncalibrated",
        "not_available",
    }:
        raise ValueError(
            "confidence_provenance must be calibrated, uncalibrated, or not_available."
        )


def _decision(
    case: dict[str, Any],
    policy: dict[str, Any],
    state: str,
    reason_codes: list[str],
) -> dict[str, Any]:
    labels = policy["states"]
    return {
        "case_id": str(case.get("case_id", "unspecified")),
        "sample_hash": case.get("sample_hash"),
        "policy_version": policy["policy_version"],
        "extractor_version": case.get("extractor_version"),
        "model_version": case.get("model_version"),
        "calibrator_version": case.get("calibrator_version"),
        "state": state,
        "state_label": labels[state],
        "reason_codes": reason_codes,
        "predicted_family": case.get("predicted_family"),
        "top_alternative": case.get("top_alternative"),
        "confidence": case.get("confidence"),
        "confidence_provenance": case["confidence_provenance"],
        "temporal_status": case["temporal_status"],
        "evidence": case.get("evidence", []),
        "input_gates": {
            field: case[field] for field in REQUIRED_BOOLEAN_FIELDS
        },
        "independent_analyst_review_required": state != "A",
    }


def decide(case: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    """Map one evidence record to an analyst-facing state with an audit trace.

    Precedence is intentional: scope/novelty gates dominate, followed by
    deferral gates. State A is available only when all gates pass and a
    calibrated score exceeds a separately validated threshold.
    """

    validate_case(case)
    if not case["schema_supported"]:
        return _decision(case, policy, "D", ["unsupported_schema"])
    if not case["taxonomy_supported"]:
        return _decision(case, policy, "D", ["unsupported_taxonomy"])
    if case["novelty_flag"]:
        return _decision(case, policy, "D", ["novelty_signal"])

    if not case["open_set_monitor_pass"]:
        return _decision(case, policy, "C", ["open_set_capability_below_gate"])
    if not case["data_quality_ok"]:
        return _decision(case, policy, "C", ["data_quality_failure"])
    if not case["evidence_available"]:
        return _decision(case, policy, "C", ["evidence_unavailable"])
    if case["abstain"]:
        return _decision(case, policy, "C", ["model_abstained"])
    if case["temporal_status"] in set(policy["temporal_review_statuses"]):
        return _decision(case, policy, "C", ["temporal_review_trigger"])

    confidence = case["confidence"]
    thresholds = policy["thresholds"]
    if confidence is not None and confidence < thresholds["defer_below_confidence"]:
        return _decision(case, policy, "C", ["confidence_below_deferral_gate"])

    qualifies_for_a = (
        confidence is not None
        and case["confidence_provenance"] == "calibrated"
        and case["threshold_validated"]
        and case["explanation_available"]
        and confidence >= thresholds["supported_confidence"]
    )
    if qualifies_for_a:
        return _decision(case, policy, "A", ["all_support_gates_passed"])

    reasons: list[str] = []
    if confidence is None:
        reasons.append("case_confidence_unavailable")
    elif case["confidence_provenance"] != "calibrated":
        reasons.append("confidence_not_calibrated")
    elif confidence < thresholds["supported_confidence"]:
        reasons.append("confidence_below_support_gate")
    if not case["threshold_validated"]:
        reasons.append("threshold_not_independently_validated")
    if not case["explanation_available"]:
        reasons.append("explanation_unavailable")
    if not reasons:
        reasons.append("analyst_confirmation_required")
    return _decision(case, policy, "B", reasons)
