from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from decision_support_policy import decide, load_policy  # noqa: E402


class DecisionSupportPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = load_policy(PROJECT_ROOT / "configs" / "decision_support_policy.json")

    def case(self, **updates):
        value = {
            "case_id": "test",
            "sample_hash": "0" * 64,
            "extractor_version": "test-extractor",
            "model_version": "test-model",
            "calibrator_version": "test-calibrator",
            "predicted_family": "emotet",
            "top_alternative": "qakbot",
            "evidence": ["api:example"],
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
        value.update(updates)
        return value

    def test_state_a_requires_all_support_gates(self):
        result = decide(
            self.case(
                confidence=0.8,
                confidence_provenance="calibrated",
                threshold_validated=True,
            ),
            self.policy,
        )
        self.assertEqual(result["state"], "A")

    def test_uncalibrated_high_score_remains_state_b(self):
        result = decide(
            self.case(confidence=0.99, confidence_provenance="uncalibrated"),
            self.policy,
        )
        self.assertEqual(result["state"], "B")
        self.assertIn("confidence_not_calibrated", result["reason_codes"])

    def test_low_confidence_is_deferred(self):
        result = decide(
            self.case(confidence=0.59, confidence_provenance="calibrated"),
            self.policy,
        )
        self.assertEqual(result["state"], "C")

    def test_temporal_trigger_is_deferred(self):
        result = decide(self.case(temporal_status="triggered"), self.policy)
        self.assertEqual(result["state"], "C")

    def test_novelty_signal_maps_to_state_d(self):
        result = decide(self.case(novelty_flag=True), self.policy)
        self.assertEqual(result["state"], "D")

    def test_scope_gate_has_precedence_over_deferral_gate(self):
        result = decide(
            self.case(schema_supported=False, data_quality_ok=False, novelty_flag=True),
            self.policy,
        )
        self.assertEqual(result["state"], "D")
        self.assertEqual(result["reason_codes"], ["unsupported_schema"])

    def test_invalid_confidence_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "within"):
            decide(self.case(confidence=1.1), self.policy)

    def test_missing_safety_field_is_rejected(self):
        case = self.case()
        del case["data_quality_ok"]
        with self.assertRaisesRegex(ValueError, "data_quality_ok"):
            decide(case, self.policy)

    def test_audit_record_retains_versions_evidence_and_gates(self):
        result = decide(self.case(), self.policy)
        self.assertEqual(result["sample_hash"], "0" * 64)
        self.assertEqual(result["model_version"], "test-model")
        self.assertEqual(result["evidence"], ["api:example"])
        self.assertTrue(result["input_gates"]["schema_supported"])


if __name__ == "__main__":
    unittest.main()
