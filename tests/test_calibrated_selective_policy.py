from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "calibrated_selective_policy",
    ROOT / "scripts" / "20_calibrated_selective_policy.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CalibratedSelectivePolicyTests(unittest.TestCase):
    def test_perfect_probabilities_have_zero_calibration_and_brier_error(self) -> None:
        classes = np.array(["a", "b"])
        truth = np.array(["a", "b"])
        probabilities = np.array([[1.0, 0.0], [0.0, 1.0]])
        self.assertEqual(
            MODULE.expected_calibration_error(truth, probabilities, classes), 0.0
        )
        self.assertEqual(MODULE.multiclass_brier(truth, probabilities, classes), 0.0)

    def test_threshold_maximises_coverage_subject_to_error_bound(self) -> None:
        classes = np.array(["a", "b"])
        truth = np.array(["a", "b", "b", "b", "b"])
        probabilities = np.array(
            [
                [0.90, 0.10],
                [0.80, 0.20],
                [0.30, 0.70],
                [0.40, 0.60],
                [0.55, 0.45],
            ]
        )
        selected = MODULE.select_threshold(
            truth, probabilities, classes, target_error=0.25, min_cases=1
        )
        self.assertAlmostEqual(selected["threshold"], 0.60)
        self.assertEqual(selected["n_selected"], 4)
        self.assertAlmostEqual(selected["coverage"], 0.80)
        self.assertAlmostEqual(selected["selective_error"], 0.25)

    def test_frozen_threshold_reports_selected_and_deferred_cases(self) -> None:
        classes = np.array(["a", "b"])
        truth = np.array(["a", "b", "b"])
        probabilities = np.array([[0.8, 0.2], [0.7, 0.3], [0.4, 0.6]])
        result = MODULE.evaluate_threshold(truth, probabilities, classes, 0.65)
        self.assertEqual(result["n_selected"], 2)
        self.assertEqual(result["n_deferred"], 1)
        self.assertAlmostEqual(result["coverage"], 2 / 3)
        self.assertAlmostEqual(result["selective_error"], 0.5)

    def test_validation_roles_exclude_tied_boundaries_and_preserve_date_order(self) -> None:
        labels = pd.DataFrame(
            {
                "sha256": [f"{index:064x}" for index in range(11)],
                "family": ["a", "b", "a", "a", "b", "a", "b", "a", "b", "a", "b"],
                "date": pd.to_datetime(
                    [
                        "2020-01-01",
                        "2020-01-02",
                        "2020-01-02",
                        "2020-01-03",
                        "2020-01-03",
                        "2020-01-04",
                        "2020-01-05",
                        "2020-01-06",
                        "2020-01-07",
                        "2020-01-07",
                        "2020-01-08",
                    ]
                ),
            }
        )
        split = {"train": [0, 1], "val": [2, 3, 4, 5, 6, 7, 8], "test": [9, 10]}
        calibration, policy, metadata = MODULE.strict_validation_roles(labels, split)
        np.testing.assert_array_equal(calibration, np.array([3, 4]))
        np.testing.assert_array_equal(policy, np.array([5, 6, 7]))
        self.assertEqual(metadata["excluded_validation_boundary_ties"], 2)
        self.assertLess(
            labels.iloc[calibration]["date"].max(), labels.iloc[policy]["date"].min()
        )
        self.assertLess(labels.iloc[policy]["date"].max(), labels.iloc[split["test"]]["date"].min())

        bundle, metadata, classes, decision_columns, probability_columns = (
            MODULE.load_public_replay_bundle()
        )
        self.assertEqual(len(bundle), 13650)
        self.assertEqual(metadata["roles"]["n_calibration"], 1878)
        self.assertEqual(metadata["roles"]["n_policy_selection"], 1976)
        self.assertEqual(metadata["roles"]["n_test"], 9796)

        calibration_bundle = bundle[bundle["role"] == "calibration"]
        policy_bundle = bundle[bundle["role"] == "policy_selection"]
        test_bundle = bundle[bundle["role"] == "future_test"]
        calibrated_policy, _ = MODULE.fit_sigmoid_probabilities(
            calibration_bundle[decision_columns].to_numpy(dtype=float),
            calibration_bundle["true_family"].to_numpy(dtype=str),
            policy_bundle[decision_columns].to_numpy(dtype=float),
            classes,
        )
        calibrated_test, _ = MODULE.fit_sigmoid_probabilities(
            calibration_bundle[decision_columns].to_numpy(dtype=float),
            calibration_bundle["true_family"].to_numpy(dtype=str),
            test_bundle[decision_columns].to_numpy(dtype=float),
            classes,
        )
        selected = MODULE.select_threshold(
            policy_bundle["true_family"].to_numpy(dtype=str),
            calibrated_policy,
            classes,
        )
        future = MODULE.evaluate_threshold(
            test_bundle["true_family"].to_numpy(dtype=str),
            calibrated_test,
            classes,
            float(selected["threshold"]),
        )
        self.assertAlmostEqual(selected["threshold"], 0.655995732535332, places=12)
        self.assertEqual(future["n_selected"], 8759)
        self.assertAlmostEqual(future["selective_error"], 0.02557369562735472)
        self.assertEqual(len(probability_columns), len(classes))


if __name__ == "__main__":
    unittest.main()
