"""Tests for evaluation metric and scoring logic."""

import unittest
from src.evaluate import (
    compute_confusion_components,
    evaluate_predictions,
    load_matches_tsv,
    parse_id_string,
    run_tiny_example_checks,
    score_single_s1,
)


class TestEvaluate(unittest.TestCase):

    def test_tiny_example_runner(self):
        """Test the built-in self check runner."""
        self.assertTrue(run_tiny_example_checks())

    def test_perfect_match(self):
        # Ground truth: 2 matches, Pred: same 2 matches
        # TP=2, FP=0, FN=0 -> 5*2 / (5*2 + 4*0 + 0) = 1.0
        score = score_single_s1(["S2-101", "S3-202"], ["S2-101", "S3-202"])
        self.assertAlmostEqual(score, 1.0)

    def test_extra_match(self):
        # Ground truth: 1 match, Pred: 1 true + 1 extra
        # TP=1, FP=1, FN=0 -> 5*1 / (5*1 + 4*1 + 0) = 5/9 ≈ 0.5555555555555556
        score = score_single_s1(["S2-101"], ["S2-101", "S2-999"])
        self.assertAlmostEqual(score, 5.0 / 9.0)

    def test_missed_match(self):
        # Ground truth: 2 matches, Pred: 1 true
        # TP=1, FP=0, FN=1 -> 5*1 / (5*1 + 4*0 + 1) = 5/6 ≈ 0.8333333333333334
        score = score_single_s1(["S2-101", "S3-202"], ["S2-101"])
        self.assertAlmostEqual(score, 5.0 / 6.0)

    def test_empty_prediction_non_singleton(self):
        # Ground truth: 1 match, Pred: empty
        # TP=0, FP=0, FN=1 -> 0.0
        score = score_single_s1(["S2-101"], [])
        self.assertAlmostEqual(score, 0.0)

    def test_singleton_empty_prediction(self):
        # Ground truth: 0 matches, Pred: empty -> correctly identified singleton
        score = score_single_s1([], [])
        self.assertAlmostEqual(score, 1.0)

    def test_singleton_non_empty_prediction(self):
        # Ground truth: 0 matches, Pred: 1 match -> false merge on singleton
        score = score_single_s1([], ["S2-101"])
        self.assertAlmostEqual(score, 0.0)

    def test_macro_averaging(self):
        gt = {
            "S1-1": ["S2-1", "S3-1"],  # 1.0
            "S1-2": ["S2-2"],          # 5/9
            "S1-3": ["S2-3", "S3-3"],  # 5/6
            "S1-4": ["S2-4"],          # 0.0
            "S1-5": [],                # 1.0
            "S1-6": [],                # 0.0
        }
        pred = {
            "S1-1": ["S2-1", "S3-1"],
            "S1-2": ["S2-2", "S2-99"],
            "S1-3": ["S2-3"],
            "S1-4": [],
            "S1-5": [],
            "S1-6": ["S2-99"],
        }
        res = evaluate_predictions(gt, pred)
        expected = (1.0 + (5.0 / 9.0) + (5.0 / 6.0) + 0.0 + 1.0 + 0.0) / 6.0
        self.assertAlmostEqual(res["macro_f05"], expected)
        self.assertEqual(res["num_entities"], 6)
        self.assertEqual(res["num_singletons"], 2)
        self.assertEqual(res["num_non_singletons"], 4)
        self.assertAlmostEqual(res["singleton_accuracy"], 0.5)

    def test_parse_id_string(self):
        self.assertEqual(parse_id_string(""), [])
        self.assertEqual(parse_id_string(None), [])
        self.assertEqual(parse_id_string("S2-1, S3-2, S2-1"), ["S2-1", "S3-2"])


if __name__ == "__main__":
    unittest.main()
