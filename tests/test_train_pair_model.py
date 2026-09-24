import tempfile
import unittest

import pandas as pd

from src.train_pair_model import (
    generate_oof_predictions,
    identify_feature_columns,
    predictions_from_threshold,
    search_threshold,
)


class RecordingClassifier:
    fits = []

    def fit(self, x, y, sample_weight=None):
        self.__class__.fits.append(set(x[:, 0].tolist()))
        self.p = float(y.mean())
        self.classes_ = [0, 1]
        return self

    def predict_proba(self, x):
        import numpy as np

        return np.column_stack([1 - self.p * (x[:, 0] > 0), self.p * (x[:, 0] > 0)])


class TestTrainPairModel(unittest.TestCase):
    def setUp(self):
        RecordingClassifier.fits = []

    def test_oof_models_never_fit_validation_fold_and_excludes_metadata(self):
        frame = pd.DataFrame(
            {
                "source1_entity_id": ["S1-a", "S1-b", "S1-c", "S1-d", "S1-e", "S1-f"],
                "candidate_entity_id": ["S2-a", "S2-b", "S2-c", "S2-d", "S2-e", "S2-f"],
                "fold": [0, 0, 1, 1, 2, 2],
                "label": [1, 0, 1, 0, 1, 0],
                "useful_feature": [0, 1, 2, 3, 4, 5],
            }
        )
        self.assertEqual(identify_feature_columns(frame), ["useful_feature"])
        scores, columns = generate_oof_predictions(frame, classifier_factory=RecordingClassifier)
        self.assertEqual(columns, ["useful_feature"])
        self.assertEqual(len(scores), len(frame))
        self.assertTrue(scores["score"].between(0, 1).all())
        # Each fit sees four rows: never the two values in its validation fold.
        self.assertEqual([len(values) for values in RecordingClassifier.fits], [4, 4, 4])
        self.assertEqual(set(RecordingClassifier.fits[0]), {2, 3, 4, 5})

    def test_threshold_allows_empty_and_multiple_predictions(self):
        scores = pd.DataFrame(
            {
                "source1_entity_id": ["S1-a", "S1-a", "S1-b"],
                "candidate_entity_id": ["S2-a", "S3-a", "S2-b"],
                "score": [0.9, 0.8, 0.1],
            }
        )
        self.assertEqual(predictions_from_threshold(scores, 0.95), {})
        self.assertEqual(predictions_from_threshold(scores, 0.8), {"S1-a": ["S2-a", "S3-a"]})

    def test_threshold_search_uses_exact_evaluator(self):
        scores = pd.DataFrame(
            {
                "source1_entity_id": ["S1-a", "S1-a", "S1-b"],
                "candidate_entity_id": ["S2-a", "S3-a", "S2-b"],
                "score": [0.9, 0.2, 0.1],
            }
        )
        threshold, metrics = search_threshold(scores, {"S1-a": ["S2-a"], "S1-b": []}, thresholds=[0.0, 0.5, 0.95])
        self.assertEqual(threshold, 0.5)
        self.assertAlmostEqual(metrics["macro_f05"], 1.0)

    def test_endpoint_folds_exclude_candidate_validation_records(self):
        frame = pd.DataFrame(
            {
                "source1_entity_id": ["S1-a", "S1-a", "S1-b", "S1-b", "S1-c", "S1-c"],
                "candidate_entity_id": ["S2-a", "S2-b", "S2-c", "S2-d", "S2-e", "S2-f"],
                "source1_fold": [0, 0, 1, 1, 2, 2],
                "candidate_fold": [0, 1, 1, 2, 2, 0],
                "label": [1, 0, 1, 0, 1, 0],
                "useful_feature": [0, 1, 2, 3, 4, 5],
            }
        )
        generate_oof_predictions(frame, classifier_factory=RecordingClassifier)
        self.assertEqual([len(values) for values in RecordingClassifier.fits], [3, 3, 3])
        self.assertEqual(set(RecordingClassifier.fits[0]), {2, 3, 4})


if __name__ == "__main__":
    unittest.main()
