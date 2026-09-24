import json

import joblib
import pandas as pd

from src.predict_pair_model import predict_pair_model


class ProbabilityModel:
    classes_ = [0, 1]

    def predict_proba(self, values):
        score = values[:, 0].clip(0.0, 1.0)
        return __import__("numpy").column_stack([1.0 - score, score])


def test_inference_uses_saved_order_deduplicates_and_allows_multiple_matches(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    joblib.dump(ProbabilityModel(), model_dir / "final_model.joblib")
    (model_dir / "feature_columns.json").write_text(json.dumps(["saved_first", "saved_second"]))
    (model_dir / "best_threshold.json").write_text(json.dumps({"threshold": 0.75}))
    features = pd.DataFrame([
        {"source1_entity_id": "S1-1", "candidate_entity_id": "S2-1", "saved_first": 0.9, "saved_second": 0.0, "label": 1, "source1_fold": 0},
        {"source1_entity_id": "S1-1", "candidate_entity_id": "S2-1", "saved_first": 0.2, "saved_second": 0.0},
        {"source1_entity_id": "S1-1", "candidate_entity_id": "S3-1", "saved_first": 0.8, "saved_second": 0.0},
        {"source1_entity_id": "S1-2", "candidate_entity_id": "S2-2", "saved_first": 0.1, "saved_second": 1.0},
    ])
    result = predict_pair_model(features, model_dir, tmp_path / "scores", ["S1-1", "S1-2", "S1-3"])
    assert result["predictions"] == {"S1-1": ["S2-1", "S3-1"], "S1-2": [], "S1-3": []}
    assert len(result["scores"]) == 3
    assert "label" not in result["scores"].columns
