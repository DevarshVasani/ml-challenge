import json

import joblib
import pandas as pd
import pytest

from src.artifact_contract import validate_candidate_frame
from src.candidate_io import iter_candidate_partitions
from src.context_features import add_candidate_context_features
from src.export import validate_submission_data
from src.fold_lookup import fold_for_entity
from src.predict_pair_model import predict_pair_model
from src.streaming_features import write_feature_parts
from src.train_pair_model import predictions_from_threshold
from src.training_data import select_hard_training_rows, training_rows_for_fold


class TinyProbabilityModel:
    classes_ = [0, 1]

    def predict_proba(self, values):
        score = values[:, 0].clip(0.0, 1.0)
        return __import__("numpy").column_stack([1.0 - score, score])


def _candidates():
    return pd.DataFrame([
        {"source1_entity_id": "S1-1", "candidate_entity_id": "S2-1", "candidate_source": "S2", "name_tfidf_score": .9, "address_tfidf_score": .8, "name_rank": 1, "address_rank": 1, "retrieved_by_name": 1, "retrieved_by_address": 1, "retrieval_score": .9},
        {"source1_entity_id": "S1-1", "candidate_entity_id": "S3-1", "candidate_source": "S3", "name_tfidf_score": .9, "address_tfidf_score": 0., "name_rank": 2, "address_rank": 2, "retrieved_by_name": 1, "retrieved_by_address": 0, "retrieval_score": .9},
        {"source1_entity_id": "S1-1", "candidate_entity_id": "S2-9", "candidate_source": "S2", "name_tfidf_score": .1, "address_tfidf_score": .1, "name_rank": 3, "address_rank": 3, "retrieved_by_name": 0, "retrieved_by_address": 1, "retrieval_score": .2},
        {"source1_entity_id": "S1-2", "candidate_entity_id": "S3-2", "candidate_source": "S3", "name_tfidf_score": .4, "address_tfidf_score": .0, "name_rank": 1, "address_rank": 1, "retrieved_by_name": 1, "retrieved_by_address": 0, "retrieval_score": .4},
        {"source1_entity_id": "S1-3", "candidate_entity_id": "S2-3", "candidate_source": "S2", "name_tfidf_score": .8, "address_tfidf_score": .0, "name_rank": 1, "address_rank": 1, "retrieved_by_name": 1, "retrieved_by_address": 0, "retrieval_score": .8},
        {"source1_entity_id": "S1-3", "candidate_entity_id": "S3-3", "candidate_source": "S3", "name_tfidf_score": .3, "address_tfidf_score": .0, "name_rank": 2, "address_rank": 2, "retrieved_by_name": 1, "retrieved_by_address": 0, "retrieval_score": .1},
    ])


def test_contract_parquet_context_and_sampling_are_deterministic(tmp_path):
    frame = _candidates()
    validate_candidate_frame(frame)
    path = tmp_path / "candidates.parquet"
    frame.to_parquet(path, index=False)
    partitions = list(iter_candidate_partitions(path, batch_size=2))
    assert [len(part) for part in partitions] == [3, 1, 2]
    assert all(part["source1_entity_id"].nunique() == 1 for part in partitions)
    feature_dir = tmp_path / "features"
    first_manifest = write_feature_parts(path, feature_dir)
    second_manifest = write_feature_parts(path, feature_dir)
    assert first_manifest["configuration_hash"] == second_manifest["configuration_hash"]
    assert len(list(feature_dir.glob("part-*.parquet"))) == 3

    context = add_candidate_context_features(frame.sample(frac=1, random_state=7).reset_index(drop=True), score_margin=0.0)
    first = context[context.source1_entity_id == "S1-1"].set_index("candidate_entity_id")
    assert first.loc["S2-1", "candidate_count"] == 3
    assert first.loc["S2-1", "retrieval_score_rank"] == 1
    assert first.loc["S3-1", "retrieval_score_rank"] == 2
    assert first.loc["S2-1", "is_strongest_s2_candidate"] == 1
    assert first.loc["S3-1", "is_strongest_s3_candidate"] == 1

    labelled = frame.assign(label=[1, 1, 0, 0, 1, 0], source1_fold=[0, 0, 0, 1, 2, 2], candidate_fold=[0, 0, 1, 1, 2, 0])
    sampled = select_hard_training_rows(labelled, negatives_per_s1=1)
    assert set(sampled.loc[sampled.label == 1, "candidate_entity_id"]) == {"S2-1", "S3-1", "S2-3"}
    assert len(sampled[sampled.source1_entity_id == "S1-1"]) == 3
    excluded = training_rows_for_fold(labelled, validation_fold=0, negatives_per_s1=1)
    assert not ((excluded.source1_fold == 0) | (excluded.candidate_fold == 0)).any()


def test_candidate_partitions_reassemble_tsv_chunks_and_reject_repeats(tmp_path):
    frame = _candidates()
    path = tmp_path / "candidates.tsv"
    frame.to_csv(path, sep="\t", index=False)

    partitions = list(iter_candidate_partitions(path, batch_size=2))
    assert [len(part) for part in partitions] == [3, 1, 2]
    assert [part.source1_entity_id.iloc[0] for part in partitions] == ["S1-1", "S1-2", "S1-3"]

    repeated = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
    repeated_path = tmp_path / "repeated.tsv"
    repeated.to_csv(repeated_path, sep="\t", index=False)
    with pytest.raises(ValueError, match="non-contiguous"):
        list(iter_candidate_partitions(repeated_path, batch_size=2))


def test_candidate_partitions_reassemble_parquet_batches_and_files(tmp_path):
    first = tmp_path / "01.parquet"
    second = tmp_path / "02.parquet"
    frame = _candidates()
    frame.iloc[:4].to_parquet(first, index=False)
    frame.iloc[3:].to_parquet(second, index=False)

    partitions = list(iter_candidate_partitions(tmp_path, batch_size=1))
    assert [len(part) for part in partitions] == [3, 2, 2]
    assert [part.source1_entity_id.iloc[0] for part in partitions] == ["S1-1", "S1-2", "S1-3"]


def test_fold_fallback_and_batch_inference_output_validation(tmp_path):
    explicit = {"S1-1": 0, "S2-1": 0}
    assert fold_for_entity("S2-1", explicit, 42, 3) == 0
    assert fold_for_entity("S9-unmatched", explicit, 42, 3) == fold_for_entity("S9-unmatched", explicit, 42, 3)

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    joblib.dump(TinyProbabilityModel(), model_dir / "final_model.joblib")
    (model_dir / "feature_columns.json").write_text(json.dumps(["retrieval_score"]), encoding="utf-8")
    (model_dir / "training_config.json").write_text(json.dumps({
        "feature_columns": ["retrieval_score"], "feature_schema_version": "pair-features-v1",
        "candidate_schema_version": "candidate-artifact-v1", "seed": 42,
    }), encoding="utf-8")
    (model_dir / "best_threshold.json").write_text(json.dumps({"threshold": .5}), encoding="utf-8")
    result = predict_pair_model(_candidates(), model_dir, tmp_path / "output", ["S1-1", "S1-2", "S1-3", "S1-France"])
    assert result["predictions"]["S1-1"] == ["S2-1", "S3-1"]
    validate_submission_data(
        ["S1-1", "S1-2", "S1-3", "S1-France"], result["predictions"],
        {key: value for key, value in {"S1-1": ["S2-1", "S3-1"], "S1-2": ["S3-2"], "S1-3": ["S2-3", "S3-3"], "S1-France": []}.items()},
    )
    assert predictions_from_threshold(result["scores"], .5)["S1-3"] == ["S2-3"]
