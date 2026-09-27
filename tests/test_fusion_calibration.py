import json
import pandas as pd

from src.fusion import fit_fusion, apply_fusion
from src.calibration import fit_calibration, apply_calibration


def make_rows():
    rows=[]
    for i in range(20):
        rows.append({
            "candidate_source":"S2", "candidate_entity_id":f"S2-{i}", "source1_entity_id":f"S1-{i}",
            "tree_score": .9 if i%2 else .1, "neural_probability": .8 if i%2 else .2,
            "neural_requested":1, "has_neural_score":1, "decision_route":"neural_fusion",
            "competing_s1_count":1, "competing_s1_score_gap":0., "competing_best_second_gap":0.,
            "competing_near_tie_count":1, "is_top_competing_s1":1,
            "postcode_match_house_mismatch":0, "legal_form_agreement":1, "house_number_match":1,
            "name_tokens_extra_in_b":0, "name_tokens_missing_from_b":0,
        })
    return pd.DataFrame(rows)


def test_labels_are_separate_from_production_scores(tmp_path):
    scores=make_rows(); sp=tmp_path/"scores.parquet"; scores.to_parquet(sp)
    labels=scores[["candidate_source","candidate_entity_id","source1_entity_id"]].copy(); labels["label"]=[i%2 for i in range(len(labels))]; lp=tmp_path/"labels.parquet"; labels.to_parquet(lp)
    split=pd.DataFrame({"source1_entity_id":scores.source1_entity_id,"member_d_partition":["fusion_fit"]*10+["calibration"]*10}); splitp=tmp_path/"split.tsv"; split.to_csv(splitp,sep="\t",index=False)
    config={"version":"v1","neural_features":["tree_score","tree_logit","neural_logit_feature","source_s2_indicator","source_s3_indicator"]}
    fit_fusion(sp,lp,splitp,config,tmp_path/"fusion")
    fused=apply_fusion(scores,tmp_path/"fusion"); fp=tmp_path/"fused.parquet"; fused.to_parquet(fp)
    fit_calibration(fp,lp,splitp,{"version":"v1","method":"platt"},tmp_path/"cal")
    calibrated=apply_calibration(fused,tmp_path/"cal")
    assert calibrated["match_probability"].notna().all()
    assert "label" not in scores.columns
