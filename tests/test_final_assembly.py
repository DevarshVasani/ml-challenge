import pandas as pd
import pytest

from src.final_assembly import assemble_final_outputs


def test_final_match_must_be_subset_and_tree_only_candidates_are_exported(tmp_path):
    s1=tmp_path/"s1.tsv"; s2=tmp_path/"s2.tsv"; s3=tmp_path/"s3.tsv"
    s1.write_text("entity_id\nS1-1\nS1-2\n"); s2.write_text("entity_id\nS2-1\nS2-2\n"); s3.write_text("entity_id\nS3-1\n")
    decisions=pd.DataFrame([
        {"candidate_source":"S2","candidate_entity_id":"S2-1","source1_entity_id":"S1-1","decision_route":"tree_only"},
        {"candidate_source":"S3","candidate_entity_id":"S3-1","source1_entity_id":"S1-1","decision_route":"neural_fusion"},
    ])
    bad=pd.DataFrame([{"candidate_source":"S2","candidate_entity_id":"S2-2","source1_entity_id":"S1-1"}])
    dp=tmp_path/"d.parquet"; bp=tmp_path/"b.parquet"; decisions.to_parquet(dp); bad.to_parquet(bp)
    with pytest.raises(ValueError, match="subset"):
        assemble_final_outputs(s1,[dp],[bp],s2_registry_path=s2,s3_registry_path=s3,completion_manifests=[],output_dir=tmp_path/"out",ownership_enabled=False)

    good=decisions.iloc[[0]][["candidate_source","candidate_entity_id","source1_entity_id"]]; gp=tmp_path/"g.parquet"; good.to_parquet(gp)
    result=assemble_final_outputs(s1,[dp],[gp],s2_registry_path=s2,s3_registry_path=s3,completion_manifests=[],output_dir=tmp_path/"goodout",ownership_enabled=False)
    text=(tmp_path/"goodout"/"candidate_pairs.tsv").read_text()
    assert "S2-1" in text and "S3-1" in text
    assert "S1-2\t\n" in text
