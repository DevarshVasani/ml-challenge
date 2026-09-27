import csv
import json
from pathlib import Path

from src.member_d_split import build_member_d_split


def test_component_safe_split_and_manifest(tmp_path):
    q = tmp_path / "threshold.tsv"; truth = tmp_path / "truth.tsv"
    q.write_text("entity_id\tcountry\nS1-1\tUS\nS1-2\tUS\nS1-3\tIN\nS1-4\tIN\n", encoding="utf-8")
    # S1-1 and S1-2 share S2-9 and therefore must stay together.
    truth.write_text(
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-1\tS2-9\nS1-2\tS2-9\nS1-3\tS3-1,S3-2\nS1-4\t\n",
        encoding="utf-8",
    )
    result = build_member_d_split({"threshold_queries_path": str(q), "truth_path": str(truth), "output_dir": str(tmp_path), "seed": 42, "proportions": {"fusion_fit": .4, "calibration": .3, "selection": .3}})
    a = result["assignment"]
    assert a["S1-1"] == a["S1-2"]
    out = tmp_path / "configs" / "member_d"
    assert (out / "split.tsv").exists()
    assert (out / "base_model_exclusion_ids.tsv").exists()
    manifest = json.loads((out / "split_manifest.json").read_text())
    assert "gold_edge_counts" in manifest and "country_counts" in manifest and "singleton_counts" in manifest
