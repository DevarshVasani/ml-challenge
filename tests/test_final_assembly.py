import json

import pandas as pd
import pytest

from src.final_assembly import assemble_final_outputs
from src.member_d_manifest import MemberDManifest, write_manifest
from src.neural_contracts import sha256_file


def _manifest(path, artifact):
    write_manifest(
        path,
        MemberDManifest(
            artifact_kind="test",
            version="v1",
            completion_state="complete",
            row_count=1,
            files={artifact.name: sha256_file(artifact)},
        ),
    )


def test_final_match_must_be_subset_and_tree_only_candidates_are_exported(
    tmp_path,
):
    s1 = tmp_path / "s1.tsv"
    s2 = tmp_path / "s2.tsv"
    s3 = tmp_path / "s3.tsv"
    s1.write_text("entity_id\nS1-1\nS1-2\n")
    s2.write_text("entity_id\nS2-1\nS2-2\n")
    s3.write_text("entity_id\nS3-1\n")

    decisions = pd.DataFrame(
        [
            {
                "candidate_source": "S2",
                "candidate_entity_id": "S2-1",
                "source1_entity_id": "S1-1",
                "decision_route": "tree_only",
            },
            {
                "candidate_source": "S3",
                "candidate_entity_id": "S3-1",
                "source1_entity_id": "S1-1",
                "decision_route": "neural_fusion",
            },
        ]
    )
    bad = pd.DataFrame(
        [
            {
                "candidate_source": "S2",
                "candidate_entity_id": "S2-2",
                "source1_entity_id": "S1-1",
            }
        ]
    )
    decision_path = tmp_path / "d.parquet"
    bad_path = tmp_path / "b.parquet"
    decisions.to_parquet(decision_path)
    bad.to_parquet(bad_path)
    decision_manifest = tmp_path / "d.manifest.json"
    bad_manifest = tmp_path / "b.manifest.json"
    _manifest(decision_manifest, decision_path)
    _manifest(bad_manifest, bad_path)

    with pytest.raises(ValueError, match="subset"):
        assemble_final_outputs(
            s1,
            [decision_path],
            [bad_path],
            s2_registry_path=s2,
            s3_registry_path=s3,
            completion_manifests=[
                decision_manifest,
                bad_manifest,
            ],
            output_dir=tmp_path / "out",
            ownership_enabled=False,
        )

    good = decisions.iloc[[0]][
        [
            "candidate_source",
            "candidate_entity_id",
            "source1_entity_id",
        ]
    ]
    good_path = tmp_path / "g.parquet"
    good.to_parquet(good_path)
    good_manifest = tmp_path / "g.manifest.json"
    _manifest(good_manifest, good_path)

    result = assemble_final_outputs(
        s1,
        [decision_path],
        [good_path],
        s2_registry_path=s2,
        s3_registry_path=s3,
        completion_manifests=[
            decision_manifest,
            good_manifest,
        ],
        output_dir=tmp_path / "goodout",
        ownership_enabled=False,
    )
    text = (
        tmp_path / "goodout" / "candidate_pairs.tsv"
    ).read_text()
    assert "S2-1" in text and "S3-1" in text
    assert "S1-2\t\n" in text

    with pytest.raises(FileExistsError):
        assemble_final_outputs(
            s1,
            [decision_path],
            [good_path],
            s2_registry_path=s2,
            s3_registry_path=s3,
            completion_manifests=[
                decision_manifest,
                good_manifest,
            ],
            output_dir=tmp_path / "goodout",
            ownership_enabled=False,
        )
