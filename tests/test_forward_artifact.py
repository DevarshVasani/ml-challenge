import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.forward_fts import audit_output
from src.retrieval_experiments import sha


def _partition(root, number, *, k=20):
    folder = root / f"bucket-{number:02d}"
    folder.mkdir(parents=True)
    name = "S1-000000000-000000001.parquet"
    file = folder / name
    pq.write_table(pa.table({"s1_id": [f"S1-{number}"], "source_record_id": ["S2-a"]}), file)
    manifest = {
        "complete": True,
        "config": {"split": "test", "partition": number, "partitions": 2,
                   "k_per_field": k, "indexes": {"S2": "same-index"}},
        "queries_this_run": 1,
        "shards": {name: {"start": 0, "end": 1, "queries": 1, "pairs": 1,
                          "sha256": sha(file)}},
    }
    (folder / "manifest.json").write_text(json.dumps(manifest))
    return file


def test_forward_audit_checks_complete_compatible_shards(tmp_path):
    _partition(tmp_path, 0)
    second = _partition(tmp_path, 1)
    result = audit_output(tmp_path, "test", 2)
    assert result["queries"] == 2 and result["pairs"] == 2
    assert len(result["partition_manifest_sha256"]) == 2
    second.write_bytes(b"corrupt")
    with pytest.raises(Exception, match="corrupt forward shard"):
        audit_output(tmp_path, "test", 2)


def test_forward_audit_rejects_policy_or_coverage_mismatch(tmp_path):
    _partition(tmp_path, 0)
    _partition(tmp_path, 1, k=50)
    with pytest.raises(ValueError, match="different retrieval policies"):
        audit_output(tmp_path, "test", 2)
    manifest_file = tmp_path / "bucket-01" / "manifest.json"
    manifest = json.loads(manifest_file.read_text())
    manifest["config"]["k_per_field"] = 20
    manifest_file.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="expected query count"):
        audit_output(tmp_path, "test", 3)
