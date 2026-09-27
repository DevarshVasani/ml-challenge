"""Build threshold pair labels as a separate keyed artifact for Member D.

Truth never enters the production score artifact.  This tool uses the frozen
threshold query manifest to scope evaluation_truth.tsv and writes only
PAIR_KEY + label for the supplied threshold candidate/gate universe.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .member_d_contracts import PAIR_KEY, derive_candidate_source, validate_pair_keys
from .neural_contracts import sha256_file


def _read(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(p)
    return pd.read_csv(p, sep="\t", dtype=object, keep_default_na=False)


def _write(frame: pd.DataFrame, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() in {".parquet", ".pq"}:
        frame.to_parquet(p, index=False)
    else:
        frame.to_csv(p, sep="\t", index=False)


def build_keyed_labels(
    pair_path: str | Path,
    query_manifest_path: str | Path,
    truth_path: str | Path,
    output_path: str | Path,
) -> dict[str, object]:
    pairs = _read(pair_path)
    validate_pair_keys(pairs)

    queries = pd.read_csv(
        query_manifest_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )
    id_col = (
        "source1_entity_id"
        if "source1_entity_id" in queries.columns
        else "entity_id"
    )
    query_ids = set(queries[id_col].astype(str))
    if "split" in queries.columns and any(
        value.lower() == "final" for value in queries["split"].astype(str)
    ):
        raise ValueError(
            "Refusing to build tuning labels from a final query manifest"
        )

    unknown = sorted(set(pairs["source1_entity_id"].astype(str)) - query_ids)
    if unknown:
        raise ValueError(
            f"Pair artifact contains S1 IDs outside threshold manifest: {unknown[:5]}"
        )

    truth_frame = pd.read_csv(
        truth_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )
    truth = {}
    for row in truth_frame.itertuples(index=False):
        sid = str(row.source1_entity_id)
        if sid not in query_ids:
            continue
        truth[sid] = {
            value.strip()
            for value in str(row.matched_entity_ids or "").split(",")
            if value.strip()
        }

    labels = pairs[PAIR_KEY].copy()
    labels["label"] = [
        int(cid in truth.get(sid, set()))
        for _, cid, sid in labels[PAIR_KEY].astype(str).itertuples(
            index=False, name=None
        )
    ]
    if labels.duplicated(PAIR_KEY).any():
        raise ValueError("Duplicate labels would be produced")

    _write(labels, output_path)
    manifest = {
        "artifact_kind": "member_d_keyed_threshold_labels",
        "row_count": len(labels),
        "positive_count": int(labels["label"].sum()),
        "negative_count": int((labels["label"] == 0).sum()),
        "pair_key": PAIR_KEY,
        "pair_artifact_sha256": sha256_file(pair_path),
        "query_manifest_sha256": sha256_file(query_manifest_path),
        "truth_sha256": sha256_file(truth_path),
        "output_sha256": sha256_file(output_path),
        "truth_kept_separate_from_scores": True,
        "selection_eligible": True,
    }
    manifest_path = Path(f"{output_path}.manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--query-manifest", required=True)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build_keyed_labels(
                args.pairs,
                args.query_manifest,
                args.truth,
                args.output,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
