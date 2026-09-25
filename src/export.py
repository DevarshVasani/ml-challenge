"""Official grouped TSV export and output validation."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple

import pandas as pd

from .candidate_io import iter_complete_groups
from .neural_contracts import validate_prediction_frame


def clean_id_list(ids: Optional[Iterable[str]]) -> List[str]:
    if not ids:
        return []
    seen: Set[str] = set(); cleaned: List[str] = []
    for item in ids:
        if item is None:
            continue
        value = str(item).strip()
        if value and value not in seen:
            seen.add(value); cleaned.append(value)
    return cleaned


def load_test_s1_ids(test_source1_path: str, id_col: str = "entity_id") -> List[str]:
    with open(test_source1_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if id_col not in (reader.fieldnames or []):
            raise KeyError(f"Expected column '{id_col}' in '{test_source1_path}'. Found: {reader.fieldnames}")
        return [row[id_col].strip() for row in reader if row[id_col].strip()]


def _validate_mappings(s1_ids: Sequence[str], predictions: Mapping[str, Iterable[str]], candidates: Mapping[str, Iterable[str]] | None = None, s2_ids: Set[str] | None = None, s3_ids: Set[str] | None = None) -> None:
    required = list(s1_ids)
    if len(required) != len(set(required)):
        raise ValueError("test S1 IDs contain duplicates")
    required_set = set(required)
    for name, mapping in (("predictions", predictions), ("candidates", candidates or {})):
        unknown = set(mapping) - required_set
        if unknown:
            raise ValueError(f"{name} contains unknown S1 IDs: {sorted(unknown)[:5]}")
        for s1_id, values in mapping.items():
            ids = clean_id_list(values)
            if any(value == s1_id or value in required_set for value in ids):
                raise ValueError(f"{name} contains a Source 1 ID for {s1_id}")
            if s2_ids is not None or s3_ids is not None:
                valid = (s2_ids or set()) | (s3_ids or set())
                unknown_ids = set(ids) - valid
                if unknown_ids:
                    raise ValueError(f"{name} contains unknown candidate IDs: {sorted(unknown_ids)[:5]}")
    if candidates is not None:
        for s1_id, values in predictions.items():
            missing = set(clean_id_list(values)) - set(clean_id_list(candidates.get(s1_id, [])))
            if missing:
                raise ValueError(f"final matches for {s1_id} are not in its final candidate list: {sorted(missing)[:5]}")


def validate_submission_data(s1_ids: Sequence[str], predictions: Mapping[str, Iterable[str]], candidates: Mapping[str, Iterable[str]], s2_ids: Set[str] | None = None, s3_ids: Set[str] | None = None) -> None:
    _validate_mappings(s1_ids, predictions, candidates, s2_ids, s3_ids)


def write_submission_tsv(s1_ids: Sequence[str], predictions: Mapping[str, Iterable[str]], output_path: str, id_col: str = "source1_entity_id", list_col: str = "matched_entity_ids") -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, mode="w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        writer.writerow([id_col, list_col])
        for s1_id in s1_ids:
            writer.writerow([s1_id, ",".join(clean_id_list(predictions.get(s1_id, [])))])


def candidate_table_to_mapping(candidate_table: pd.DataFrame, s1_ids: Sequence[str] | None = None) -> dict[str, list[str]]:
    required = {"source1_entity_id", "candidate_entity_id"}
    if not required.issubset(candidate_table.columns):
        raise ValueError(f"candidate table must contain {sorted(required)}")
    mapping: dict[str, list[str]] = {str(s1): [] for s1 in (s1_ids or candidate_table["source1_entity_id"].astype(str).tolist())}
    for row in candidate_table.itertuples(index=False):
        source1 = str(row.source1_entity_id); candidate = str(row.candidate_entity_id)
        mapping.setdefault(source1, []).append(candidate)
    return {source1: clean_id_list(values) for source1, values in mapping.items()}


def export_candidate_table(candidate_table: pd.DataFrame, s1_ids: Sequence[str], output_path: str = "output/candidate_pairs.tsv") -> None:
    mapping = candidate_table_to_mapping(candidate_table, s1_ids)
    export_candidate_pairs(s1_ids, mapping, output_path)


def export_matching_results(s1_ids: Sequence[str], predictions: Mapping[str, Iterable[str]], output_path: str = "output/matching_results.tsv") -> None:
    write_submission_tsv(s1_ids, predictions, output_path, "source1_entity_id", "matched_entity_ids")


def export_candidate_pairs(s1_ids: Sequence[str], candidates: Mapping[str, Iterable[str]], output_path: str = "output/candidate_pairs.tsv") -> None:
    write_submission_tsv(s1_ids, candidates, output_path, "source1_entity_id", "candidate_entity_ids")


def export_all(s1_ids: Sequence[str], predictions: Mapping[str, Iterable[str]], candidates: Optional[Mapping[str, Iterable[str]]] = None, output_dir: str = "output") -> Tuple[str, Optional[str]]:
    os.makedirs(output_dir, exist_ok=True)
    if candidates is not None:
        _validate_mappings(s1_ids, predictions, candidates)
    else:
        _validate_mappings(s1_ids, predictions)
    matching_path = os.path.join(output_dir, "matching_results.tsv")
    export_matching_results(s1_ids, predictions, matching_path)
    candidate_path = None
    if candidates is not None:
        candidate_path = os.path.join(output_dir, "candidate_pairs.tsv")
        export_candidate_pairs(s1_ids, candidates, candidate_path)
    return matching_path, candidate_path


def export_streaming_shards(
    query_ids: Sequence[str],
    score_shards: Sequence[str],
    candidate_shards: Sequence[str],
    output_dir: str,
    *,
    threshold: float,
    allowed_candidate_ids: Set[str] | Mapping[str, Set[str]] | None = None,
) -> Tuple[str, str]:
    """Export final outputs without holding all test pairs in memory.

    Score and candidate pair keys must match exactly. Empty queries have no
    group in either artifact and still receive one empty output row. A present
    candidate group without scores is failed inference and is rejected.
    """
    required = list(map(str, query_ids))
    if len(required) != len(set(required)):
        raise ValueError("query manifest contains duplicate S1 IDs")
    positions = {sid: index for index, sid in enumerate(required)}
    candidate_iter = iter(iter_complete_groups(candidate_shards))
    score_iter = iter(iter_complete_groups(score_shards))
    candidate_group = next(candidate_iter, None)
    score_group = next(score_iter, None)
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    matching_path = output / "matching_results.tsv"; candidate_path = output / "candidate_pairs.tsv"
    if matching_path.exists() or candidate_path.exists():
        raise FileExistsError(f"refusing to overwrite existing final export in {output}")
    matching_tmp = output / ".matching_results.tsv.tmp"; candidate_tmp = output / ".candidate_pairs.tsv.tmp"
    try:
      with matching_tmp.open("w", encoding="utf-8", newline="") as match_handle, candidate_tmp.open("w", encoding="utf-8", newline="") as candidate_handle:
        match_handle.write("source1_entity_id\tmatched_entity_ids\n")
        candidate_handle.write("source1_entity_id\tcandidate_entity_ids\n")
        for index, sid in enumerate(required):
            next_candidate_id = str(candidate_group.iloc[0]["source1_entity_id"]) if candidate_group is not None else None
            if next_candidate_id is not None and next_candidate_id not in positions:
                raise ValueError(f"candidate artifact contains unknown query {next_candidate_id}")
            if next_candidate_id is not None and positions[next_candidate_id] < index:
                raise ValueError(f"candidate groups are not in query-manifest order near {next_candidate_id}")
            if next_candidate_id != sid:
                next_score_id = str(score_group.iloc[0]["source1_entity_id"]) if score_group is not None else None
                if next_score_id == sid:
                    raise ValueError(f"scores exist without candidates for {sid}")
                match_handle.write(f"{sid}\t\n"); candidate_handle.write(f"{sid}\t\n")
                continue
            if score_group is None or str(score_group.iloc[0]["source1_entity_id"]) != sid:
                raise ValueError(f"missing scoring coverage for candidate group {sid}")
            expected = set(map(tuple, candidate_group[["source1_entity_id", "candidate_entity_id", "candidate_source"]].astype(str).itertuples(index=False, name=None)))
            validate_prediction_frame(score_group, expected)
            candidate_values: list[str] = []
            selected: list[str] = []
            for row in candidate_group.itertuples(index=False):
                candidate_id = str(row.candidate_entity_id); source = str(row.candidate_source)
                if allowed_candidate_ids is not None:
                    allowed = allowed_candidate_ids.get(source, set()) if isinstance(allowed_candidate_ids, Mapping) else allowed_candidate_ids
                    if candidate_id not in allowed:
                        raise ValueError(f"candidate {source}/{candidate_id} is outside the declared source registry")
                candidate_values.append(candidate_id)
            for row in score_group.itertuples(index=False):
                if float(row.score) >= threshold:
                    selected.append(str(row.candidate_entity_id))
            candidates = clean_id_list(candidate_values); matches = clean_id_list(selected)
            if not set(matches).issubset(set(candidates)):
                raise ValueError(f"final matches for {sid} are not among scored candidates")
            match_handle.write(f"{sid}\t{','.join(matches)}\n")
            candidate_handle.write(f"{sid}\t{','.join(candidates)}\n")
            candidate_group = next(candidate_iter, None); score_group = next(score_iter, None)
      if candidate_group is not None or score_group is not None:
          raise ValueError("candidate/score artifacts contain groups after the query manifest ended")
      os.replace(matching_tmp, matching_path); os.replace(candidate_tmp, candidate_path)
    except BaseException:
      matching_tmp.unlink(missing_ok=True); candidate_tmp.unlink(missing_ok=True)
      raise
    return str(matching_path), str(candidate_path)


def _read_output_mapping(path: str, expected_columns: list[str]) -> tuple[list[str], dict[str, list[str]]]:
    with open(path, newline="", encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n\r").split("\t")
        if header != expected_columns:
            raise ValueError(f"{path} must have exact columns {expected_columns}; got {header}")
        rows = list(csv.reader(handle, delimiter="\t"))
    ids: list[str] = []; mapping: dict[str, list[str]] = {}
    for row in rows:
        if len(row) != 2:
            raise ValueError(f"{path} contains a non-tab-separated or malformed row")
        raw_ids = [value.strip() for value in row[1].split(",") if value.strip()]
        if len(raw_ids) != len(set(raw_ids)):
            raise ValueError(f"{path} contains duplicate IDs in the row for {row[0]}")
        ids.append(row[0]); mapping[row[0]] = clean_id_list(row[1].split(","))
    return ids, mapping


def validate_submission_files(matching_path: str, candidate_path: str, test_source1_path: str, test_source2_path: str, test_source3_path: str) -> None:
    required_s1 = load_test_s1_ids(test_source1_path)
    matching_ids, predictions = _read_output_mapping(matching_path, ["source1_entity_id", "matched_entity_ids"])
    candidate_ids, candidates = _read_output_mapping(candidate_path, ["source1_entity_id", "candidate_entity_ids"])
    if matching_ids != required_s1 or candidate_ids != required_s1:
        raise ValueError("both output files must contain every test S1 exactly once and in test order")
    s2 = set(pd.read_csv(test_source2_path, sep="\t", dtype=str, usecols=["entity_id"])["entity_id"])
    s3 = set(pd.read_csv(test_source3_path, sep="\t", dtype=str, usecols=["entity_id"])["entity_id"])
    validate_submission_data(required_s1, predictions, candidates, s2, s3)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Entity Resolution predictions to TSV.")
    parser.add_argument("--test-s1", default=None); parser.add_argument("--predictions-json", default=None); parser.add_argument("--candidates-json", default=None); parser.add_argument("--out-dir", default="output")
    parser.add_argument("--config", default=None, help="JSON configuration for grouped streaming export")
    parser.add_argument("--dry-run", action="store_true"); parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.config:
        if args.dry_run and args.execute:
            parser.error("--dry-run and --execute are mutually exclusive")
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        if not args.execute:
            print(json.dumps({"command": "export_streaming", "execute_required": True, "dataset_iteration": "deferred", "output_dir": config.get("output_dir")}, indent=2)); return
        manifest = json.loads(Path(config["query_manifest"]).read_text(encoding="utf-8"))
        query_ids = list(map(str, manifest["query_ids"][config.get("query_subset", "test")]))
        def shards(value, pattern):
            values = value if isinstance(value, list) else [value]
            result = []
            for raw in values:
                path = Path(raw)
                result.extend(str(item) for item in sorted(path.glob(pattern))) if path.is_dir() else result.append(str(path))
            return result
        allowed: dict[str, set[str]] | None = None
        if config.get("source_registry"):
            allowed = {}
            for source, path in config["source_registry"].items():
                allowed[str(source)] = set(pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, usecols=["entity_id"])["entity_id"].astype(str))
        matching_path, candidate_path = export_streaming_shards(query_ids, shards(config["scores"], "scores-*.parquet"), shards(config["candidates"], "*.parquet"), config["output_dir"], threshold=float(config["threshold"]), allowed_candidate_ids=allowed)
        print(json.dumps({"matching_results": matching_path, "candidate_pairs": candidate_path, "queries": len(query_ids)}, indent=2)); return
    if args.dry_run or args.execute:
        parser.error("--dry-run/--execute require --config")
    if not args.test_s1:
        parser.error("--test-s1 is required for legacy JSON export")
    s1_ids = load_test_s1_ids(args.test_s1)
    predictions = json.loads(Path(args.predictions_json).read_text(encoding="utf-8")) if args.predictions_json else {}
    candidates = json.loads(Path(args.candidates_json).read_text(encoding="utf-8")) if args.candidates_json else None
    matching_path, candidate_path = export_all(s1_ids, predictions, candidates, args.out_dir)
    print(f"Exported matching results to: {matching_path}")
    if candidate_path: print(f"Exported candidate pairs to:  {candidate_path}")


if __name__ == "__main__":
    main()
