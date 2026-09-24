"""Submission exporter for Entity Resolution challenge.

Produces standard tab-separated submission files:
1. matching_results.tsv:
   - Columns: source1_entity_id, matched_entity_ids
   - Final matched entities per Source 1 record (empty for singletons)
2. candidate_pairs.tsv:
   - Columns: source1_entity_id, candidate_entity_ids
   - Candidate pairs considered during blocking / model scoring

Formatting Rules:
- Exactly one row per Source 1 entity in the evaluation / test set.
- Leave matched_entity_ids / candidate_entity_ids empty for singletons / no-candidates.
- No duplicate IDs within a list.
- Tab-separated values with NO surrounding quotes.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set


def clean_id_list(ids: Optional[Iterable[str]]) -> List[str]:
    """Clean and deduplicate a sequence of entity IDs while preserving order."""
    if not ids:
        return []
    seen: Set[str] = set()
    cleaned: List[str] = []
    for item in ids:
        if item is None:
            continue
        tok = str(item).strip()
        if tok and tok not in seen:
            seen.add(tok)
            cleaned.append(tok)
    return cleaned


def load_test_s1_ids(test_source1_path: str, id_col: str = "entity_id") -> List[str]:
    """Load all Source 1 entity IDs from test_source1.tsv in order."""
    s1_ids: List[str] = []
    with open(test_source1_path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if id_col not in (reader.fieldnames or []):
            raise KeyError(f"Expected column '{id_col}' in '{test_source1_path}'. Found: {reader.fieldnames}")
        for row in reader:
            eid = row[id_col].strip()
            if eid:
                s1_ids.append(eid)
    return s1_ids


def write_submission_tsv(
    s1_ids: Sequence[str],
    predictions: Mapping[str, Iterable[str]],
    output_path: str,
    id_col: str = "source1_entity_id",
    list_col: str = "matched_entity_ids",
) -> None:
    """Write submission TSV with exact 1-row-per-S1 structure and empty strings for singletons.

    Args:
        s1_ids: Ordered sequence of all Source 1 entity IDs in the target set.
        predictions: Mapping of s1_id -> iterable of matched entity IDs.
        output_path: Destination path.
        id_col: Name of ID column (e.g., 'source1_entity_id').
        list_col: Name of list column (e.g., 'matched_entity_ids' or 'candidate_entity_ids').
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(
            f,
            delimiter="\t",
            quoting=csv.QUOTE_MINIMAL,
            lineterminator="\n",
        )
        # Header
        writer.writerow([id_col, list_col])

        # Rows
        for s1_id in s1_ids:
            raw_matches = predictions.get(s1_id, [])
            cleaned = clean_id_list(raw_matches)
            match_str = ",".join(cleaned)
            writer.writerow([s1_id, match_str])


def export_matching_results(
    s1_ids: Sequence[str],
    predictions: Mapping[str, Iterable[str]],
    output_path: str = "output/matching_results.tsv",
) -> None:
    """Export final matching results TSV."""
    write_submission_tsv(
        s1_ids=s1_ids,
        predictions=predictions,
        output_path=output_path,
        id_col="source1_entity_id",
        list_col="matched_entity_ids",
    )


def export_candidate_pairs(
    s1_ids: Sequence[str],
    candidates: Mapping[str, Iterable[str]],
    output_path: str = "output/candidate_pairs.tsv",
) -> None:
    """Export candidate pairs TSV from blocking/candidate generation."""
    write_submission_tsv(
        s1_ids=s1_ids,
        predictions=candidates,
        output_path=output_path,
        id_col="source1_entity_id",
        list_col="candidate_entity_ids",
    )


def export_all(
    s1_ids: Sequence[str],
    predictions: Mapping[str, Iterable[str]],
    candidates: Optional[Mapping[str, Iterable[str]]] = None,
    output_dir: str = "output",
) -> Tuple[str, Optional[str]]:
    """Export both matching_results.tsv and candidate_pairs.tsv into output directory."""
    os.makedirs(output_dir, exist_ok=True)
    matching_path = os.path.join(output_dir, "matching_results.tsv")
    export_matching_results(s1_ids, predictions, matching_path)

    candidate_path = None
    if candidates is not None:
        candidate_path = os.path.join(output_dir, "candidate_pairs.tsv")
        export_candidate_pairs(s1_ids, candidates, candidate_path)

    return matching_path, candidate_path


def main():
    parser = argparse.ArgumentParser(description="Export Entity Resolution predictions to TSV.")
    parser.add_argument("--test-s1", type=str, required=True, help="Path to test_source1.tsv to ensure all test entities exist")
    parser.add_argument("--predictions-json", type=str, default=None, help="Path to JSON file with predictions {s1_id: [matches]}")
    parser.add_argument("--candidates-json", type=str, default=None, help="Path to JSON file with candidate pairs {s1_id: [candidates]}")
    parser.add_argument("--out-dir", type=str, default="output", help="Directory where matching_results.tsv and candidate_pairs.tsv are saved")

    args = parser.parse_args()

    s1_ids = load_test_s1_ids(args.test_s1)
    print(f"Loaded {len(s1_ids)} test S1 entities from {args.test_s1}")

    predictions = {}
    if args.predictions_json:
        with open(args.predictions_json, "r", encoding="utf-8") as f:
            predictions = json.load(f)

    candidates = None
    if args.candidates_json:
        with open(args.candidates_json, "r", encoding="utf-8") as f:
            candidates = json.load(f)

    matching_path, cand_path = export_all(
        s1_ids=s1_ids,
        predictions=predictions,
        candidates=candidates,
        output_dir=args.out_dir,
    )
    print(f"Exported matching results to: {matching_path}")
    if cand_path:
        print(f"Exported candidate pairs to:  {cand_path}")


if __name__ == "__main__":
    main()
