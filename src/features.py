"""Deterministic pair features for Source 1 to Source 2/3 matching.

The module deliberately keeps candidate generation separate from feature
generation.  Candidate files may contain one pair per row or the export
format's comma-separated ``candidate_entity_ids`` column; both are converted
to the same ordered, long-format representation here.
"""

from __future__ import annotations

import argparse
import os
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .split import load_folds_tsv, load_ground_truth


ID_COLUMN = "entity_id"
S1_ID_COLUMN = "source1_entity_id"
CANDIDATE_ID_COLUMN = "candidate_entity_id"


def load_source_tsv(path: str | os.PathLike[str]) -> pd.DataFrame:
    """Load a source TSV and validate its required ``entity_id`` column."""
    frame = pd.read_csv(path, sep="\t", dtype=object, keep_default_na=False)
    if ID_COLUMN not in frame.columns:
        raise ValueError(f"Source file '{path}' must contain an '{ID_COLUMN}' column")
    frame = frame.copy()
    frame[ID_COLUMN] = frame[ID_COLUMN].map(_clean_id)
    if frame[ID_COLUMN].eq("").any():
        raise ValueError(f"Source file '{path}' contains an empty entity_id")
    if frame[ID_COLUMN].duplicated().any():
        duplicates = frame.loc[frame[ID_COLUMN].duplicated(), ID_COLUMN].tolist()
        raise ValueError(f"Source file '{path}' contains duplicate entity IDs: {duplicates[:5]}")
    return frame


def load_sources(
    s1_path: str | os.PathLike[str],
    s2_path: str | os.PathLike[str],
    s3_path: str | os.PathLike[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load S1, S2 and S3 TSV files (S3 may be absent and becomes empty)."""
    s1 = load_source_tsv(s1_path)
    s2 = load_source_tsv(s2_path)
    if s3_path:
        s3 = load_source_tsv(s3_path)
    else:
        s3 = pd.DataFrame(columns=s2.columns)
    return s1, s2, s3


def _clean_id(value: Any) -> str:
    if value is None or _is_missing(value):
        return ""
    return str(value).strip()


def _is_missing(value: Any) -> bool:
    try:
        missing = pd.isna(value)
        return bool(missing) if not isinstance(missing, (np.ndarray, list, tuple)) else False
    except (TypeError, ValueError):
        return False


def _split_ids(value: Any) -> list[str]:
    text = _clean_id(value)
    if not text or text.casefold() == "nan":
        return []
    return [token.strip() for token in text.split(",") if token.strip()]


def load_candidate_pairs(path: str | os.PathLike[str]) -> pd.DataFrame:
    """Load candidates into one row per pair while preserving input order.

    Accepted forms are ``source1_entity_id,candidate_entity_id`` and the
    grouped exporter form ``source1_entity_id,candidate_entity_ids``.  All
    other columns are retained and repeated for each exploded candidate.
    """
    frame = pd.read_csv(path, sep="\t", dtype=object, keep_default_na=False)
    source_col = S1_ID_COLUMN if S1_ID_COLUMN in frame.columns else ID_COLUMN
    if source_col not in frame.columns:
        raise ValueError(f"Candidate file '{path}' must contain source1_entity_id")
    grouped_col = "candidate_entity_ids" if "candidate_entity_ids" in frame.columns else None
    if CANDIDATE_ID_COLUMN not in frame.columns and grouped_col is None:
        for alias in ("candidate_id", "matched_entity_id"):
            if alias in frame.columns:
                frame = frame.rename(columns={alias: CANDIDATE_ID_COLUMN})
                break
    if CANDIDATE_ID_COLUMN not in frame.columns and grouped_col is None:
        raise ValueError(
            f"Candidate file '{path}' must contain candidate_entity_id or candidate_entity_ids"
        )

    output: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        source_id = _clean_id(row.get(source_col))
        candidate_ids = _split_ids(row.get(grouped_col)) if grouped_col else [_clean_id(row.get(CANDIDATE_ID_COLUMN))]
        for candidate_id in candidate_ids:
            if not candidate_id:
                continue
            item = dict(row)
            item[S1_ID_COLUMN] = source_id
            item[CANDIDATE_ID_COLUMN] = candidate_id
            if grouped_col:
                item.pop(grouped_col, None)
            if source_col != S1_ID_COLUMN:
                item.pop(source_col, None)
            output.append(item)

    columns = [S1_ID_COLUMN, CANDIDATE_ID_COLUMN]
    columns.extend(c for c in frame.columns if c not in {source_col, grouped_col, CANDIDATE_ID_COLUMN})
    return pd.DataFrame(output, columns=list(dict.fromkeys(columns)))


def validate_candidate_ids(
    candidates: pd.DataFrame,
    s1: pd.DataFrame,
    s2: pd.DataFrame,
    s3: pd.DataFrame,
) -> None:
    """Reject unknown endpoints, S1-as-candidate IDs, and ambiguous records."""
    required = {S1_ID_COLUMN, CANDIDATE_ID_COLUMN}
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"Candidate table is missing columns: {sorted(missing)}")
    s1_ids = set(s1[ID_COLUMN])
    s2_ids = set(s2[ID_COLUMN])
    s3_ids = set(s3[ID_COLUMN])
    candidate_ids = set(candidates[CANDIDATE_ID_COLUMN].map(_clean_id))
    unknown_s1 = set(candidates[S1_ID_COLUMN].map(_clean_id)) - s1_ids
    if unknown_s1:
        raise ValueError(f"Candidate file contains unknown Source 1 IDs: {sorted(unknown_s1)[:5]}")
    if candidate_ids & s1_ids:
        raise ValueError(f"S1 IDs cannot be used as candidates: {sorted(candidate_ids & s1_ids)[:5]}")
    unknown_candidates = candidate_ids - s2_ids - s3_ids
    if unknown_candidates:
        raise ValueError(f"Candidate file contains unknown candidate IDs: {sorted(unknown_candidates)[:5]}")
    duplicate_other_ids = s2_ids & s3_ids
    if duplicate_other_ids:
        explicit = _candidate_source_column(candidates)
        if explicit is None:
            raise ValueError(
                "Candidate IDs shared by S2 and S3 require a candidate_source column: "
                f"{sorted(duplicate_other_ids)[:5]}"
            )


def _candidate_source_column(frame: pd.DataFrame) -> str | None:
    for name in ("candidate_source", "source", "candidate_dataset"):
        if name in frame.columns:
            return name
    return None


def _source_for_candidate(candidate_id: str, row: Mapping[str, Any], s2_ids: set[str], s3_ids: set[str]) -> str:
    explicit_col = next((c for c in ("candidate_source", "source", "candidate_dataset") if c in row), None)
    if explicit_col:
        value = _clean_id(row.get(explicit_col)).casefold().replace("source", "")
        if value in {"2", "s2"}:
            if candidate_id not in s2_ids:
                raise ValueError(f"Candidate {candidate_id} is not present in S2")
            return "S2"
        if value in {"3", "s3"}:
            if candidate_id not in s3_ids:
                raise ValueError(f"Candidate {candidate_id} is not present in S3")
            return "S3"
        elif value:
            raise ValueError(f"Unsupported candidate source '{row.get(explicit_col)}'")
        else:
            value = ""
    else:
        value = ""
    in_s2 = candidate_id in s2_ids
    in_s3 = candidate_id in s3_ids
    if in_s2 and not in_s3:
        return "S2"
    if in_s3 and not in_s2:
        return "S3"
    raise ValueError(f"Cannot determine source for candidate ID '{candidate_id}'")


def normalize_text(value: Any) -> str:
    """Normalize text without discarding digits or non-English characters."""
    if value is None or _is_missing(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).casefold().replace("&", " and ")
    text = "".join(" " if unicodedata.category(char).startswith(("P", "S")) else char for char in text)
    return " ".join(text.split())


def _field(row: Mapping[str, Any], aliases: Sequence[str]) -> Any:
    for alias in aliases:
        if alias in row:
            return row[alias]
    return ""


NAME_ALIASES = ("business_name", "name", "company_name", "商号")
ADDRESS_ALIASES = ("business_address", "address", "street_address", "所在地")
COUNTRY_ALIASES = ("country", "country_code", "国")


def _token_jaccard(left: Sequence[str], right: Sequence[str]) -> float:
    a, b = set(left), set(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _token_containment(left: Sequence[str], right: Sequence[str]) -> float:
    a, b = set(left), set(right)
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def _length_ratio(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return min(len(left), len(right)) / max(len(left), len(right))


def _char_ngrams(text: str, n: int = 3) -> set[str]:
    compact = text.replace(" ", "")
    return {compact[i : i + n] for i in range(max(0, len(compact) - n + 1))}


def _set_jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _sequence_ratio(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return float(SequenceMatcher(None, left, right, autojunk=False).ratio())


def _token_sort_ratio(left_tokens: Sequence[str], right_tokens: Sequence[str]) -> float:
    if not left_tokens or not right_tokens:
        return 0.0
    return _sequence_ratio(" ".join(sorted(left_tokens)), " ".join(sorted(right_tokens)))


_NUMBER_RE = re.compile(r"\d+")


def _numbers(text: str) -> set[str]:
    return set(_NUMBER_RE.findall(text))


def _number_overlap(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _number_conflict(left: set[str], right: set[str]) -> float:
    return float(bool(left and right and not (left & right)))


def _pair_text_features(prefix: str, left: str, right: str) -> dict[str, float | int]:
    left_tokens, right_tokens = left.split(), right.split()
    left_ngrams, right_ngrams = _char_ngrams(left), _char_ngrams(right)
    return {
        f"{prefix}_exact": float(bool(left and right and left == right)),
        f"{prefix}_both_missing": float(not left and not right),
        f"{prefix}_one_missing": float(bool(left) != bool(right)),
        f"{prefix}_length_ratio": _length_ratio(left, right),
        f"{prefix}_token_jaccard": _token_jaccard(left_tokens, right_tokens),
        f"{prefix}_token_containment": _token_containment(left_tokens, right_tokens),
        f"{prefix}_char_3gram_jaccard": _set_jaccard(left_ngrams, right_ngrams),
        f"{prefix}_sequence_ratio": _sequence_ratio(left, right),
        f"{prefix}_token_sort_ratio": _token_sort_ratio(left_tokens, right_tokens),
        f"{prefix}_s1_token_count": len(left_tokens),
        f"{prefix}_candidate_token_count": len(right_tokens),
    }


def _record_values(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        normalize_text(_field(row, NAME_ALIASES)),
        normalize_text(_field(row, ADDRESS_ALIASES)),
        normalize_text(_field(row, COUNTRY_ALIASES)),
    )


def compute_pair_features(
    pair: Mapping[str, Any],
    s1_record: Mapping[str, Any],
    candidate_record: Mapping[str, Any],
    candidate_source: str,
) -> dict[str, float | int]:
    """Compute all numeric features for one already-validated pair."""
    s1_name, s1_address, s1_country = _record_values(s1_record)
    candidate_name, candidate_address, candidate_country = _record_values(candidate_record)
    features: dict[str, float | int] = {}
    features.update(_pair_text_features("name", s1_name, candidate_name))
    features.update(_pair_text_features("address", s1_address, candidate_address))
    s1_name_numbers, candidate_name_numbers = _numbers(s1_name), _numbers(candidate_name)
    s1_address_numbers, candidate_address_numbers = _numbers(s1_address), _numbers(candidate_address)
    features.update(
        {
            "name_number_overlap": _number_overlap(s1_name_numbers, candidate_name_numbers),
            "name_number_conflict": _number_conflict(s1_name_numbers, candidate_name_numbers),
            "address_number_overlap": _number_overlap(s1_address_numbers, candidate_address_numbers),
            "address_number_conflict": _number_conflict(s1_address_numbers, candidate_address_numbers),
            "address_numbers_exact": float(
                bool(s1_address_numbers and candidate_address_numbers and s1_address_numbers == candidate_address_numbers)
            ),
            "country_exact": float(bool(s1_country and candidate_country and s1_country == candidate_country)),
            "candidate_source_s2": float(candidate_source == "S2"),
            "candidate_source_s3": float(candidate_source == "S3"),
            "combined_name_address_similarity": _combined_similarity(
                s1_name, candidate_name, s1_address, candidate_address
            ),
        }
    )
    return features


def _combined_similarity(name_left: str, name_right: str, address_left: str, address_right: str) -> float:
    scores = []
    if name_left and name_right:
        scores.append(_sequence_ratio(name_left, name_right))
    if address_left and address_right:
        scores.append(_sequence_ratio(address_left, address_right))
    return float(sum(scores) / len(scores)) if scores else 0.0


def join_candidate_pairs(
    candidates: pd.DataFrame,
    s1: pd.DataFrame,
    s2: pd.DataFrame,
    s3: pd.DataFrame,
) -> list[tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]]:
    """Resolve each pair to its business records in stable input order."""
    validate_candidate_ids(candidates, s1, s2, s3)
    s1_records = s1.set_index(ID_COLUMN, drop=False).to_dict(orient="index")
    s2_records = s2.set_index(ID_COLUMN, drop=False).to_dict(orient="index")
    s3_records = s3.set_index(ID_COLUMN, drop=False).to_dict(orient="index")
    s2_ids, s3_ids = set(s2_records), set(s3_records)
    joined = []
    for row in candidates.to_dict(orient="records"):
        source1_id = _clean_id(row[S1_ID_COLUMN])
        candidate_id = _clean_id(row[CANDIDATE_ID_COLUMN])
        source = _source_for_candidate(candidate_id, row, s2_ids, s3_ids)
        joined.append((row, s1_records[source1_id], (s2_records if source == "S2" else s3_records)[candidate_id], source))
    return joined


def add_labels_and_folds(
    features: pd.DataFrame,
    ground_truth: Mapping[str, Iterable[str]] | None = None,
    folds: Mapping[str, int] | None = None,
) -> pd.DataFrame:
    """Add labels and fold assignments, rejecting cross-fold pairs."""
    output = features.copy()
    if ground_truth is not None:
        true_sets = {str(k): set(str(x).strip() for x in values if str(x).strip()) for k, values in ground_truth.items()}
        output["label"] = [int(candidate in true_sets.get(source1, set())) for source1, candidate in zip(output[S1_ID_COLUMN], output[CANDIDATE_ID_COLUMN])]
    if folds is not None:
        missing = set(output[S1_ID_COLUMN]) | set(output[CANDIDATE_ID_COLUMN])
        missing -= set(folds)
        if missing:
            raise ValueError(f"Missing fold assignments for pair endpoints: {sorted(missing)[:5]}")
        s1_folds = output[S1_ID_COLUMN].map(folds)
        candidate_folds = output[CANDIDATE_ID_COLUMN].map(folds)
        bad = s1_folds != candidate_folds
        if bad.any():
            first = output.loc[bad].iloc[0]
            raise ValueError(
                "Cross-fold candidate pair rejected: "
                f"{first[S1_ID_COLUMN]} (fold {s1_folds[bad].iloc[0]}) and "
                f"{first[CANDIDATE_ID_COLUMN]} (fold {candidate_folds[bad].iloc[0]})"
            )
        output["fold"] = s1_folds.astype(int).to_numpy()
    return output


def build_pair_features(
    s1_path: str | os.PathLike[str],
    s2_path: str | os.PathLike[str],
    s3_path: str | os.PathLike[str] | None,
    candidates_path: str | os.PathLike[str],
    ground_truth_path: str | os.PathLike[str] | None = None,
    folds_path: str | os.PathLike[str] | None = None,
) -> pd.DataFrame:
    """Build the ordered pair feature table used by the baseline model."""
    s1, s2, s3 = load_sources(s1_path, s2_path, s3_path)
    candidates = load_candidate_pairs(candidates_path)
    joined = join_candidate_pairs(candidates, s1, s2, s3)
    rows: list[dict[str, Any]] = []
    for candidate_row, s1_record, candidate_record, source in joined:
        result: dict[str, Any] = {
            S1_ID_COLUMN: _clean_id(candidate_row[S1_ID_COLUMN]),
            CANDIDATE_ID_COLUMN: _clean_id(candidate_row[CANDIDATE_ID_COLUMN]),
        }
        result.update(compute_pair_features(candidate_row, s1_record, candidate_record, source))
        for column, value in candidate_row.items():
            if column in {S1_ID_COLUMN, CANDIDATE_ID_COLUMN, "candidate_source", "source", "candidate_dataset"}:
                continue
            numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
            if pd.notna(numeric) and np.isfinite(float(numeric)):
                result[column] = float(numeric)
        rows.append(result)
    output = pd.DataFrame(rows)
    if output.empty:
        output = pd.DataFrame(columns=[S1_ID_COLUMN, CANDIDATE_ID_COLUMN])
    output = add_labels_and_folds(
        output,
        ground_truth=load_ground_truth(str(ground_truth_path)) if ground_truth_path else None,
        folds=load_folds_tsv(str(folds_path)) if folds_path else None,
    )
    return output


# Friendly aliases for callers that prefer verb-oriented names.
create_pair_features = build_pair_features
make_pair_features = build_pair_features
load_s1_tsv = load_source_tsv
load_s2_tsv = load_source_tsv
load_s3_tsv = load_source_tsv
explode_candidates = load_candidate_pairs
load_candidates = load_candidate_pairs
validate_ids = validate_candidate_ids
add_ground_truth_labels = add_labels_and_folds


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Source 1/candidate pair features.")
    parser.add_argument("--s1", required=True)
    parser.add_argument("--s2", required=True)
    parser.add_argument("--s3", default=None)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--ground-truth", "--gt", dest="ground_truth", default=None)
    parser.add_argument("--folds", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    table = build_pair_features(
        args.s1, args.s2, args.s3, args.candidates, args.ground_truth, args.folds
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(output_path, index=False)
    print(f"Saved {len(table)} pair feature rows to {output_path}")


if __name__ == "__main__":
    main()
