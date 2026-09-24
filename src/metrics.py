"""Exact F_0.5 macro evaluation for the Amazon ML Challenge (Entity Resolution).

Spec (from the problem statement):
    F_0.5 = (1.25 * P * R) / (0.25 * P + R)

Computed as a MACRO average: F_0.5 is calculated per Source 1 entity, then
averaged across all Source 1 entities in the evaluation set.

Singletons (Source 1 entities with no true matches) are included:
  - ground truth empty & prediction empty  -> 1.0 (correctly identified singleton)
  - ground truth empty & prediction non-empty -> 0.0 (false merge on singleton)

Non-singleton cases:
  - prediction empty & ground truth non-empty -> recall = 0 -> F_0.5 = 0.0
  - otherwise standard set-based precision/recall and the F_0.5 formula above.
"""

from __future__ import annotations

from typing import Iterable


def f05_single(true_ids: Iterable[str], pred_ids: Iterable[str]) -> float:
    """F_0.5 for one Source 1 entity, given true and predicted match ID sets."""
    true_set = set(true_ids)
    pred_set = set(pred_ids)

    # Singleton handling (ground truth has no matches)
    if not true_set:
        return 1.0 if not pred_set else 0.0

    # Non-singleton ground truth but empty prediction -> recall 0
    if not pred_set:
        return 0.0

    tp = len(true_set & pred_set)
    precision = tp / len(pred_set)
    recall = tp / len(true_set)

    denom = 0.25 * precision + recall
    if denom == 0.0:
        return 0.0
    return (1.25 * precision * recall) / denom


def f05_macro(
    true_matches: dict[str, Iterable[str]],
    pred_matches: dict[str, Iterable[str]],
) -> float:
    """Macro-averaged F_0.5 over all Source 1 entities in `true_matches`.

    Every Source 1 entity in the ground truth dict is scored exactly once;
    predicted IDs are deduplicated per entity (mirrors submission rules).
    Entities absent from `pred_matches` count as empty predictions.
    """
    if not true_matches:
        raise ValueError("true_matches is empty — nothing to score.")

    total = 0.0
    for s1_id, true_ids in true_matches.items():
        pred_ids = pred_matches.get(s1_id, [])
        total += f05_single(true_ids, pred_ids)
    return total / len(true_matches)


def parse_id_list(cell: str | float | None) -> list[str]:
    """Parse a comma-separated matched_entity_ids cell; empty/NaN -> []."""
    if cell is None or (isinstance(cell, float) and cell != cell):  # NaN
        return []
    cell = str(cell).strip()
    if not cell:
        return []
    return [tok.strip() for tok in cell.split(",") if tok.strip()]


def load_matches_tsv(path: str, id_col: str, list_col: str) -> dict[str, list[str]]:
    """Load a ground-truth or submission TSV into {source1_id: [match_ids]}."""
    import csv

    matches: dict[str, list[str]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            matches[row[id_col]] = parse_id_list(row[list_col])
    return matches
