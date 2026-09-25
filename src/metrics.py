"""Backward-compatible aliases for the canonical evaluator."""

from __future__ import annotations

from .evaluate import evaluate_predictions, load_matches_tsv as _load_matches_tsv
from .evaluate import parse_id_string as parse_id_list, score_single_s1 as f05_single


def f05_macro(true_matches: dict[str, object], pred_matches: dict[str, object]) -> float:
    return float(evaluate_predictions(true_matches, pred_matches)["macro_f05"])


def load_matches_tsv(path: str, id_col: str, list_col: str) -> dict[str, list[str]]:
    return _load_matches_tsv(path, id_col=id_col, matches_col=list_col)
