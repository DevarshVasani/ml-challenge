"""Member D diagnostic slices reproducing Member C's fake-pattern definitions."""
from __future__ import annotations

from typing import Any
import pandas as pd

REQUIRED_PATTERN_COLUMNS = {
    "name_raw_similarity",
    "name_tokens_extra_in_b",
    "name_tokens_missing_from_b",
    "legal_form_agreement",
    "house_number_match",
    "house_number_a_present",
    "house_number_b_present",
    "postcode_match_house_mismatch",
}


def _num(frame: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce").fillna(default)


def build_fake_pattern_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    resembles_real = _num(frame, "name_raw_similarity", 0.0) >= 0.6
    branch_raw = (_num(frame, "name_tokens_extra_in_b") > 0) | (_num(frame, "name_tokens_missing_from_b") > 0)
    legal_raw = _num(frame, "legal_form_agreement", -1) == 0
    house_raw = (
        (_num(frame, "house_number_match", -1) == 0)
        & (_num(frame, "house_number_a_present") == 1)
        & (_num(frame, "house_number_b_present") == 1)
    )
    postcode_raw = _num(frame, "postcode_match_house_mismatch") == 1

    masks = {
        "branch_word_added": resembles_real & branch_raw,
        "legal_form_swapped": resembles_real & legal_raw,
        "house_number_shifted": resembles_real & house_raw,
        "postcode_masks_house": resembles_real & postcode_raw,
    }
    union = masks["branch_word_added"] | masks["legal_form_swapped"] | masks["house_number_shifted"] | masks["postcode_masks_house"]
    masks["fake_pattern_union"] = union
    masks["legit_variation_guardrail"] = union
    return masks


def evaluate_fake_pattern_slices(
    frame: pd.DataFrame,
    *,
    label_col: str = "label",
    prediction_col: str = "predicted_match",
    score_col: str | None = None,
) -> dict[str, Any]:
    masks = build_fake_pattern_masks(frame)
    if label_col not in frame.columns or prediction_col not in frame.columns:
        raise ValueError(f"Slice evaluation requires {label_col} and {prediction_col}")
    labels = pd.to_numeric(frame[label_col], errors="raise").astype(int)
    preds = pd.to_numeric(frame[prediction_col], errors="raise").astype(int)

    raw: dict[str, Any] = {}
    for name, mask in masks.items():
        subset_idx = frame.index[mask]
        if len(subset_idx) == 0:
            raw[name] = "N/A"
            continue
        y = labels.loc[subset_idx]
        p = preds.loc[subset_idx]
        neg = y == 0
        pos = y == 1
        result: dict[str, Any] = {
            "negative_count": int(neg.sum()),
            "negative_catch_rate": float((p[neg] == 0).mean()) if neg.any() else "N/A",
            "false_accept_count": int((p[neg] == 1).sum()) if neg.any() else "N/A",
            "positive_count": int(pos.sum()),
            "positive_keep_rate": float((p[pos] == 1).mean()) if pos.any() else "N/A",
            "false_reject_count": int((p[pos] == 0).sum()) if pos.any() else "N/A",
            "mean_match_probability": "N/A",
        }
        if score_col and score_col in frame.columns:
            score = pd.to_numeric(frame.loc[subset_idx, score_col], errors="coerce").dropna()
            result["mean_match_probability"] = float(score.mean()) if len(score) else "N/A"
        raw[name] = result

    union = raw["fake_pattern_union"]
    guard = raw["legit_variation_guardrail"]
    out = {
        "fake_pattern_negative_count": union.get("negative_count", "N/A") if union != "N/A" else "N/A",
        "fake_pattern_catch_rate": union.get("negative_catch_rate", "N/A") if union != "N/A" else "N/A",
        "fake_pattern_false_accept_count": union.get("false_accept_count", "N/A") if union != "N/A" else "N/A",
        "legit_variation_positive_count": guard.get("positive_count", "N/A") if guard != "N/A" else "N/A",
        "legit_variation_keep_rate": guard.get("positive_keep_rate", "N/A") if guard != "N/A" else "N/A",
        "legit_variation_false_reject_count": guard.get("false_reject_count", "N/A") if guard != "N/A" else "N/A",
    }
    for name in ("branch_word_added", "legal_form_swapped", "house_number_shifted", "postcode_masks_house"):
        out[name] = raw[name]
    return out
