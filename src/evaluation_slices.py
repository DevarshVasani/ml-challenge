"""Fake-pattern evaluation slices for Member D.

Reproduces Member C's four fake-pattern definitions for diagnostic reporting
without modifying the core macro F0.5 metrics.
"""

from __future__ import annotations

import pandas as pd
from typing import Any

def build_fake_pattern_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    """Build boolean masks for each defined fake-pattern slice."""
    masks = {}
    
    # Branch word added or removed
    extra_in_b = pd.to_numeric(frame.get("name_tokens_extra_in_b", pd.Series(0, index=frame.index)), errors="coerce").fillna(0)
    missing_from_b = pd.to_numeric(frame.get("name_tokens_missing_from_b", pd.Series(0, index=frame.index)), errors="coerce").fillna(0)
    masks["branch_word_added"] = (extra_in_b > 0) | (missing_from_b > 0)
    
    # Legal form disagreement
    legal_form_ag = pd.to_numeric(frame.get("legal_form_agreement", pd.Series(-1, index=frame.index)), errors="coerce").fillna(-1)
    masks["legal_form_swapped"] = legal_form_ag == 0
    
    # House number shifted (mismatch with both numbers present)
    hn_match = pd.to_numeric(frame.get("house_number_match", pd.Series(-1, index=frame.index)), errors="coerce").fillna(-1)
    masks["house_number_shifted"] = hn_match == 0
    
    # Postcode match plus house number mismatch
    pc_hm = pd.to_numeric(frame.get("postcode_match_house_mismatch", pd.Series(0, index=frame.index)), errors="coerce").fillna(0)
    masks["postcode_masks_house"] = pc_hm == 1
    
    # Union of all fake patterns
    masks["fake_pattern_union"] = (
        masks["branch_word_added"] | 
        masks["legal_form_swapped"] | 
        masks["house_number_shifted"] | 
        masks["postcode_masks_house"]
    )
    
    # Legitimate variation guardrail
    masks["legit_variation_guardrail"] = masks["fake_pattern_union"]
    
    return masks


def evaluate_fake_pattern_slices(
    frame: pd.DataFrame, 
    label_col: str = "label", 
    prediction_col: str = "predicted_match", 
    score_col: str | None = None
) -> dict[str, Any]:
    """Evaluate performance on the defined slices."""
    if frame is None:
        frame = pd.DataFrame()
        
    masks = build_fake_pattern_masks(frame)
    raw_results: dict[str, Any] = {}
    
    has_pred = prediction_col in frame.columns
    has_label = label_col in frame.columns
    has_score = score_col and score_col in frame.columns

    for pattern_name, mask in masks.items():
        if len(mask) == 0:
            raw_results[pattern_name] = "N/A"
            continue
            
        subset = frame[mask]
        
        if len(subset) == 0:
            raw_results[pattern_name] = "N/A"
            continue
            
        metrics: dict[str, Any] = {}
        
        if has_label and has_pred:
            labels = pd.to_numeric(subset[label_col], errors="coerce").fillna(0)
            preds = pd.to_numeric(subset[prediction_col], errors="coerce").fillna(0)
            
            negatives_mask = labels == 0
            positives_mask = labels == 1
            
            neg_count = negatives_mask.sum()
            pos_count = positives_mask.sum()
            
            metrics["negative_count"] = int(neg_count)
            if neg_count > 0:
                metrics["negative_catch_rate"] = float((preds[negatives_mask] == 0).mean())
                metrics["false_accept_count"] = int((preds[negatives_mask] == 1).sum())
            else:
                metrics["negative_catch_rate"] = "N/A"
                metrics["false_accept_count"] = "N/A"
                
            metrics["positive_count"] = int(pos_count)
            if pos_count > 0:
                metrics["positive_keep_rate"] = float((preds[positives_mask] == 1).mean())
                metrics["false_reject_count"] = int((preds[positives_mask] == 0).sum())
            else:
                metrics["positive_keep_rate"] = "N/A"
                metrics["false_reject_count"] = "N/A"
                
        if has_score:
            metrics["mean_match_probability"] = float(pd.to_numeric(subset[score_col], errors="coerce").mean())
        else:
            metrics["mean_match_probability"] = "N/A"
            
        raw_results[pattern_name] = metrics
        
    # Format output precisely as requested in member_D_implementation_plan_2.md
    final_results = {}
    
    union_res = raw_results.get("fake_pattern_union", "N/A")
    if union_res != "N/A":
        final_results["fake_pattern_negative_count"] = union_res.get("negative_count", "N/A")
        final_results["fake_pattern_catch_rate"] = union_res.get("negative_catch_rate", "N/A")
        final_results["fake_pattern_false_accept_count"] = union_res.get("false_accept_count", "N/A")
    else:
        final_results["fake_pattern_negative_count"] = "N/A"
        final_results["fake_pattern_catch_rate"] = "N/A"
        final_results["fake_pattern_false_accept_count"] = "N/A"

    guard_res = raw_results.get("legit_variation_guardrail", "N/A")
    if guard_res != "N/A":
        final_results["legit_variation_positive_count"] = guard_res.get("positive_count", "N/A")
        final_results["legit_variation_keep_rate"] = guard_res.get("positive_keep_rate", "N/A")
        final_results["legit_variation_false_reject_count"] = guard_res.get("false_reject_count", "N/A")
    else:
        final_results["legit_variation_positive_count"] = "N/A"
        final_results["legit_variation_keep_rate"] = "N/A"
        final_results["legit_variation_false_reject_count"] = "N/A"
        
    for pattern in ["branch_word_added", "legal_form_swapped", "house_number_shifted", "postcode_masks_house"]:
        final_results[pattern] = raw_results.get(pattern, "N/A")
        
    return final_results
