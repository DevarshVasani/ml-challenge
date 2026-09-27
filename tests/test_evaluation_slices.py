import pandas as pd
import pytest
from src.evaluation_slices import build_fake_pattern_masks, evaluate_fake_pattern_slices

def test_fake_pattern_masks():
    rows = [
        # 0: Branch word added
        {"name_tokens_extra_in_b": 1, "name_tokens_missing_from_b": 0, "legal_form_agreement": 1, "house_number_match": 1, "postcode_match_house_mismatch": 0, "label": 0, "predicted_match": 0},
        # 1: Legal form mismatch
        {"name_tokens_extra_in_b": 0, "name_tokens_missing_from_b": 0, "legal_form_agreement": 0, "house_number_match": 1, "postcode_match_house_mismatch": 0, "label": 0, "predicted_match": 1},
        # 2: House number shifted
        {"name_tokens_extra_in_b": 0, "name_tokens_missing_from_b": 0, "legal_form_agreement": 1, "house_number_match": 0, "postcode_match_house_mismatch": 0, "label": 0, "predicted_match": 0},
        # 3: Postcode match + house mismatch
        {"name_tokens_extra_in_b": 0, "name_tokens_missing_from_b": 0, "legal_form_agreement": 1, "house_number_match": 0, "postcode_match_house_mismatch": 1, "label": 0, "predicted_match": 0},
        # 4: Legitimate positive resembling a fake pattern (e.g. branch word added)
        {"name_tokens_extra_in_b": 1, "name_tokens_missing_from_b": 0, "legal_form_agreement": 1, "house_number_match": 1, "postcode_match_house_mismatch": 0, "label": 1, "predicted_match": 1},
        # 5: Normal positive (none of the fake patterns)
        {"name_tokens_extra_in_b": 0, "name_tokens_missing_from_b": 0, "legal_form_agreement": 1, "house_number_match": 1, "postcode_match_house_mismatch": 0, "label": 1, "predicted_match": 1}
    ]
    df = pd.DataFrame(rows)
    
    masks = build_fake_pattern_masks(df)
    
    # 1. branch_word_added
    assert masks["branch_word_added"].iloc[0] == True
    assert masks["branch_word_added"].iloc[4] == True
    
    # 2. legal_form_swapped
    assert masks["legal_form_swapped"].iloc[1] == True
    
    # 3. house_number_shifted
    assert masks["house_number_shifted"].iloc[2] == True
    
    # 4. postcode_masks_house
    assert masks["postcode_masks_house"].iloc[3] == True
    
    # Evaluate metrics
    results = evaluate_fake_pattern_slices(df)
    
    # 5. Legitimate positive contributes to guard-rail keep rate
    assert results["legit_variation_positive_count"] == 1
    assert results["legit_variation_keep_rate"] == 1.0
    
    # 6. Zero row slices return N/A
    df_empty = pd.DataFrame(columns=df.columns)
    empty_results = evaluate_fake_pattern_slices(df_empty)
    assert empty_results.get("fake_pattern_negative_count") == "N/A"
    assert empty_results.get("branch_word_added") == "N/A"

def test_no_macro_f05_alteration():
    # 7. slice evaluation does not alter canonical macro-F0.5 calculation
    from src.evaluate_pipeline import run_evaluation
    df = pd.DataFrame([
        {"name_tokens_extra_in_b": 1, "label": 0, "predicted_match": 0}
    ])
    df_copy = df.copy(deep=True)
    evaluate_fake_pattern_slices(df)
    pd.testing.assert_frame_equal(df, df_copy)
