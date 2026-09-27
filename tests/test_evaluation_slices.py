import pandas as pd
from src.evaluation_slices import build_fake_pattern_masks, evaluate_fake_pattern_slices


def base_row(**kw):
    row = {
        "name_raw_similarity": 0.8,
        "name_tokens_extra_in_b": 0,
        "name_tokens_missing_from_b": 0,
        "legal_form_agreement": 1,
        "house_number_match": 1,
        "house_number_a_present": 1,
        "house_number_b_present": 1,
        "postcode_match_house_mismatch": 0,
        "label": 0,
        "predicted_match": 0,
        "match_probability": 0.1,
    }
    row.update(kw); return row


def test_exact_c_pattern_definitions_and_guardrail():
    df = pd.DataFrame([
        base_row(name_tokens_extra_in_b=1),
        base_row(legal_form_agreement=0, predicted_match=1),
        base_row(house_number_match=0),
        base_row(house_number_match=0, postcode_match_house_mismatch=1),
        base_row(name_tokens_extra_in_b=1, label=1, predicted_match=1, match_probability=.9),
        # Similarity below C's resembles_real gate: must not be included.
        base_row(name_raw_similarity=.59, name_tokens_extra_in_b=1),
        # House mismatch without both numbers present: not house_number_shifted.
        base_row(house_number_match=0, house_number_a_present=0),
    ])
    masks = build_fake_pattern_masks(df)
    assert masks["branch_word_added"].tolist() == [True, False, False, False, True, False, False]
    assert masks["legal_form_swapped"].iloc[1]
    assert masks["house_number_shifted"].iloc[2]
    assert not masks["house_number_shifted"].iloc[6]
    result = evaluate_fake_pattern_slices(df, score_col="match_probability")
    assert result["legit_variation_positive_count"] == 1
    assert result["legit_variation_keep_rate"] == 1.0


def test_zero_slices_return_na():
    df = pd.DataFrame([base_row(name_raw_similarity=.1)])
    result = evaluate_fake_pattern_slices(df)
    assert result["fake_pattern_negative_count"] == "N/A"
    assert result["branch_word_added"] == "N/A"
