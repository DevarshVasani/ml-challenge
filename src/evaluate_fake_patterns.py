"""Targeted evaluation of fake-record detection.

Fakes in this challenge typically copy a real business but apply one small
trick: an extra "branch" word in the name, a changed legal form, or a
shifted house number / postcode-masks-house contradiction. This script
isolates exactly those pattern types from the validation split and reports
catch rate (how often we correctly reject a fake) separately from the
overall F0.5, plus a guard-rail check that legitimate name variation isn't
being over-rejected for superficially resembling the same patterns.
"""

import glob
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import fbeta_score


def main(model_path: str = "artifacts/models/lightgbm_baseline_v4.txt") -> None:
    files = sorted(glob.glob("artifacts/neural-data-enriched/train_pairs/*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    _, val_idx = next(gss.split(df, groups=df["source1_entity_id"]))
    val_df = df.iloc[val_idx].copy()

    booster = lgb.Booster(model_file=model_path)
    feature_cols = booster.feature_name()
    val_df["tree_score"] = booster.predict(val_df[feature_cols])
    val_df["pred"] = (val_df["tree_score"] >= 0.5).astype(int)

    resembles_real = val_df["name_raw_similarity"] >= 0.6
    branch_word_added = (val_df["name_tokens_extra_in_b"] > 0) | (val_df["name_tokens_missing_from_b"] > 0)
    legal_form_swapped = val_df["legal_form_agreement"] == 0
    house_number_shifted = (
        (val_df["house_number_match"] == 0)
        & (val_df["house_number_a_present"] == 1)
        & (val_df["house_number_b_present"] == 1)
    )
    postcode_masks_house = val_df["postcode_match_house_mismatch"] == 1

    fake_pattern = resembles_real & (
        branch_word_added | legal_form_swapped | house_number_shifted | postcode_masks_house
    )

    print(f"total val rows: {len(val_df)}")
    print(f"rows matching fake-pattern criteria: {fake_pattern.sum()}")
    print(f"  actual negatives (true fakes): {(fake_pattern & (val_df['label'] == 0)).sum()}")
    print(f"  actual positives (legit variation): {(fake_pattern & (val_df['label'] == 1)).sum()}")

    print("\n--- Per-pattern breakdown (negatives only = actual fakes) ---")
    patterns = {
        "branch_word_added": branch_word_added,
        "legal_form_swapped": legal_form_swapped,
        "house_number_shifted": house_number_shifted,
        "postcode_masks_house": postcode_masks_house,
    }
    for name, mask in patterns.items():
        subset = val_df[mask & resembles_real & (val_df["label"] == 0)]
        if len(subset) == 0:
            print(f"{name:<25} n=0")
            continue
        caught = (subset["pred"] == 0).sum()
        print(f"{name:<25} n={len(subset):<6} correctly_rejected={caught:<6} "
              f"catch_rate={caught / len(subset):.4f}")

    print("\n--- Fakes vs easy negatives ---")
    fakes = val_df[fake_pattern & (val_df["label"] == 0)]
    easy_negatives = val_df[~fake_pattern & (val_df["label"] == 0)]
    print(f"fakes: n={len(fakes)}  catch_rate={(fakes['pred'] == 0).mean():.4f}  "
          f"mean_tree_score={fakes['tree_score'].mean():.4f}")
    print(f"easy negatives: n={len(easy_negatives)}  catch_rate={(easy_negatives['pred'] == 0).mean():.4f}  "
          f"mean_tree_score={easy_negatives['tree_score'].mean():.4f}")

    print("\n--- Guard-rail: legit variation correctly kept as match? ---")
    legit_variation = val_df[fake_pattern & (val_df["label"] == 1)]
    print(f"n={len(legit_variation)}  keep_rate={(legit_variation['pred'] == 1).mean():.4f}")

    print(f"\nOverall val F0.5: {fbeta_score(val_df['label'], val_df['pred'], beta=0.5):.4f}")


if __name__ == "__main__":
    main()
