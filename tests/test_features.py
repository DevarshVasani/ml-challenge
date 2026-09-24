import os
import tempfile
import unittest

import pandas as pd

from src.features import (
    add_labels_and_folds,
    build_pair_features,
    compute_pair_features,
    load_candidate_pairs,
    normalize_text,
    validate_candidate_ids,
)


class TestFeatures(unittest.TestCase):
    def test_unicode_punctuation_ampersand_nulls_and_digits(self):
        self.assertEqual(normalize_text("  Café & Co., 12-A  "), "café and co 12 a")
        self.assertEqual(normalize_text(None), "")
        self.assertEqual(normalize_text("東京\u3000商店 ４２"), "東京 商店 42")

    def test_reordered_names_and_conflicting_numbers(self):
        row = compute_pair_features(
            {},
            {"business_name": "Blue River Foods", "business_address": "12 Main Road", "country": "IN"},
            {"business_name": "Foods Blue River", "business_address": "14 Main Road", "country": "IN"},
            "S2",
        )
        self.assertEqual(row["name_token_jaccard"], 1.0)
        self.assertEqual(row["name_token_sort_ratio"], 1.0)
        self.assertEqual(row["address_number_conflict"], 1.0)
        self.assertEqual(row["address_numbers_exact"], 0.0)

    def test_two_empty_fields_are_not_exact_matches(self):
        row = compute_pair_features(
            {}, {"business_name": None, "business_address": ""}, {"business_name": "", "business_address": None}, "S3"
        )
        self.assertEqual(row["name_exact"], 0.0)
        self.assertEqual(row["name_both_missing"], 1.0)
        self.assertEqual(row["address_exact"], 0.0)

    def test_grouped_candidates_explode_in_stable_order_and_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            s1_path = os.path.join(directory, "s1.tsv")
            s2_path = os.path.join(directory, "s2.tsv")
            s3_path = os.path.join(directory, "s3.tsv")
            candidates_path = os.path.join(directory, "candidates.tsv")
            gt_path = os.path.join(directory, "gt.tsv")
            folds_path = os.path.join(directory, "folds.tsv")
            pd.DataFrame(
                [
                    {"entity_id": "S1-a", "business_name": "A", "business_address": "1 Main", "country": "IN"},
                    {"entity_id": "S1-b", "business_name": "B", "business_address": "2 Main", "country": "IN"},
                ]
            ).to_csv(s1_path, sep="\t", index=False)
            pd.DataFrame(
                [
                    {"entity_id": "S2-a", "business_name": "A", "business_address": "1 Main", "country": "IN"},
                    {"entity_id": "S2-b", "business_name": "B", "business_address": "2 Main", "country": "IN"},
                ]
            ).to_csv(s2_path, sep="\t", index=False)
            pd.DataFrame([{"entity_id": "S3-a", "business_name": "A3", "business_address": "3 Main", "country": "IN"}]).to_csv(s3_path, sep="\t", index=False)
            pd.DataFrame(
                [
                    {"source1_entity_id": "S1-a", "candidate_entity_ids": "S2-a,S3-a", "retrieval_score": "0.9"},
                    {"source1_entity_id": "S1-b", "candidate_entity_ids": "S2-b", "retrieval_score": "0.8"},
                ]
            ).to_csv(candidates_path, sep="\t", index=False)
            pd.DataFrame(
                [
                    {"source1_entity_id": "S1-a", "matched_entity_ids": "S2-a,S3-a"},
                    {"source1_entity_id": "S1-b", "matched_entity_ids": ""},
                ]
            ).to_csv(gt_path, sep="\t", index=False)
            pd.DataFrame(
                [
                    {"entity_id": "S1-a", "fold": 0}, {"entity_id": "S2-a", "fold": 0}, {"entity_id": "S3-a", "fold": 0},
                    {"entity_id": "S1-b", "fold": 1}, {"entity_id": "S2-b", "fold": 1},
                ]
            ).to_csv(folds_path, sep="\t", index=False)

            table = build_pair_features(s1_path, s2_path, s3_path, candidates_path, gt_path, folds_path)
            self.assertEqual(table[["source1_entity_id", "candidate_entity_id"]].values.tolist(), [["S1-a", "S2-a"], ["S1-a", "S3-a"], ["S1-b", "S2-b"]])
            self.assertEqual(table["label"].tolist(), [1, 1, 0])
            self.assertEqual(table["fold"].tolist(), [0, 0, 1])
            self.assertEqual(table["retrieval_score"].tolist(), [0.9, 0.9, 0.8])

    def test_validation_and_cross_fold_rejection(self):
        s1 = pd.DataFrame({"entity_id": ["S1-a"]})
        s2 = pd.DataFrame({"entity_id": ["S2-a"]})
        s3 = pd.DataFrame({"entity_id": ["S3-a"]})
        with self.assertRaisesRegex(ValueError, "unknown candidate"):
            validate_candidate_ids(pd.DataFrame({"source1_entity_id": ["S1-a"], "candidate_entity_id": ["S9-a"]}), s1, s2, s3)
        with self.assertRaisesRegex(ValueError, "Cross-fold"):
            add_labels_and_folds(
                pd.DataFrame({"source1_entity_id": ["S1-a"], "candidate_entity_id": ["S2-a"]}),
                folds={"S1-a": 0, "S2-a": 1},
            )


if __name__ == "__main__":
    unittest.main()
