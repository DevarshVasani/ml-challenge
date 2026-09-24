"""Tests for dataset splitting and leakage prevention."""

import os
import tempfile
import unittest
from src.split import (
    build_connected_components,
    create_folds,
    get_train_val_entities,
    load_folds_tsv,
    partition_components,
    save_folds_tsv,
    verify_split_integrity,
)


class TestSplit(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_connected_components_transitivity(self):
        # S1-A matches S2-1, S1-B also matches S2-1 -> must be in same component
        s1_ids = {"S1-A", "S1-B", "S1-C", "S1-Single"}
        gt = {
            "S1-A": ["S2-1"],
            "S1-B": ["S2-1", "S3-2"],
            "S1-C": ["S3-3"],
            "S1-Single": [],
        }
        comps = build_connected_components(s1_ids, gt)
        # Should have 3 components: {S1-A, S1-B, S2-1, S3-2}, {S1-C, S3-3}, {S1-Single}
        self.assertEqual(len(comps), 3)

        # Check that S1-A and S1-B share the same root
        root_a = [r for r, members in comps.items() if "S1-A" in members][0]
        root_b = [r for r, members in comps.items() if "S1-B" in members][0]
        self.assertEqual(root_a, root_b)
        self.assertIn("S2-1", comps[root_a])
        self.assertIn("S3-2", comps[root_a])

    def test_leakage_prevention(self):
        # Test that held-out entities never cross into training fold
        entity_to_fold = {
            "S1-A": 0,
            "S2-1": 0,
            "S1-B": 1,
            "S2-2": 1,
            "S1-C": 2,
            "S2-3": 2,
        }
        gt = {
            "S1-A": ["S2-1"],
            "S1-B": ["S2-2"],
            "S1-C": ["S2-3"],
        }
        passed, issues = verify_split_integrity(entity_to_fold, gt)
        self.assertTrue(passed)
        self.assertEqual(len(issues), 0)

        # Test intentional cross-fold leak detection
        bad_folds = {
            "S1-A": 0,
            "S2-1": 1,  # leaked to fold 1!
        }
        passed, issues = verify_split_integrity(bad_folds, gt)
        self.assertFalse(passed)
        self.assertGreater(len(issues), 0)

    def test_train_val_strict_disjoint(self):
        entity_to_fold = {
            "S1-1": 0, "S2-1": 0,
            "S1-2": 1, "S2-2": 1,
            "S1-3": 2, "S2-3": 2,
        }
        train_ids, val_ids = get_train_val_entities(entity_to_fold, val_fold=1)
        self.assertEqual(train_ids, {"S1-1", "S2-1", "S1-3", "S2-3"})
        self.assertEqual(val_ids, {"S1-2", "S2-2"})
        self.assertTrue(train_ids.isdisjoint(val_ids))

    def test_end_to_end_split_generation(self):
        s1_path = os.path.join(self.temp_dir.name, "train_s1.tsv")
        gt_path = os.path.join(self.temp_dir.name, "train_gt.tsv")
        out_folds = os.path.join(self.temp_dir.name, "folds.tsv")

        # Write dummy S1
        with open(s1_path, "w", encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            f.write("S1-1\tAcme US\t123 Main\tUS\n")
            f.write("S1-2\tAcme Branch\t123 Main\tUS\n")
            f.write("S1-3\tBharat Corp\t456 MG Rd\tIndia\n")
            f.write("S1-4\tDelhi Traders\t789 Connaught\tIndia\n")
            f.write("S1-5\tSingleton US\t999 Oak\tUS\n")
            f.write("S1-6\tSingleton IN\t111 Brigade\tIndia\n")

        # Write dummy GT
        with open(gt_path, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
            f.write("S1-1\tS2-1\n")
            f.write("S1-2\tS2-1,S3-1\n")  # Links S1-1 and S1-2
            f.write("S1-3\tS2-3\n")
            f.write("S1-4\tS3-4\n")
            f.write("S1-5\t\n")
            f.write("S1-6\t\n")

        entity_to_fold, summary = create_folds(s1_path, gt_path, n_splits=3, seed=42)
        save_folds_tsv(entity_to_fold, out_folds)
        loaded = load_folds_tsv(out_folds)

        self.assertEqual(entity_to_fold, loaded)
        self.assertEqual(entity_to_fold["S1-1"], entity_to_fold["S1-2"])
        self.assertEqual(entity_to_fold["S1-1"], entity_to_fold["S2-1"])
        self.assertEqual(entity_to_fold["S1-1"], entity_to_fold["S3-1"])
        self.assertEqual(summary["n_splits"], 3)

    def test_unmatched_entities_are_spread_across_folds(self):
        s1_path = os.path.join(self.temp_dir.name, "train_source1.tsv")
        s2_path = os.path.join(self.temp_dir.name, "train_source2.tsv")
        s3_path = os.path.join(self.temp_dir.name, "train_source3.tsv")
        gt_path = os.path.join(self.temp_dir.name, "train_ground_truth.tsv")

        with open(s1_path, "w", encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tcountry\n")
        with open(gt_path, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
        with open(s2_path, "w", encoding="utf-8") as f:
            f.write("entity_id\n")
            for index in range(6):
                f.write(f"S2-{index}\n")
        with open(s3_path, "w", encoding="utf-8") as f:
            f.write("entity_id\n")
            for index in range(3):
                f.write(f"S3-{index}\n")

        entity_to_fold, _ = create_folds(
            s1_path,
            gt_path,
            s2_tsv_path=s2_path,
            s3_tsv_path=s3_path,
            n_splits=3,
            seed=42,
        )

        fold_counts = [sum(fold == index for fold in entity_to_fold.values()) for index in range(3)]
        self.assertEqual(fold_counts, [3, 3, 3])

        repeated, _ = create_folds(
            s1_path,
            gt_path,
            s2_tsv_path=s2_path,
            s3_tsv_path=s3_path,
            n_splits=3,
            seed=42,
        )
        self.assertEqual(entity_to_fold, repeated)


if __name__ == "__main__":
    unittest.main()
