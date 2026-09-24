"""Tests for export and submission formatting."""

import os
import tempfile
import unittest
from src.export import (
    clean_id_list,
    export_all,
    export_candidate_pairs,
    export_matching_results,
    load_test_s1_ids,
)


class TestExport(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_clean_id_list(self):
        self.assertEqual(clean_id_list([]), [])
        self.assertEqual(clean_id_list(None), [])
        self.assertEqual(clean_id_list([" S2-1 ", "S2-1", "S3-2"]), ["S2-1", "S3-2"])
        self.assertEqual(clean_id_list(["", None, "  "]), [])

    def test_export_matching_results(self):
        s1_ids = ["S1-0001", "S1-0002", "S1-0003"]
        predictions = {
            "S1-0001": ["S2-0047", "S3-0812"],
            "S1-0002": ["S3-0004"],
            # S1-0003 is omitted -> singleton
        }
        out_path = os.path.join(self.temp_dir.name, "matching_results.tsv")
        export_matching_results(s1_ids, predictions, out_path)

        with open(out_path, "r", encoding="utf-8") as f:
            lines = [line.rstrip("\r\n") for line in f.readlines()]

        expected_lines = [
            "source1_entity_id\tmatched_entity_ids",
            "S1-0001\tS2-0047,S3-0812",
            "S1-0002\tS3-0004",
            "S1-0003\t",
        ]
        self.assertEqual(lines, expected_lines)

    def test_export_all_candidates_and_matches(self):
        s1_ids = ["S1-1", "S1-2"]
        predictions = {"S1-1": ["S2-1"]}
        candidates = {"S1-1": ["S2-1", "S2-2"], "S1-2": ["S3-5"]}

        matching_path, cand_path = export_all(
            s1_ids=s1_ids,
            predictions=predictions,
            candidates=candidates,
            output_dir=self.temp_dir.name,
        )

        self.assertTrue(os.path.exists(matching_path))
        self.assertTrue(os.path.exists(cand_path))

        with open(matching_path, "r", encoding="utf-8") as f:
            match_lines = [l.strip() for l in f.readlines()]
        with open(cand_path, "r", encoding="utf-8") as f:
            cand_lines = [l.strip() for l in f.readlines()]

        self.assertEqual(match_lines, ["source1_entity_id\tmatched_entity_ids", "S1-1\tS2-1", "S1-2"])
        self.assertEqual(cand_lines, ["source1_entity_id\tcandidate_entity_ids", "S1-1\tS2-1,S2-2", "S1-2\tS3-5"])


if __name__ == "__main__":
    unittest.main()
