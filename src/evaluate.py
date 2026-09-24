"""Evaluation module for Entity Resolution challenge.

Metric specification:
- Score each Source 1 (S1) entity separately, then macro-average over all S1 entities.
- For non-singletons (entities with >= 1 true matches):
    F0.5 = 5 * TP / (5 * TP + 4 * FP + FN)
    where:
      TP = correct predicted matches (|true & pred|)
      FP = incorrect predicted matches (|pred - true|)
      FN = missed true matches (|true - pred|)
- For singletons (entities with no true matches):
    Score = 1.0 if prediction is empty, otherwise 0.0.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union


def compute_confusion_components(
    true_ids: Iterable[str], pred_ids: Iterable[str]
) -> Tuple[int, int, int]:
    """Compute (TP, FP, FN) for an S1 entity given true and predicted match ID iterables."""
    true_set: Set[str] = set(true_ids)
    pred_set: Set[str] = set(pred_ids)

    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    return tp, fp, fn


def score_single_s1(true_ids: Iterable[str], pred_ids: Iterable[str]) -> float:
    """Compute F0.5 score for a single S1 entity.

    Rules:
    1. Singleton (len(true) == 0):
       - 1.0 if prediction is empty
       - 0.0 if prediction is non-empty
    2. Non-singleton (len(true) > 0):
       - F0.5 = 5*TP / (5*TP + 4*FP + FN)
       - If denominator is 0 (or TP=0, FP=0, FN=0), returns 0.0.
    """
    true_set: Set[str] = {x for x in true_ids if x}
    pred_set: Set[str] = {x for x in pred_ids if x}

    # Singleton check
    if not true_set:
        return 1.0 if not pred_set else 0.0

    # Non-singleton with empty prediction
    if not pred_set:
        return 0.0

    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)

    denom = 5 * tp + 4 * fp + fn
    if denom == 0:
        return 0.0
    return (5.0 * tp) / float(denom)


def evaluate_predictions(
    ground_truth: Dict[str, Iterable[str]],
    predictions: Dict[str, Iterable[str]],
) -> Dict[str, Union[float, int, Dict[str, float]]]:
    """Evaluate predictions against ground truth dictionary.

    Args:
        ground_truth: mapping of {s1_id: [true_matched_ids]}
        predictions: mapping of {s1_id: [predicted_matched_ids]}

    Returns:
        Dictionary containing:
          - macro_f05: float
          - num_entities: int
          - num_singletons: int
          - num_non_singletons: int
          - singleton_accuracy: float
          - non_singleton_macro_f05: float
          - per_entity_scores: Dict[str, float]
    """
    if not ground_truth:
        raise ValueError("Ground truth is empty; cannot evaluate.")

    per_entity_scores: Dict[str, float] = {}
    total_score = 0.0

    singleton_count = 0
    singleton_correct = 0

    non_singleton_count = 0
    non_singleton_total_score = 0.0

    for s1_id, true_ids in ground_truth.items():
        true_list = list(true_ids)
        pred_list = list(predictions.get(s1_id, []))

        s = score_single_s1(true_list, pred_list)
        per_entity_scores[s1_id] = s
        total_score += s

        if not true_list:
            singleton_count += 1
            if s == 1.0:
                singleton_correct += 1
        else:
            non_singleton_count += 1
            non_singleton_total_score += s

    num_entities = len(ground_truth)
    macro_f05 = total_score / num_entities
    singleton_acc = (singleton_correct / singleton_count) if singleton_count > 0 else 1.0
    non_singleton_macro = (
        (non_singleton_total_score / non_singleton_count)
        if non_singleton_count > 0
        else 1.0
    )

    return {
        "macro_f05": macro_f05,
        "num_entities": num_entities,
        "num_singletons": singleton_count,
        "num_non_singletons": non_singleton_count,
        "singleton_accuracy": singleton_acc,
        "non_singleton_macro_f05": non_singleton_macro,
        "per_entity_scores": per_entity_scores,
    }


def parse_id_string(val: Optional[Union[str, float]]) -> List[str]:
    """Parse comma-delimited ID list into deduplicated list of clean IDs."""
    if val is None:
        return []
    if isinstance(val, float):
        if val != val:  # NaN
            return []
        val = str(val)
    s = str(val).strip()
    if not s:
        return []
    seen = set()
    result = []
    for item in s.split(","):
        clean_id = item.strip()
        if clean_id and clean_id not in seen:
            seen.add(clean_id)
            result.append(clean_id)
    return result


def load_matches_tsv(
    path: str,
    id_col: str = "source1_entity_id",
    matches_col: str = "matched_entity_ids",
) -> Dict[str, List[str]]:
    """Load TSV file into mapping of s1_id -> list of matched IDs."""
    records: Dict[str, List[str]] = {}
    with open(path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if id_col not in (reader.fieldnames or []):
            raise KeyError(f"Expected column '{id_col}' in TSV file '{path}'. Found: {reader.fieldnames}")
        if matches_col not in (reader.fieldnames or []):
            raise KeyError(f"Expected column '{matches_col}' in TSV file '{path}'. Found: {reader.fieldnames}")
        for row in reader:
            s1_id = row[id_col].strip()
            if s1_id:
                records[s1_id] = parse_id_string(row[matches_col])
    return records


def run_tiny_example_checks() -> bool:
    """Verify scorer behavior on all canonical tiny examples."""
    test_cases = [
        {
            "name": "perfect_match",
            "true": ["S2-0001", "S3-0002"],
            "pred": ["S2-0001", "S3-0002"],
            "expected": 1.0,  # TP=2, FP=0, FN=0 -> 10/10 = 1.0
        },
        {
            "name": "extra_match",
            "true": ["S2-0001"],
            "pred": ["S2-0001", "S2-0002"],
            "expected": 5.0 / 9.0,  # TP=1, FP=1, FN=0 -> 5*1 / (5*1 + 4*1 + 0) = 5/9 ≈ 0.555556
        },
        {
            "name": "missed_match",
            "true": ["S2-0001", "S3-0002"],
            "pred": ["S2-0001"],
            "expected": 5.0 / 6.0,  # TP=1, FP=0, FN=1 -> 5*1 / (5*1 + 0 + 1) = 5/6 ≈ 0.833333
        },
        {
            "name": "empty_prediction_non_singleton",
            "true": ["S2-0001"],
            "pred": [],
            "expected": 0.0,  # TP=0, FP=0, FN=1 -> 0.0
        },
        {
            "name": "singleton_empty_prediction",
            "true": [],
            "pred": [],
            "expected": 1.0,  # correctly identified singleton
        },
        {
            "name": "singleton_non_empty_prediction",
            "true": [],
            "pred": ["S2-0001"],
            "expected": 0.0,  # false merge on singleton
        },
    ]

    all_passed = True
    print("=== Running Scorer Tiny Example Checks ===")
    for case in test_cases:
        score = score_single_s1(case["true"], case["pred"])
        expected = case["expected"]
        diff = abs(score - expected)
        passed = diff < 1e-7
        status = "PASS" if passed else "FAIL"
        print(
            f"[{status}] {case['name']:<32} | True: {str(case['true']):<22} | Pred: {str(case['pred']):<24} | Score: {score:.6f} | Expected: {expected:.6f}"
        )
        if not passed:
            all_passed = False

    # Check macro averaging across a miniature dataset
    mini_gt = {
        "S1-1": ["S2-1", "S3-1"],  # perfect (1.0)
        "S1-2": ["S2-2"],          # extra match (5/9)
        "S1-3": ["S2-3", "S3-3"],  # missed match (5/6)
        "S1-4": ["S2-4"],          # empty pred (0.0)
        "S1-5": [],                # correct singleton (1.0)
        "S1-6": [],                # false merge singleton (0.0)
    }
    mini_pred = {
        "S1-1": ["S2-1", "S3-1"],
        "S1-2": ["S2-2", "S3-99"],
        "S1-3": ["S2-3"],
        "S1-4": [],
        "S1-5": [],
        "S1-6": ["S2-99"],
    }
    expected_macro = (1.0 + (5.0 / 9.0) + (5.0 / 6.0) + 0.0 + 1.0 + 0.0) / 6.0
    res = evaluate_predictions(mini_gt, mini_pred)
    macro_passed = abs(res["macro_f05"] - expected_macro) < 1e-7
    print(
        f"[{'PASS' if macro_passed else 'FAIL'}] {'macro_average_all_cases':<32} | Macro F0.5: {res['macro_f05']:.6f} | Expected: {expected_macro:.6f}"
    )
    if not macro_passed:
        all_passed = False

    print("==========================================")
    if all_passed:
        print("All tiny example checks passed successfully!")
    else:
        print("Some checks failed!")
    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Evaluate Entity Resolution matching results.")
    parser.add_argument("--gt", "--ground-truth", dest="gt_path", type=str, help="Path to ground truth TSV")
    parser.add_argument("--pred", "--predictions", dest="pred_path", type=str, help="Path to predictions TSV")
    parser.add_argument("--check", "--self-check", action="store_true", help="Run tiny example validation checks")
    parser.add_argument("--out", type=str, default=None, help="Optional output JSON path for evaluation report")
    parser.add_argument("--id-col", type=str, default="source1_entity_id", help="Entity ID column name")
    parser.add_argument("--matches-col", type=str, default="matched_entity_ids", help="Matched IDs column name")

    args = parser.parse_args()

    if args.check:
        success = run_tiny_example_checks()
        if not success:
            sys.exit(1)
        if not args.gt_path:
            sys.exit(0)

    if not args.gt_path or not args.pred_path:
        if not args.check:
            parser.print_help()
        sys.exit(0)

    print(f"Loading ground truth: {args.gt_path}")
    gt = load_matches_tsv(args.gt_path, id_col=args.id_col, matches_col=args.matches_col)
    print(f"Loading predictions:  {args.pred_path}")
    pred = load_matches_tsv(args.pred_path, id_col=args.id_col, matches_col=args.matches_col)

    metrics = evaluate_predictions(gt, pred)

    print("\n--- Evaluation Summary ---")
    print(f"Macro F0.5 Score:         {metrics['macro_f05']:.6f}")
    print(f"Total S1 Entities:        {metrics['num_entities']}")
    print(f"Singletons:               {metrics['num_singletons']} (Accuracy: {metrics['singleton_accuracy']:.4f})")
    print(f"Non-Singletons:           {metrics['num_non_singletons']} (Macro F0.5: {metrics['non_singleton_macro_f05']:.6f})")

    if args.out:
        # Avoid dumping huge per_entity_scores if not needed or dump summary
        summary = {k: v for k, v in metrics.items() if k != "per_entity_scores"}
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved summary metrics to {args.out}")


if __name__ == "__main__":
    main()
