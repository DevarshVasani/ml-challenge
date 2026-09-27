"""Evaluation module for Entity Resolution challenge.

Metric specification:
- Score each Source 1 (S1) entity separately, then macro-average over all S1 entities.
- For non-singletons (entities with >= 1 true matches):
    F0.5 = 5 * TP / (5 * TP + 4 * FP + FN)
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
    true_set: Set[str] = set(true_ids)
    pred_set: Set[str] = set(pred_ids)
    return (
        len(true_set & pred_set),
        len(pred_set - true_set),
        len(true_set - pred_set),
    )


def score_single_s1(true_ids: Iterable[str], pred_ids: Iterable[str]) -> float:
    true_set: Set[str] = {x for x in true_ids if x}
    pred_set: Set[str] = {x for x in pred_ids if x}

    if not true_set:
        return 1.0 if not pred_set else 0.0
    if not pred_set:
        return 0.0

    tp, fp, fn = compute_confusion_components(true_set, pred_set)
    denom = 5 * tp + 4 * fp + fn
    return (5.0 * tp) / float(denom) if denom else 0.0


def evaluate_predictions(
    ground_truth: Dict[str, Iterable[str]],
    predictions: Dict[str, Iterable[str]],
) -> Dict[str, Union[float, int, Dict[str, float]]]:
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
        score = score_single_s1(true_list, pred_list)
        per_entity_scores[s1_id] = score
        total_score += score

        if not true_list:
            singleton_count += 1
            if score == 1.0:
                singleton_correct += 1
        else:
            non_singleton_count += 1
            non_singleton_total_score += score

    num_entities = len(ground_truth)
    return {
        "macro_f05": total_score / num_entities,
        "num_entities": num_entities,
        "num_singletons": singleton_count,
        "num_non_singletons": non_singleton_count,
        "singleton_accuracy": (
            singleton_correct / singleton_count if singleton_count else 1.0
        ),
        "non_singleton_macro_f05": (
            non_singleton_total_score / non_singleton_count
            if non_singleton_count
            else 1.0
        ),
        "per_entity_scores": per_entity_scores,
    }


def parse_id_string(val: Optional[Union[str, float]]) -> List[str]:
    if val is None:
        return []
    if isinstance(val, float):
        if val != val:
            return []
        val = str(val)
    text = str(val).strip()
    if not text:
        return []
    seen: Set[str] = set()
    result: List[str] = []
    for item in text.split(","):
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
    records: Dict[str, List[str]] = {}
    with open(path, mode="r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if id_col not in (reader.fieldnames or []):
            raise KeyError(
                f"Expected column '{id_col}' in TSV file '{path}'. "
                f"Found: {reader.fieldnames}"
            )
        if matches_col not in (reader.fieldnames or []):
            raise KeyError(
                f"Expected column '{matches_col}' in TSV file '{path}'. "
                f"Found: {reader.fieldnames}"
            )
        for row in reader:
            s1_id = row[id_col].strip()
            if s1_id:
                records[s1_id] = parse_id_string(row[matches_col])
    return records


def run_tiny_example_checks() -> bool:
    test_cases = [
        ("perfect_match", ["S2-0001", "S3-0002"], ["S2-0001", "S3-0002"], 1.0),
        ("extra_match", ["S2-0001"], ["S2-0001", "S2-0002"], 5.0 / 9.0),
        ("missed_match", ["S2-0001", "S3-0002"], ["S2-0001"], 5.0 / 6.0),
        ("empty_prediction_non_singleton", ["S2-0001"], [], 0.0),
        ("singleton_empty_prediction", [], [], 1.0),
        ("singleton_non_empty_prediction", [], ["S2-0001"], 0.0),
    ]
    all_passed = True
    print("=== Running Scorer Tiny Example Checks ===")
    for name, true_ids, pred_ids, expected in test_cases:
        score = score_single_s1(true_ids, pred_ids)
        passed = abs(score - expected) < 1e-7
        print(
            f"[{'PASS' if passed else 'FAIL'}] {name:<32} "
            f"| Score: {score:.6f} | Expected: {expected:.6f}"
        )
        all_passed &= passed

    mini_gt = {
        "S1-1": ["S2-1", "S3-1"],
        "S1-2": ["S2-2"],
        "S1-3": ["S2-3", "S3-3"],
        "S1-4": ["S2-4"],
        "S1-5": [],
        "S1-6": [],
    }
    mini_pred = {
        "S1-1": ["S2-1", "S3-1"],
        "S1-2": ["S2-2", "S3-99"],
        "S1-3": ["S2-3"],
        "S1-4": [],
        "S1-5": [],
        "S1-6": ["S2-99"],
    }
    expected_macro = (
        1.0 + (5.0 / 9.0) + (5.0 / 6.0) + 0.0 + 1.0 + 0.0
    ) / 6.0
    result = evaluate_predictions(mini_gt, mini_pred)
    macro_passed = abs(float(result["macro_f05"]) - expected_macro) < 1e-7
    print(
        f"[{'PASS' if macro_passed else 'FAIL'}] "
        f"{'macro_average_all_cases':<32} | Macro F0.5: "
        f"{float(result['macro_f05']):.6f} | Expected: {expected_macro:.6f}"
    )
    all_passed &= macro_passed
    return all_passed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Entity Resolution matching results."
    )
    parser.add_argument("--gt", "--ground-truth", dest="gt_path")
    parser.add_argument("--pred", "--predictions", dest="pred_path")
    parser.add_argument("--check", "--self-check", action="store_true")
    parser.add_argument("--out")
    parser.add_argument("--id-col", default="source1_entity_id")
    parser.add_argument("--matches-col", default="matched_entity_ids")
    args = parser.parse_args()

    if args.check:
        if not run_tiny_example_checks():
            sys.exit(1)
        if not args.gt_path:
            return

    if not args.gt_path or not args.pred_path:
        if not args.check:
            parser.print_help()
        return

    gt = load_matches_tsv(
        args.gt_path, id_col=args.id_col, matches_col=args.matches_col
    )
    pred = load_matches_tsv(
        args.pred_path, id_col=args.id_col, matches_col=args.matches_col
    )
    metrics = evaluate_predictions(gt, pred)

    print("--- Evaluation Summary ---")
    print(f"Macro F0.5 Score: {float(metrics['macro_f05']):.6f}")
    print(f"Total S1 Entities: {metrics['num_entities']}")
    print(
        f"Singletons: {metrics['num_singletons']} "
        f"(Accuracy: {float(metrics['singleton_accuracy']):.4f})"
    )
    print(
        f"Non-Singletons: {metrics['num_non_singletons']} "
        f"(Macro F0.5: {float(metrics['non_singleton_macro_f05']):.6f})"
    )

    if args.out:
        summary = {k: v for k, v in metrics.items() if k != "per_entity_scores"}
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()
