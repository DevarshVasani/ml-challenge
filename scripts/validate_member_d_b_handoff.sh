#!/usr/bin/env bash
set -euo pipefail

# Validate Member B's handoff without requiring D to copy the 4.8-4.9 GB
# train/test retrieval buckets locally.
#
# Expected local extraction:
#   member-d-handoff/
#     dataset/train/train_ground_truth.tsv
#     member-b-starter-v1/{manifest.json,evaluation_truth.tsv,...}
#     reverse-v1/{frozen_policy.json,country_truth_audit.json,
#                 train_union_manifest.json,test_union_manifest.json}
#
# Override D_B_HANDOFF_ROOT if your folder has a different name.

ROOT="${D_B_HANDOFF_ROOT:-member-d-handoff}"
OUT="${D_B_HANDOFF_OUT:-artifacts/member_d/current/b-handoff}"

python -m src.member_d_b_handoff \
  --starter-manifest "$ROOT/member-b-starter-v1/manifest.json" \
  --evaluation-truth "$ROOT/member-b-starter-v1/evaluation_truth.tsv" \
  --threshold-candidate-pairs "$ROOT/member-b-starter-v1/threshold_candidate_pairs.parquet" \
  --threshold-queries "$ROOT/member-b-starter-v1/threshold_queries.tsv" \
  --threshold-records "$ROOT/member-b-starter-v1/threshold_records.parquet" \
  --train-labels "$ROOT/member-b-starter-v1/train_labels.parquet" \
  --train-ground-truth "$ROOT/dataset/train/train_ground_truth.tsv" \
  --country-truth-audit "$ROOT/reverse-v1/country_truth_audit.json" \
  --frozen-policy "$ROOT/reverse-v1/frozen_policy.json" \
  --train-union-manifest "$ROOT/reverse-v1/train_union_manifest.json" \
  --test-union-manifest "$ROOT/reverse-v1/test_union_manifest.json" \
  --output-dir "$OUT"

echo
echo "B handoff validated."
echo "Report:"
echo "  $OUT/b_handoff_report.json"
echo "Ownership audit:"
echo "  $OUT/full_training_ownership_audit.json"
echo
echo "To rerun D threshold selection with ownership enabled:"
echo "  D_FULL_TRAINING_OWNERSHIP_AUDIT=$OUT/full_training_ownership_audit.json \\"
echo "    bash scripts/run_member_d_source_complete_threshold.sh"
