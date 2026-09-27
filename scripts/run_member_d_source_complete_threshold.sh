#!/usr/bin/env bash
set -euo pipefail

# Member D source-complete historical-threshold selection run.
#
# Inputs:
#   C: threshold gate, route table, tree provenance
#   A: already-validated complete threshold neural logits
#   B starter: threshold queries + evaluation truth
#
# This run IS selection-eligible on the frozen threshold split because C now
# declares complete_by_source_record=true and a valid leakage-safe feature
# contract. It is NOT the final production/test run because C's provenance
# explicitly identifies this as the historical bootstrap candidate universe.

D_GATE="${D_GATE:-threshold_gate_v2.parquet}"
D_ROUTE_TABLE="${D_ROUTE_TABLE:-threshold_route_table_v1.parquet}"
D_TREE_PROVENANCE="${D_TREE_PROVENANCE:-threshold_tree_provenance_v1.json}"

D_NEURAL_ROOT="${D_NEURAL_ROOT:-neural-d-handoff/artifacts}"
D_CARD="${D_CARD:-$D_NEURAL_ROOT/neural-card/checkpoint_card.json}"
D_EXPOSURE="${D_EXPOSURE:-$D_NEURAL_ROOT/neural-card/training_exposure.parquet}"
D_THRESHOLD_NEURAL="${D_THRESHOLD_NEURAL:-$D_NEURAL_ROOT/neural-scores/threshold}"
D_ROUTED_NEURAL="${D_ROUTED_NEURAL:-$D_NEURAL_ROOT/neural-scores/gate-threshold-v1}"
D_EXPECTED_CHECKPOINT="${D_EXPECTED_CHECKPOINT:-ckpt-e315e181e5896f9c}"

OUT="${D_OUT:-artifacts/member_d/current/source-complete-threshold}"
REPORTS="${D_REPORTS:-reports/member_d/current/source-complete-threshold}"
PREP="$OUT/c-handoff"
JOIN="$OUT/joined"
SELECTION="$OUT/selection"

mkdir -p "$OUT" "$REPORTS"

for f in "$D_GATE" "$D_ROUTE_TABLE" "$D_TREE_PROVENANCE" \
         "$D_CARD" "$D_EXPOSURE" \
         "$D_THRESHOLD_NEURAL/neural_manifest.json"; do
  if [[ ! -e "$f" ]]; then
    echo "ERROR: required input not found: $f" >&2
    exit 2
  fi
done

if [[ ! -f configs/member_d/split.tsv ]]; then
  echo "Frozen D split not found; generating it..."
  python -m src.member_d_split --config configs/member_d/split.json
fi

echo "== 1. Revalidate A's full threshold neural handoff against C's gate =="
handoff_args=(
  --checkpoint-card "$D_CARD"
  --training-exposure "$D_EXPOSURE"
  --threshold-score-dir "$D_THRESHOLD_NEURAL"
  --gate "$D_GATE"
  --expected-checkpoint "$D_EXPECTED_CHECKPOINT"
  --expected-rows 993174
  --output "$REPORTS/neural_handoff_validation.json"
)
if [[ -d "$D_ROUTED_NEURAL" ]]; then
  handoff_args+=(--routed-score-dir "$D_ROUTED_NEURAL")
fi
python -m src.member_d_handoff "${handoff_args[@]}"

echo "== 2. Validate/adapt C source-complete threshold handoff =="
python -m src.member_d_c_handoff \
  --gate "$D_GATE" \
  --route-table "$D_ROUTE_TABLE" \
  --tree-provenance "$D_TREE_PROVENANCE" \
  --output-dir "$PREP" \
  | tee "$REPORTS/c_handoff_validation.log"

echo "== 3. Run threshold-only oracle/cutoff audit on the source-complete gate =="
python - <<'PY'
import json
import os
from pathlib import Path

from src.member_d_gate_audit import audit_gate, write_report

cfg = json.loads(Path("configs/member_d/gate_audit.json").read_text())
report = audit_gate(
    os.environ.get("D_GATE", "threshold_gate_v2.parquet"),
    "member-b-starter-v1/threshold_queries.tsv",
    "member-b-starter-v1/evaluation_truth.tsv",
    cutoffs=cfg["cutoffs"],
    pairs_per_second=cfg["pairs_per_second"],
    production_top3_pairs=cfg["production_top3_pairs"],
)
write_report(
    report,
    "reports/member_d/current/source-complete-threshold/gate-audit",
)
print(json.dumps(report["gate"], indent=2))
print(json.dumps(report["cutoff_audit"], indent=2))
PY

echo "== 4. Build separate keyed labels =="
python -m src.member_d_labels \
  --pairs "$PREP/candidates.parquet" \
  --query-manifest member-b-starter-v1/threshold_queries.tsv \
  --truth member-b-starter-v1/evaluation_truth.tsv \
  --output "$OUT/threshold_labels.parquet"

echo "== 5. Strict Member-D score join using C provenance/routes + A logits =="
python - <<'PY'
import glob
import os

from src.score_join import join_scores

root = os.environ.get(
    "D_OUT",
    "artifacts/member_d/current/source-complete-threshold",
)
prep = f"{root}/c-handoff"
neural_dir = os.environ.get(
    "D_THRESHOLD_NEURAL",
    "neural-d-handoff/artifacts/neural-scores/threshold",
)

neural_shards = sorted(
    glob.glob(f"{neural_dir}/*.neural.parquet")
)
if not neural_shards:
    raise SystemExit(
        f"No threshold neural score shards found under {neural_dir}"
    )

result = join_scores(
    [f"{prep}/candidates.parquet"],
    [f"{prep}/tree_scores.parquet"],
    output_dir=f"{root}/joined",
    candidate_manifest_path=f"{prep}/candidate_manifest.json",
    tree_score_manifest_path=f"{prep}/tree_manifest.normalized.json",
    neural_score_shards=neural_shards,
    neural_score_manifest_path=f"{neural_dir}/neural_manifest.json",
    route_table_path=f"{prep}/routes.parquet",
    provenance={
        "version": "source_complete_threshold_join_v1",
        "selection_scope": "historical_threshold",
        "production_ready": False,
    },
)
print(result)
PY

echo "== 6. Fit real source-complete fusion/calibration and select decoder =="
ownership_arg=()
FULL_OWNERSHIP="${D_FULL_TRAINING_OWNERSHIP_AUDIT:-}"
if [[ -n "$FULL_OWNERSHIP" && -f "$FULL_OWNERSHIP" ]]; then
  ownership_arg+=(--ownership-audit "$FULL_OWNERSHIP")
fi

python -m src.member_d_threshold_selection \
  --joined-scores "$JOIN/joined_scores.parquet" \
  --labels "$OUT/threshold_labels.parquet" \
  --split configs/member_d/split.tsv \
  --truth member-b-starter-v1/evaluation_truth.tsv \
  --fusion-config configs/member_d/fusion.json \
  --calibration-config configs/member_d/calibration.json \
  --decoder-config configs/member_d/decoder.json \
  --output-dir "$SELECTION" \
  "${ownership_arg[@]}"

echo
echo "Source-complete threshold selection completed."
echo
echo "Important:"
echo "  - This run IS eligible for threshold-model selection."
echo "  - It is NOT the final production/test run."
echo "  - Ownership is production-eligible only with a full-training ownership audit."
echo
echo "Review:"
echo "  $PREP/handoff_report.json"
echo "  $JOIN/join_manifest.json"
echo "  $SELECTION/selection_decoder.json"
echo "  $SELECTION/selection_summary.json"
echo "  $REPORTS/gate-audit/gate_cutoffs.tsv"
