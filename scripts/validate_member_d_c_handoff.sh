#!/usr/bin/env bash
set -euo pipefail

# Audit Member C's threshold-v2 + final-standin handoff.
#
# The stand-in is deliberately validated for integration rehearsal only.
# This script NEVER promotes it to final production status.
#
# Expected local folder:
#
# member-c-handoff/
#   threshold_gate_v2.parquet
#   threshold_route_table_v2.parquet
#   threshold_tree_provenance_v2.json
#   production_candidate_manifest.json
#   production_tree_manifest.json
#   production_route_manifest.json
#   production_candidates_v1-final-standin/
#   production_tree_scores_v1-final-standin/
#   production_routes_v1-final-standin/

ROOT="${D_C_HANDOFF_ROOT:-member-c-handoff}"
OUT="${D_C_AUDIT_OUT:-artifacts/member_d/current/c-handoff-audit}"
B_REPORT="${D_B_HANDOFF_REPORT:-artifacts/member_d/current/b-handoff/b_handoff_report.json}"

args=(
  --threshold-gate "$ROOT/threshold_gate_v2.parquet"
  --threshold-route "$ROOT/threshold_route_table_v2.parquet"
  --threshold-provenance "$ROOT/threshold_tree_provenance_v2.json"
  --candidate-manifest "$ROOT/production_candidate_manifest.json"
  --candidate-dir "$ROOT/production_candidates_v1-final-standin"
  --tree-manifest "$ROOT/production_tree_manifest.json"
  --tree-dir "$ROOT/production_tree_scores_v1-final-standin"
  --route-manifest "$ROOT/production_route_manifest.json"
  --route-dir "$ROOT/production_routes_v1-final-standin"
  --output-dir "$OUT"
)

if [[ -f "$B_REPORT" ]]; then
  args+=(--b-handoff-report "$B_REPORT")
fi

python -m src.member_d_c_artifact_audit "${args[@]}"

echo
echo "C handoff audit complete."
echo "Review:"
echo "  $OUT/c_handoff_audit.json"
echo
echo "Expected current status:"
echo "  threshold-v2               -> selection eligible"
echo "  final-standin              -> rehearsal only"
echo "  real reverse-v1 production -> still required from C"
