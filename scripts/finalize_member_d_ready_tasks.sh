#!/usr/bin/env bash
set -euo pipefail

# Complete the READY-TO-GO Member-D baseline checks.
# Run from repository root on Daksh.
#
# Handoff-specific A/C validation is in:
#   scripts/run_member_d_threshold_rehearsal.sh
#
# Outputs are intentionally written under ignored/local paths.

branch="$(git branch --show-current)"
if [[ "$branch" != "Daksh" ]]; then
  echo "ERROR: expected branch Daksh, current branch is '$branch'" >&2
  exit 2
fi

mkdir -p reports/member_d/current

echo "== 1. Generate frozen Member-D split =="
python -m src.member_d_split --config configs/member_d/split.json

echo "== 2. Validate frozen split =="
python - <<'PY'
from pathlib import Path
import json
import pandas as pd

queries = pd.read_csv(
    "member-b-starter-v1/threshold_queries.tsv",
    sep="\t",
    dtype=str,
    keep_default_na=False,
)
split = pd.read_csv(
    "configs/member_d/split.tsv",
    sep="\t",
    dtype=str,
    keep_default_na=False,
)
qid = "entity_id" if "entity_id" in queries.columns else "source1_entity_id"

assert len(split) == len(queries), (len(split), len(queries))
assert split["source1_entity_id"].is_unique
assert set(split["source1_entity_id"]) == set(queries[qid])
assert set(split["member_d_partition"]) == {
    "fusion_fit", "calibration", "selection"
}

manifest = json.loads(
    Path("configs/member_d/split_manifest.json").read_text(encoding="utf-8")
)
required = {
    "seed",
    "git_commit",
    "threshold_queries_sha256",
    "evaluation_truth_sha256",
    "component_hash",
    "component_count",
    "partition_counts",
    "country_counts",
    "singleton_counts",
    "gold_edge_counts",
}
missing = required - set(manifest)
assert not missing, f"split manifest missing: {sorted(missing)}"

print(split["member_d_partition"].value_counts().to_string())
print("split manifest OK")
PY

echo "== 3. Run evaluation-truth ownership audit =="
python - <<'PY'
from pathlib import Path
import json
from src.ownership import audit_gold_ownership

result = audit_gold_ownership(
    "member-b-starter-v1/evaluation_truth.tsv"
)
result["audit_scope"] = "evaluation_truth_only"
result["description"] = (
    "Evaluation-truth ownership audit; "
    "not full-training ownership verification."
)

out = Path("reports/member_d/current/ownership_audit.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(
    json.dumps(result, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

print(json.dumps(result, indent=2))
if result["multi_owner_violation_count"] != 0:
    raise SystemExit(
        "Ownership audit found multi-owner violations. "
        "Do not enable ownership until reviewed."
    )
PY

echo "== 4. Canonical evaluator self-check =="
python -m src.evaluate --check

echo "== 5. Focused Member-D tests =="
python -m pytest -q \
  tests/test_evaluate.py \
  tests/test_member_d_split.py \
  tests/test_member_d_integration.py \
  tests/test_member_d_handoff_tools.py \
  tests/test_fusion_calibration.py \
  tests/test_evaluation_slices.py \
  tests/test_final_assembly.py

echo "== 6. Full test suite =="
python -m pytest -q 2>&1 | tee reports/member_d/current/pytest_full.log

echo "== 7. Verify generated artifacts remain ignored =="
unexpected="$(
  git status --short | \
  grep -E '(\.pytest-tmp|audit_output|reports/member_d/current|configs/member_d/(split\.tsv|split_manifest\.json|selection_split\.tsv|selection_split_manifest\.json|base_model_exclusion_ids\.tsv))' \
  || true
)"

if [[ -n "$unexpected" ]]; then
  echo "ERROR: generated files are unexpectedly visible to Git:" >&2
  echo "$unexpected" >&2
  exit 3
fi

echo
echo "Baseline Member-D validation completed successfully."
echo "For the current A/C handoff, next run:"
echo "  bash scripts/run_member_d_threshold_rehearsal.sh"
echo
echo "Review branch state:"
git status --short
