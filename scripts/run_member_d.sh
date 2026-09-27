#!/usr/bin/env bash
set -euo pipefail

# Member D Plan 1-3 runner. No large artifacts are stored in Git.
# Required environment variables for the full pipeline:
#   D_CANDIDATE_SHARDS       colon-separated source-complete candidate parquet paths
#   D_CANDIDATE_MANIFESTS    colon-separated source-complete manifest paths
#   D_TREE_SCORE_SHARDS      colon-separated tree score parquet paths
#   D_TREE_SCORE_MANIFEST    tree prediction/provenance manifest
#   D_LABELS                 keyed labels table for threshold split
# Optional:
#   D_NEURAL_SCORE_SHARDS    colon-separated neural score paths
#   D_NEURAL_SCORE_MANIFEST  neural manifest
#   D_ROUTE_TABLE            keyed route table

python -m src.member_d_split --config configs/member_d/split.json
python -m src.runtime_budget --config configs/member_d/runtime_budget.json --output reports/member_d/current/runtime_budget.json --refuse-on-violation

: "${D_CANDIDATE_SHARDS:?Set D_CANDIDATE_SHARDS}"
: "${D_CANDIDATE_MANIFESTS:?Set D_CANDIDATE_MANIFESTS}"
: "${D_TREE_SCORE_SHARDS:?Set D_TREE_SCORE_SHARDS}"
: "${D_TREE_SCORE_MANIFEST:?Set D_TREE_SCORE_MANIFEST}"
: "${D_LABELS:?Set D_LABELS}"

IFS=':' read -r -a CANDIDATES <<< "$D_CANDIDATE_SHARDS"
IFS=':' read -r -a CAND_MANIFESTS <<< "$D_CANDIDATE_MANIFESTS"
IFS=':' read -r -a TREE_SCORES <<< "$D_TREE_SCORE_SHARDS"
NEURAL_ARG="[]"
if [[ -n "${D_NEURAL_SCORE_SHARDS:-}" ]]; then
  IFS=':' read -r -a NEURAL_SCORES <<< "$D_NEURAL_SCORE_SHARDS"
  NEURAL_ARG=$(python -c 'import json,sys; print(json.dumps(sys.argv[1:]))' "${NEURAL_SCORES[@]}")
fi

python - "${CANDIDATES[@]}" -- "${CAND_MANIFESTS[@]}" <<'PY'
# Intentionally only verifies source-complete grouping here; feature output path is fixed.
import sys
from src.source_competition import build_source_competition_artifact
args=sys.argv[1:]; cut=args.index('--'); shards=args[:cut]; manifests=args[cut+1:]
build_source_competition_artifact(shards, manifests, 'artifacts/member_d/current/source_competition/candidates.parquet', version='source_competition_v1')
PY

python - <<'PY'
import json, os
from src.score_join import join_scores
split=lambda v: [x for x in os.environ[v].split(':') if x]
join_scores(
    ['artifacts/member_d/current/source_competition/candidates.parquet'],
    split('D_TREE_SCORE_SHARDS'),
    output_dir='artifacts/member_d/current/joined_scores',
    candidate_manifest_path='artifacts/member_d/current/source_competition/source_competition_manifest.json',
    tree_score_manifest_path=os.environ['D_TREE_SCORE_MANIFEST'],
    neural_score_shards=split('D_NEURAL_SCORE_SHARDS') if os.environ.get('D_NEURAL_SCORE_SHARDS') else None,
    neural_score_manifest_path=os.environ.get('D_NEURAL_SCORE_MANIFEST'),
    route_table_path=os.environ.get('D_ROUTE_TABLE'),
    provenance={'version':'joined_v1'},
)
PY

echo "Member D preparation/join complete. Fit fusion/calibration using D_LABELS and configs/member_d/*.json after keyed score coverage is available."
