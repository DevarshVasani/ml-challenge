#!/usr/bin/env bash
set -euo pipefail

# Member D work that is safe to do now with:
#   C threshold_gate_v2.parquet
#   A neural-card + complete threshold neural scores
#
# This is intentionally diagnostic/provisional.  It does NOT certify C's
# incomplete competition features and it does NOT tune on final.

branch="$(git branch --show-current)"
if [[ "$branch" != "Daksh" ]]; then
  echo "ERROR: expected branch Daksh, current branch is '$branch'" >&2
  exit 2
fi

D_GATE="${D_GATE:-threshold_gate_v2.parquet}"
D_NEURAL_ROOT="${D_NEURAL_ROOT:-neural-d-handoff/artifacts}"
D_CARD="${D_CARD:-$D_NEURAL_ROOT/neural-card/checkpoint_card.json}"
D_EXPOSURE="${D_EXPOSURE:-$D_NEURAL_ROOT/neural-card/training_exposure.parquet}"
D_THRESHOLD_NEURAL="${D_THRESHOLD_NEURAL:-$D_NEURAL_ROOT/neural-scores/threshold}"
D_ROUTED_NEURAL="${D_ROUTED_NEURAL:-$D_NEURAL_ROOT/neural-scores/gate-threshold-v1}"
D_EXPECTED_CHECKPOINT="${D_EXPECTED_CHECKPOINT:-ckpt-e315e181e5896f9c}"

OUT="${D_OUT:-artifacts/member_d/current/threshold-rehearsal}"
REPORTS="${D_REPORTS:-reports/member_d/current}"
mkdir -p "$OUT" "$REPORTS"

if [[ ! -f "$D_GATE" ]]; then
  echo "ERROR: gate not found: $D_GATE" >&2
  exit 2
fi

if [[ ! -f configs/member_d/split.tsv ]]; then
  echo "Frozen D split not found; generating it..."
  python -m src.member_d_split --config configs/member_d/split.json
fi

echo "== 1. Validate A checkpoint + full threshold + routed threshold handoff =="
python -m src.member_d_handoff \
  --checkpoint-card "$D_CARD" \
  --training-exposure "$D_EXPOSURE" \
  --threshold-score-dir "$D_THRESHOLD_NEURAL" \
  --routed-score-dir "$D_ROUTED_NEURAL" \
  --gate "$D_GATE" \
  --expected-checkpoint "$D_EXPECTED_CHECKPOINT" \
  --expected-rows 993174 \
  --clean-zone-identifiers \
  --handoff-root "$D_NEURAL_ROOT" \
  --output "$REPORTS/neural_handoff_validation.json"

echo "== 2. Correct threshold-only gate/cutoff audit =="
python - <<'PY'
import json, os
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
write_report(report, "reports/member_d/current/gate-audit")
print(json.dumps(report["gate"], indent=2))
print(json.dumps(report["cutoff_audit"], indent=2))
PY

echo "== 3. Build separate keyed threshold labels =="
python -m src.member_d_labels \
  --pairs "$D_GATE" \
  --query-manifest member-b-starter-v1/threshold_queries.tsv \
  --truth member-b-starter-v1/evaluation_truth.tsv \
  --output "$OUT/threshold_labels.parquet"

echo "== 4. Build diagnostic joined scores with incomplete-group features removed =="
python -m src.member_d_rehearsal \
  --gate "$D_GATE" \
  --neural-score-dir "$D_THRESHOLD_NEURAL" \
  --output "$OUT/joined_scores.parquet"

echo "== 5. Fit/apply provisional minimal fusion and calibration =="
python - <<'PY'
import json
from pathlib import Path
import pandas as pd

from src.fusion import fit_fusion, apply_fusion
from src.calibration import fit_calibration, apply_calibration

root = Path("artifacts/member_d/current/threshold-rehearsal")
joined = root / "joined_scores.parquet"
labels = root / "threshold_labels.parquet"
split = Path("configs/member_d/split.tsv")
fusion_dir = root / "fusion-provisional"
cal_dir = root / "calibration-provisional"

fusion_cfg = json.loads(
    Path("configs/member_d/fusion_provisional_minimal.json").read_text()
)
cal_cfg = json.loads(
    Path("configs/member_d/calibration.json").read_text()
)

fit_fusion(joined, labels, split, fusion_cfg, fusion_dir)
frame = pd.read_parquet(joined)
fused = apply_fusion(frame, fusion_dir)
fused_path = root / "fused_scores.parquet"
fused.to_parquet(fused_path, index=False)

fit_calibration(fused_path, labels, split, cal_cfg, cal_dir)
calibrated = apply_calibration(fused, cal_dir)
calibrated_path = root / "calibrated_scores.parquet"
calibrated.to_parquet(calibrated_path, index=False)

print("wrote", fused_path)
print("wrote", calibrated_path)
PY

echo "== 6. Provisional decoder sweep on D selection partition only =="
python - <<'PY'
import json
from pathlib import Path
import pandas as pd

from src.decoder import select_decoder

root = Path("artifacts/member_d/current/threshold-rehearsal")
frame = pd.read_parquet(root / "calibrated_scores.parquet")
split = pd.read_csv(
    "configs/member_d/split.tsv",
    sep="\t",
    dtype=str,
    keep_default_na=False,
)
selection_ids = split.loc[
    split.member_d_partition == "selection",
    "source1_entity_id",
].astype(str).tolist()
selection = frame[
    frame.source1_entity_id.astype(str).isin(selection_ids)
].copy()

truth_df = pd.read_csv(
    "member-b-starter-v1/evaluation_truth.tsv",
    sep="\t",
    dtype=str,
    keep_default_na=False,
)
truth = {}
wanted = set(selection_ids)
for row in truth_df.itertuples(index=False):
    sid = str(row.source1_entity_id)
    if sid in wanted:
        truth[sid] = [
            x.strip()
            for x in str(row.matched_entity_ids or "").split(",")
            if x.strip()
        ]

decoder_cfg = json.loads(Path("configs/member_d/decoder.json").read_text())
result = select_decoder(
    selection,
    truth,
    selection_ids,
    thresholds=list(map(float, decoder_cfg["thresholds"])),
    ownership_audit=None,   # source-record groups are not complete here
    enable_expected_f05=False,
    output_path=root / "selection_decoder_provisional.json",
)
result["diagnostic_only"] = True
result["selection_eligible"] = False
result["reason"] = (
    "Threshold competition groups are incomplete; rerun after C publishes "
    "source-complete threshold/test artifacts."
)
(root / "selection_decoder_provisional.json").write_text(
    json.dumps(result, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(result, indent=2))
PY

echo
echo "Threshold rehearsal complete."
echo "IMPORTANT: outputs are diagnostic only and must not be used as the final production freeze."
echo "Reports:"
echo "  $REPORTS/neural_handoff_validation.json"
echo "  $REPORTS/gate-audit/gate_audit.json"
echo "  $REPORTS/gate-audit/gate_cutoffs.tsv"
echo "Artifacts:"
echo "  $OUT/"
