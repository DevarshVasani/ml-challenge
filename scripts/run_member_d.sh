#!/usr/bin/env bash
# run_member_d.sh - Orchestrates Member D pipeline execution (Steps 15-19)
set -e

# Usage: bash scripts/run_member_d.sh

echo "=== Member D Pipeline ==="
echo ""

# 15. Freeze partitions (already run, but ensuring it runs)
echo "[1/6] Running split to freeze partitions..."
python -m src.member_d_split --config configs/member_d/split.json

# 16. Evaluate tree-only fallback on keyed artifacts (Step 17)
# Note: This requires the tree scoring pipeline (Member A) to have produced
# 'artifacts/tree_scores_fallback/' first.
echo "[2/6] Evaluating fallback tree-only (Placeholder)..."
# python -m src.evaluate_pipeline --config configs/member_d/evaluate_threshold.json

# 18. Run Score Join -> Fusion -> Calibration -> Decoder Selection
echo "[3/6] Joining Scores..."
# python -m src.score_join \
#     --candidates artifacts/candidates/ \
#     --tree-scores artifacts/tree_scores/ \
#     --neural-scores artifacts/neural_scores/ \
#     --output-dir artifacts/member_d/joined_scores/

echo "[4/6] Running Fusion..."
# python -m src.fusion --config configs/member_d/fusion.json \
#     --input artifacts/member_d/joined_scores/ \
#     --output-dir artifacts/member_d/fusion/

echo "[5/6] Running Calibration..."
# python -m src.calibration --config configs/member_d/calibration.json \
#     --input artifacts/member_d/fusion/ \
#     --output-dir artifacts/member_d/calibrated_scores/

echo "[6/6] Selecting Decoder and Freezing Config..."
# python -c "from src.decoder import select_decoder; select_decoder('artifacts/member_d/calibrated_scores/', 'configs/member_d/decoder.json')"

echo ""
echo "=== Member D Pipeline Complete ==="
echo "Note: The execution commands are currently commented out because the required upstream data (Member A's tree scores, Member C's neural scores) are not yet generated in the 'artifacts/' directory. Once those are available, uncomment the execution lines in this script."
