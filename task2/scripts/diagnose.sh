#!/usr/bin/env bash
# Quick sanity check: in-distribution vs zero-shot L1 (50 batches each).
# Run after training to verify the model learned something before full eval.
#
# Usage: bash scripts/diagnose.sh <A|B|C|ABC|ABC_fair>
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
activate_venv

EXP="${1:?Usage: bash scripts/diagnose.sh <A|B|C|ABC|ABC_fair>}"
RESOLVED="$(resolve_experiment "${EXP}")"
if [[ "${RESOLVED}" == "UNKNOWN" ]]; then
  echo "EXP must be A, B, C, ABC, or ABC_fair" >&2
  exit 1
fi

IFS='|' read -r DATA_SPLIT RUN_TAG _STEPS <<< "${RESOLVED}"
DEVICE="$(detect_device)"
VIDEO_BACKEND="$(detect_video_backend "${DEVICE}")"
CKPT="$(latest_checkpoint "$(output_dir_for_tag "${RUN_TAG}")")"
LABEL="$(echo "${RUN_TAG}" | tr '[:lower:]' '[:upper:]')"

echo "=== Diagnose ${EXP} | seed=${SEED} | ckpt=${CKPT} ==="
echo ""
echo "--- In-distribution: ${DATA_SPLIT} -> ${DATA_SPLIT} (50 batches) ---"
python scripts/eval_zeroshot.py \
  --checkpoint "${CKPT}" \
  --eval-root "$(dataset_path "${DATA_SPLIT}")" \
  --device "${DEVICE}" \
  --video-backend "${VIDEO_BACKEND}" \
  --train-split "${LABEL}" \
  --seed "${SEED}" \
  --max-batches=50 \
  --batch-size=$([[ "${DEVICE}" == "cuda" ]] && echo 16 || echo 4) \
  --num-workers=$([[ "${DEVICE}" == "cuda" ]] && echo 4 || echo 0) \
  --output "${TASK2_ROOT}/outputs/diagnose_${LABEL}_in_dist.json"

echo ""
echo "--- Zero-shot: ${DATA_SPLIT} -> D (50 batches) ---"
python scripts/eval_zeroshot.py \
  --checkpoint "${CKPT}" \
  --eval-root "$(dataset_path D)" \
  --device "${DEVICE}" \
  --video-backend "${VIDEO_BACKEND}" \
  --train-split "${LABEL}" \
  --seed "${SEED}" \
  --max-batches=50 \
  --batch-size=$([[ "${DEVICE}" == "cuda" ]] && echo 16 || echo 4) \
  --num-workers=$([[ "${DEVICE}" == "cuda" ]] && echo 4 || echo 0) \
  --output "${TASK2_ROOT}/outputs/diagnose_${LABEL}_on_D.json"

echo ""
echo "--- Summary (quick) ---"
python scripts/summarize_eval.py \
  --in-dist "${TASK2_ROOT}/outputs/diagnose_${LABEL}_in_dist.json" \
  --zero-shot "${TASK2_ROOT}/outputs/diagnose_${LABEL}_on_D.json" \
  --output "${TASK2_ROOT}/outputs/diagnose_${LABEL}_summary.json" \
  2>/dev/null | tail -20 || true

echo ""
echo "Interpretation:"
echo "  in_dist action_l1_loss should be LOW (~0.2-0.5) if training worked"
echo "  zero-shot on D will be HIGHER — that is expected cross-env shift"
echo "  chunk_horizon_degradation > 1 → later chunk steps harder (Action Chunking)"
echo "  Full report: bash scripts/eval_report.sh ${EXP} server"
