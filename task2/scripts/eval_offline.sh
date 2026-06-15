#!/usr/bin/env bash
# Offline eval: trained checkpoint on any calvin_env_{A,B,C,ABC,D}.
#
# Usage:
#   bash scripts/eval_offline.sh <train_exp> <eval_env> [smoke|server]
#
# Examples:
#   bash scripts/eval_offline.sh ABC D server      # zero-shot (homework step 4)
#   bash scripts/eval_offline.sh ABC ABC server    # in-distribution sanity check
#   bash scripts/eval_offline.sh A A server        # A train -> A eval
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
activate_venv

TRAIN_EXP="${1:?Usage: bash scripts/eval_offline.sh <A|B|C|ABC|ABC_fair> <A|B|C|ABC|D> [smoke|server]}"
EVAL_ENV="${2:?Missing eval env (A|B|C|ABC|D)}"
MODE="${3:-server}"

RESOLVED="$(resolve_experiment "${TRAIN_EXP}")"
if [[ "${RESOLVED}" == "UNKNOWN" ]]; then
  echo "TRAIN_EXP must be A, B, C, ABC, or ABC_fair, got: ${TRAIN_EXP}" >&2
  exit 1
fi

IFS='|' read -r _DATA_SPLIT RUN_TAG _STEPS <<< "${RESOLVED}"

case "${EVAL_ENV}" in
  A|B|C|ABC|D) ;;
  *) echo "EVAL_ENV must be A, B, C, ABC, or D, got: ${EVAL_ENV}" >&2; exit 1 ;;
esac

DEVICE="$(detect_device)"
VIDEO_BACKEND="$(detect_video_backend "${DEVICE}")"
RUN_DIR="$(output_dir_for_tag "${RUN_TAG}")"
CKPT="$(latest_checkpoint "${RUN_DIR}")"

LABEL="$(echo "${RUN_TAG}" | tr '[:lower:]' '[:upper:]')"
EVAL_UPPER="$(echo "${EVAL_ENV}" | tr '[:lower:]' '[:upper:]')"
OUTPUT="${TASK2_ROOT}/outputs/eval_${LABEL}_on_${EVAL_UPPER}.json"

EXTRA_ARGS=(--seed="${SEED}")
if [[ "${MODE}" == "smoke" ]]; then
  EXTRA_ARGS+=(--episodes="[0,1,2]" --max-batches=5 --batch-size=2 --num-workers=0)
else
  EXTRA_ARGS+=(--batch-size=$([[ "${DEVICE}" == "cuda" ]] && echo 16 || echo 4))
  EXTRA_ARGS+=(--num-workers=$([[ "${DEVICE}" == "cuda" ]] && echo 8 || echo 2))
fi

echo "=== Offline eval | train=${TRAIN_EXP} eval=${EVAL_ENV} seed=${SEED} mode=${MODE} ==="
echo "Checkpoint: ${CKPT}"

python scripts/eval_zeroshot.py \
  --checkpoint "${CKPT}" \
  --eval-root "$(dataset_path "${EVAL_ENV}")" \
  --device "${DEVICE}" \
  --video-backend "${VIDEO_BACKEND}" \
  --train-split "${LABEL}" \
  --output "${OUTPUT}" \
  "${EXTRA_ARGS[@]}"

echo "Results: ${OUTPUT}"
