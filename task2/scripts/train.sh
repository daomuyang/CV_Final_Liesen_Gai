#!/usr/bin/env bash
# Train ACT on calvin_env_{A,B,C,ABC}.
#
# Usage:
#   SEED=42 bash scripts/train.sh <A|B|C|ABC|ABC_fair> [smoke|server]
#
# Homework:
#   Step 1: bash scripts/train.sh A|B|C server  (各自 loss 曲线)
#   Step 2: bash scripts/train.sh ABC|ABC_fair server
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
activate_venv

EXP="${1:?Usage: SEED=42 bash scripts/train.sh <A|B|C|ABC|ABC_fair> [smoke|server]}"
MODE="${2:-server}"

RESOLVED="$(resolve_experiment "${EXP}")"
if [[ "${RESOLVED}" == "UNKNOWN" ]]; then
  echo "EXP must be A, B, C, ABC, or ABC_fair, got: ${EXP}" >&2
  exit 1
fi

IFS='|' read -r DATA_SPLIT RUN_TAG STEPS_SERVER <<< "${RESOLVED}"

DEVICE="$(detect_device)"
VIDEO_BACKEND="$(detect_video_backend "${DEVICE}")"
DS_ROOT="$(dataset_path "${DATA_SPLIT}")"
REPO_ID="$(dataset_repo_id "${DATA_SPLIT}")"
OUTPUT_DIR="$(output_dir_for_tag "${RUN_TAG}")"
JOB_NAME="act_calvin_${RUN_TAG}"

# ACT defaults per LeRobot / original paper (chunk_size=100).
# Override for old runs: CHUNK_SIZE=10 N_ACTION_STEPS=10 bash scripts/train.sh ...
CHUNK_SIZE="${CHUNK_SIZE:-100}"
N_ACTION_STEPS="${N_ACTION_STEPS:-10}"

if [[ "${MODE}" == "smoke" ]]; then
  STEPS=20
  BATCH_SIZE=2
  NUM_WORKERS=0
  SAVE_FREQ=20
  LOG_FREQ=5
  EPISODES='[0,1,2]'
  WANDB_ENABLE=false
elif [[ "${MODE}" == "server" ]]; then
  STEPS="${STEPS_SERVER}"
  BATCH_SIZE=$([[ "${DEVICE}" == "cuda" ]] && echo 16 || echo 4)
  NUM_WORKERS=$([[ "${DEVICE}" == "cuda" ]] && echo 8 || echo 2)
  SAVE_FREQ=10000
  LOG_FREQ=100
  EPISODES=""
  WANDB_ENABLE=true
else
  echo "MODE must be smoke or server, got: ${MODE}" >&2
  exit 1
fi

if [[ -d "${OUTPUT_DIR}" ]]; then
  if [[ "${MODE}" == "smoke" ]]; then
    rm -rf "${OUTPUT_DIR}"
  else
    echo "Output exists: ${OUTPUT_DIR}. Use --resume or remove it first." >&2
    exit 1
  fi
fi
mkdir -p "${TASK2_ROOT}/outputs/logs"
LOG_FILE="${TASK2_ROOT}/outputs/logs/train_${RUN_TAG}_${MODE}.log"

WANDB_ARGS=(
  --wandb.enable="${WANDB_ENABLE}"
  --wandb.project=cv_hw3_task2
  --wandb.mode=offline
)

EPISODE_ARGS=()
if [[ -n "${EPISODES}" ]]; then
  EPISODE_ARGS=(--dataset.episodes="${EPISODES}")
fi

echo "=== Train ACT | exp=${EXP} data=${DATA_SPLIT} tag=${RUN_TAG} steps=${STEPS} seed=${SEED} ==="
echo "Dataset: ${DS_ROOT}"
echo "Output:  ${OUTPUT_DIR}"
echo "ACT: chunk_size=${CHUNK_SIZE} n_action_steps=${N_ACTION_STEPS} cudnn_deterministic=${CUDNN_DETERMINISTIC}"

set -x
lerobot-train \
  --dataset.repo_id="${REPO_ID}" \
  --dataset.root="${DS_ROOT}" \
  "${EPISODE_ARGS[@]}" \
  --dataset.video_backend="${VIDEO_BACKEND}" \
  --policy.type=act \
  --policy.device="${DEVICE}" \
  --policy.push_to_hub=false \
  --policy.chunk_size="${CHUNK_SIZE}" \
  --policy.n_action_steps="${N_ACTION_STEPS}" \
  --output_dir="${OUTPUT_DIR}" \
  --job_name="${JOB_NAME}" \
  --steps="${STEPS}" \
  --batch_size="${BATCH_SIZE}" \
  --num_workers="${NUM_WORKERS}" \
  --seed="${SEED}" \
  --cudnn_deterministic="${CUDNN_DETERMINISTIC}" \
  --eval_freq=0 \
  --save_freq="${SAVE_FREQ}" \
  --log_freq="${LOG_FREQ}" \
  --tolerance_s=0.02 \
  "${WANDB_ARGS[@]}" \
  2>&1 | tee "${LOG_FILE}"
set +x

echo "Training done. Checkpoint: $(latest_checkpoint "${OUTPUT_DIR}")"
echo "Log: ${LOG_FILE}"
if [[ "${WANDB_ENABLE}" == true ]]; then
  echo "WandB offline: ${TASK2_ROOT}/wandb/  (wandb sync wandb/offline-run-*)"
fi
