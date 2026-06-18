#!/usr/bin/env bash
set -euo pipefail

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "${PIPELINE_DIR}/../common.sh"
# shellcheck source=../timing.sh
source "${PIPELINE_DIR}/../timing.sh"
# shellcheck source=../wandb.sh
source "${PIPELINE_DIR}/../wandb.sh"

SCENE_NAME="${1:-garden}"
GPU_ID="${GPU_ID:-0}"
PORT="${PORT:-6020}"
MAX_STEPS="${MAX_STEPS:-10000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-2000}"

GS_REPO="${THIRD_PARTY_DIR}/2d-gaussian-splatting"
SCENE_DIR="${DATASET_ROOT}/${SCENE_NAME}"
TRAIN_OUTPUT_DIR="${SCENE_DIR}/2dgs_output"

build_test_iterations "$MAX_STEPS" "$EVAL_INTERVAL"

activate_conda_env
setup_cuda_toolchain

if [ ! -d "$SCENE_DIR" ]; then
  echo "Scene directory not found: $SCENE_DIR" >&2
  exit 1
fi

if [ ! -d "${SCENE_DIR}/sparse" ]; then
  echo "COLMAP sparse directory not found: ${SCENE_DIR}/sparse" >&2
  exit 1
fi

if [ -d "${SCENE_DIR}/images" ]; then
  IMAGE_FOLDER="images"
elif [ -d "${SCENE_DIR}/images_4" ]; then
  IMAGE_FOLDER="images_4"
elif [ -d "${SCENE_DIR}/images_2" ]; then
  IMAGE_FOLDER="images_2"
else
  echo "Image directory not found under: $SCENE_DIR" >&2
  exit 1
fi

rm -rf "$TRAIN_OUTPUT_DIR"
mkdir -p "$TRAIN_OUTPUT_DIR"

cd "$GS_REPO"

run_timed "background_${SCENE_NAME}" 2dgs_train \
  env CUDA_VISIBLE_DEVICES=$GPU_ID python train.py \
    -s "$SCENE_DIR" \
    -m "$TRAIN_OUTPUT_DIR" \
    --images "$IMAGE_FOLDER" \
    --iterations "$MAX_STEPS" \
    --position_lr_max_steps "$MAX_STEPS" \
    --save_iterations "$MAX_STEPS" \
    --test_iterations "${TEST_ITERATIONS[@]}" \
    --port "$PORT" \
    --depth_ratio 0

log_metrics_to_wandb "background_${SCENE_NAME}" --profile twodgs "$TRAIN_OUTPUT_DIR"

run_timed "background_${SCENE_NAME}" 2dgs_render_export \
  env CUDA_VISIBLE_DEVICES=$GPU_ID python render.py \
    -s "$SCENE_DIR" \
    -m "$TRAIN_OUTPUT_DIR" \
    --images "$IMAGE_FOLDER" \
    --iteration "$MAX_STEPS" \
    --unbounded \
    --mesh_res 1024 \
    --depth_ratio 0
