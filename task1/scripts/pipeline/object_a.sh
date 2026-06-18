#!/usr/bin/env bash
set -euo pipefail

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "${PIPELINE_DIR}/../common.sh"
# shellcheck source=../timing.sh
source "${PIPELINE_DIR}/../timing.sh"
# shellcheck source=../wandb.sh
source "${PIPELINE_DIR}/../wandb.sh"

SCENE_DIR="${DATASET_ROOT}/object_A"
RAW_IMAGES_DIR="${SCENE_DIR}/images_raw"
FOREGROUND_DIR="${SCENE_DIR}/images"
SPARSE_DIR="${SCENE_DIR}/sparse"
COLMAP_DB="${SCENE_DIR}/database.db"
TRAIN_OUTPUT_DIR="${SCENE_DIR}/2dgs_output"
GS_REPO="${THIRD_PARTY_DIR}/2d-gaussian-splatting"

GPU_ID="${GPU_ID:-0}"
PORT="${PORT:-6010}"
MAX_STEPS="${MAX_STEPS:-10000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-2000}"
MASK_LOSS_WEIGHT=0.1
MESH_VOXEL_SIZE=0.006
MESH_SDF_TRUNC=0.024

build_test_iterations "$MAX_STEPS" "$EVAL_INTERVAL"

activate_conda_env
setup_rembg_cache

cd "$TASK1_ROOT"

run_timed object_A preprocess \
  python "${TOOLS_DIR}/preprocess_image.py" "$RAW_IMAGES_DIR" objectA

rm -f "$COLMAP_DB"
rm -rf "$SPARSE_DIR" "$TRAIN_OUTPUT_DIR"
mkdir -p "$SPARSE_DIR"

unset LD_LIBRARY_PATH

run_timed object_A colmap_feature_extractor \
  colmap feature_extractor \
    --database_path "$COLMAP_DB" \
    --image_path "$FOREGROUND_DIR" \
    --ImageReader.single_camera 1 \
    --ImageReader.camera_model PINHOLE \
    --FeatureExtraction.use_gpu 1 \
    --FeatureExtraction.gpu_index "$GPU_ID"

run_timed object_A colmap_exhaustive_matcher \
  colmap exhaustive_matcher \
    --database_path "$COLMAP_DB" \
    --FeatureMatching.use_gpu 1 \
    --FeatureMatching.gpu_index "$GPU_ID"

run_timed object_A colmap_mapper \
  colmap mapper \
    --database_path "$COLMAP_DB" \
    --image_path "$FOREGROUND_DIR" \
    --output_path "$SPARSE_DIR"

run_timed object_A colmap_model_analyzer \
  colmap model_analyzer \
    --path "${SPARSE_DIR}/0"

setup_cuda_toolchain
ensure_2dgs_alpha_patch

cd "$GS_REPO"

run_timed object_A 2dgs_train \
  env CUDA_VISIBLE_DEVICES=$GPU_ID python train.py \
    -s "$SCENE_DIR" \
    -m "$TRAIN_OUTPUT_DIR" \
    --iterations "$MAX_STEPS" \
    --position_lr_max_steps "$MAX_STEPS" \
    --save_iterations "$MAX_STEPS" \
    --test_iterations "${TEST_ITERATIONS[@]}" \
    --port "$PORT" \
    --lambda_mask "$MASK_LOSS_WEIGHT"

log_metrics_to_wandb object_A --profile twodgs "$TRAIN_OUTPUT_DIR"

run_timed object_A 2dgs_render_export \
  env CUDA_VISIBLE_DEVICES=$GPU_ID python render.py \
    -s "$SCENE_DIR" \
    -m "$TRAIN_OUTPUT_DIR" \
    --iteration "$MAX_STEPS" \
    --num_cluster 1 \
    --voxel_size "$MESH_VOXEL_SIZE" \
    --sdf_trunc "$MESH_SDF_TRUNC"
