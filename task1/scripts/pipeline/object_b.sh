#!/usr/bin/env bash
set -euo pipefail

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "${PIPELINE_DIR}/../common.sh"
# shellcheck source=../timing.sh
source "${PIPELINE_DIR}/../timing.sh"
# shellcheck source=../wandb.sh
source "${PIPELINE_DIR}/../wandb.sh"

OBJECT_B_NAME="${OBJECT_B_NAME:-object_B}"
OBJECT_B_DIR="${DATASET_ROOT}/${OBJECT_B_NAME}"
THREESTUDIO_REPO="${THIRD_PARTY_DIR}/threestudio"

GPU_ID="${GPU_ID:-1}"
MAX_STEPS=12000
RANDOM_SEED=42
GUIDANCE_SCALE=75.0
SPARSITY_WEIGHT=10.0

PROMPT="a zoomed out DSLR product photo of a single front-facing puppy plush toy, one round head, one short muzzle, two long floppy ears, one compact solid bean-shaped body, two small front paws visible, rear legs hidden, soft light brown fabric, full object, centered, clean silhouette, opaque solid shape, plain white background"
NEGATIVE_PROMPT="multiple dogs, duplicate head, second face, face on back, extra muzzle, extra ears, extra paws, many legs, side view, hollow body, holes, transparent body, missing torso, dust, fur cloud, floating debris, distorted anatomy, broken shape, cropped, blurry, cartoon, text, watermark"

NEGATIVE_PROMPT_ARGS=()
if [ -n "$NEGATIVE_PROMPT" ]; then
  NEGATIVE_PROMPT_ARGS=(system.prompt_processor.negative_prompt="$NEGATIVE_PROMPT")
fi

TRAIN_ARGS=(
  --config configs/dreamfusion-sd.yaml
  --train
  --gpu 0
  seed=$RANDOM_SEED
  exp_root_dir="${DATASET_ROOT}/${OBJECT_B_NAME}"
  name=.
  tag=.
  use_timestamp=false
  trainer.max_steps=$MAX_STEPS
  system.prompt_processor.pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5"
  system.guidance.pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5"
  system.prompt_processor.prompt="$PROMPT"
  "${NEGATIVE_PROMPT_ARGS[@]}"
  system.guidance.guidance_scale=$GUIDANCE_SCALE
  system.loss.lambda_sparsity=$SPARSITY_WEIGHT
)

EXPORT_ARGS=(
  --config "${OBJECT_B_DIR}/configs/parsed.yaml"
  --export
  --gpu 0
  resume="${OBJECT_B_DIR}/ckpts/last.ckpt"
)

activate_conda_env
setup_cuda_toolchain
setup_hf_offline

mkdir -p "$OBJECT_B_DIR"
rm -rf \
  "${OBJECT_B_DIR}/ckpts" \
  "${OBJECT_B_DIR}/configs" \
  "${OBJECT_B_DIR}/csv_logs" \
  "${OBJECT_B_DIR}/tb_logs" \
  "${OBJECT_B_DIR}/save"

cd "$THREESTUDIO_REPO"

run_timed "$OBJECT_B_NAME" threestudio_train \
  env CUDA_VISIBLE_DEVICES=$GPU_ID python launch.py "${TRAIN_ARGS[@]}"

log_metrics_to_wandb "$OBJECT_B_NAME" --profile dreamfusion "${OBJECT_B_DIR}/csv_logs"

run_timed "$OBJECT_B_NAME" threestudio_export \
  env CUDA_VISIBLE_DEVICES=$GPU_ID python launch.py "${EXPORT_ARGS[@]}"
