#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -gt 1 ]; then
  echo "Usage: bash scripts/pipeline/object_c.sh [lambda_2d_3d]" >&2
  echo "Example: bash scripts/pipeline/object_c.sh 0.5" >&2
  exit 2
fi

LAMBDA_2D_3D="${1:-1.0}"

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "${PIPELINE_DIR}/../common.sh"
# shellcheck source=../timing.sh
source "${PIPELINE_DIR}/../timing.sh"
# shellcheck source=../wandb.sh
source "${PIPELINE_DIR}/../wandb.sh"

OBJECT_C_NAME="${OBJECT_C_NAME:-object_C}"
OBJECT_C_DIR="${DATASET_ROOT}/${OBJECT_C_NAME}"
MAGIC123_REPO="${THIRD_PARTY_DIR}/Magic123"
IMAGE_DIR="${OBJECT_C_DIR}/images"
INPUT_IMAGE="${INPUT_IMAGE:-}"

if [ -z "$INPUT_IMAGE" ]; then
  for candidate in "${IMAGE_DIR}/0001.jpg" "${IMAGE_DIR}/0001.png"; do
    if [ -f "$candidate" ]; then
      INPUT_IMAGE="$candidate"
      break
    fi
  done
fi

if [ -z "$INPUT_IMAGE" ]; then
  echo "Input image not found under: $IMAGE_DIR" >&2
  echo "Expected one of: 0001.jpg, 0001.png" >&2
  exit 1
fi

INPUT_STEM="$(basename "${INPUT_IMAGE%.*}")"
FOREGROUND_IMAGE="${IMAGE_DIR}/${INPUT_STEM}_foreground.png"
RGBA_IMAGE="${IMAGE_DIR}/rgba.png"
OUTPUT_ROOT="${OBJECT_C_DIR}/magic123_output"
COARSE_WORKSPACE="${OUTPUT_ROOT}/object_C_coarse"
FINE_WORKSPACE="${OUTPUT_ROOT}/object_C_fine"
COARSE_INIT_CKPT="${COARSE_WORKSPACE}/checkpoints/object_C_coarse_latest.pth"

GPU_ID="${GPU_ID:-1}"
MAX_STEPS=5000
MASK_LOSS_WEIGHT=3
COARSE_SD_WEIGHT=$(awk "BEGIN { print 1.0 * ${LAMBDA_2D_3D} }")
FINE_SD_WEIGHT=$(awk "BEGIN { print 0.001 * ${LAMBDA_2D_3D} }")

TEXT_PROMPT="Front-angled view of a Yamaha dreadnought acoustic guitar, solid natural spruce top, dark rosewood sides and back, black pickguard with red tortoiseshell pattern, dark brown bridge and saddle, six metal strings, chrome tuning pegs, dark fretboard with dot inlays, polished wood texture, no scratches or damage, pure white background, soft even lighting, photorealistic high-detail 3D model"

activate_conda_env
setup_rembg_cache
setup_cuda_toolchain

python - "$INPUT_IMAGE" <<'PY'
import sys
from PIL import Image

image_path = sys.argv[1]
try:
    with Image.open(image_path) as image:
        image.verify()
except Exception as exc:
    raise SystemExit(
        f"Input image is not readable by PIL: {image_path}\n"
        "Please export it as a real JPEG or PNG before running Object C. "
        "A HEIC/HEIF file renamed to .jpg or .png will fail here.\n"
        f"Original error: {exc}"
    )
PY

rm -rf "$COARSE_WORKSPACE" "$FINE_WORKSPACE"
mkdir -p "$OUTPUT_ROOT"

cd "$TASK1_ROOT"

run_timed "$OBJECT_C_NAME" preprocess_foreground \
  python "${TOOLS_DIR}/preprocess_image.py" "$INPUT_IMAGE" objectC

cd "$MAGIC123_REPO"

run_timed "$OBJECT_C_NAME" magic123_preprocess_rgba_depth \
  env CUDA_VISIBLE_DEVICES=$GPU_ID python preprocess_image.py \
    --path "$FOREGROUND_IMAGE"

run_timed "$OBJECT_C_NAME" magic123_coarse \
  env CUDA_VISIBLE_DEVICES=$GPU_ID python main.py -O \
    --text "$TEXT_PROMPT" \
    --sd_version 1.5 \
    --image "$RGBA_IMAGE" \
    --workspace "$COARSE_WORKSPACE" \
    --optim adam \
    --iters "$MAX_STEPS" \
    --guidance SD zero123 \
    --lambda_guidance "$COARSE_SD_WEIGHT" 40 \
    --guidance_scale 100 5 \
    --lambda_rgb 8 \
    --lambda_mask "$MASK_LOSS_WEIGHT" \
    --latent_iter_ratio 0 \
    --normal_iter_ratio 0.2 \
    --t_range 0.2 0.6 \
    --bg_radius -1 \
    --save_mesh

LATEST_COARSE_CKPT=$(ls -t "${COARSE_WORKSPACE}"/checkpoints/*.pth | head -n 1)
cp "$LATEST_COARSE_CKPT" "$COARSE_INIT_CKPT"

run_timed "$OBJECT_C_NAME" magic123_fine_export \
  env CUDA_VISIBLE_DEVICES=$GPU_ID python main.py -O \
    --text "$TEXT_PROMPT" \
    --sd_version 1.5 \
    --image "$RGBA_IMAGE" \
    --workspace "$FINE_WORKSPACE" \
    --dmtet \
    --init_ckpt "$COARSE_INIT_CKPT" \
    --iters "$MAX_STEPS" \
    --optim adam \
    --known_view_interval 2 \
    --latent_iter_ratio 0 \
    --guidance SD zero123 \
    --lambda_guidance "$FINE_SD_WEIGHT" 0.01 \
    --guidance_scale 100 5 \
    --lambda_rgb 8 \
    --lambda_mask "$MASK_LOSS_WEIGHT" \
    --bg_radius -1 \
    --save_mesh

log_metrics_to_wandb "${OBJECT_C_NAME}_coarse" --profile magic123 "${COARSE_WORKSPACE}/run"
log_metrics_to_wandb "${OBJECT_C_NAME}_fine" --profile magic123 "${FINE_WORKSPACE}/run"

cd "$TASK1_ROOT"
run_timed "$OBJECT_C_NAME" magic123_plot_ema \
  python "${TOOLS_DIR}/plot_magic123_loss.py"
