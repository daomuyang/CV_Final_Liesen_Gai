#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ROOT_DIR}/environment.yml"
ENV_NAME="${CONDA_ENV_NAME:-cvpj1}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Install Miniconda first." >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Conda env already exists: $ENV_NAME"
else
  conda env create -f "$ENV_FILE" -n "$ENV_NAME"
fi

conda activate "$ENV_NAME"

# shellcheck source=common.sh
source "${ROOT_DIR}/scripts/common.sh"
ensure_2dgs_alpha_patch

echo "Task1 server environment ready: $ENV_NAME"
echo "Third-party repos are under: ${ROOT_DIR}/third_party/"
echo "  - 2d-gaussian-splatting (alpha-mask patch applied if needed)"
echo "  - Magic123"
echo "  - threestudio"
echo "  - colmap"
