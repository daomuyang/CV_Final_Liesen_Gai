#!/usr/bin/env bash

# Shared paths and environment helpers for task1 pipelines.

TASK1_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export TASK1_ROOT
export PROJECT_ROOT="${TASK1_ROOT}"

THIRD_PARTY_DIR="${TASK1_ROOT}/third_party"
DATASET_ROOT="${TASK1_ROOT}/submission_assets"
TOOLS_DIR="${TASK1_ROOT}/scripts/tools"
PATCHES_DIR="${TASK1_ROOT}/patches"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-task1}"

activate_conda_env() {
  # shellcheck disable=SC1091
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate "$CONDA_ENV_NAME"
}

setup_cuda_toolchain() {
  export CUDA_HOME=$CONDA_PREFIX
  export CUDA_PATH=$CONDA_PREFIX
  export CUDACXX=$CONDA_PREFIX/bin/nvcc
  export PATH=$CUDA_HOME/bin:$PATH
  export LD_LIBRARY_PATH=$CUDA_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
  export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
  export TCNN_CUDA_ARCHITECTURES="${TCNN_CUDA_ARCHITECTURES:-86}"
}

setup_rembg_cache() {
  export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}"
  export U2NET_HOME="${U2NET_HOME:-${TASK1_ROOT}/.cache/rembg}"
  mkdir -p "$NUMBA_CACHE_DIR" "$U2NET_HOME"
}

setup_hf_offline() {
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export DIFFUSERS_OFFLINE=1
}

build_test_iterations() {
  local max_steps="$1"
  local eval_interval="$2"
  local step
  TEST_ITERATIONS=()
  for step in $(seq "$eval_interval" "$eval_interval" "$max_steps"); do
    TEST_ITERATIONS+=("$step")
  done
}

ensure_2dgs_alpha_patch() {
  local gs_repo="${THIRD_PARTY_DIR}/2d-gaussian-splatting"
  local patch_file="${PATCHES_DIR}/2d-gaussian-splatting-alpha-mask.patch"

  if grep -q "lambda_mask" "${gs_repo}/arguments/__init__.py"; then
    return
  fi
  git -C "$gs_repo" apply "$patch_file"
}
