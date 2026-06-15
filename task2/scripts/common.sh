#!/usr/bin/env bash
# Shared paths and environment for task2 (local + Aliyun server).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK2_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${TASK2_ROOT}"

export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# Reproducibility (override: SEED=123 bash scripts/train.sh A server)
export SEED="${SEED:-42}"
export CUDNN_DETERMINISTIC="${CUDNN_DETERMINISTIC:-true}"

DATA_ROOT="${TASK2_ROOT}/data"
export DATA_ROOT

dataset_path() {
  local split="$1"
  echo "${DATA_ROOT}/calvin_env_${split}"
}

dataset_repo_id() {
  local split="$1"
  echo "local/calvin_env_${split}"
}

# Map CLI experiment name -> data_split | run_tag | server_steps
resolve_experiment() {
  local name="$1"
  case "${name}" in
    A|a)   echo "A|a|80000" ;;
    B|b)   echo "B|b|80000" ;;
    C|c)   echo "C|c|80000" ;;
    ABC|abc|ABC_fair|abc_fair|ABC-fair|abc-fair) echo "ABC|abc_fair|80000" ;;
    *)     echo "UNKNOWN" ;;
  esac
}

output_dir_for_tag() {
  local tag="$1"
  echo "${TASK2_ROOT}/outputs/train_${tag}"
}

detect_device() {
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    echo "cuda"
  elif [[ "$(uname -s)" == "Darwin" ]] && python -c "import torch; print(torch.backends.mps.is_available())" 2>/dev/null | grep -q True; then
    echo "mps"
  else
    echo "cpu"
  fi
}

detect_video_backend() {
  local device="$1"
  if [[ "${device}" == "cuda" ]] && python -c "import importlib.util; print(bool(importlib.util.find_spec('torchcodec')))" 2>/dev/null | grep -q True; then
    echo "torchcodec"
  else
    echo "pyav"
  fi
}

activate_venv() {
  if [[ -f "${TASK2_ROOT}/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${TASK2_ROOT}/.venv/bin/activate"
    return
  fi

  if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    if conda env list | awk '{print $1}' | grep -qx "cv_hw3_task2"; then
      conda activate cv_hw3_task2
      return
    fi
  fi

  echo "Python env not found." >&2
  echo "  macOS/local: bash scripts/setup_local.sh && source .venv/bin/activate" >&2
  echo "  GPU server:  bash scripts/setup_server.sh" >&2
  exit 1
}

latest_checkpoint() {
  local run_dir="$1"
  local ckpt="${run_dir}/checkpoints/last"
  if [[ ! -d "${ckpt}/pretrained_model" ]]; then
    echo "No checkpoint under ${run_dir}/checkpoints/last" >&2
    exit 1
  fi
  echo "${ckpt}"
}
