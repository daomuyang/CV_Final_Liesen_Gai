#!/usr/bin/env bash
# Aliyun GPU server setup (CUDA + WandB offline).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK2_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${TASK2_ROOT}"

echo "=== Task2 server setup ==="
nvidia-smi || echo "WARN: nvidia-smi not available"

if command -v conda >/dev/null 2>&1; then
  echo "Using conda env from environment-server.yml"
  conda env update -f environment-server.yml --prune -y || conda env create -f environment-server.yml -y
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate cv_hw3_task2
else
  echo "Conda not found; falling back to venv + pip"
  python3 -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  pip install -U pip wheel
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  pip install -r requirements-server.txt
fi

export HF_HUB_OFFLINE=1
python scripts/verify_dataset.py

echo ""
echo "=== WandB (offline first, sync later) ==="
echo "  export WANDB_API_KEY=<your_key>   # once per machine"
echo "  wandb login                       # or use API key"
echo "  # after training:"
echo "  wandb sync wandb/offline-run-*"
echo ""
echo "=== Training commands ==="
echo "  bash scripts/train.sh A server"
echo "  bash scripts/train.sh B server"
echo "  bash scripts/train.sh ABC_fair server   # 80k steps (ABC mixed)"
echo "  bash scripts/eval_zeroshot.sh B server"
echo "  bash scripts/eval_zeroshot.sh ABC_fair server"
echo "  See docs/experiments.md and docs/calvin_sim_setup.md"
