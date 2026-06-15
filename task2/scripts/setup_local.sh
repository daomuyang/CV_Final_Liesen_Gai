#!/usr/bin/env bash
# macOS local environment (CPU/MPS smoke tests).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK2_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${TASK2_ROOT}"

PYTHON="${PYTHON:-python3}"
if ! command -v "${PYTHON}" >/dev/null 2>&1; then
  echo "python3 not found" >&2
  exit 1
fi

if [[ ! -d .venv ]]; then
  "${PYTHON}" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

pip install -U pip wheel
pip install -r requirements-local.txt

echo "Local env ready. Activate with: source .venv/bin/activate"
python -c "import lerobot; import torch; print('lerobot', lerobot.__version__, '| torch', torch.__version__)"
