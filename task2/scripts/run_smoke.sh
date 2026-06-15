#!/usr/bin/env bash
# End-to-end local smoke: verify data → train B/ABC_fair (tiny) → zero-shot eval on D.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
activate_venv

echo "========== 1/4 Verify datasets =========="
python scripts/verify_dataset.py

echo "========== 2/4 Smoke train env B =========="
bash scripts/train.sh B smoke

echo "========== 3/4 Smoke train env ABC_fair =========="
bash scripts/train.sh ABC_fair smoke

echo "========== 4/4 Zero-shot eval on D =========="
bash scripts/eval_zeroshot.sh B smoke
bash scripts/eval_zeroshot.sh ABC_fair smoke

echo ""
echo "Smoke test passed."
echo "  outputs/train_b/checkpoints/last"
echo "  outputs/train_abc_fair/checkpoints/last"
echo "  outputs/eval_B_on_D.json"
echo "  outputs/eval_ABC_FAIR_on_D.json"
