#!/usr/bin/env bash
# Zero-shot offline eval on calvin_env_D (homework step 4).
#
# Usage:
#   SEED=42 bash scripts/eval_zeroshot.sh <A|B|C|ABC|ABC_fair> [smoke|server]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

EXP="${1:?Usage: SEED=42 bash scripts/eval_zeroshot.sh <A|B|C|ABC|ABC_fair> [smoke|server]}"
MODE="${2:-server}"

bash "${SCRIPT_DIR}/eval_offline.sh" "${EXP}" D "${MODE}"
