#!/usr/bin/env bash
# Full eval report for one trained experiment (no retraining needed).
#
# Runs:
#   1. In-distribution eval on training env
#   2. Zero-shot eval on D
#   3. Summary JSON with generalization gap + chunk-step analysis
#   4. Plots for PDF report
#
# Usage:
#   bash scripts/eval_report.sh <A|B|C|ABC|ABC_fair> [smoke|server]
#
# Outputs:
#   outputs/eval_{LABEL}_on_{TRAIN}.json
#   outputs/eval_{LABEL}_on_D.json
#   outputs/report_{LABEL}.json
#   outputs/chunk_l1_{LABEL}.png
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
activate_venv

EXP="${1:?Usage: bash scripts/eval_report.sh <A|B|C|ABC|ABC_fair> [smoke|server]}"
MODE="${2:-server}"

RESOLVED="$(resolve_experiment "${EXP}")"
if [[ "${RESOLVED}" == "UNKNOWN" ]]; then
  echo "EXP must be A, B, C, ABC, or ABC_fair" >&2
  exit 1
fi

IFS='|' read -r DATA_SPLIT RUN_TAG _STEPS <<< "${RESOLVED}"
LABEL="$(echo "${RUN_TAG}" | tr '[:lower:]' '[:upper:]')"
TRAIN_UPPER="$(echo "${DATA_SPLIT}" | tr '[:lower:]' '[:upper:]')"

echo "========== Eval report: ${EXP} (train env=${DATA_SPLIT}, zero-shot=D) =========="

bash "${SCRIPT_DIR}/eval_offline.sh" "${EXP}" "${DATA_SPLIT}" "${MODE}"
bash "${SCRIPT_DIR}/eval_offline.sh" "${EXP}" D "${MODE}"

IN_DIST_JSON="${TASK2_ROOT}/outputs/eval_${LABEL}_on_${TRAIN_UPPER}.json"
ZERO_SHOT_JSON="${TASK2_ROOT}/outputs/eval_${LABEL}_on_D.json"
REPORT_JSON="${TASK2_ROOT}/outputs/report_${LABEL}.json"

python scripts/summarize_eval.py \
  --in-dist "${IN_DIST_JSON}" \
  --zero-shot "${ZERO_SHOT_JSON}" \
  --output "${REPORT_JSON}"

python scripts/plot_eval_metrics.py \
  --report "${REPORT_JSON}" \
  --label "${LABEL}" \
  --output-dir "${TASK2_ROOT}/outputs"

echo ""
echo "Report complete:"
echo "  ${IN_DIST_JSON}"
echo "  ${ZERO_SHOT_JSON}"
echo "  ${REPORT_JSON}"
echo "  ${TASK2_ROOT}/outputs/chunk_l1_${LABEL}.png"
