#!/usr/bin/env bash
# Compare multiple eval reports (e.g. A vs ABC) — plots only, no retraining.
#
# Usage:
#   bash scripts/eval_compare.sh A ABC
#
# Prerequisite: run eval_report.sh for each experiment first.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

if [[ $# -lt 2 ]]; then
  echo "Usage: bash scripts/eval_compare.sh <exp1> <exp2> [exp3 ...]" >&2
  exit 1
fi

activate_venv

REPORT_ARGS=()
LABEL_ARGS=()
for EXP in "$@"; do
  RESOLVED="$(resolve_experiment "${EXP}")"
  if [[ "${RESOLVED}" == "UNKNOWN" ]]; then
    echo "Unknown experiment: ${EXP}" >&2
    exit 1
  fi
  IFS='|' read -r _DATA_SPLIT RUN_TAG _STEPS <<< "${RESOLVED}"
  LABEL="$(echo "${RUN_TAG}" | tr '[:lower:]' '[:upper:]')"
  REPORT="${TASK2_ROOT}/outputs/report_${LABEL}.json"
  if [[ ! -f "${REPORT}" ]]; then
    echo "Missing ${REPORT}. Run: bash scripts/eval_report.sh ${EXP} server" >&2
    exit 1
  fi
  REPORT_ARGS+=(--report "${REPORT}")
  LABEL_ARGS+=(--label "${LABEL}")
done

python scripts/plot_eval_metrics.py \
  "${REPORT_ARGS[@]}" \
  "${LABEL_ARGS[@]}" \
  --output-dir "${TASK2_ROOT}/outputs"

echo "Wrote ${TASK2_ROOT}/outputs/gap_comparison.png (and per-run chunk_l1_*.png)"
