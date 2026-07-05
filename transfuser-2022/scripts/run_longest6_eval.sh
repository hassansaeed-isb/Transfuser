#!/usr/bin/env bash
# Run full Longest6 evaluation (36 routes) with a chosen model directory.
#
# Usage:
#   ./scripts/run_longest6_eval.sh [MODEL_DIR] [CARLA_ROOT]
#
# Examples:
#   ./scripts/run_longest6_eval.sh model_ckpt/crossvit_fusion
#   ./scripts/run_longest6_eval.sh model_ckpt/official_transfuser /root/workspace/CARLA_0.9.16
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${1:-model_ckpt/crossvit_fusion}"
CARLA_ROOT_ARG="${2:-}"

if [[ -n "${CARLA_ROOT_ARG}" ]]; then
  export CARLA_ROOT="${CARLA_ROOT_ARG}"
fi

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env_longest6.sh"

if [[ "${MODEL_DIR}" != /* ]]; then
  MODEL_DIR="${WORK_DIR}/${MODEL_DIR}"
fi
export TEAM_CONFIG="${MODEL_DIR}"

if [[ ! -f "${MODEL_DIR}/args.txt" ]]; then
  echo "ERROR: args.txt not found in ${MODEL_DIR}"
  exit 1
fi

PTH_COUNT="$(find "${MODEL_DIR}" -maxdepth 1 -name '*.pth' | wc -l)"
if [[ "${PTH_COUNT}" -eq 0 ]]; then
  echo "ERROR: No .pth checkpoint found in ${MODEL_DIR}"
  echo "Copy your model.pth into that folder and retry."
  exit 1
fi

HOST="${CARLA_HOST:-localhost}"
PORT="${CARLA_PORT:-2000}"

echo "============================================================"
echo "TransFuser Longest6 evaluation"
echo "  WORK_DIR:   ${WORK_DIR}"
echo "  CARLA_ROOT: ${CARLA_ROOT}"
echo "  MODEL_DIR:  ${TEAM_CONFIG}"
echo "  RESULTS:    ${CHECKPOINT_ENDPOINT}"
echo "  CARLA:      ${HOST}:${PORT}"
echo "============================================================"

python3 "${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator_local.py" \
  --host="${HOST}" \
  --port="${PORT}" \
  --trafficManagerPort=8000 \
  --trafficManagerSeed=0 \
  --scenarios="${SCENARIOS}" \
  --routes="${ROUTES}" \
  --repetitions="${REPETITIONS}" \
  --track="${CHALLENGE_TRACK_CODENAME}" \
  --checkpoint="${CHECKPOINT_ENDPOINT}" \
  --agent="${TEAM_AGENT}" \
  --agent-config="${TEAM_CONFIG}" \
  --debug="${DEBUG_CHALLENGE}" \
  --resume="${RESUME}"

echo
echo "Evaluation finished. Printing metrics ..."
python3 "${SCRIPT_DIR}/print_metrics.py" --file "${CHECKPOINT_ENDPOINT}"
