#!/usr/bin/env bash
# Quick smoke test: evaluate a single Longest6 route before running all 36 routes.
#
# Usage:
#   ./scripts/run_single_route_eval.sh [MODEL_DIR] [ROUTE_INDEX]
#
# ROUTE_INDEX is 0-35 (default: 0)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${1:-model_ckpt/crossvit_fusion}"
ROUTE_INDEX="${2:-0}"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env_longest6.sh"

if [[ "${MODEL_DIR}" != /* ]]; then
  MODEL_DIR="${WORK_DIR}/${MODEL_DIR}"
fi
export TEAM_CONFIG="${MODEL_DIR}"

ROUTE_XML="${WORK_DIR}/leaderboard/data/longest6/longest6_split/longest_weathers_${ROUTE_INDEX}.xml"
if [[ ! -f "${ROUTE_XML}" ]]; then
  echo "ERROR: Route file not found: ${ROUTE_XML}"
  exit 1
fi

export ROUTES="${ROUTE_XML}"
export CHECKPOINT_ENDPOINT="${WORK_DIR}/results/longest6_route${ROUTE_INDEX}_smoke.json"
export RESUME=0

HOST="${CARLA_HOST:-localhost}"
PORT="${CARLA_PORT:-2000}"

echo "Smoke test route ${ROUTE_INDEX}"
echo "  MODEL_DIR: ${TEAM_CONFIG}"
echo "  ROUTES:    ${ROUTES}"
echo "  RESULTS:   ${CHECKPOINT_ENDPOINT}"

python3 "${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator_local.py" \
  --host="${HOST}" \
  --port="${PORT}" \
  --trafficManagerPort=8000 \
  --trafficManagerSeed=0 \
  --scenarios="${SCENARIOS}" \
  --routes="${ROUTES}" \
  --repetitions=1 \
  --track="${CHALLENGE_TRACK_CODENAME}" \
  --checkpoint="${CHECKPOINT_ENDPOINT}" \
  --agent="${TEAM_AGENT}" \
  --agent-config="${TEAM_CONFIG}" \
  --debug=0 \
  --resume=0

python3 "${SCRIPT_DIR}/print_metrics.py" --file "${CHECKPOINT_ENDPOINT}"
