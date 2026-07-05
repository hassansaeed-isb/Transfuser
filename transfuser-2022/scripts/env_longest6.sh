#!/usr/bin/env bash
# Configure PYTHONPATH and CARLA paths for TransFuser Longest6 evaluation.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WORK_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export CARLA_ROOT="${CARLA_ROOT:-${HOME}/workspace/CARLA_0.9.16}"

export CARLA_SERVER="${CARLA_ROOT}/CarlaUE4.sh"
export SCENARIO_RUNNER_ROOT="${WORK_DIR}/scenario_runner"
export LEADERBOARD_ROOT="${WORK_DIR}/leaderboard"

export PYTHONPATH="${WORK_DIR}/team_code_transfuser"
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla:${SCENARIO_RUNNER_ROOT}:${LEADERBOARD_ROOT}:${PYTHONPATH}"

export SCENARIOS="${WORK_DIR}/leaderboard/data/longest6/eval_scenarios.json"
export ROUTES="${WORK_DIR}/leaderboard/data/longest6/longest6.xml"
export REPETITIONS="${REPETITIONS:-1}"
export CHALLENGE_TRACK_CODENAME=SENSORS
export DEBUG_CHALLENGE="${DEBUG_CHALLENGE:-0}"
export RESUME="${RESUME:-1}"
export DATAGEN=0

export TEAM_AGENT="${WORK_DIR}/team_code_transfuser/submission_agent.py"
export TEAM_CONFIG="${TEAM_CONFIG:-${WORK_DIR}/model_ckpt/crossvit_fusion}"
export CHECKPOINT_ENDPOINT="${CHECKPOINT_ENDPOINT:-${WORK_DIR}/results/longest6_results.json}"

mkdir -p "$(dirname "${CHECKPOINT_ENDPOINT}")" "${WORK_DIR}/results"
