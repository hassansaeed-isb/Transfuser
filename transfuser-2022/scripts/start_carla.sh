#!/usr/bin/env bash
# Start CARLA 0.9.16 server for headless evaluation.
set -euo pipefail

CARLA_ROOT="${CARLA_ROOT:-${HOME}/workspace/CARLA_0.9.16}"
PORT="${CARLA_PORT:-2000}"
QUALITY="${CARLA_QUALITY:-Low}"

if [[ ! -x "${CARLA_ROOT}/CarlaUE4.sh" ]]; then
  echo "ERROR: CarlaUE4.sh not found at ${CARLA_ROOT}"
  echo "Set CARLA_ROOT or run scripts/setup_carla_0.9.16.sh first."
  exit 1
fi

cd "${CARLA_ROOT}"
echo "Starting CARLA on port ${PORT} (quality=${QUALITY}) ..."
exec ./CarlaUE4.sh -world-port="${PORT}" -RenderOffScreen -quality-level="${QUALITY}"
