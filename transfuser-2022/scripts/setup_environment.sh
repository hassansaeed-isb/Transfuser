#!/usr/bin/env bash
# Create a Python environment for Longest6 evaluation on RTX 50-series GPUs (no mmcv).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${1:-${WORK_DIR}/.venv_eval}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "Work directory: ${WORK_DIR}"
echo "Virtual env:    ${VENV_DIR}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: ${PYTHON_BIN} not found. Install Python 3.10+ first."
  exit 1
fi

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip wheel setuptools

echo "Installing PyTorch (CUDA 12.x build for RTX 50-series) ..."
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

echo "Installing evaluation dependencies ..."
python -m pip install -r "${WORK_DIR}/requirements_eval.txt"

CARLA_ROOT="${CARLA_ROOT:-${HOME}/workspace/CARLA_0.9.16}"
if [[ -d "${CARLA_ROOT}/PythonAPI/carla/dist" ]]; then
  echo "Installing CARLA Python API from ${CARLA_ROOT} ..."
  python -m pip install "${CARLA_ROOT}"/PythonAPI/carla/dist/carla-*.whl
else
  echo "WARNING: CARLA wheel not found at ${CARLA_ROOT}/PythonAPI/carla/dist"
  echo "Install CARLA first, then run:"
  echo "  source ${VENV_DIR}/bin/activate"
  echo "  pip install \${CARLA_ROOT}/PythonAPI/carla/dist/carla-*.whl"
fi

echo
echo "Verifying GPU + imports ..."
python "${SCRIPT_DIR}/verify_environment.py" --carla-root "${CARLA_ROOT}" || true

echo
echo "Environment ready. Activate with:"
echo "  source ${VENV_DIR}/bin/activate"
