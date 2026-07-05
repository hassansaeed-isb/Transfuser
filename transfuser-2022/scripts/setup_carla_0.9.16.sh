#!/usr/bin/env bash
# Download and install CARLA 0.9.16 + additional maps into ~/workspace (or CARLA_INSTALL_DIR).
set -euo pipefail

INSTALL_DIR="${1:-${HOME}/workspace/CARLA_0.9.16}"
CARLA_URL="${CARLA_URL:-https://tiny.carla.org/carla-0-9-16-linux}"
MAPS_URL="${MAPS_URL:-https://tiny.carla.org/additional-maps-0-9-16-linux}"

echo "Installing CARLA 0.9.16 into: ${INSTALL_DIR}"

mkdir -p "$(dirname "${INSTALL_DIR}")"
cd "$(dirname "${INSTALL_DIR}")"

if [[ ! -f "CARLA_0.9.16.tar.gz" ]]; then
  echo "Downloading CARLA_0.9.16.tar.gz ..."
  wget -O CARLA_0.9.16.tar.gz "${CARLA_URL}"
fi

if [[ ! -d "${INSTALL_DIR}" ]]; then
  echo "Extracting CARLA ..."
  tar -xzf CARLA_0.9.16.tar.gz
fi

if [[ ! -f "AdditionalMaps_0.9.16.tar.gz" ]]; then
  echo "Downloading AdditionalMaps_0.9.16.tar.gz ..."
  wget -O AdditionalMaps_0.9.16.tar.gz "${MAPS_URL}"
fi

echo "Importing additional maps ..."
mkdir -p "${INSTALL_DIR}/Import"
tar -xzf AdditionalMaps_0.9.16.tar.gz -C "${INSTALL_DIR}/Import"
cd "${INSTALL_DIR}"
chmod +x ImportAssets.sh CarlaUE4.sh
./ImportAssets.sh

echo "CARLA 0.9.16 is ready at: ${INSTALL_DIR}"
echo "Start it with: ${INSTALL_DIR}/CarlaUE4.sh -RenderOffScreen -quality-level=Low"
