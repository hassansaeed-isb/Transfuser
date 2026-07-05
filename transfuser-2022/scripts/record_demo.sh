#!/usr/bin/env bash
# Helper for recording a demonstration video with ffmpeg while evaluation runs.
# Run this in a second terminal AFTER CARLA is already running.
#
# Usage:
#   ./scripts/record_demo.sh [output_file.mp4]
set -euo pipefail

OUTPUT="${1:-demo_longest6.mp4}"
DISPLAY="${DISPLAY:-:0}"

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "Installing ffmpeg ..."
  sudo apt update
  sudo apt install -y ffmpeg
fi

echo "Recording desktop to ${OUTPUT}"
echo "Press Ctrl+C to stop recording."

ffmpeg -y -video_size 1920x1080 -framerate 30 -f x11grab -i "${DISPLAY}" -c:v libx264 -preset veryfast "${OUTPUT}"
