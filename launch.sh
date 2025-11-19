#!/bin/bash

# Go to script directory (project root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check venv
if [ ! -d ".venv" ]; then
  echo ".venv not found. Please create it first."
  echo "Follow the instructions in README.md before running this script."
  exit 1
fi

# Activate venv
source .venv/bin/activate

# Run TensorRT camera script
python3 yolo12_cam_gst_trt.py

