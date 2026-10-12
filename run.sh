
#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "================================"
echo "      Starting TileLens AI"
echo "================================"

# Check virtual environment
if [ ! -f ".venv/bin/python" ]; then
    echo "Virtual environment not found."
    echo "Please run ./setup.sh first."
    exit 1
fi

# Verify trained model checkpoints
CLASSIFIER="models/tile_classifier/mobilenetv3_small_tile_v2.pt"
DETECTOR="models/tile_damage_detector/yolo11n_tile_damage_v1_best.pt"

if [ ! -f "$CLASSIFIER" ]; then
    echo "Missing MobileNetV3 checkpoint:"
    echo "$CLASSIFIER"
    exit 1
fi

if [ ! -f "$DETECTOR" ]; then
    echo "Missing YOLO11n checkpoint:"
    echo "$DETECTOR"
    exit 1
fi

echo "Model checkpoints found."
echo "Starting Streamlit..."
echo "Open http://localhost:8501"

exec .venv/bin/python -m streamlit run dashboard/app.py
