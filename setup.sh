
#!/usr/bin/env bash
set -euo pipefail

# Navigate to the project directory
cd "$(dirname "$0")"

echo "================================"
echo "    TileLens AI - Setup"
echo "================================"

# Check Python
if ! command -v python3 &>/dev/null; then
    echo "Error: Python 3 is not installed."
    exit 1
fi

# Create virtual environment if missing
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

PYTHON=".venv/bin/python"

echo "Upgrading pip..."
"$PYTHON" -m pip install --upgrade pip

echo "Installing CPU-compatible PyTorch..."
"$PYTHON" -m pip install \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

echo "Installing dashboard dependencies..."
"$PYTHON" -m pip install \
    -r requirements-dashboard.txt

echo "Checking dependencies..."
"$PYTHON" -m pip check

echo "================================"
echo "Setup completed!"
echo "Run the dashboard with ./run.sh"
echo "================================"
