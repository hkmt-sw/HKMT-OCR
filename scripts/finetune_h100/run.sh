#!/bin/bash
# LightOnOCR-2 Hungarian Fine-tuning
# Run on H100 GPU

set -e

echo "=============================================="
echo "LightOnOCR-2 Hungarian Fine-tuning Setup"
echo "=============================================="

# Check for NVIDIA GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Is this a GPU machine?"
    exit 1
fi

echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo ""
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo ""
echo "Creating virtual environment with Python 3.11..."
uv venv --python 3.11 .venv
source .venv/bin/activate

echo ""
echo "Installing dependencies..."
uv pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
uv pip install transformers==4.44.2
uv pip install peft==0.11.1
uv pip install datasets accelerate pillow opencv-python-headless numpy

echo ""
echo "Starting training..."
python train.py

echo ""
echo "=============================================="
echo "Training complete!"
echo "=============================================="
echo ""
echo "Model saved in: ./merged"
