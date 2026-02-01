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
echo "Creating virtual environment..."
uv venv .venv
source .venv/bin/activate

echo ""
echo "Installing dependencies..."
# torch 2.2+ supports Python 3.12
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
uv pip install transformers==4.44.2
uv pip install peft==0.11.1
uv pip install datasets accelerate pillow opencv-python-headless numpy

echo ""
echo "Checking versions..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"

echo ""
echo "Starting training..."
python train.py

echo ""
echo "=============================================="
echo "Training complete!"
echo "=============================================="
echo ""
echo "Model saved in: ./merged"
