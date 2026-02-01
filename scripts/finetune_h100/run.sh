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
    source $HOME/.local/bin/env 2>/dev/null || export PATH="$HOME/.local/bin:$PATH"
fi

echo ""
echo "Creating virtual environment..."
uv venv --python 3.11

echo ""
echo "Installing dependencies..."
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
uv pip install transformers>=4.45.0 peft datasets accelerate pillow opencv-python-headless numpy

echo ""
echo "Starting training..."
uv run python train.py

echo ""
echo "=============================================="
echo "Training complete!"
echo "=============================================="
echo ""
echo "Model saved in: ./merged"
echo ""
echo "To download to your Mac:"
echo "  scp -r user@this-server:$(pwd)/merged ~/Downloads/"
echo ""
echo "Then convert to MLX:"
echo "  mlx_vlm convert --hf-path merged --mlx-path lighton-hun-mlx -q --q-bits 4"
