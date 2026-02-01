# LightOnOCR-2 Hungarian Fine-tuning (H100)

## Quick Start

```bash
# 1. Clone or copy this folder to the GPU server
git clone https://github.com/YOUR_USER/HKMT-OCR.git
cd HKMT-OCR/scripts/finetune_h100

# 2. Run
chmod +x run.sh
./run.sh
```

That's it! The script will:
1. Install `uv` (if needed)
2. Create virtual environment
3. Install dependencies
4. Download fonts
5. Generate training data
6. Train the model
7. Save to `./merged`

## Manual Steps

If you prefer to run manually:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env

# Setup
uv venv --python 3.11
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
uv pip install transformers>=4.45.0 peft datasets accelerate pillow opencv-python-headless

# Train
uv run python train.py
```

## Configuration

Edit `train.py` Config class to adjust:

```python
@dataclass
class Config:
    num_images: int = 1500      # Training images
    img_width: int = 1000       # Image size
    img_height: int = 500
    lora_r: int = 32            # LoRA rank
    batch_size: int = 8         # Batch size
    num_epochs: int = 5         # Training epochs
```

## After Training

1. Download the model to your Mac:
```bash
scp -r user@gpu-server:/path/to/merged ~/Downloads/
```

2. Convert to MLX:
```bash
mlx_vlm convert --hf-path merged --mlx-path lighton-hun-mlx -q --q-bits 4
```

3. Use in your OCR:
```bash
python run_ocr.py document.pdf --engine lighton --model lighton-hun-mlx
```

## GPU Requirements

| GPU | VRAM | Batch Size | Time |
|-----|------|------------|------|
| H100 | 80GB | 8 | ~15 min |
| A100 | 40GB | 4 | ~25 min |
| A10 | 24GB | 2 | ~40 min |
| T4 | 15GB | 1 | Use v12 notebook |

## Providers

- **Lambda Labs**: ~$2/hr H100
- **RunPod**: ~$2.5/hr H100
- **Vast.ai**: ~$1.5/hr H100
- **AWS**: p5.xlarge ~$30/hr (overkill)
