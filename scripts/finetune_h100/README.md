# LightOnOCR-2 Hungarian Fine-tuning (H100)

Based on the [official LightOnOCR fine-tuning notebook](https://colab.research.google.com/drive/1WjbsFJZ4vOAAlKtcCauFLn_evo5UBRNa).

## Quick Start

```bash
# 1. Clone or copy this folder to the GPU server
git clone https://github.com/hkmt-sw/HKMT-OCR.git
cd HKMT-OCR/scripts/finetune_h100

# 2. Run
chmod +x run.sh
./run.sh
```

That's it! The script will:
1. Install `uv` (if needed)
2. Create virtual environment
3. Install dependencies (transformers 5.0.0)
4. Download fonts
5. Generate training data with Hungarian text
6. Fine-tune vision encoder on Hungarian characters (ő, ű)
7. Save to `./merged`

## Manual Steps

If you prefer to run manually:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env

# Setup
uv venv
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
uv pip install transformers==5.0.0 huggingface-hub>=1.3.1 accelerate pillow numpy

# Train
python train.py
```

## Configuration

Edit `train.py` Config class to adjust:

```python
@dataclass
class Config:
    num_images: int = 800       # Training images
    img_width: int = 700        # Image size
    img_height: int = 350
    batch_size: int = 4         # Batch size (H100: 4-8)
    gradient_accumulation: int = 4  # Effective batch = 16
    num_epochs: int = 2         # Training epochs
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
