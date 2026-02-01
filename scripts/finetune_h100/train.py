#!/usr/bin/env python3
"""
LightOnOCR-2 Hungarian Fine-tuning Script
Optimized for H100 GPU (80GB VRAM)

Usage:
    uv run python train.py
"""

import json
import random
import os
from pathlib import Path
from dataclasses import dataclass

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model

# ============================================================================
# CONFIGURATION - H100 OPTIMIZED
# ============================================================================

@dataclass
class Config:
    # Model
    model_id: str = "lightonai/LightOnOCR-2-1B-base"

    # Output
    output_dir: str = "./output"
    merged_dir: str = "./merged"

    # Data generation
    num_images: int = 1500
    img_width: int = 1000
    img_height: int = 500
    font_sizes: tuple = (18, 20, 22, 24, 26, 28, 30)
    augment_ratio: float = 0.6

    # LoRA - Full power for H100
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    target_modules: tuple = ("q_proj", "k_proj", "v_proj", "o_proj")

    # Training - H100 optimized
    batch_size: int = 8
    gradient_accumulation: int = 2
    num_epochs: int = 5
    learning_rate: float = 5e-5
    warmup_ratio: float = 0.1
    max_tokens: int = 512

    # Paths
    data_dir: str = "./data"
    font_dir: str = "./fonts"


# ============================================================================
# HUNGARIAN VOCABULARY
# ============================================================================

HUNGARIAN_WORDS = [
    # ő karakterek
    "őr", "őriz", "ők", "ősz", "ősi", "őszinte", "őrült", "ő",
    "erő", "idő", "mező", "tető", "fő", "nő", "bő", "hő", "jő",
    "belső", "külső", "felső", "alsó", "utolsó", "első", "hátsó",
    "költő", "festő", "vezető", "börtön", "könyv", "között", "előtt",
    "Győr", "dőlt", "dől", "töröl", "pörög", "görög", "örök", "örül",
    "köszönöm", "töltő", "öntő", "öröm", "ötven", "önök", "öböl",
    # ű karakterek
    "űr", "űrlap", "űrhajó", "gyűrű", "tűz", "fűz", "gyűjt", "gyűlés",
    "tűnik", "fűszer", "hűtő", "hűvös", "hűség", "hűtlen",
    "szürke", "szűk", "szűr", "sűrű", "bűvös", "működik", "műszer",
    "fűrész", "tűrés", "bűn", "fűt", "nyű", "csűr", "dűne",
    "gyűlik", "tűző", "fűző", "sűrít", "szűkít", "hűsít",
    # Dokumentum szavak
    "Csatornadíj", "vízdíj", "díj", "tükörfúrógép", "árvíztűrő",
    "halványszürke", "fizetendő", "összeg", "összesen", "bruttó", "nettó",
    "adószám", "cégjegyzékszám", "azonosító", "határidő", "lejárat",
    "számla", "nyugta", "bizonylat", "kiállító", "vevő", "szállító",
    "egységár", "mennyiség", "áfa", "kedvezmény", "végösszeg",
    # Egyéb gyakori szavak ékezetekkel
    "él", "élet", "év", "én", "és", "után", "előtt", "között",
    "már", "más", "még", "míg", "így", "úgy", "új", "régi",
    "kérem", "köszönöm", "üdvözöljük", "ügyfélfogadás",
]

TEMPLATES = [
    "Fizetendő összeg: {amt} Ft",
    "Csatornadíj: {amt} Ft",
    "Vízdíj alapdíj: {amt} Ft",
    "Kedvezmény: {amt} Ft",
    "Bruttó összeg: {amt} Ft",
    "Nettó összeg: {amt} Ft",
    "ÁFA (27%): {amt} Ft",
    "Végösszeg: {amt} Ft",
    "Adószám: {tax}",
    "Cégjegyzékszám: {ceg}",
    "Határidő: {date}",
    "Kiállítás dátuma: {date}",
    "Azonosító: {id}",
    "Számla sorszám: {id}",
    "IBAN: HU{iban}",
]


# ============================================================================
# FONT HANDLING
# ============================================================================

def download_fonts(font_dir: str) -> list:
    """Download fonts from GitHub releases."""
    font_dir = Path(font_dir)
    font_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading fonts...")

    # Liberation fonts (Arial, Times, Courier compatible)
    os.system(f"wget -q 'https://github.com/liberationfonts/liberation-fonts/files/7261482/liberation-fonts-ttf-2.1.5.tar.gz' -O /tmp/liberation.tar.gz")
    os.system(f"tar -xzf /tmp/liberation.tar.gz -C /tmp/")
    os.system(f"cp /tmp/liberation-fonts-ttf-2.1.5/*.ttf {font_dir}/")

    # DejaVu fonts
    os.system(f"wget -q 'https://github.com/dejavu-fonts/dejavu-fonts/releases/download/version_2_37/dejavu-fonts-ttf-2.37.zip' -O /tmp/dejavu.zip")
    os.system(f"unzip -q -o /tmp/dejavu.zip -d /tmp/")
    os.system(f"cp /tmp/dejavu-fonts-ttf-2.37/ttf/*.ttf {font_dir}/")

    # GNU FreeFont
    os.system(f"wget -q 'https://ftp.gnu.org/gnu/freefont/freefont-ttf-20120503.zip' -O /tmp/freefont.zip")
    os.system(f"unzip -q -o /tmp/freefont.zip -d /tmp/")
    os.system(f"cp /tmp/freefont-20120503/*.ttf {font_dir}/")

    return list(font_dir.glob("*.ttf"))


def test_font(font_path: str, test_text: str = "őűŐŰ") -> bool:
    """Test if a font supports Hungarian characters."""
    try:
        font = ImageFont.truetype(font_path, 28)
        img = Image.new("RGB", (100, 40), "white")
        ImageDraw.Draw(img).text((5, 5), test_text, fill="black", font=font)
        return np.sum(np.array(img) < 100) > 80
    except:
        return False


def get_working_fonts(font_dir: str) -> list:
    """Get list of fonts that support Hungarian characters."""
    font_dir = Path(font_dir)

    if not font_dir.exists() or not list(font_dir.glob("*.ttf")):
        font_files = download_fonts(font_dir)
    else:
        font_files = list(font_dir.glob("*.ttf"))

    fonts = []
    for fp in sorted(font_files):
        if test_font(str(fp)):
            fonts.append((fp.stem, str(fp)))
            print(f"  ✓ {fp.stem}")

    print(f"\n{len(fonts)} fonts support Hungarian characters")
    return fonts


# ============================================================================
# DATA GENERATION
# ============================================================================

def generate_text() -> str:
    """Generate random Hungarian text with ő/ű characters."""
    lines = []

    # Random words line
    lines.append(" ".join(random.sample(HUNGARIAN_WORDS, random.randint(6, 10))))

    # Template lines
    for _ in range(random.randint(4, 6)):
        template = random.choice(TEMPLATES)
        text = template.format(
            amt=f"{random.randint(1, 999)} {random.randint(100, 999):03d}",
            tax=f"{random.randint(10000000, 99999999)}-{random.randint(1, 2)}-{random.randint(10, 99)}",
            ceg=f"{random.randint(1, 99):02d}-{random.randint(1, 99):02d}-{random.randint(100000, 999999)}",
            date=f"2025.{random.randint(1, 12):02d}.{random.randint(1, 28):02d}",
            id=f"SZ-{random.randint(100000, 999999)}",
            iban=f"{random.randint(10, 99)} {random.randint(1000, 9999)} {random.randint(1000, 9999)} {random.randint(1000, 9999)} {random.randint(1000, 9999)}",
        )
        lines.append(text)

    # Always include test line
    lines.append("öüóőúéáűí - ÖÜÓŐÚÉÁŰÍ")
    lines.append("Árvíztűrő tükörfúrógép - ÁRVÍZTŰRŐ TÜKÖRFÚRÓGÉP")

    # Another random words line
    lines.append(" ".join(random.sample(HUNGARIAN_WORDS, random.randint(5, 8))))

    return "\n".join(lines)


def render_text(text: str, font_path: str, config: Config) -> Image.Image:
    """Render text to image."""
    font_size = random.choice(config.font_sizes)
    font = ImageFont.truetype(font_path, font_size)

    # Random background
    bg_color = random.choice(["white", "#fafafa", "#f5f5f5", "#fffef0", "#f0f0f0"])
    img = Image.new("RGB", (config.img_width, config.img_height), bg_color)
    draw = ImageDraw.Draw(img)

    # Draw text
    y = 30
    line_height = int(font_size * 1.4)
    for line in text.split("\n"):
        if y + line_height > config.img_height - 20:
            break
        # Random text color (mostly black, sometimes dark gray)
        text_color = random.choice(["black", "black", "black", "#333333", "#222222"])
        draw.text((30, y), line, fill=text_color, font=font)
        y += line_height

    return img


def apply_augmentations(img: Image.Image) -> Image.Image:
    """Apply random augmentations to image."""
    # Gaussian noise
    if random.random() < 0.3:
        arr = np.array(img).astype(np.float32)
        noise = np.random.normal(0, random.uniform(3, 8), arr.shape)
        img = Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))

    # Salt & pepper noise
    if random.random() < 0.2:
        arr = np.array(img)
        amount = random.uniform(0.002, 0.008)
        salt = np.random.random(arr.shape[:2]) < amount / 2
        arr[salt] = 255
        pepper = np.random.random(arr.shape[:2]) < amount / 2
        arr[pepper] = 0
        img = Image.fromarray(arr)

    # Rotation
    if random.random() < 0.4:
        angle = random.uniform(-2.5, 2.5)
        img = img.rotate(angle, fillcolor="white", expand=False)

    # Blur
    if random.random() < 0.2:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.8)))

    # Brightness
    if random.random() < 0.3:
        factor = random.uniform(0.9, 1.1)
        arr = np.array(img).astype(np.float32)
        img = Image.fromarray(np.clip(arr * factor, 0, 255).astype(np.uint8))

    return img


def generate_dataset(config: Config, fonts: list) -> None:
    """Generate training dataset."""
    data_dir = Path(config.data_dir)
    img_dir = data_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    annotations = []

    print(f"\nGenerating {config.num_images} images ({config.img_width}x{config.img_height})...")

    for i in range(config.num_images):
        text = generate_text()
        font_name, font_path = random.choice(fonts)

        img = render_text(text, font_path, config)

        # Apply augmentations
        augmented = random.random() < config.augment_ratio
        if augmented:
            img = apply_augmentations(img)

        img.save(img_dir / f"{i:05d}.png")
        annotations.append({
            "image": f"{i:05d}.png",
            "text": text,
            "font": font_name,
            "augmented": augmented,
        })

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{config.num_images}")

    # Save annotations
    with open(data_dir / "annotations.jsonl", "w", encoding="utf-8") as f:
        for a in annotations:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")

    aug_count = sum(1 for a in annotations if a["augmented"])
    print(f"✓ Generated {config.num_images} images ({aug_count} augmented)")


# ============================================================================
# DATASET CLASS
# ============================================================================

class OCRDataset(TorchDataset):
    def __init__(self, jsonl_path: str, img_dir: str, processor, max_tokens: int):
        self.processor = processor
        self.img_dir = Path(img_dir)
        self.max_tokens = max_tokens

        with open(jsonl_path, encoding="utf-8") as f:
            self.data = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img = Image.open(self.img_dir / item["image"]).convert("RGB")

        img_in = self.processor.image_processor(img, return_tensors="pt")
        txt_in = self.processor.tokenizer(
            item["text"],
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_tokens,
            truncation=True,
        )

        return {
            "pixel_values": img_in["pixel_values"].squeeze(0),
            "input_ids": txt_in["input_ids"].squeeze(0),
            "attention_mask": txt_in["attention_mask"].squeeze(0),
            "labels": txt_in["input_ids"].squeeze(0),
        }


# ============================================================================
# TRAINING
# ============================================================================

def train(config: Config):
    """Main training function."""
    print("=" * 60)
    print("LightOnOCR-2 Hungarian Fine-tuning")
    print("=" * 60)

    # Get fonts
    fonts = get_working_fonts(config.font_dir)
    if len(fonts) < 3:
        raise RuntimeError("Not enough fonts found!")

    # Generate data
    data_dir = Path(config.data_dir)
    if not (data_dir / "annotations.jsonl").exists():
        generate_dataset(config, fonts)
    else:
        print(f"Using existing dataset in {config.data_dir}")

    # Load model
    print(f"\nLoading model: {config.model_id}")
    model = AutoModelForImageTextToText.from_pretrained(
        config.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained(config.model_id)

    # Apply LoRA
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=list(config.target_modules),
        lora_dropout=config.lora_dropout,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Dataset
    dataset = OCRDataset(
        data_dir / "annotations.jsonl",
        data_dir / "images",
        processor,
        config.max_tokens,
    )
    print(f"Dataset: {len(dataset)} images")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        bf16=True,
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=4,
    )

    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    print(f"\nTraining: {len(dataset)} images, {config.num_epochs} epochs, batch={config.batch_size}")
    trainer.train()
    print("✓ Training complete!")

    # Save
    print("\nSaving model...")
    model.save_pretrained(config.output_dir)

    # Merge and save
    print("Merging LoRA weights...")
    merged = model.merge_and_unload()
    merged.save_pretrained(config.merged_dir)
    processor.save_pretrained(config.merged_dir)
    print(f"✓ Saved to {config.merged_dir}")

    # Test
    print("\n" + "=" * 60)
    print("Testing...")
    print("=" * 60)

    for idx in [0, len(dataset) // 2, len(dataset) - 1]:
        img = Image.open(data_dir / "images" / f"{idx:05d}.png")
        inputs = processor.image_processor(img, return_tensors="pt")
        inputs = {k: v.to(merged.device) for k, v in inputs.items()}
        inputs["input_ids"] = processor.tokenizer("", return_tensors="pt")["input_ids"].to(merged.device)

        with torch.no_grad():
            out = merged.generate(**inputs, max_new_tokens=400, do_sample=False)

        result = processor.tokenizer.decode(out[0], skip_special_tokens=True)
        print(f"\n--- Image #{idx} ---")
        print(result[:500])

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    print(f"\nModel saved to: {config.merged_dir}")
    print("\nNext steps on Mac:")
    print(f"  scp -r user@server:{config.merged_dir} .")
    print(f"  mlx_vlm convert --hf-path {config.merged_dir} --mlx-path lighton-hun-mlx -q --q-bits 4")


def main():
    config = Config()
    train(config)


if __name__ == "__main__":
    main()
