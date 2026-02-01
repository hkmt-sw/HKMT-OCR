#!/usr/bin/env python3
"""
LightOnOCR-2 Hungarian Fine-tuning Script
Optimized for H100 GPU (80GB VRAM)
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
from peft import LoraConfig, get_peft_model, TaskType

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    model_id: str = "lightonai/LightOnOCR-2-1B-base"
    output_dir: str = "./output"
    merged_dir: str = "./merged"

    # Data - smaller to avoid OOM
    num_images: int = 600
    img_width: int = 600
    img_height: int = 300
    font_sizes: tuple = (18, 20, 22, 24)
    augment_ratio: float = 0.5

    # LoRA
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05

    # Training
    batch_size: int = 1
    gradient_accumulation: int = 16
    num_epochs: int = 3
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    max_tokens: int = 256

    data_dir: str = "./data"
    font_dir: str = "./fonts"


# ============================================================================
# HUNGARIAN VOCABULARY
# ============================================================================

HUNGARIAN_WORDS = [
    "őr", "őriz", "ők", "ősz", "ősi", "őszinte", "őrült", "ő",
    "erő", "idő", "mező", "tető", "fő", "nő", "bő", "hő", "jő",
    "belső", "külső", "felső", "alsó", "utolsó", "első", "hátsó",
    "költő", "festő", "vezető", "börtön", "könyv", "között", "előtt",
    "Győr", "dőlt", "dől", "töröl", "pörög", "görög", "örök", "örül",
    "köszönöm", "töltő", "öntő", "öröm", "ötven", "önök", "öböl",
    "űr", "űrlap", "űrhajó", "gyűrű", "tűz", "fűz", "gyűjt", "gyűlés",
    "tűnik", "fűszer", "hűtő", "hűvös", "hűség", "hűtlen",
    "szürke", "szűk", "szűr", "sűrű", "bűvös", "működik", "műszer",
    "fűrész", "tűrés", "bűn", "fűt", "nyű", "csűr", "dűne",
    "Csatornadíj", "vízdíj", "díj", "tükörfúrógép", "árvíztűrő",
    "fizetendő", "összeg", "összesen", "bruttó", "nettó",
    "adószám", "azonosító", "határidő",
]


# ============================================================================
# FONT HANDLING
# ============================================================================

def download_fonts(font_dir: str) -> list:
    font_dir = Path(font_dir)
    font_dir.mkdir(parents=True, exist_ok=True)
    print("Downloading fonts...")
    os.system(f"wget -q 'https://github.com/liberationfonts/liberation-fonts/files/7261482/liberation-fonts-ttf-2.1.5.tar.gz' -O /tmp/liberation.tar.gz")
    os.system(f"tar -xzf /tmp/liberation.tar.gz -C /tmp/")
    os.system(f"cp /tmp/liberation-fonts-ttf-2.1.5/*.ttf {font_dir}/")
    os.system(f"wget -q 'https://github.com/dejavu-fonts/dejavu-fonts/releases/download/version_2_37/dejavu-fonts-ttf-2.37.zip' -O /tmp/dejavu.zip")
    os.system(f"unzip -q -o /tmp/dejavu.zip -d /tmp/")
    os.system(f"cp /tmp/dejavu-fonts-ttf-2.37/ttf/*.ttf {font_dir}/")
    return list(font_dir.glob("*.ttf"))


def test_font(font_path: str) -> bool:
    try:
        font = ImageFont.truetype(font_path, 28)
        img = Image.new("RGB", (100, 40), "white")
        ImageDraw.Draw(img).text((5, 5), "őűŐŰ", fill="black", font=font)
        return np.sum(np.array(img) < 100) > 80
    except:
        return False


def get_working_fonts(font_dir: str) -> list:
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
    lines = []
    lines.append(" ".join(random.sample(HUNGARIAN_WORDS, random.randint(5, 7))))
    lines.append(f"Összeg: {random.randint(1, 99)} {random.randint(100, 999):03d} Ft")
    lines.append(f"Adószám: {random.randint(10000000, 99999999)}-{random.randint(1, 2)}-{random.randint(10, 99)}")
    lines.append("őűŐŰ öüóéáíú - ŐŰÖÜÓÉÁÍÚ")
    lines.append(" ".join(random.sample(HUNGARIAN_WORDS, random.randint(4, 6))))
    return "\n".join(lines)


def render_text(text: str, font_path: str, config: Config) -> Image.Image:
    font_size = random.choice(config.font_sizes)
    font = ImageFont.truetype(font_path, font_size)
    bg_color = random.choice(["white", "#fafafa", "#f5f5f5"])
    img = Image.new("RGB", (config.img_width, config.img_height), bg_color)
    draw = ImageDraw.Draw(img)
    y = 25
    line_height = int(font_size * 1.4)
    for line in text.split("\n"):
        if y + line_height > config.img_height - 20:
            break
        draw.text((25, y), line, fill="black", font=font)
        y += line_height
    return img


def apply_augmentations(img: Image.Image) -> Image.Image:
    if random.random() < 0.3:
        arr = np.array(img).astype(np.float32)
        noise = np.random.normal(0, random.uniform(3, 6), arr.shape)
        img = Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))
    if random.random() < 0.4:
        img = img.rotate(random.uniform(-1.5, 1.5), fillcolor="white", expand=False)
    if random.random() < 0.2:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.6)))
    return img


def generate_dataset(config: Config, fonts: list) -> None:
    data_dir = Path(config.data_dir)
    img_dir = data_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    annotations = []

    print(f"\nGenerating {config.num_images} images ({config.img_width}x{config.img_height})...")
    for i in range(config.num_images):
        text = generate_text()
        font_name, font_path = random.choice(fonts)
        img = render_text(text, font_path, config)
        if random.random() < config.augment_ratio:
            img = apply_augmentations(img)
        img.save(img_dir / f"{i:05d}.png")
        annotations.append({"image": f"{i:05d}.png", "text": text})
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{config.num_images}")

    with open(data_dir / "annotations.jsonl", "w", encoding="utf-8") as f:
        for a in annotations:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
    print(f"✓ Generated {config.num_images} images")


# ============================================================================
# DATASET CLASS - with image_sizes support
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

        # Get image size
        width, height = img.size

        # Process image
        img_inputs = self.processor.image_processor(img, return_tensors="pt")

        # Process text
        txt_inputs = self.processor.tokenizer(
            item["text"],
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_tokens,
            truncation=True,
        )

        return {
            "pixel_values": img_inputs["pixel_values"].squeeze(0),
            "image_sizes": torch.tensor([[height, width]]),  # Add image sizes!
            "input_ids": txt_inputs["input_ids"].squeeze(0),
            "attention_mask": txt_inputs["attention_mask"].squeeze(0),
            "labels": txt_inputs["input_ids"].squeeze(0),
        }


# ============================================================================
# CUSTOM DATA COLLATOR
# ============================================================================

class VisionDataCollator:
    def __call__(self, features):
        batch = {
            "pixel_values": torch.stack([f["pixel_values"] for f in features]),
            "image_sizes": torch.cat([f["image_sizes"] for f in features], dim=0),
            "input_ids": torch.stack([f["input_ids"] for f in features]),
            "attention_mask": torch.stack([f["attention_mask"] for f in features]),
            "labels": torch.stack([f["labels"] for f in features]),
        }
        return batch


# ============================================================================
# TRAINING
# ============================================================================

def train(config: Config):
    print("=" * 60)
    print("LightOnOCR-2 Hungarian Fine-tuning")
    print("=" * 60)

    fonts = get_working_fonts(config.font_dir)
    if len(fonts) < 3:
        raise RuntimeError("Not enough fonts found!")

    data_dir = Path(config.data_dir)
    if not (data_dir / "annotations.jsonl").exists():
        generate_dataset(config, fonts)
    else:
        print(f"Using existing dataset in {config.data_dir}")

    torch.cuda.empty_cache()

    print(f"\nLoading model: {config.model_id}")
    model = AutoModelForImageTextToText.from_pretrained(
        config.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    processor = AutoProcessor.from_pretrained(config.model_id, trust_remote_code=True)

    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()

    # Find target modules
    target_modules = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if any(x in name for x in ["q_proj", "v_proj"]):
                short_name = name.split(".")[-1]
                if short_name not in target_modules:
                    target_modules.append(short_name)
    if not target_modules:
        target_modules = ["q_proj", "v_proj"]
    print(f"LoRA target modules: {target_modules}")

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = OCRDataset(
        data_dir / "annotations.jsonl",
        data_dir / "images",
        processor,
        config.max_tokens,
    )
    print(f"Dataset: {len(dataset)} images")

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        logging_steps=20,
        save_steps=200,
        save_total_limit=2,
        bf16=True,
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=2,
        gradient_checkpointing=True,
        optim="adamw_torch",
        max_grad_norm=1.0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=VisionDataCollator(),
    )

    print(f"\nTraining: {len(dataset)} images, {config.num_epochs} epochs")
    print(f"Batch: {config.batch_size} x {config.gradient_accumulation} = {config.batch_size * config.gradient_accumulation}")

    trainer.train()
    print("✓ Training complete!")

    print("\nSaving model...")
    model.save_pretrained(config.output_dir)

    print("Merging LoRA weights...")
    merged = model.merge_and_unload()
    merged.save_pretrained(config.merged_dir)
    processor.save_pretrained(config.merged_dir)
    print(f"✓ Saved to {config.merged_dir}")

    # Quick test
    print("\n" + "=" * 60)
    print("Quick test...")
    print("=" * 60)
    test_img = Image.open(data_dir / "images" / "00000.png")
    w, h = test_img.size
    inputs = processor.image_processor(test_img, return_tensors="pt")
    inputs = {k: v.to(merged.device) for k, v in inputs.items()}
    inputs["image_sizes"] = torch.tensor([[h, w]]).to(merged.device)
    inputs["input_ids"] = processor.tokenizer("", return_tensors="pt")["input_ids"].to(merged.device)

    with torch.no_grad():
        out = merged.generate(**inputs, max_new_tokens=200, do_sample=False)
    result = processor.tokenizer.decode(out[0], skip_special_tokens=True)
    print(f"Test result:\n{result[:400]}")

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    print(f"\nModel saved to: {config.merged_dir}")


def main():
    config = Config()
    train(config)


if __name__ == "__main__":
    main()
