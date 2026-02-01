#!/usr/bin/env python3
"""
LightOnOCR-2 Hungarian Fine-tuning Script
Optimized for H100 GPU (80GB VRAM)
Based on official LightOnOCR fine-tuning notebook
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
    LightOnOcrProcessor,
    LightOnOcrForConditionalGeneration,
    TrainingArguments,
    Trainer,
)

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    model_id: str = "lightonai/LightOnOCR-2-1B-base"
    output_dir: str = "./output"
    merged_dir: str = "./merged"

    # Data
    num_images: int = 800
    img_width: int = 700
    img_height: int = 350
    font_sizes: tuple = (18, 20, 22, 24, 26)
    augment_ratio: float = 0.5

    # Training
    batch_size: int = 4
    gradient_accumulation: int = 4
    num_epochs: int = 2
    learning_rate: float = 5e-5
    warmup_steps: int = 20
    max_length: int = 512
    longest_edge: int = 700

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
    lines.append(" ".join(random.sample(HUNGARIAN_WORDS, random.randint(5, 8))))
    lines.append(f"Összeg: {random.randint(1, 99)} {random.randint(100, 999):03d} Ft")
    lines.append(f"Adószám: {random.randint(10000000, 99999999)}-{random.randint(1, 2)}-{random.randint(10, 99)}")
    lines.append("őűŐŰ öüóéáíú - ŐŰÖÜÓÉÁÍÚ")
    lines.append(" ".join(random.sample(HUNGARIAN_WORDS, random.randint(4, 7))))
    return "\n".join(lines)


def render_text(text: str, font_path: str, config: Config) -> Image.Image:
    font_size = random.choice(config.font_sizes)
    font = ImageFont.truetype(font_path, font_size)
    bg_color = random.choice(["white", "#fafafa", "#f5f5f5"])
    img = Image.new("RGB", (config.img_width, config.img_height), bg_color)
    draw = ImageDraw.Draw(img)
    y = 25
    line_height = int(font_size * 1.5)
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
# DATASET CLASS - Following official notebook pattern
# ============================================================================

class OCRDataset(TorchDataset):
    def __init__(self, jsonl_path: str, img_dir: str):
        self.img_dir = Path(img_dir)
        with open(jsonl_path, encoding="utf-8") as f:
            self.data = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img = Image.open(self.img_dir / item["image"]).convert("RGB")
        return {"image": img, "text": item["text"]}


# ============================================================================
# DATA COLLATOR - Following official notebook pattern
# ============================================================================

def create_collate_fn(processor, config):
    """Create data collator following official LightOnOCR pattern."""

    # Assistant start pattern for masking: <|im_end|>\n<|im_start|>assistant\n
    ASSISTANT_START_PATTERN = [151645, 198, 151644, 77091, 198]

    def collate_fn(examples):
        batch_messages = []
        batch_images = []

        for example in examples:
            image = example["image"]
            text = example["text"].strip()
            batch_images.append(image)

            messages = [
                {"role": "user", "content": [{"type": "image"}]},
                {"role": "assistant", "content": [{"type": "text", "text": text}]},
            ]
            batch_messages.append(messages)

        # Apply chat template
        texts = [
            processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            for messages in batch_messages
        ]

        # Process with processor
        inputs = processor(
            text=texts,
            images=batch_images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.max_length,
            size={"longest_edge": config.longest_edge},
        )

        # Create labels with proper masking
        labels = inputs["input_ids"].clone()
        pad_token_id = processor.tokenizer.pad_token_id

        for i in range(len(labels)):
            full_ids = inputs["input_ids"][i].tolist()

            # Find where assistant content starts
            assistant_content_start = None
            for idx in range(len(full_ids) - len(ASSISTANT_START_PATTERN)):
                if full_ids[idx:idx + len(ASSISTANT_START_PATTERN)] == ASSISTANT_START_PATTERN:
                    assistant_content_start = idx + len(ASSISTANT_START_PATTERN)
                    break

            if assistant_content_start is None:
                labels[i, :] = -100
            else:
                # Mask everything before assistant response
                labels[i, :assistant_content_start] = -100
                # Mask padding
                labels[i, inputs["input_ids"][i] == pad_token_id] = -100

        inputs["labels"] = labels
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        return inputs

    return collate_fn


# ============================================================================
# TRAINING
# ============================================================================

def train(config: Config):
    print("=" * 60)
    print("LightOnOCR-2 Hungarian Fine-tuning")
    print("Using official LightOnOCR classes (transformers 5.0)")
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

    print(f"\nLoading processor: {config.model_id}")
    processor = LightOnOcrProcessor.from_pretrained(config.model_id)
    processor.tokenizer.padding_side = "left"

    print(f"Loading model: {config.model_id}")
    model = LightOnOcrForConditionalGeneration.from_pretrained(
        config.model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
    )

    # Freeze language model, train vision components
    print("\nFreezing language model, training vision encoder + projection...")
    for param in model.model.language_model.parameters():
        param.requires_grad = False

    # Count trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()

    dataset = OCRDataset(
        data_dir / "annotations.jsonl",
        data_dir / "images",
    )
    print(f"Dataset: {len(dataset)} images")

    # Split into train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=50,
        save_steps=200,
        save_total_limit=2,
        bf16=True,
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=2,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        lr_scheduler_type="linear",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    collate_fn = create_collate_fn(processor, config)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
    )

    print(f"\nTraining: {len(train_dataset)} images, {config.num_epochs} epochs")
    print(f"Effective batch: {config.batch_size} x {config.gradient_accumulation} = {config.batch_size * config.gradient_accumulation}")

    trainer.train()
    print("✓ Training complete!")

    print("\nSaving model...")
    trainer.save_model(config.merged_dir)
    processor.save_pretrained(config.merged_dir)
    print(f"✓ Saved to {config.merged_dir}")

    # Quick test
    print("\n" + "=" * 60)
    print("Quick test...")
    print("=" * 60)

    test_img = Image.open(data_dir / "images" / "00000.png").convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image"}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(
        text=[text],
        images=[[test_img]],
        return_tensors="pt",
        size={"longest_edge": config.longest_edge},
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=256, do_sample=False)

    input_length = inputs["input_ids"].shape[1]
    result = processor.tokenizer.decode(out[0, input_length:], skip_special_tokens=True)
    print(f"Test result:\n{result[:500]}")

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    print(f"\nModel saved to: {config.merged_dir}")


def main():
    config = Config()
    train(config)


if __name__ == "__main__":
    main()
