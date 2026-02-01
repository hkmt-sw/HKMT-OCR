#!/usr/bin/env python3
"""
Prepare training data for Hungarian OCR fine-tuning.

This script generates synthetic training images from Hungarian text,
focusing on words with ő/ű characters that the model struggles with.

Usage:
    python scripts/prepare_training_data.py --output-dir training_data/
"""

import argparse
import json
import random
from pathlib import Path
from typing import List

from PIL import Image, ImageDraw, ImageFont


# Hungarian words with ő and ű that are commonly confused
HUNGARIAN_WORDS = [
    # ő vs ö
    "őr", "őriz", "ők", "ősz", "ősi", "őszinte", "őrült", "őröl",
    "erő", "idő", "mező", "tető", "fő", "nő", "bő", "hő",
    "belső", "külső", "felső", "alsó", "utolsó", "első",
    "költő", "festő", "vezető", "eladó", "vásárló",
    "Győr", "Debrecen", "Székesfehérvár",
    "tükörfúrógép", "árvíztűrő", "börtön",
    "könyv", "költség", "közel", "között",
    "Csatornadíj", "vízdíj", "díj", "díjszabás",
    "dőlt", "dől", "bedől", "kidől",
    # ű vs ü
    "űr", "űrlap", "űrhajó", "gyűrű", "tűz", "fűz",
    "gyűjt", "gyűjtemény", "tűnik", "fűszer",
    "hűtő", "hűvös", "hűség", "hűtlen",
    "szürke", "szűk", "szűr", "szűz",
    "fülke", "küld", "kürt", "süt",
    # Mixed
    "őrző", "tűző", "fűző", "űző",
    "öüóőúéáűí", "ÖÜÓŐÚÉÁŰÍ",
    "halványszürke", "sötétszürke",
    # Common document words
    "fizetendő", "követendő", "teljesítendő",
    "összeg", "összesen", "összesítés",
    "kedvezmény", "átutalás", "számlaszám",
    "adószám", "cégjegyzékszám", "azonosító",
    "határidő", "lejárat", "esedékesség",
]

# Sentence templates with Hungarian special characters
SENTENCE_TEMPLATES = [
    "A fizetendő összeg: {amount} Ft",
    "Kedvezmény összege: {amount} Ft",
    "Számla azonosító: {id}",
    "Csatornadíj: {amount} Ft",
    "Vízdíj alapdíj: {amount} Ft",
    "Árvíztűrő tükörfúrógép",
    "öüóőúéáűí - ÖÜÓŐÚÉÁŰÍ",
    "Halványszürke szöveg tesztelése",
    "Fizetési határidő: {date}",
    "Adószám: {taxid}",
    "Dőlt betűs szöveg formázás",
    "A költő verse szép és őszinte",
    "Győri gyűjtemény és kiállítás",
    "Belső és külső műveletek",
    "Első és utolsó tételek összege",
]


def generate_random_text() -> str:
    """Generate random Hungarian text with special characters."""
    lines = []

    # Add some words
    words = random.sample(HUNGARIAN_WORDS, min(10, len(HUNGARIAN_WORDS)))
    lines.append(" ".join(words))

    # Add some sentences
    for _ in range(random.randint(2, 5)):
        template = random.choice(SENTENCE_TEMPLATES)
        text = template.format(
            amount=f"{random.randint(1, 99)} {random.randint(100, 999):03d}",
            id=f"ABC{random.randint(100000, 999999)}",
            date=f"2025.{random.randint(1,12):02d}.{random.randint(1,28):02d}",
            taxid=f"{random.randint(10000000, 99999999)}-{random.randint(1,2)}-{random.randint(10, 99)}",
        )
        lines.append(text)

    # Add the critical test line
    lines.append("öüóőúéáűí - ÖÜÓŐÚÉÁŰÍ")

    return "\n".join(lines)


def render_text_to_image(
    text: str,
    width: int = 800,
    font_size: int = 24,
    bg_color: str = "white",
    text_color: str = "black",
) -> Image.Image:
    """Render text to an image."""
    # Try to find a font that supports Hungarian characters
    font_paths = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ]

    font = None
    for font_path in font_paths:
        try:
            font = ImageFont.truetype(font_path, font_size)
            break
        except (OSError, IOError):
            continue

    if font is None:
        font = ImageFont.load_default()

    # Calculate image height based on text
    lines = text.split("\n")
    line_height = font_size + 10
    height = len(lines) * line_height + 40

    # Create image
    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    # Draw text
    y = 20
    for line in lines:
        draw.text((20, y), line, fill=text_color, font=font)
        y += line_height

    return img


def generate_training_data(
    output_dir: Path,
    num_samples: int = 500,
    variations: bool = True,
) -> None:
    """Generate training data with images and annotations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    annotations = []

    for i in range(num_samples):
        text = generate_random_text()

        # Generate variations
        if variations:
            # Different font sizes
            font_size = random.choice([18, 20, 22, 24, 28, 32])
            # Different backgrounds
            bg = random.choice(["white", "#f5f5f5", "#eeeeee", "#fafafa"])
            # Different text colors
            color = random.choice(["black", "#333333", "#222222"])
        else:
            font_size, bg, color = 24, "white", "black"

        # Render image
        img = render_text_to_image(text, font_size=font_size, bg_color=bg, text_color=color)

        # Save
        image_name = f"{i:05d}.png"
        img.save(images_dir / image_name)

        annotations.append({
            "image": image_name,
            "text": text,
        })

        if (i + 1) % 100 == 0:
            print(f"Generated {i + 1}/{num_samples} samples")

    # Save annotations
    annotations_file = output_dir / "annotations.jsonl"
    with open(annotations_file, "w", encoding="utf-8") as f:
        for ann in annotations:
            f.write(json.dumps(ann, ensure_ascii=False) + "\n")

    print(f"Done! Generated {num_samples} training samples")
    print(f"  Images: {images_dir}")
    print(f"  Annotations: {annotations_file}")


def add_real_documents(
    output_dir: Path,
    pdf_dir: Path,
    ground_truth_dir: Path,
) -> None:
    """Add real PDF documents with ground truth to training data."""
    from pdf2image import convert_from_path

    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    annotations_file = output_dir / "annotations.jsonl"

    # Load existing annotations if any
    existing = []
    if annotations_file.exists():
        with open(annotations_file, "r", encoding="utf-8") as f:
            existing = [json.loads(line) for line in f]

    new_annotations = []
    idx = len(existing)

    for pdf_path in pdf_dir.glob("*.pdf"):
        gt_path = ground_truth_dir / f"{pdf_path.stem}.txt"
        if not gt_path.exists():
            print(f"Skipping {pdf_path.name} - no ground truth")
            continue

        # Load ground truth
        gt_text = gt_path.read_text(encoding="utf-8")
        pages_gt = gt_text.split("---PAGE---")

        # Convert PDF to images
        images = convert_from_path(str(pdf_path), dpi=200)

        for page_idx, (img, page_gt) in enumerate(zip(images, pages_gt)):
            image_name = f"real_{idx:05d}.png"
            img.save(images_dir / image_name)

            new_annotations.append({
                "image": image_name,
                "text": page_gt.strip(),
                "source": pdf_path.name,
                "page": page_idx,
            })
            idx += 1

    # Save all annotations
    all_annotations = existing + new_annotations
    with open(annotations_file, "w", encoding="utf-8") as f:
        for ann in all_annotations:
            f.write(json.dumps(ann, ensure_ascii=False) + "\n")

    print(f"Added {len(new_annotations)} real document pages")


def main():
    parser = argparse.ArgumentParser(description="Prepare Hungarian OCR training data")
    parser.add_argument("--output-dir", type=Path, default=Path("training_data"))
    parser.add_argument("--num-samples", type=int, default=500, help="Number of synthetic samples")
    parser.add_argument("--add-real", action="store_true", help="Add real PDFs from test-pdf/")
    parser.add_argument("--pdf-dir", type=Path, default=Path("test-pdf"))
    parser.add_argument("--gt-dir", type=Path, default=Path("ground-truth"))
    args = parser.parse_args()

    print("Generating synthetic training data...")
    generate_training_data(args.output_dir, args.num_samples)

    if args.add_real:
        print("\nAdding real documents...")
        add_real_documents(args.output_dir, args.pdf_dir, args.gt_dir)


if __name__ == "__main__":
    main()
