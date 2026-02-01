#!/usr/bin/env python3
"""
Fine-tune LightOnOCR-2 for Hungarian language (ő/ű characters).

Requirements:
    pip install transformers peft datasets accelerate bitsandbytes

Usage:
    # Prepare training data first (see prepare_training_data.py)
    python scripts/finetune_hungarian.py --data-dir training_data/ --output-dir models/lighton-hun

Then convert to MLX:
    mlx_vlm convert --hf-path models/lighton-hun --mlx-path models/lighton-hun-mlx -q --q-bits 4
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset, Features, Image, Value
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from PIL import Image as PILImage
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    LightOnOcrForConditionalGeneration,
    TrainingArguments,
    Trainer,
)


def load_training_data(data_dir: Path) -> Dataset:
    """Load training data from directory.

    Expected structure:
        data_dir/
            images/
                001.png
                002.png
                ...
            annotations.jsonl  # {"image": "001.png", "text": "..."}
    """
    annotations_file = data_dir / "annotations.jsonl"
    images_dir = data_dir / "images"

    data = []
    with open(annotations_file, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            image_path = images_dir / entry["image"]
            if image_path.exists():
                data.append({
                    "image": str(image_path),
                    "text": entry["text"]
                })

    return Dataset.from_list(data)


def create_training_example(example, processor):
    """Convert a single example to model inputs."""
    image = PILImage.open(example["image"]).convert("RGB")
    text = example["text"]

    # Create conversation format
    conversation = [
        {"role": "user", "content": [{"type": "image"}]},
        {"role": "assistant", "content": text}
    ]

    # Process with chat template
    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    # Add image
    image_inputs = processor.image_processor(image, return_tensors="pt")
    inputs["pixel_values"] = image_inputs["pixel_values"]

    # Labels are the same as input_ids for causal LM
    inputs["labels"] = inputs["input_ids"].clone()

    return {k: v.squeeze(0) for k, v in inputs.items()}


def main():
    parser = argparse.ArgumentParser(description="Fine-tune LightOnOCR-2 for Hungarian")
    parser.add_argument("--data-dir", type=Path, required=True, help="Training data directory")
    parser.add_argument("--output-dir", type=Path, default=Path("models/lighton-hun"))
    parser.add_argument("--base-model", default="lightonai/LightOnOCR-2-1B-base")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--use-4bit", action="store_true", help="Use 4-bit quantization for training")
    args = parser.parse_args()

    print(f"Loading base model: {args.base_model}")

    # Quantization config for memory-efficient training
    if args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None

    # Load model and processor
    model = LightOnOcrForConditionalGeneration.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(args.base_model)

    # Prepare for LoRA
    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)

    # LoRA config - target language model layers
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load and process data
    print(f"Loading training data from: {args.data_dir}")
    dataset = load_training_data(args.data_dir)
    print(f"Loaded {len(dataset)} training examples")

    # Process dataset
    def process_fn(example):
        return create_training_example(example, processor)

    processed_dataset = dataset.map(process_fn, remove_columns=dataset.column_names)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        bf16=True,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save
    print(f"Saving model to: {args.output_dir}")
    trainer.save_model()
    processor.save_pretrained(args.output_dir)

    # Merge LoRA weights for easier conversion
    print("Merging LoRA weights...")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(args.output_dir / "merged")

    print("Done! Next steps:")
    print(f"  1. Test the model: python -c \"from transformers import ...; ...\"")
    print(f"  2. Convert to MLX:")
    print(f"     mlx_vlm convert --hf-path {args.output_dir}/merged --mlx-path {args.output_dir}-mlx -q --q-bits 4")


if __name__ == "__main__":
    main()
