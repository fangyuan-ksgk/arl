#!/usr/bin/env python3
"""
SFT Training for VLM (Qwen3.5) on Visual QA/Task

This script fine-tunes a Vision-Language Model using Supervised Fine-Tuning.
Example task: Visual reasoning, image captioning, or visual QA.
"""

import os
import json
import random
import argparse
from dataclasses import dataclass

import torch
from PIL import Image, ImageDraw
import numpy as np
from tqdm import tqdm

from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from datasets import Dataset
from trl import SFTTrainer, SFTConfig as TRLSFTConfig

@dataclass
class VLMSFTConfig:
    model_name: str = "Qwen/Qwen3.5-0.8B"
    output_dir: str = "./vlm_sft_output"
    num_train_epochs: int = 3
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    max_seq_length: int = 512
    lora_r: int = 16
    lora_alpha: int = 32
    use_lora: bool = True
    use_4bit: bool = True
    logging_steps: int = 10
    save_steps: int = 100
    task_type: str = "counting"


def create_synthetic_counting_dataset(num_samples: int = 1000, split: str = "train"):
    """
    Create synthetic visual counting dataset.
    Generates images with geometric shapes and corresponding Q&A pairs.
    """
    data = []
    
    print(f"Creating {num_samples} synthetic {split} samples...")
    
    for i in tqdm(range(num_samples)):
        # Create blank image
        img_size = 256
        img = Image.new('RGB', (img_size, img_size), color='white')
        draw = ImageDraw.Draw(img)
        
        # Randomly place shapes
        num_shapes = random.randint(1, 8)
        shape_counts = {"circle": 0, "square": 0, "triangle": 0}
        colors = ["red", "blue", "green", "yellow", "purple", "orange"]
        
        for _ in range(num_shapes):
            shape_type = random.choice(list(shape_counts.keys()))
            color = random.choice(colors)
            x, y = random.randint(20, img_size-70), random.randint(20, img_size-70)
            size = random.randint(20, 50)
            
            if shape_type == "circle":
                draw.ellipse([x, y, x+size, y+size], fill=color, outline="black")
            elif shape_type == "square":
                draw.rectangle([x, y, x+size, y+size], fill=color, outline="black")
            elif shape_type == "triangle":
                draw.polygon([(x, y+size), (x+size//2, y), (x+size, y+size)], fill=color, outline="black")
            
            shape_counts[shape_type] += 1
        
        # Create question-answer pairs
        qa_pairs = []
        
        # Total count question
        total = sum(shape_counts.values())
        qa_pairs.append({
            "question": "How many shapes are in this image?",
            "answer": str(total)
        })
        
        # Individual shape questions
        for shape, count in shape_counts.items():
            if count > 0:
                qa_pairs.append({
                    "question": f"How many {shape}s are in this image?",
                    "answer": str(count)
                })
        
        # Select one QA pair for this sample
        qa = random.choice(qa_pairs)
        
        data.append({
            "image": img,
            "question": qa["question"],
            "answer": qa["answer"],
            "metadata": {
                "shape_counts": shape_counts,
                "total_shapes": total
            }
        })
    
    return Dataset.from_list(data)


def create_color_recognition_dataset(num_samples: int = 1000):
    """Create synthetic color recognition dataset."""
    data = []
    colors = {
        "red": (255, 0, 0),
        "blue": (0, 0, 255),
        "green": (0, 255, 0),
        "yellow": (255, 255, 0),
        "purple": (128, 0, 128),
        "orange": (255, 165, 0),
    }
    
    print(f"Creating {num_samples} color recognition samples...")
    
    for i in tqdm(range(num_samples)):
        img_size = 256
        img = Image.new('RGB', (img_size, img_size), color='white')
        draw = ImageDraw.Draw(img)
        
        # Pick a dominant color
        dominant_color_name = random.choice(list(colors.keys()))
        dominant_color_rgb = colors[dominant_color_name]
        
        # Draw colored circles
        num_circles = random.randint(3, 7)
        target_count = 0
        
        for _ in range(num_circles):
            if random.random() < 0.6:  # 60% chance of dominant color
                color_rgb = dominant_color_rgb
                color_name = dominant_color_name
                target_count += 1
            else:
                other_color = random.choice([c for c in colors.keys() if c != dominant_color_name])
                color_rgb = colors[other_color]
            
            x, y = random.randint(20, img_size-50), random.randint(20, img_size-50)
            size = random.randint(30, 60)
            draw.ellipse([x, y, x+size, y+size], fill=color_rgb, outline="black")
        
        # Create question
        question = f"How many {dominant_color_name} circles are in this image?"
        answer = str(target_count)
        
        data.append({
            "image": img,
            "question": question,
            "answer": answer,
            "metadata": {"target_color": dominant_color_name, "count": target_count}
        })
    
    return Dataset.from_list(data)


def format_dataset_for_sft(dataset):
    """Format dataset into chat messages for SFTTrainer.

    SFTTrainer expects a 'messages' column with the conversation,
    and a separate 'images' column with the PIL images.
    """
    rows = []
    print("Formatting dataset for SFT...")
    for ex in tqdm(dataset):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": ex["question"]}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": ex["answer"]}
                ]
            }
        ]
        rows.append({"messages": messages, "images": [ex["image"]]})
    return Dataset.from_list(rows)


def main():
    parser = argparse.ArgumentParser(description="SFT Training for VLM")
    parser.add_argument("--task", default="counting", choices=["counting", "color"])
    parser.add_argument("--output_dir", default="./vlm_sft_output")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--no_4bit", action="store_true", help="Disable 4-bit quantization")
    args = parser.parse_args()

    config = VLMSFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        batch_size=args.batch_size,
        task_type=args.task,
        use_4bit=not args.no_4bit,
    )

    os.makedirs(config.output_dir, exist_ok=True)

    print("=" * 60)
    print("VLM SFT Training - Qwen3.5")
    print("=" * 60)
    print(f"Task: {config.task_type}")
    print(f"Model: {config.model_name}")
    print(f"Output: {config.output_dir}")

    # Create dataset
    if config.task_type == "counting":
        train_raw = create_synthetic_counting_dataset(args.max_samples, "train")
        eval_raw = create_synthetic_counting_dataset(100, "eval")
    elif config.task_type == "color":
        train_raw = create_color_recognition_dataset(args.max_samples)
        eval_raw = create_color_recognition_dataset(100)
    else:
        raise ValueError(f"Unknown task: {config.task_type}")

    train_data = format_dataset_for_sft(train_raw)
    eval_data = format_dataset_for_sft(eval_raw)
    print(f"\nTrain: {len(train_data)}, Eval: {len(eval_data)}")

    # Load processor and model
    print("\nLoading model and processor...")
    processor = AutoProcessor.from_pretrained(config.model_name, trust_remote_code=True)

    model_kwargs = dict(
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )
    if config.use_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model = AutoModelForImageTextToText.from_pretrained(config.model_name, **model_kwargs)

    # Setup LoRA
    if config.use_lora:
        from peft import LoraConfig, get_peft_model, TaskType
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # SFTTrainer handles multimodal collation automatically
    training_args = TRLSFTConfig(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        logging_steps=config.logging_steps,
        eval_steps=config.save_steps,
        save_steps=config.save_steps,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        report_to="none",
        bf16=True,
        dataset_text_field="messages",
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        processing_class=processor,
    )

    print("\nStarting training...")
    trainer.train()

    # Save
    trainer.save_model(config.output_dir)
    processor.save_pretrained(config.output_dir)

    with open(os.path.join(config.output_dir, "training_config.json"), "w") as f:
        json.dump(vars(config), f, indent=2, default=str)

    print(f"\n✓ Training complete! Model saved to {config.output_dir}")


if __name__ == "__main__":
    main()
