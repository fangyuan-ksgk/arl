#!/usr/bin/env python3
"""
GRPO Training for VLM (Qwen3.5) on Visual Task

This script trains a Vision-Language Model using Group Relative Policy Optimization.
Task: Visual counting/reasoning with verifiable rewards.
"""

import os
import sys
import json
import re
import random
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any
from collections import defaultdict

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Patch for compatibility
try:
    from transformers import configuration_utils
    if not hasattr(configuration_utils, 'ALLOWED_LAYER_TYPES'):
        configuration_utils.ALLOWED_LAYER_TYPES = ['qwen2', 'qwen3', 'qwen3_5', 'llama', 'mistral']
except:
    pass

from transformers import AutoProcessor, AutoModelForImageTextToText
from datasets import Dataset

# Try importing TRL, fall back to custom implementation if needed
try:
    from trl import GRPOTrainer as _GRPOTrainer, GRPOConfig
    HAS_TRL = True

    class GRPOTrainer(_GRPOTrainer):
        """Patch GRPOTrainer to propagate mm_token_type_ids for Qwen3.5-VL.

        TRL 0.29 doesn't propagate mm_token_type_ids through the training pipeline,
        but Qwen3.5-VL requires it for computing 3D mrope position IDs.
        We re-derive it from input_ids by detecting image/video token positions.
        """

        def _build_mm_token_type_ids(self, input_ids):
            """Build mm_token_type_ids from input_ids by detecting image/video tokens."""
            processor = self.processing_class
            image_token_id = getattr(processor, 'image_token_id', None)
            if image_token_id is None:
                return None
            mm_token_type_ids = torch.zeros_like(input_ids)
            mm_token_type_ids[input_ids == image_token_id] = 1
            video_token_id = getattr(processor, 'video_token_id', None)
            if video_token_id is not None:
                mm_token_type_ids[input_ids == video_token_id] = 2
            return mm_token_type_ids

        def _generate_and_score_completions(self, inputs):
            output = super()._generate_and_score_completions(inputs)
            if "pixel_values" in output and "mm_token_type_ids" not in output:
                input_ids = torch.cat([output["prompt_ids"], output["completion_ids"]], dim=1)
                mm_ids = self._build_mm_token_type_ids(input_ids)
                if mm_ids is not None:
                    output["mm_token_type_ids"] = mm_ids
            return output

        def _get_per_token_logps_and_entropies(self, model, input_ids, attention_mask,
                                                logits_to_keep, batch_size=None,
                                                compute_entropy=False, **kwargs):
            # Inject mm_token_type_ids if we have pixel_values (Qwen3.5-VL needs it)
            if kwargs.get("pixel_values") is not None and "mm_token_type_ids" not in kwargs:
                mm_ids = self._build_mm_token_type_ids(input_ids)
                if mm_ids is not None:
                    kwargs["mm_token_type_ids"] = mm_ids
            return super()._get_per_token_logps_and_entropies(
                model, input_ids, attention_mask, logits_to_keep,
                batch_size=batch_size, compute_entropy=compute_entropy, **kwargs
            )

except ImportError:
    print("Warning: TRL not available, using custom GRPO implementation")
    HAS_TRL = False


@dataclass
class GRPOVLMConfig:
    model_name: str = "Qwen/Qwen3.5-0.8B"
    output_dir: str = "./vlm_grpo_output"
    num_generations: int = 4  # Group size for GRPO
    max_completion_length: int = 128
    num_train_epochs: int = 3
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-5
    logging_steps: int = 10
    save_steps: int = 50
    use_lora: bool = True
    lora_r: int = 8
    temperature: float = 0.7
    top_p: float = 0.9
    clip_epsilon: float = 0.2  # PPO-style clipping
    kl_penalty: float = 0.01


class VisualCountingEnvironment:
    """Environment that generates visual counting tasks with verifiable answers."""
    
    def __init__(self, img_size: int = 256):
        self.img_size = img_size
        self.shape_types = ["circle", "square", "triangle"]
        self.colors = ["red", "blue", "green", "yellow", "purple", "orange"]
    
    def generate_task(self, seed: int = None) -> Dict[str, Any]:
        """Generate a counting task with image and ground truth."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Create image
        img = Image.new('RGB', (self.img_size, self.img_size), color='white')
        draw = ImageDraw.Draw(img)
        
        # Place random shapes
        num_shapes = random.randint(2, 10)
        shape_counts = defaultdict(int)
        shape_details = []
        
        for _ in range(num_shapes):
            shape = random.choice(self.shape_types)
            color = random.choice(self.colors)
            x = random.randint(20, self.img_size - 70)
            y = random.randint(20, self.img_size - 70)
            size = random.randint(25, 55)
            
            # Draw shape
            color_rgb = self._color_to_rgb(color)
            if shape == "circle":
                draw.ellipse([x, y, x+size, y+size], fill=color_rgb, outline="black")
            elif shape == "square":
                draw.rectangle([x, y, x+size, y+size], fill=color_rgb, outline="black")
            elif shape == "triangle":
                draw.polygon([(x, y+size), (x+size//2, y), (x+size, y+size)], fill=color_rgb, outline="black")
            
            shape_counts[shape] += 1
            shape_details.append({"shape": shape, "color": color, "pos": (x, y), "size": size})
        
        # Choose question type
        question_type = random.choice(["total", "specific_shape", "color_count"])
        
        if question_type == "total":
            question = "How many shapes are in this image? Count carefully and provide only the number."
            answer = str(num_shapes)
            metadata = {"type": "total", "count": num_shapes}
        
        elif question_type == "specific_shape":
            target_shape = random.choice(self.shape_types)
            count = shape_counts[target_shape]
            question = f"How many {target_shape}s are in this image? Count carefully and provide only the number."
            answer = str(count)
            metadata = {"type": "specific_shape", "shape": target_shape, "count": count}
        
        else:  # color_count
            # Count shapes of a specific color
            target_color = random.choice(self.colors)
            count = sum(1 for s in shape_details if s["color"] == target_color)
            question = f"How many {target_color} shapes are in this image? Count carefully and provide only the number."
            answer = str(count)
            metadata = {"type": "color_count", "color": target_color, "count": count}
        
        return {
            "image": img,
            "question": question,
            "answer": answer,
            "metadata": metadata,
            "shape_counts": dict(shape_counts),
            "shape_details": shape_details
        }
    
    def _color_to_rgb(self, color: str) -> tuple:
        """Convert color name to RGB."""
        color_map = {
            "red": (255, 0, 0),
            "blue": (0, 0, 255),
            "green": (0, 255, 0),
            "yellow": (255, 255, 0),
            "purple": (128, 0, 128),
            "orange": (255, 165, 0),
        }
        return color_map.get(color, (128, 128, 128))


def extract_number(text: str) -> int:
    """Extract number from model completion."""
    # Look for explicit number patterns
    patterns = [
        r'answer is (\d+)',
        r'count is (\d+)',
        r'are (\d+)',
        r'contains? (\d+)',
        r'\\boxed{(\d+)}',
        r'^(\d+)$',
        r'(\d+) shapes',
        r'(\d+) circles',
        r'(\d+) squares',
        r'(\d+) triangles',
    ]
    
    text_lower = text.lower()
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            try:
                return int(match.group(1))
            except:
                pass
    
    # Fallback: find any number
    numbers = re.findall(r'\d+', text)
    if numbers:
        return int(numbers[-1])  # Take last number (usually the answer)
    
    return -1  # No number found


def create_reward_function():
    """Create reward function for visual counting task."""

    def reward_fn(prompts, completions, **kwargs) -> List[float]:
        """
        Reward function for GRPO.

        TRL calls this with a flat list: one entry per completion across the batch.
        gold_answer is repeated per num_generations by TRL automatically.
        """
        rewards = []
        gold_answers = kwargs.get('gold_answer', [])

        for i, completion in enumerate(completions):
            # Extract text from completion (conversational or plain string)
            if isinstance(completion, list):
                text = completion[0]["content"] if completion else ""
            elif isinstance(completion, dict):
                text = completion.get("content", "")
            else:
                text = str(completion)

            gold_answer = gold_answers[i] if i < len(gold_answers) else "0"
            gold_num = int(gold_answer) if str(gold_answer).isdigit() else 0
            predicted_num = extract_number(text)

            # Calculate reward
            if predicted_num == gold_num:
                reward = 1.0
            elif predicted_num >= 0 and abs(predicted_num - gold_num) <= 1:
                reward = 0.5
            elif predicted_num >= 0 and abs(predicted_num - gold_num) <= 2:
                reward = 0.2
            elif predicted_num >= 0:
                reward = 0.0
            else:
                reward = -0.5

            rewards.append(reward)

        return rewards

    return reward_fn


def create_dataset(num_samples: int = 500, env: VisualCountingEnvironment = None):
    """Create dataset for GRPO training."""
    if env is None:
        env = VisualCountingEnvironment()

    data = []
    print(f"Creating {num_samples} visual counting tasks...")

    for i in tqdm(range(num_samples)):
        task = env.generate_task(seed=i)

        # TRL expects text-only prompts + a separate "image" column.
        # TRL's prepare_multimodal_messages will inject images into the prompt.
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": task["question"]}
                ]
            }
        ]

        data.append({
            "prompt": messages,
            "image": task["image"],
            "gold_answer": task["answer"],
            "metadata": task["metadata"],
        })

    return Dataset.from_list(data)


def visualize_task(task: Dict, save_path: str = None):
    """Visualize a counting task."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Show image
    axes[0].imshow(task["image"])
    axes[0].set_title("Image")
    axes[0].axis('off')
    
    # Show info
    info_text = f"""
    Question: {task['question']}
    
    Answer: {task['answer']}
    
    Metadata: {json.dumps(task['metadata'], indent=2)}
    
    Shape Counts: {task['shape_counts']}
    """
    axes[1].text(0.1, 0.5, info_text, fontsize=10, verticalalignment='center',
                 family='monospace', transform=axes[1].transAxes)
    axes[1].axis('off')
    axes[1].set_title("Task Details")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def custom_grpo_train(model, processor, dataset, config, output_dir):
    """Custom GRPO implementation when TRL is not available."""
    print("Using custom GRPO implementation...")
    
    from torch.optim import AdamW
    
    # Setup
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    device = model.device
    
    # Create environment and reward function
    env = VisualCountingEnvironment()
    reward_fn = create_reward_function(env)
    
    # Training loop
    model.train()
    global_step = 0
    all_metrics = []
    
    for epoch in range(config.num_train_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_train_epochs}")
        
        epoch_rewards = []
        epoch_losses = []
        
        for i in tqdm(range(0, len(dataset), config.batch_size)):
            batch = dataset[i:i+config.batch_size]
            
            # For each sample in batch
            for sample in batch:
                # Generate multiple completions (group)
                prompt = sample["prompt"]
                gold_answer = sample["gold_answer"]
                
                # Process inputs
                try:
                    inputs = processor.apply_chat_template(
                        prompt,
                        return_tensors="pt",
                        add_generation_prompt=True
                    ).to(device)
                except:
                    # Fallback
                    text_input = prompt[0]["content"][1]["text"]  # Get text part
                    inputs = processor(text_input, return_tensors="pt").input_ids.to(device)
                
                # Generate completions
                completions_texts = []
                log_probs = []
                
                for _ in range(config.num_generations):
                    with torch.no_grad():
                        outputs = model.generate(
                            inputs,
                            max_new_tokens=config.max_completion_length,
                            temperature=config.temperature,
                            top_p=config.top_p,
                            do_sample=True,
                            return_dict_in_generate=True,
                            output_scores=True,
                        )
                    
                    # Decode completion
                    completion_ids = outputs.sequences[0][inputs.shape[1]:]
                    completion_text = processor.decode(completion_ids, skip_special_tokens=True)
                    completions_texts.append(completion_text)
                    
                    # Calculate log probability
                    scores = torch.stack(outputs.scores, dim=1)
                    log_prob = F.log_softmax(scores, dim=-1)
                    token_log_probs = log_prob.gather(2, completion_ids.unsqueeze(0).unsqueeze(-1)).squeeze(-1)
                    avg_log_prob = token_log_probs.mean().item()
                    log_probs.append(avg_log_prob)
                
                # Calculate rewards
                rewards = []
                for text in completions_texts:
                    pred_num = extract_number(text)
                    gold_num = int(gold_answer)
                    
                    if pred_num == gold_num:
                        reward = 1.0
                    elif abs(pred_num - gold_num) <= 1:
                        reward = 0.5
                    else:
                        reward = 0.0
                    rewards.append(reward)
                
                # GRPO: relative advantage
                mean_reward = np.mean(rewards)
                advantages = [r - mean_reward for r in rewards]
                
                # Simple policy gradient update
                loss = 0
                for log_prob, advantage in zip(log_probs, advantages):
                    loss -= log_prob * advantage
                
                loss = loss / len(log_probs)
                
                # Backward
                loss.backward()
                
                if (global_step + 1) % config.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                
                epoch_rewards.extend(rewards)
                epoch_losses.append(loss.item())
                
                global_step += 1
                
                if global_step % config.logging_steps == 0:
                    avg_reward = np.mean(epoch_rewards[-20:]) if epoch_rewards else 0
                    avg_loss = np.mean(epoch_losses[-20:]) if epoch_losses else 0
                    print(f"Step {global_step}: loss={avg_loss:.4f}, reward={avg_reward:.4f}")
                    all_metrics.append({
                        "step": global_step,
                        "loss": avg_loss,
                        "reward": avg_reward,
                    })
        
        # Epoch summary
        print(f"Epoch {epoch+1} complete. Avg reward: {np.mean(epoch_rewards):.4f}")
        
        # Save checkpoint
        checkpoint_dir = os.path.join(output_dir, f"checkpoint-epoch-{epoch+1}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        model.save_pretrained(checkpoint_dir)
        processor.save_pretrained(checkpoint_dir)
    
    # Save final model
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    
    # Save metrics
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)
    
    print(f"\n✓ Training complete! Model saved to {output_dir}")
    
    return all_metrics


def main():
    parser = argparse.ArgumentParser(description="GRPO Training for VLM")
    parser.add_argument("--output_dir", default="./vlm_grpo_output")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--visualize", action="store_true", help="Generate sample visualizations")
    args = parser.parse_args()
    
    config = GRPOVLMConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        num_generations=args.num_generations,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    
    os.makedirs(config.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("VLM GRPO Training - Qwen3.5")
    print("=" * 60)
    print(f"Task: Visual Counting with Verifiable Rewards")
    print(f"Model: {config.model_name}")
    print(f"Group size: {config.num_generations}")
    print(f"Output: {config.output_dir}")
    
    # Setup environment
    env = VisualCountingEnvironment()
    
    # Visualize sample tasks
    if args.visualize:
        print("\nGenerating sample visualizations...")
        viz_dir = os.path.join(config.output_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        for i in range(5):
            task = env.generate_task(seed=i)
            visualize_task(task, os.path.join(viz_dir, f"sample_task_{i}.png"))
    
    # Create dataset
    train_dataset = create_dataset(args.max_samples, env)
    eval_dataset = create_dataset(50, env)
    
    print(f"\nTrain samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")
    
    # Show sample
    print("\nSample task:")
    sample = train_dataset[0]
    text_parts = [c for c in sample['prompt'][0]['content'] if c.get('type') == 'text']
    print(f"  Question: {text_parts[0]['text'] if text_parts else 'N/A'}")
    print(f"  Answer: {sample['gold_answer']}")
    
    # Load model and processor
    print("\nLoading model and processor...")
    processor = AutoProcessor.from_pretrained(config.model_name, trust_remote_code=True)

    # Don't use device_map with accelerate multi-GPU (accelerate handles placement)
    model = AutoModelForImageTextToText.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )
    
    # Setup LoRA
    if config.use_lora:
        try:
            from peft import LoraConfig, get_peft_model, TaskType
            lora_config = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_r * 2,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
        except ImportError:
            print("PEFT not available, training full model")
    
    # Train
    if HAS_TRL:
        print("\nUsing TRL GRPOTrainer...")

        # Create reward function
        reward_fn = create_reward_function()

        # TRL 0.29 requires per_device_train_batch_size >= num_generations
        training_args = GRPOConfig(
            output_dir=config.output_dir,
            num_generations=config.num_generations,
            max_completion_length=config.max_completion_length,
            per_device_train_batch_size=config.num_generations,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            num_train_epochs=config.num_train_epochs,
            learning_rate=config.learning_rate,
            logging_steps=config.logging_steps,
            save_steps=config.save_steps,
            bf16=True,
            report_to="none",
        )

        trainer = GRPOTrainer(
            model=model,
            reward_funcs=reward_fn,
            args=training_args,
            train_dataset=train_dataset,
            processing_class=processor,
        )

        trainer.train()
        trainer.save_model(config.output_dir)
        
    else:
        print("\nUsing custom GRPO implementation...")
        metrics = custom_grpo_train(model, processor, train_dataset, config, config.output_dir)
    
    # Save config
    with open(os.path.join(config.output_dir, "training_config.json"), "w") as f:
        json.dump(vars(config), f, indent=2, default=str)
    
    print(f"\n✓ Training complete! Model saved to {config.output_dir}")


if __name__ == "__main__":
    main()
