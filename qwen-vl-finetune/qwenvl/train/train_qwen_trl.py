#!/usr/bin/env python
# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Training script for Qwen-VL models using TRL's SFTTrainer.

Example usage:
    python qwenvl/train/train_qwen_trl.py \
        --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
        --dataset_name demo_single_images \
        --output_dir ./output/qwen-vl-7b-trl \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 4 \
        --num_train_epochs 1 \
        --learning_rate 2e-5 \
        --bf16 True \
        --gradient_checkpointing True \
        --use_peft True \
        --lora_r 64 \
        --lora_alpha 16 \
        --lora_target_modules all-linear
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset, load_dataset
from transformers import AutoProcessor

from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)


@dataclass
class QwenScriptArguments(ScriptArguments):
    """Extended script arguments for Qwen-VL training."""

    dataset_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to local dataset JSON file (e.g., demo/single_images.json)"}
    )
    data_root: Optional[str] = field(
        default="",
        metadata={"help": "Root directory for image/video files"}
    )
    max_pixels: int = field(
        default=50176,  # ~224x224
        metadata={"help": "Maximum number of pixels for image encoding"}
    )
    min_pixels: int = field(
        default=784,  # ~28x28
        metadata={"help": "Minimum number of pixels for image encoding"}
    )
    video_fps: float = field(
        default=2.0,
        metadata={"help": "FPS for video frame extraction"}
    )
    video_max_pixels: int = field(
        default=50176,  # Can be higher for videos
        metadata={"help": "Maximum pixels per video frame"}
    )
    video_min_pixels: int = field(
        default=784,
        metadata={"help": "Minimum pixels per video frame"}
    )


def load_qwen_dataset(dataset_path: str, data_root: str = "") -> Dataset:
    """
    Load Qwen-VL format dataset and convert to HF format.

    Expected format:
    [
        {
            "image": "path/to/image.jpg",
            "conversations": [
                {"from": "human", "value": "<image>\\nQuestion?"},
                {"from": "gpt", "value": "Answer"}
            ]
        }
    ]
    """
    with open(dataset_path, 'r') as f:
        data = json.load(f)

    # Convert to TRL format
    converted_data = []
    for item in data:
        # Handle images
        if "image" in item:
            image_path = os.path.join(data_root, item["image"]) if data_root else item["image"]
            images = [image_path]
        elif "images" in item:
            images = [os.path.join(data_root, img) if data_root else img for img in item["images"]]
        else:
            images = []

        # Handle videos
        if "video" in item:
            video_path = os.path.join(data_root, item["video"]) if data_root else item["video"]
            videos = [video_path]
        elif "videos" in item:
            videos = [os.path.join(data_root, vid) if data_root else vid for vid in item["videos"]]
        else:
            videos = []

        # Convert conversations to messages format
        messages = []
        for conv in item.get("conversations", []):
            role = "user" if conv["from"] == "human" else "assistant"
            messages.append({
                "role": role,
                "content": conv["value"]
            })

        example = {
            "messages": messages
        }

        # Add images/videos if present
        if images:
            example["images"] = images
        if videos:
            example["videos"] = videos

        converted_data.append(example)

    return Dataset.from_list(converted_data)


def format_dataset_for_qwen(example, processor):
    """Format dataset examples for Qwen-VL models."""
    # Process messages and images together
    text = processor.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False
    )

    return {
        "text": text,
        "images": example["images"]
    }


if __name__ == "__main__":
    # Parse arguments
    parser = TrlParser((QwenScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    # Set training configurations
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.remove_unused_columns = False
    training_args.dataset_text_field = ""  # We'll use formatting function
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    ################
    # Model & Processor
    ################
    dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)

    # Determine model class based on model name
    # Note: Qwen3-VL-30B-A3B uses MoE architecture (A3B = Active 3B parameters)
    if "qwen3" in model_args.model_name_or_path.lower() and ("moe" in model_args.model_name_or_path.lower() or "a3b" in model_args.model_name_or_path.lower()):
        from transformers import Qwen3VLMoeForConditionalGeneration
        model_class = Qwen3VLMoeForConditionalGeneration
    elif "qwen3" in model_args.model_name_or_path.lower():
        from transformers import Qwen3VLForConditionalGeneration
        model_class = Qwen3VLForConditionalGeneration
    elif "qwen2.5" in model_args.model_name_or_path.lower():
        from transformers import Qwen2_5_VLForConditionalGeneration
        model_class = Qwen2_5_VLForConditionalGeneration
    else:
        from transformers import Qwen2VLForConditionalGeneration
        model_class = Qwen2VLForConditionalGeneration

    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation or "flash_attention_2",
        torch_dtype=dtype,
    )

    # Add quantization config if specified
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config

    # Load model
    print(f"Loading model: {model_args.model_name_or_path}")
    print(f"Model class: {model_class.__name__}")
    model = model_class.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        **model_kwargs
    )

    # Enable MoE auxiliary loss for MoE models
    # This is required for TRL to include the load balancing/auxiliary loss
    is_moe_model = "moe" in model_class.__name__.lower()
    if is_moe_model:
        model.config.output_router_logits = True
        print(f"✓ Enabled MoE auxiliary loss (output_router_logits=True)")
        print(f"  TRL will automatically include router load balancing loss in training")

    # Load processor
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
    )

    # Set processor min/max pixels for images
    if hasattr(processor, "image_processor"):
        processor.image_processor.max_pixels = script_args.max_pixels
        processor.image_processor.min_pixels = script_args.min_pixels

    # Set video processing parameters
    if hasattr(processor, "video_processor"):
        processor.video_processor.fps = script_args.video_fps
        processor.video_processor.max_pixels = script_args.video_max_pixels
        processor.video_processor.min_pixels = script_args.video_min_pixels

    ################
    # Dataset
    ################
    if script_args.dataset_path:
        # Load local dataset
        print(f"Loading dataset from: {script_args.dataset_path}")
        dataset = load_qwen_dataset(script_args.dataset_path, script_args.data_root)
        train_dataset = dataset
        eval_dataset = None
    else:
        # Load from HuggingFace Hub
        print(f"Loading dataset: {script_args.dataset_name}")
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
        train_dataset = dataset[script_args.dataset_train_split]
        eval_dataset = dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None

    print(f"Train dataset size: {len(train_dataset)}")
    if eval_dataset:
        print(f"Eval dataset size: {len(eval_dataset)}")

    ################
    # PEFT Config
    ################
    peft_config = get_peft_config(model_args)
    if peft_config:
        print(f"Using PEFT with config: {peft_config}")

    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        peft_config=peft_config,
    )

    print("Starting training...")
    trainer.train()

    # Save model and processor
    print(f"Saving model to: {training_args.output_dir}")
    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)

    if training_args.push_to_hub:
        print("Pushing to hub...")
        trainer.push_to_hub(dataset_name=script_args.dataset_name)

    print("Training completed!")
