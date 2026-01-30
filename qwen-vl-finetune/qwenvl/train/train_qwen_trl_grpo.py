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

pip install math_verify

Example usage:
    python qwenvl/train/train_qwen_trl_grpo.py \
        --model_name_or_path Qwen/Qwen3-VL-8B-Instruct \
        --output_dir ./output/qwen-vl-8b-trl \
        --per_device_train_batch_size 2 \
        --per_device_train_batch_size 8 \
        --max_completion_length 1024 \
        --num_generations 8 \
        --max_prompt_length 2048 \
        --gradient_accumulation_steps 8 \
        --num_train_epochs 1 \
        --learning_rate 2e-5 \
        --bf16 True \
        --gradient_checkpointing True \
        --use_peft True \
        --lora_r 8 \
        --lora_alpha 32 \
        --lora_target_modules "q_proj", "v_proj" \
        --log_completions
"""

from dataclasses import dataclass, field
from typing import Optional

import re
import torch
from datasets import load_dataset
from transformers import AutoProcessor
from math_verify import LatexExtractionConfig, parse, verify
from latex2sympy2_extended import NormalizationConfig
from typing import Optional

from trl import (
    ModelConfig,
    ScriptArguments,
    GRPOConfig,
    GRPOTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)


@dataclass
class QwenScriptArguments(ScriptArguments):
    """Extended script arguments for Qwen-VL training."""

    data_root: Optional[str] = field(
        default="", metadata={"help": "Root directory for image/video files"}
    )
    max_pixels: int = field(
        default=50176,  # ~224x224
        metadata={"help": "Maximum number of pixels for image encoding"},
    )
    min_pixels: int = field(
        default=784,  # ~28x28
        metadata={"help": "Minimum number of pixels for image encoding"},
    )
    video_fps: float = field(
        default=2.0, metadata={"help": "FPS for video frame extraction"}
    )


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
    rewards = [1.0 if match else 0.0 for match in matches]
    return rewards


def len_reward(completions, solution, **kwargs) -> float:
    """Compute length-based rewards to discourage overthinking and promote token efficiency.

    Taken from the Kimi 1.5 tech report: https://huggingface.co/papers/2501.12599

    Args:
        completions: List of model completions
        solution: List of ground truth solutions

    Returns:
        List of rewards where:
        - For correct answers: reward = 0.5 - (len - min_len)/(max_len - min_len)
        - For incorrect answers: reward = min(0, 0.5 - (len - min_len)/(max_len - min_len))
    """
    contents = completions

    # First check correctness of answers
    correctness = []
    for content, sol in zip(contents, solution):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if len(gold_parsed) == 0:
            # Skip unparseable examples
            correctness.append(True)  # Treat as correct to avoid penalizing
            print("Failed to parse gold solution: ", sol)
            continue

        answer_parsed = parse(
            content,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed=True,
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        correctness.append(verify(answer_parsed, gold_parsed))

    # Calculate lengths
    lengths = [len(content) for content in contents]
    min_len = min(lengths)
    max_len = max(lengths)

    # If all responses have the same length, return zero rewards
    if max_len == min_len:
        return [0.0] * len(completions)

    rewards = []
    for length, is_correct in zip(lengths, correctness):
        lambda_val = 0.5 - (length - min_len) / (max_len - min_len)

        if is_correct:
            reward = lambda_val
        else:
            reward = min(0, lambda_val)

        rewards.append(float(reward))

    return rewards

if __name__ == "__main__":
    # Parse arguments
    parser = TrlParser((QwenScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    # Set training configurations
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.remove_unused_columns = False
    training_args.dataset_text_field = ""  # We'll use formatting function
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    ################
    # Model & Processor
    ################
    dtype = (
        model_args.dtype
        if model_args.dtype in ["auto", None]
        else getattr(torch, model_args.dtype)
    )

    # Determine model class based on model name
    # Note: Qwen3-VL-30B-A3B uses MoE architecture (A3B = Active 3B parameters)
    if "qwen3" in model_args.model_name_or_path.lower() and (
        "moe" in model_args.model_name_or_path.lower()
        or "a3b" in model_args.model_name_or_path.lower()
    ):
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
        dtype=dtype,
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
        **model_kwargs,
    )

    # Enable MoE auxiliary loss for MoE models
    # This is required for TRL to include the load balancing/auxiliary loss
    is_moe_model = "moe" in model_class.__name__.lower()
    if is_moe_model:
        model.config.output_router_logits = True
        print("✓ Enabled MoE auxiliary loss (output_router_logits=True)")
        print(
            "  TRL will automatically include router load balancing loss in training"
        )

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

    ################
    # Dataset
    ################
    print(f"Loading dataset: {script_args.dataset_name}")
    dataset = load_dataset("lmms-lab/multimodal-open-r1-8k-verified")

    SYSTEM_PROMPT = (
        "You are a helpful AI Assistant that provides well-reasoned and detailed responses. "
        "You first think about the reasoning process as an internal monologue and then provide the user with the answer. "
        "Respond in the following format: <think>\n...\n</think>\n<answer>\n...\n</answer>"
    )

    def make_conversation(example):
        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": example["image"]},
                    {"type": "text", "text": example["problem"]},
                ],
            },
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        return {
            "prompt": prompt,
            "image": example["image"],
        }
    dataset = dataset.map(make_conversation)


    train_dataset = dataset[script_args.dataset_train_split]
    eval_dataset = (
        dataset[script_args.dataset_test_split]
        if training_args.eval_strategy != "no"
        else None
    )

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
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[format_reward, len_reward],
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
