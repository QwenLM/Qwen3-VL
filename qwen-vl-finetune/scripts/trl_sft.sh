#!/bin/bash

# ==============================================
# Qwen-VL Training with TRL SFTTrainer
# Optimized for 1 GPU training
# ==============================================

# Model configuration
MODEL_PATH="Qwen/Qwen3-VL-8B-Instruct"

# Dataset configuration
# Option 1: Use local demo dataset
DATASET_PATH="demo/single_images.json"
DATA_ROOT=""  # Leave empty if paths in JSON are absolute, or set to project root

# Option 2: Use HuggingFace dataset (comment out DATASET_PATH and use these)
# DATASET_NAME="HuggingFaceH4/llava-instruct-mix-vsft"
# DATASET_CONFIG=""

# Output configuration
OUTPUT_DIR="./output/qwen-vl-8b-trl-lora"
RUN_NAME="qwen-vl-8b-trl-sft"

# Training hyperparameters (optimized for 1 GPU)
BATCH_SIZE=1
GRAD_ACCUM_STEPS=16  # Effective batch size = 1 * 16 = 16
LEARNING_RATE=2e-4
NUM_EPOCHS=1
MAX_LENGTH=2048

# LoRA configuration (for memory efficiency on 1 GPU)
USE_LORA=true
LORA_R=64
LORA_ALPHA=16
LORA_DROPOUT=0.05
# Target all linear layers for best performance
LORA_TARGET_MODULES="all-linear"

# Image processing
MAX_PIXELS=50176   # ~224x224 for 8B model on single GPU
MIN_PIXELS=784     # ~28x28

# Advanced options
GRADIENT_CHECKPOINTING=true
BF16=true
LOGGING_STEPS=10
SAVE_STEPS=100
SAVE_TOTAL_LIMIT=2

# ==============================================
# Build training command
# ==============================================

echo "========================================"
echo "Starting Qwen-VL 8B Training with TRL"
echo "========================================"
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"
echo "Batch size: $BATCH_SIZE"
echo "Gradient accumulation: $GRAD_ACCUM_STEPS"
echo "Effective batch size: $((BATCH_SIZE * GRAD_ACCUM_STEPS))"
echo "Using LoRA: $USE_LORA"
echo "========================================"

# Build command
CMD="python qwenvl/train/train_qwen_trl_sft.py‎ \
    --model_name_or_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --run_name $RUN_NAME"

# Add dataset configuration
if [ -n "$DATASET_PATH" ]; then
    CMD="$CMD \
    --dataset_path $DATASET_PATH \
    --data_root $DATA_ROOT"
else
    CMD="$CMD \
    --dataset_name $DATASET_NAME \
    --dataset_config $DATASET_CONFIG"
fi

# Add training arguments
CMD="$CMD \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --num_train_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --max_seq_length $MAX_LENGTH \
    --logging_steps $LOGGING_STEPS \
    --save_steps $SAVE_STEPS \
    --save_total_limit $SAVE_TOTAL_LIMIT \
    --max_pixels $MAX_PIXELS \
    --min_pixels $MIN_PIXELS"

# Add optimizer and scheduler
CMD="$CMD \
    --optim adamw_torch \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --weight_decay 0.01"

# Add precision settings
if [ "$BF16" = true ]; then
    CMD="$CMD --bf16 true"
else
    CMD="$CMD --fp16 true"
fi

# Add gradient checkpointing
if [ "$GRADIENT_CHECKPOINTING" = true ]; then
    CMD="$CMD --gradient_checkpointing true"
fi

# Add LoRA configuration
if [ "$USE_LORA" = true ]; then
    CMD="$CMD \
    --use_peft true \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --lora_target_modules $LORA_TARGET_MODULES"
fi

# Optional: Add reporting (uncomment if you have wandb/tensorboard)
# CMD="$CMD --report_to wandb"

# Launch training
echo "Launching training..."
echo "$CMD"
echo ""

eval $CMD

echo ""
echo "========================================"
echo "Training completed!"
echo "Model saved to: $OUTPUT_DIR"
echo "========================================"
