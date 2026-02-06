#!/bin/bash

# 图像分类训练脚本示例
# 使用方法: bash scripts/train_classification.sh

export CUDA_VISIBLE_DEVICES=0,1,2,3

python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=29500 \
    qwenvl/train/train_classification.py \
    --model_name_or_path "Qwen/Qwen2-VL-2B-Instruct" \
    --dataset_use "data/train.jsonl" \
    --output_dir "outputs/classification" \
    --num_classes 200 \
    --classifier_dropout 0.1 \
    --tune_mm_llm false \
    --tune_mm_vision false \
    --tune_mm_mlp false \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --num_train_epochs 10 \
    --save_steps 500 \
    --logging_steps 100 \
    --bf16 \
    --gradient_checkpointing \
    --dataloader_num_workers 4 \
    --remove_unused_columns false
