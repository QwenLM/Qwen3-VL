# QwenVL Training Framework

This repository provides a training framework for Qwen VL models. The are two steps to use our repo:

1. Customize your dataset: downloading data, implement the config
2. Modify training scripts: 

## Repository Structure

The `qwenvl` directory contains the following components:

### `train/`
- `trainer.py`: Main trainer updated from Huggingface Trainer
- `train_qwen.py`: Main file for training
- `argument.py`: Dataclasses for model, data and training arguments

### `data/`
- `__init__.py`: Contains datasets configs
- `data_processor.py`: Data processing module for QwenVL models
- `rope2d.py`: Provide RoPE implementation

### `tools`
- `process_bbox.ipynb`: Convert bbox into QwenVL format. If you have grounding data, please refer this file to tranform your data.
- `pack_data.py`: Pack data into even length buckets.

## Requirements

You could use follow version of packages:

- `torch==2.6.0`
- `torchvision==0.21.0`
- `transformers==4.57.0.dev0`
- `deepspeed==0.17.1`
- `flash_attn==2.7.4.post1`
- `triton==3.2.0`
- `accelerate==1.7.0`
- `torchcodec==0.2`
- `peft==0.17.1`

## Custom Dataset Configuration

The customized data should have the format like:

### JSON Data Structure

**Media Specification**:
- `image/video`: Contains path to the media file (required)
- Media tags in prompts:
    - `<image>` for image understanding tasks
    - `<video>` for video understanding tasks
- `conversations`: contains the questions and answers

### Example Instances:

1. **Single Image Example**:
```json
{
    "image": "images/001.jpg",
    "conversations": [
        {
            "from": "human",
            "value": "<image>\nWhat's the main object in this picture?"
        },
        {
            "from": "gpt",
            "value": "A red apple on a wooden table"
        }
    ]
}
```

2. **Multi-Image Example**:
```json
{
    "image": ["cats/001.jpg", "cats/002.jpg"],
    "conversations": [
        {
            "from": "human",
            "value": "<image>\n<image>\nWhat are the differences between these two cats?"
        },
        {
            "from": "gpt",
            "value": "The first cat is an orange tabby with short fur and green eyes, while the second is a gray Siamese with blue eyes and pointed coloration. They also appear to be in different environments - the first is indoors on a couch, the second is outdoors in a garden."
        }
    ]
}
```

3. **Video Example**:
```json
{
    "video": "videos/005.mp4",
    "conversations": [
        {
            "from": "human",
            "value": "<video>\nWhat caused the blue object to move?\nOptions:\n(A) Gravity\n(B) Collision\n(C) Magnetic force"
        },
        {
            "from": "gpt",
            "value": "Answer: (B) Collision"
        }
    ]
}
```

4. **Grounding Example**:
```json
{
    "image": "demo/COCO_train2014_000000580957.jpg",
    "conversations": [
        {
            "from": "human",
            "value": "<image>\nLocate house in this image and output the bbox coordinates in JSON format."
        },
        {
            "from": "gpt",
            "value": "{\n"bbox_2d": [135, 114, 1016, 672]\n}"
        }
    ]
}
```

5. **Packed Data Example**:
```json
[
    {
        "image": "images/001.jpg",
        "conversations": [
            {
                "from": "human",
                "value": "<image>\nWhat's the main object in this picture?"
            },
            {
                "from": "gpt",
                "value": "A red apple on a wooden table"
            }
        ]
    },
    {
        "image": "images/002.jpg",
        "conversations": [
            {
                "from": "human",
                "value": "<image>\nWhat's the main object in this picture?"
            },
            {
                "from": "gpt",
                "value": "A green orange on a plastic table"
            }
        ]
    }
]
```

Some examples are shown in `demo/single_images.json` and `demo/video.json` and these json files could be used for training.

### Dataset config for training

To add or modify datasets for training, follow these steps:

### Dataset Definition Structure

1. **Create a dataset dictionary** in the format in the file `data/__init__.py`:
```python
DATASET_NAME = {
    "annotation_path": "/path/to/annotations.json",
    "data_path": "/path/to/image/data",  # Can be empty if paths are in annotations
}
```

2. **Register your dataset** by adding it to the `data_dict`:
```python
data_dict = {
    "your_dataset_name": DATASET_NAME,
    # ... other datasets
}
```

### Sampling Rate Control

You can optionally specify sampling rates by appending `%X` to the dataset name:
- `"dataset_name%50"` will sample 50% of the data
- `"dataset_name%20"` will sample 20% of the data

### Usage Example

1. Define your dataset:
```python
MY_DATASET = {
    "annotation_path": "/data/my_dataset/annotations.json",
    "data_path": "/data/my_dataset/images/",
}

data_dict = {
    "my_dataset": MY_DATASET,
    "cambrian_737k": CAMBRIAN_737K,  # existing dataset
}
```

2. Use it in training:
```python
dataset_names = ["my_dataset%50"]  # Will use 50% of your dataset
configs = data_list(dataset_names)
```

### Notes  
- The `annotation_path` should point to a JSON or JSONL file containing your dataset annotations.  
- The `data_path` can be left empty if the image paths in the annotations are absolute.  
- Sampling rates are applied per-dataset when multiple datasets are specified.  
- Some datasets you can use directly: `nyu-visionx/Cambrian-10M`, `lmms-lab/LLaVA-NeXT-Data`, `FreedomIntelligence/ALLaVA-4V`, `TIGER-Lab/VisualWebInstruct`.  
- The training data should strictly follow this format:  
  - One `<image>` tag in the question must correspond to exactly one image file  
  - Similarly, `<video>` tags must correspond to video files  
  - These special tokens should not appear in the answer text  
- For open source data that might have missing images or other issues, you can verify data completeness using `tools/check_image.py`.  


## Usage

To train a model:

```bash
#!/bin/bash
# Complete QwenVL Training Launch Script with Full Parameter Documentation

# ======================
# Distributed Configuration
# ======================
MASTER_ADDR="127.0.0.1"                     # [Required] Master node IP for multi-GPU training
MASTER_PORT=$(shuf -i 20000-29999 -n 1)     # Random port to avoid conflicts
NPROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)  # Automatically detects available GPUs

# ======================
# Path Configuration
# ======================
MODEL_PATH="/path/to/Qwen2.5-VL-3B-Instruct"  # [ModelArguments] Pretrained model path
OUTPUT_DIR="./checkpoints"                   # Directory for saving checkpoints
CACHE_DIR="./cache"                          # [TrainingArguments] Cache directory for models

# ======================
# Model Configuration
# ======================
DATASETS="your_dataset%100"                  # [DataArguments] Dataset with sampling rate

# ======================
# Training Hyperparameters
# ======================
torchrun --nproc_per_node=$NPROC_PER_NODE \
         --master_addr=$MASTER_ADDR \
         --master_port=$MASTER_PORT \
         qwenvl/train/train_qwen.py \
         # Core Arguments
         --model_name_or_path $MODEL_PATH \  # [ModelArguments] Model identifier
         --tune_mm_llm True \                # [TrainingArguments] Train LLM or not
         --tune_mm_vision False \            # [TrainingArguments] Train VIT or not
         --tune_mm_mlp False \               # [TrainingArguments] Train MLP or not
         --dataset_use $DATASETS \           # [DataArguments] Dataset specification
         --output_dir $OUTPUT_DIR \          # Output directory for checkpoints
         --cache_dir $CACHE_DIR \            # [TrainingArguments] Model cache location
         
         # Precision & Memory
         --bf16 \                            # Use bfloat16 precision (Ampere+ GPUs)
         --per_device_train_batch_size 4 \   # Batch size per GPU
         --gradient_accumulation_steps 4 \   # Effective batch size multiplier
         
         # Learning Rate Configuration
         --learning_rate 2e-7 \              # Base learning rate
         --mm_projector_lr 1e-5 \            # [TrainingArguments] Projector-specific LR
         --vision_tower_lr 1e-6 \            # [TrainingArguments] Vision encoder LR
         --optim adamw_torch \               # [TrainingArguments] Optimizer selection
         
         # Sequence Configuration
         --model_max_length 4096 \           # [TrainingArguments] Max sequence length
         --data_flatten True \               # [DataArguments] Concatenate batch sequences
         --data_packing True \               # [DataArguments] Using packing data
         
         # Image Processing
         --max_pixels 576\*28\*28 \               # [DataArguments] Max image pixels (H*W) for image
         --min_pixels 16\*28\*28 \                # [DataArguments] Min image pixels for image
         # Video Processing
         --video_fps 2 \                          # [DataArguments] video fps
         --video_max_frames 8 \                   # [DataArguments] Max frames per video
         --video_min_frames 4 \                   # [DataArguments] Min frames per video
         --video_max_pixels 1664\*28\*28 \        # [DataArguments] Max pixels per video
         --video_min_pixels 256\*28\*28 \         # [DataArguments] Min pixels per video
         
         # Training Schedule
         --num_train_epochs 3 \              # Total training epochs
         --warmup_ratio 0.03 \               # LR warmup proportion
         --lr_scheduler_type "cosine" \      # Learning rate schedule
         --weight_decay 0.01 \               # L2 regularization strength
         
         # Logging & Checkpoints
         --logging_steps 10 \               # Log metrics interval
         --save_steps 500 \                 # Checkpoint save interval
         --save_total_limit 3 \             # Max checkpoints to keep

         # Lora Config
         --lora_enable True \                 # [TrainingArguments] Enable LoRA
         --lora_r 8 \                         # [TrainingArguments] LoRA r
         --lora_alpha 16 \                    # [TrainingArguments] LoRA alpha 
         --lora_dropout 0.0 \                # [TrainingArguments] LoRA dropout

         # Advanced Options
         --deepspeed zero3.json \           # DeepSpeed configuration
```

The script accepts arguments in three categories:

   - Flags to control which components to tune (`tune_mm_vision`, `tune_mm_mlp`, `tune_mm_llm`). If trained with both image and video data, tune_mm_vision should be False: `tune_mm_vision=False`
   - `data_flatten` flag means data in a batch are concat into one sequence
   - `data_packing` requires preprocess with `tools/pack_data.py`
   - Training hyperparameters, the suggested learning rate is from 1e-6 to 2e-7
   - Training resolution is critical for the model performances, hence `--max_pixels` and `--min_pixels` should be properly set
   - Training with Qwen2.5-VL-32B model, you should have 8 80G GPU refering to `scripts/sft_32b.sh`
   - `"_attn_implementation": "flash_attention_2",` could be add in the config.json of the model to use flash attention.
   - The Qwen3VL MoE model does not support DeepSpeed with ZeRO-3. Additionally, Hugging Face’s official implementation does not include support for load balancing loss currently.

