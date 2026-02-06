# Qwen-VL 图像分类训练指南

本指南介绍如何使用 Qwen-VL 进行图像分类任务的训练。

## 功能说明

- **输入**: 图片、文本指令、标签类别（0 到 num_classes-1）
- **输出**: Qwen-VL 模型第一个 token 的隐藏层，通过线性层压缩到类别数维度
- **损失函数**: 交叉熵损失

## 数据格式

数据文件应为 JSONL 格式，每行一个样本：

```json
{"image": "path/to/image.jpg", "instruction": "请识别这张图片的类别", "label": 0, "data_path": "data/images"}
```

字段说明：
- `image`: 图片路径（字符串）或图片路径列表
- `instruction`: 文本指令（可选，默认为空字符串）
- `label`: 类别标签（整数，范围 0 到 num_classes-1）
- `data_path`: 数据根路径（可选，用于解析相对路径）

## 使用方法

### 1. 准备数据

创建训练数据文件 `data/train.jsonl`，格式如上所示。

### 2. 运行训练

```bash
bash scripts/train_classification.sh
```

或直接使用 Python：

```bash
python qwenvl/train/train_classification.py \
    --model_name_or_path "Qwen/Qwen2-VL-2B-Instruct" \
    --dataset_use "data/train.jsonl" \
    --output_dir "outputs/classification" \
    --num_classes 200 \
    --classifier_dropout 0.1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --num_train_epochs 10 \
    --bf16
```

### 3. 主要参数

#### 分类相关参数
- `--num_classes`: 类别数量（默认: 200）
- `--classifier_dropout`: 分类器 dropout 率（默认: 0.1）
- `--eval_dataset_path`: 测试集路径（可选，JSONL格式，与训练集格式相同）

#### 模型训练参数
- `--tune_mm_llm`: 是否微调语言模型（默认: False）
- `--tune_mm_vision`: 是否微调视觉编码器（默认: False）
- `--tune_mm_mlp`: 是否微调 MLP 投影层（默认: False）

如果所有 `tune_mm_*` 都为 False，则只训练分类头，基础模型冻结。

#### 训练参数
- `--per_device_train_batch_size`: 每个设备的批次大小
- `--gradient_accumulation_steps`: 梯度累积步数
- `--learning_rate`: 学习率
- `--num_train_epochs`: 训练轮数
- `--bf16`: 使用 bfloat16 精度

### 4. 测试集评估

如果提供了 `--eval_dataset_path` 参数，训练脚本会在**每个 epoch 结束后**自动对测试集进行评估，并保存详细的评估结果。

评估结果会保存在 `output_dir/eval_results_epoch_{epoch}.json` 文件中，包含：
- `predictions`: 预测的类别ID列表
- `labels`: 真实的类别ID列表
- `logits`: 模型输出的原始logits
- `probabilities`: 每个类别的概率
- `accuracy`: 准确率
- `num_samples`: 测试样本数量

示例：
```bash
python qwenvl/train/train_classification.py \
    --model_name_or_path "Qwen/Qwen2-VL-2B-Instruct" \
    --dataset_use "data/train.jsonl" \
    --eval_dataset_path "data/test.jsonl" \
    --output_dir "outputs/classification" \
    --num_classes 200 \
    --per_device_train_batch_size 4 \
    --num_train_epochs 10 \
    --bf16
```

训练过程中，每个epoch结束后会输出类似以下信息：
```
=== Epoch 1 结束，开始评估测试集 ===
Epoch 1 评估完成，准确率: 0.8523, 结果已保存到: outputs/classification/eval_results_epoch_1.json
=== 评估完成 ===
```

## 模型结构

```
输入 (图片 + 文本指令)
    ↓
Qwen-VL 基础模型
    ↓
第一个 token 的隐藏状态 (batch_size, hidden_size)
    ↓
Dropout
    ↓
线性层 (hidden_size → num_classes)
    ↓
输出 logits (batch_size, num_classes)
    ↓
交叉熵损失
```

## 注意事项

1. **第一个 token 的选择**: 代码中取位置 1 的隐藏状态（跳过可能的 BOS token）。如果您的模型结构不同，可能需要调整 `train_classification.py` 中的索引。

2. **内存优化**: 
   - 使用 `--gradient_checkpointing` 启用梯度检查点
   - 使用 `--bf16` 或 `--fp16` 降低精度
   - 调整 `--per_device_train_batch_size` 和 `--gradient_accumulation_steps`

3. **多 GPU 训练**: 使用 `torch.distributed.launch` 或 `accelerate` 进行多 GPU 训练。

4. **数据路径**: 确保图片路径正确，如果使用相对路径，需要设置 `data_path` 字段。

## 推理示例

训练完成后，可以使用以下代码进行推理：

```python
from transformers import AutoProcessor
from qwenvl.train.train_classification import QwenVLForClassification
import torch
from PIL import Image

# 加载模型
model = QwenVLForClassification.from_pretrained("outputs/classification")
processor = AutoProcessor.from_pretrained("outputs/classification")
model.eval()

# 准备输入
image = Image.open("path/to/image.jpg")
messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": "请识别这张图片的类别"}
    ]
}]

inputs = processor(messages, tokenize=True, return_tensors="pt")

# 推理
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs["logits"]
    predicted_class = logits.argmax(dim=-1).item()

print(f"预测类别: {predicted_class}")
```

## 故障排除

1. **CUDA 内存不足**: 减小批次大小或启用梯度检查点
2. **图片路径错误**: 检查 `data_path` 和图片路径是否正确
3. **标签超出范围**: 确保标签在 0 到 num_classes-1 之间
