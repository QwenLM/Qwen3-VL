# 图像分类训练脚本
# 输入：图片、文本指令、标签类别
# 输出：第一个token的隐藏层 -> 线性层 -> 类别概率
# 损失：交叉熵

import os
import logging
import pathlib
import torch
import torch.nn as nn
import transformers
import sys
from pathlib import Path
from typing import Dict, Optional, List
from dataclasses import dataclass, field

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from trainer import replace_qwen2_vl_attention_class
from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration,
    AutoProcessor,
    Trainer,
    TrainingArguments as HFTrainingArguments,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers import TrainerCallback
from qwenvl.train.argument import ModelArguments, DataArguments

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


@dataclass
class ClassificationArguments:
    """分类任务的额外参数"""
    num_classes: int = field(default=200, metadata={"help": "类别数量"})
    classifier_dropout: float = field(default=0.1, metadata={"help": "分类器dropout率"})
    eval_dataset_path: Optional[str] = field(default=None, metadata={"help": "测试集路径（可选）"})


class ClassificationHead(nn.Module):
    """分类头：将隐藏层映射到类别数"""
    
    def __init__(self, hidden_size: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: (batch_size, hidden_size)
        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)
        return logits


class QwenVLForClassification(nn.Module):
    """Qwen-VL模型 + 分类头"""
    
    def __init__(self, base_model, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.base_model = base_model
        # 获取隐藏层维度
        if hasattr(base_model.config, 'hidden_size'):
            hidden_size = base_model.config.hidden_size
        elif hasattr(base_model.config, 'd_model'):
            hidden_size = base_model.config.d_model
        else:
            # 尝试从语言模型获取
            if hasattr(base_model, 'language_model'):
                hidden_size = base_model.language_model.config.hidden_size
            else:
                raise ValueError("无法确定隐藏层维度")
        
        self.classifier = ClassificationHead(hidden_size, num_classes, dropout)
        
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs
    ):
        # 前向传播获取隐藏状态
        outputs = self.base_model.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
            **kwargs
        )
        
        # 获取第一个token的隐藏状态
        # hidden_states shape: (batch_size, seq_len, hidden_size)
        hidden_states = outputs.hidden_states[-1]  # 最后一层的隐藏状态
        
        # 获取第一个token的隐藏状态
        # 方法1: 使用attention_mask找到第一个有效token
        if attention_mask is not None:
            # attention_mask: (batch_size, seq_len), 1表示有效token, 0表示padding
            # 找到每个样本第一个有效token的位置
            # 注意：argmax返回第一个非零位置，如果没有非零则返回0
            first_token_indices = attention_mask.argmax(dim=1)  # (batch_size,)
            batch_size = hidden_states.shape[0]
            first_token_hidden = hidden_states[torch.arange(batch_size), first_token_indices, :]
        else:
            # 如果没有attention_mask，取位置1（跳过可能的BOS token）
            # 或者取位置0（如果BOS token就是我们要用的）
            first_token_hidden = hidden_states[:, 0, :]  # (batch_size, hidden_size)
        
        # 通过分类头
        logits = self.classifier(first_token_hidden)  # (batch_size, num_classes)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': outputs.hidden_states,
        }


class ClassificationDataset(torch.utils.data.Dataset):
    """图像分类数据集"""
    
    def __init__(
        self,
        data_list: List[Dict],
        processor,
        tokenizer,
        model_type: str = "qwen2vl"
    ):
        self.data_list = data_list
        self.processor = processor
        self.tokenizer = tokenizer
        self.model_type = model_type
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        item = self.data_list[idx]
        
        # 获取图像路径
        image_path = item.get("image")
        if isinstance(image_path, str):
            image_path = [image_path]
        
        # 获取文本指令
        instruction = item.get("instruction", "")
        if not instruction:
            instruction = item.get("text", "")
        
        # 获取标签
        label = item.get("label", item.get("category_id", 0))
        label = int(label)
        
        # 构建消息
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img} if isinstance(img, str) else img
                    for img in image_path
                ] + [{"type": "text", "text": instruction}]
            }
        ]
        
        # 处理输入
        inputs = self.processor(
            messages,
            tokenize=True,
            return_tensors="pt",
            padding=False
        )
        
        # 转换为单样本格式
        for key in inputs:
            if isinstance(inputs[key], torch.Tensor):
                if inputs[key].dim() > 1:
                    inputs[key] = inputs[key].squeeze(0)
                else:
                    inputs[key] = inputs[key]
        
        # 如果没有position_ids，需要生成（可选，模型内部会处理）
        # 这里我们依赖rope2d的处理，所以暂时不在这里生成
        
        # 添加标签
        inputs["labels"] = torch.tensor(label, dtype=torch.long)
        
        # 添加数据路径（如果需要）
        if "data_path" in item:
            inputs["data_path"] = item["data_path"]
        
        return inputs


def collate_fn(examples, processor):
    """整理批次数据"""
    # 分离不同类型的输入
    input_ids_list = []
    pixel_values_list = []
    pixel_values_videos_list = []
    image_grid_thw_list = []
    video_grid_thw_list = []
    labels_list = []
    position_ids_list = []
    
    for example in examples:
        input_ids_list.append(example["input_ids"])
        labels_list.append(example["labels"])
        
        if "pixel_values" in example:
            pixel_values_list.append(example["pixel_values"])
        if "image_grid_thw" in example:
            image_grid_thw_list.append(example["image_grid_thw"])
        if "pixel_values_videos" in example:
            pixel_values_videos_list.append(example["pixel_values_videos"])
        if "video_grid_thw" in example:
            video_grid_thw_list.append(example["video_grid_thw"])
        if "position_ids" in example:
            position_ids_list.append(example["position_ids"])
    
    # Padding input_ids
    max_length = max(len(ids) for ids in input_ids_list)
    pad_token_id = processor.tokenizer.pad_token_id
    
    padded_input_ids = []
    padded_attention_mask = []
    for ids in input_ids_list:
        pad_length = max_length - len(ids)
        padded_ids = torch.cat([ids, torch.full((pad_length,), pad_token_id, dtype=ids.dtype)])
        padded_input_ids.append(padded_ids)
        # 创建attention_mask: 1表示有效token, 0表示padding
        mask = torch.cat([torch.ones(len(ids), dtype=torch.long), torch.zeros(pad_length, dtype=torch.long)])
        padded_attention_mask.append(mask)
    
    batch = {
        "input_ids": torch.stack(padded_input_ids),
        "labels": torch.stack(labels_list),
        "attention_mask": torch.stack(padded_attention_mask),
    }
    
    if pixel_values_list:
        batch["pixel_values"] = torch.cat(pixel_values_list, dim=0)
    if image_grid_thw_list:
        batch["image_grid_thw"] = torch.cat(image_grid_thw_list, dim=0)
    if pixel_values_videos_list:
        batch["pixel_values_videos"] = torch.cat(pixel_values_videos_list, dim=0)
    if video_grid_thw_list:
        batch["video_grid_thw"] = torch.cat(video_grid_thw_list, dim=0)
    if position_ids_list:
        # 处理position_ids的padding（如果需要）
        # 这里简化处理，实际可能需要更复杂的padding逻辑
        batch["position_ids"] = torch.stack(position_ids_list) if len(position_ids_list) == len(examples) else None
    
    return batch


class EvalCallback(TrainerCallback):
    """在每个epoch结束后进行评估的回调"""
    
    def __init__(self, eval_dataset=None):
        self.eval_dataset = eval_dataset
    
    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        """在每个epoch结束时进行评估"""
        if self.eval_dataset is not None and model is not None:
            trainer = kwargs.get('trainer')
            if trainer is not None:
                rank0_print(f"\n=== Epoch {state.epoch} 结束，开始评估测试集 ===")
                trainer.evaluate(eval_dataset=self.eval_dataset)
                rank0_print("=== 评估完成 ===\n")


class ClassificationTrainer(Trainer):
    """自定义Trainer用于分类任务"""
    
    def __init__(self, eval_dataset=None, save_eval_results=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_dataset = eval_dataset
        self.save_eval_results = save_eval_results
        self.eval_results_history = []  # 保存每个epoch的评估结果
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss
    
    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        """评估模型并保存结果"""
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if eval_dataset is None:
            return {}
        
        # 调用父类的evaluate方法
        metrics = super().evaluate(eval_dataset=eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
        
        # 进行推理并保存详细结果
        if self.save_eval_results and (self.args.local_rank == -1 or self.args.local_rank == 0):
            self._save_eval_predictions(eval_dataset, metric_key_prefix)
        
        # 保存评估指标历史
        if self.args.local_rank == -1 or self.args.local_rank == 0:
            epoch = self.state.epoch if hasattr(self.state, 'epoch') else len(self.eval_results_history)
            self.eval_results_history.append({
                'epoch': epoch,
                'metrics': metrics
            })
        
        return metrics
    
    def _save_eval_predictions(self, eval_dataset, metric_key_prefix="eval"):
        """保存测试集的预测结果"""
        import json
        import numpy as np
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_logits = []
        all_probs = []
        
        # 创建数据加载器
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        
        with torch.no_grad():
            for step, inputs in enumerate(eval_dataloader):
                # 移动到设备
                inputs = self._prepare_inputs(inputs)
                labels = inputs.pop("labels")
                
                # 推理
                outputs = self.model(**inputs)
                logits = outputs["logits"]
                
                # 计算概率
                probs = torch.softmax(logits, dim=-1)
                predictions = logits.argmax(dim=-1)
                
                # 收集结果
                all_predictions.extend(predictions.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())
                all_logits.extend(logits.cpu().numpy().tolist())
                all_probs.extend(probs.cpu().numpy().tolist())
        
        # 获取当前epoch
        if hasattr(self.state, 'epoch') and self.state.epoch is not None:
            epoch = int(self.state.epoch)
        else:
            # 如果没有epoch信息，使用历史记录长度+1
            epoch = len(self.eval_results_history) + 1
        
        # 准备保存的结果
        results = {
            'epoch': epoch,
            'predictions': all_predictions,
            'labels': all_labels,
            'logits': all_logits,
            'probabilities': all_probs,
            'num_samples': len(all_predictions)
        }
        
        # 计算准确率
        correct = sum(p == l for p, l in zip(all_predictions, all_labels))
        accuracy = correct / len(all_predictions) if len(all_predictions) > 0 else 0.0
        results['accuracy'] = accuracy
        
        # 保存结果到文件
        output_dir = Path(self.args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        results_file = output_dir / f"eval_results_epoch_{epoch}.json"
        
        # 将numpy数组转换为列表以便JSON序列化
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        results = convert_numpy(results)
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        rank0_print(f"Epoch {epoch} 评估完成，准确率: {accuracy:.4f}, 结果已保存到: {results_file}")
        
        self.model.train()


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """保存模型"""
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return
    
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


def load_data(data_path: str):
    """加载数据"""
    import json
    from pathlib import Path
    
    data_path = Path(data_path)
    if data_path.suffix == ".jsonl":
        data_list = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                data_list.append(json.loads(line))
    else:
        with open(data_path, "r", encoding="utf-8") as f:
            data_list = json.load(f)
    
    return data_list


def train():
    global local_rank
    
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, HFTrainingArguments, ClassificationArguments)
    )
    model_args, data_args, training_args, classification_args = parser.parse_args_into_dataclasses()
    
    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)
    
    # 加载模型
    if "qwen3" in model_args.model_name_or_path.lower() and "a" in Path(model_args.model_name_or_path.rstrip("/")).name.lower():
        base_model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation="flash_attention_2",
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        model_type = "qwen3vl"
    elif "qwen3" in model_args.model_name_or_path.lower():
        base_model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation="flash_attention_2",
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        model_type = "qwen3vl"
    elif "qwen2.5" in model_args.model_name_or_path.lower():
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation="flash_attention_2",
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        model_type = "qwen2.5vl"
    else:
        base_model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation="flash_attention_2",
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        model_type = "qwen2vl"
    
    rank0_print(f"加载模型: {model_args.model_name_or_path}, 类型: {base_model.__class__.__name__}")
    
    # 创建分类模型
    model = QwenVLForClassification(
        base_model,
        num_classes=classification_args.num_classes,
        dropout=classification_args.classifier_dropout
    )
    
    # 冻结基础模型（可选，根据需求调整）
    if not model_args.tune_mm_llm and not model_args.tune_mm_vision and not model_args.tune_mm_mlp:
        rank0_print("冻结基础模型，只训练分类头")
        for param in model.base_model.parameters():
            param.requires_grad = False
        # 只训练分类头
        for param in model.classifier.parameters():
            param.requires_grad = True
    else:
        # 根据参数设置哪些部分可训练
        if model_args.tune_mm_vision:
            for param in model.base_model.visual.parameters():
                param.requires_grad = True
        else:
            for param in model.base_model.visual.parameters():
                param.requires_grad = False
        
        if model_args.tune_mm_mlp:
            for param in model.base_model.visual.merger.parameters():
                param.requires_grad = True
        else:
            for param in model.base_model.visual.merger.parameters():
                param.requires_grad = False
        
        if model_args.tune_mm_llm:
            for param in model.base_model.language_model.parameters():
                param.requires_grad = True
        else:
            for param in model.base_model.language_model.parameters():
                param.requires_grad = False
        
        # 分类头始终可训练
        for param in model.classifier.parameters():
            param.requires_grad = True
    
    # 加载processor
    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path)
    tokenizer = processor.tokenizer
    
    model.config.use_cache = False
    
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.base_model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    
    # 加载训练数据
    rank0_print(f"加载训练数据: {data_args.dataset_use}")
    train_data_list = load_data(data_args.dataset_use)
    
    # 创建训练数据集
    train_dataset = ClassificationDataset(
        train_data_list,
        processor,
        tokenizer,
        model_type=model_type
    )
    
    # 加载测试集（如果提供）
    eval_dataset = None
    callbacks = []
    if classification_args.eval_dataset_path:
        rank0_print(f"加载测试集: {classification_args.eval_dataset_path}")
        eval_data_list = load_data(classification_args.eval_dataset_path)
        eval_dataset = ClassificationDataset(
            eval_data_list,
            processor,
            tokenizer,
            model_type=model_type
        )
        # 添加评估回调
        callbacks.append(EvalCallback(eval_dataset=eval_dataset))
        rank0_print(f"测试集大小: {len(eval_dataset)}")
    else:
        rank0_print("未提供测试集路径，将跳过评估")
    
    # 创建数据整理函数
    def data_collator(examples):
        return collate_fn(examples, processor)
    
    # 创建Trainer
    trainer = ClassificationTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        callbacks=callbacks,
    )
    
    # 训练
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        rank0_print("发现checkpoint，继续训练")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    
    trainer.save_state()
    model.config.use_cache = True
    
    # 保存模型
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)
    
    rank0_print(f"训练完成，模型已保存到: {training_args.output_dir}")


if __name__ == "__main__":
    train()
