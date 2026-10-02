"""CPU regressions for the fine-tuning component selection.

Load only the exact set_model/create_optimizer definitions because importing the
training entrypoint also imports the CUDA-only FlashAttention implementation.
All models, parameters, autograd operations and optimizer groups below are native.
"""

import ast
import itertools
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import (
    Qwen2VLConfig,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLConfig,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLConfig,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeConfig,
    Qwen3VLMoeForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

ROOT = Path(os.environ.get("QWEN_SOURCE_ROOT", Path(__file__).resolve().parents[2]))


def load_function(path, name, globals_dict=None):
    tree = ast.parse(path.read_text())
    node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    namespace = {} if globals_dict is None else dict(globals_dict)
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace
    )
    return namespace[name]


set_model = load_function(
    ROOT / "qwen-vl-finetune/qwenvl/train/train_qwen.py", "set_model"
)
create_optimizer = load_function(
    ROOT / "qwen-vl-finetune/qwenvl/train/trainer.py",
    "create_optimizer",
    {"Trainer": Trainer},
)


def tiny_model(family):
    text_config = dict(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=32,
        rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2]},
    )
    vision_config = dict(
        depth=1,
        hidden_size=16,
        embed_dim=16,
        intermediate_size=32,
        num_heads=2,
        patch_size=2,
        temporal_patch_size=1,
        spatial_merge_size=2,
        out_hidden_size=32,
        num_position_embeddings=16,
        deepstack_visual_indexes=[],
        fullatt_block_indexes=[0],
    )
    if family == "qwen3_moe":
        text_config.update(
            moe_intermediate_size=16, num_experts=2, num_experts_per_tok=1
        )
    if family == "qwen2":
        vision_config["hidden_size"] = 32
    classes = {
        "qwen2": (Qwen2VLConfig, Qwen2VLForConditionalGeneration),
        "qwen2_5": (Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration),
        "qwen3": (Qwen3VLConfig, Qwen3VLForConditionalGeneration),
        "qwen3_moe": (Qwen3VLMoeConfig, Qwen3VLMoeForConditionalGeneration),
    }
    config_cls, model_cls = classes[family]
    config = config_cls(
        text_config=text_config,
        vision_config=vision_config,
        tie_word_embeddings=False,
        image_token_id=60,
        video_token_id=61,
        vision_start_token_id=58,
        vision_end_token_id=59,
    )
    config._attn_implementation = "eager"
    return model_cls(config)


FAMILIES = ["qwen2", "qwen2_5", "qwen3", "qwen3_moe"]


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize(
    "vision,projector,llm", list(itertools.product([False, True], repeat=3))
)
def test_component_flags_control_head_parameters(family, vision, projector, llm):
    model = tiny_model(family)
    args = SimpleNamespace(
        tune_mm_vision=vision, tune_mm_mlp=projector, tune_mm_llm=llm
    )
    set_model(args, model)
    assert all(p.requires_grad == llm for p in model.lm_head.parameters())
    assert all(p.requires_grad == llm for p in model.language_model.parameters())
    assert all(p.requires_grad == projector for p in model.visual.merger.parameters())
    assert all(p.requires_grad == vision for p in model.visual.blocks.parameters())


@pytest.mark.parametrize("family", FAMILIES)
def test_enabling_llm_restores_previously_frozen_head(family):
    model = tiny_model(family)
    model.requires_grad_(False)
    set_model(
        SimpleNamespace(tune_mm_vision=False, tune_mm_mlp=False, tune_mm_llm=True),
        model,
    )
    assert all(p.requires_grad for p in model.lm_head.parameters())


@pytest.mark.parametrize("family", FAMILIES)
def test_frozen_head_preserves_feature_gradients_and_weights(family):
    torch.manual_seed(123)
    model = tiny_model(family)
    set_model(
        SimpleNamespace(tune_mm_vision=False, tune_mm_mlp=True, tune_mm_llm=False),
        model,
    )
    features = torch.randn(2, 3, 32, requires_grad=True)
    weights_before = model.lm_head.weight.detach().clone()
    grad_logits = torch.randn(2, 3, 64)
    loss = (model.lm_head(features) * grad_logits).sum()
    loss.backward()
    expected = grad_logits @ weights_before
    torch.testing.assert_close(features.grad, expected)
    assert features.grad.abs().sum() > 0
    assert model.lm_head.weight.grad is None
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=0.01, weight_decay=0.1
    )
    optimizer.step()
    torch.testing.assert_close(model.lm_head.weight, weights_before, rtol=0, atol=0)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("llm", [False, True])
def test_native_trainer_optimizer_contains_head_only_when_enabled(
    family, llm, tmp_path
):
    model = tiny_model(family)
    set_model(
        SimpleNamespace(tune_mm_vision=False, tune_mm_mlp=True, tune_mm_llm=llm), model
    )
    training_args = TrainingArguments(
        output_dir=str(tmp_path), report_to="none", use_cpu=True, optim="adamw_torch"
    )
    training_args.mm_projector_lr = 0.001
    training_args.vision_tower_lr = None
    trainer = Trainer(model=model, args=training_args)
    optimizer = create_optimizer(trainer)
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    assert any(p is model.lm_head.weight for p in parameters) == llm
    assert len({id(p) for p in parameters}) == len(parameters)
    assert {id(p) for p in parameters} == {
        id(p) for p in model.parameters() if p.requires_grad
    }


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("llm", [False, True])
def test_native_multimodal_backward_and_optimizer_step(family, llm):
    torch.manual_seed(123)
    model = tiny_model(family)
    model.train()
    set_model(
        SimpleNamespace(tune_mm_vision=False, tune_mm_mlp=True, tune_mm_llm=llm), model
    )
    weights_before = model.lm_head.weight.detach().clone()
    output = model(
        input_ids=torch.tensor([[58, 60, 59, 7]]),
        labels=torch.tensor([[-100, -100, -100, 7]]),
        pixel_values=torch.randn(4, 12),
        image_grid_thw=torch.tensor([[1, 2, 2]]),
    )
    output.loss.backward()
    assert torch.isfinite(output.loss)
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.visual.merger.parameters()
    )
    assert (model.lm_head.weight.grad is not None) == llm
    assert all(p.grad is None for p in model.visual.blocks.parameters())
    if not llm:
        assert all(p.grad is None for p in model.language_model.parameters())
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=0.01
    )
    optimizer.step()
    assert torch.equal(model.lm_head.weight, weights_before) == (not llm)
