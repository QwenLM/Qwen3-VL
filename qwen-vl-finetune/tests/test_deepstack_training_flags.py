"""CPU regressions for DeepStack projector training selection.

AST-load the exact production helpers to avoid the training entrypoint's
unrelated CUDA-only FlashAttention import. All models and optimizers are native.
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

ROOT = Path(
    os.environ.get("QWEN_DEEPSTACK_SOURCE_ROOT", Path(__file__).resolve().parents[2])
)


def load_function(path, name, globals_dict=None):
    tree = ast.parse(path.read_text())
    node = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name
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


FAMILIES = ["qwen2", "qwen2_5", "qwen3", "qwen3_moe"]
DEEPSTACK_FAMILIES = ["qwen3", "qwen3_moe"]


def tiny_model(family):
    text_config = dict(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=32,
        rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2]},
    )
    vision_config = dict(
        depth=2,
        hidden_size=16,
        embed_dim=16,
        intermediate_size=32,
        num_heads=2,
        patch_size=2,
        temporal_patch_size=1,
        spatial_merge_size=2,
        out_hidden_size=32,
        num_position_embeddings=16,
        deepstack_visual_indexes=[0, 1],
        fullatt_block_indexes=[0, 1],
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


def flags(vision, projector, llm=True):
    return SimpleNamespace(
        tune_mm_vision=vision, tune_mm_mlp=projector, tune_mm_llm=llm
    )


def projectors(model):
    return [model.visual.merger, *getattr(model.visual, "deepstack_merger_list", [])]


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize(
    "vision,projector,llm", list(itertools.product([False, True], repeat=3))
)
def test_projectors_follow_mlp_flag_independently_of_encoder(
    family, vision, projector, llm
):
    model = tiny_model(family)
    set_model(flags(vision, projector, llm), model)
    if family in DEEPSTACK_FAMILIES:
        assert len(model.visual.deepstack_merger_list) == 2
    for module in projectors(model):
        assert all(p.requires_grad == projector for p in module.parameters())
    assert all(p.requires_grad == vision for p in model.visual.blocks.parameters())


@pytest.mark.parametrize("family", DEEPSTACK_FAMILIES)
@pytest.mark.parametrize("vision", [False, True])
def test_projector_switch_restores_all_deepstack_parameters(family, vision):
    model = tiny_model(family)
    set_model(flags(vision, False), model)
    set_model(flags(vision, True), model)
    assert all(
        p.requires_grad for module in projectors(model) for p in module.parameters()
    )
    set_model(flags(vision, False), model)
    assert not any(
        p.requires_grad for module in projectors(model) for p in module.parameters()
    )


@pytest.mark.parametrize("family", DEEPSTACK_FAMILIES)
@pytest.mark.parametrize("vision,projector", [(False, True), (True, False)])
def test_multimodal_backward_and_step_obey_projector_flag(family, vision, projector):
    torch.manual_seed(123)
    model = tiny_model(family)
    model.train()
    set_model(flags(vision, projector), model)
    modules = projectors(model)
    before = [[p.detach().clone() for p in module.parameters()] for module in modules]
    output = model(
        input_ids=torch.tensor([[58, 60, 59, 7]]),
        labels=torch.tensor([[-100, -100, -100, 7]]),
        pixel_values=torch.randn(4, 12),
        image_grid_thw=torch.tensor([[1, 2, 2]]),
    )
    assert torch.isfinite(output.loss)
    output.loss.backward()
    for module in modules:
        assert (
            any(
                p.grad is not None and p.grad.abs().sum() > 0
                for p in module.parameters()
            )
            == projector
        )
        if not projector:
            assert all(p.grad is None for p in module.parameters())
    assert (
        any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.visual.blocks.parameters()
        )
        == vision
    )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=0.01
    )
    optimizer.step()
    for module, original in zip(modules, before):
        changed = any(
            not torch.equal(p, initial)
            for p, initial in zip(module.parameters(), original)
        )
        assert changed == projector


@pytest.mark.parametrize("family", DEEPSTACK_FAMILIES)
@pytest.mark.parametrize(
    "vision,projector", list(itertools.product([False, True], repeat=2))
)
def test_native_optimizer_includes_every_enabled_projector(
    family, vision, projector, tmp_path
):
    model = tiny_model(family)
    set_model(flags(vision, projector), model)
    args = TrainingArguments(
        output_dir=str(tmp_path), report_to="none", use_cpu=True, optim="adamw_torch"
    )
    args.mm_projector_lr = 0.001
    args.vision_tower_lr = 0.0001
    optimizer = create_optimizer(Trainer(model=model, args=args))
    groups = {id(p): group for group in optimizer.param_groups for p in group["params"]}
    assert len(groups) == sum(len(group["params"]) for group in optimizer.param_groups)
    for module in projectors(model):
        for parameter in module.parameters():
            assert (id(parameter) in groups) == projector
            if projector:
                assert groups[id(parameter)]["lr"] == 0.001
