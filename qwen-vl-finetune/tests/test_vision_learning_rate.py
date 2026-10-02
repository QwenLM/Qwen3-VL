"""Independent vision learning rates must survive optimizer construction."""
import importlib
import itertools
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration, Trainer, TrainingArguments


# Import the complete native trainer without requiring the CUDA FlashAttention
# extension: these optimizer tests never call its attention entry point.
flash_interface = ModuleType("flash_attn.flash_attn_interface")
flash_interface.flash_attn_varlen_func = lambda *args, **kwargs: None
previous = sys.modules.get(flash_interface.__name__)
sys.modules[flash_interface.__name__] = flash_interface
try:
    trainer = importlib.import_module("qwenvl.train.trainer")
finally:
    if previous is None:
        del sys.modules[flash_interface.__name__]
    else:
        sys.modules[flash_interface.__name__] = previous


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = torch.nn.Linear(2, 2)
        self.visual = torch.nn.Module()
        self.visual.blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2)])
        self.visual.merger = torch.nn.Linear(2, 2)


def make_optimizer(vision_lr, projector_lr, frozen=False):
    model = TinyModel()
    if frozen:
        model.visual.blocks[0].bias.requires_grad_(False)
    context = SimpleNamespace(
        model=model, optimizer=None,
        args=SimpleNamespace(
            vision_tower_lr=vision_lr, mm_projector_lr=projector_lr,
            weight_decay=0.2,
        ),
        get_decay_parameter_names=lambda model: [
            name for name, parameter in model.named_parameters() if parameter.ndim == 2
        ],
    )
    with patch.object(Trainer, "get_optimizer_cls_and_kwargs",
                      return_value=(torch.optim.SGD, {"lr": 0.01})):
        optimizer = trainer.create_optimizer(context)
    return model, optimizer


class VisionLearningRateTest(unittest.TestCase):
    def assert_rates(self, vision_lr, projector_lr, frozen=False):
        model, optimizer = make_optimizer(vision_lr, projector_lr, frozen)
        by_id = {}
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                self.assertNotIn(id(parameter), by_id)
                by_id[id(parameter)] = group
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                self.assertNotIn(id(parameter), by_id)
                continue
            group = by_id[id(parameter)]
            expected_lr = 0.01
            if "merger" in name and projector_lr:
                expected_lr = projector_lr
            elif "visual" in name and "merger" not in name and vision_lr:
                expected_lr = vision_lr
            self.assertEqual(group["lr"], expected_lr, name)
            self.assertEqual(group["weight_decay"], 0.2 if parameter.ndim == 2 else 0.0, name)
            parameter.data.fill_(1)
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                group = by_id[id(parameter)]
                expected = 1 - group["lr"] * (1 + group["weight_decay"])
                torch.testing.assert_close(parameter, torch.full_like(parameter, expected))

    def test_vision_override_without_projector_override(self):
        for projector_lr in (None, 0.0):
            with self.subTest(projector_lr=projector_lr):
                self.assert_rates(0.002, projector_lr)

    def test_separate_vision_and_projector_overrides(self):
        self.assert_rates(0.002, 0.003)

    def test_projector_only_and_default_rates(self):
        self.assert_rates(None, 0.003)
        self.assert_rates(None, None)
        self.assert_rates(0.0, 0.0)

    def test_frozen_parameters_are_excluded(self):
        self.assert_rates(0.002, None, frozen=True)


def make_native_model():
    config = Qwen3VLConfig(
        text_config=dict(
            vocab_size=64, hidden_size=32, intermediate_size=48,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
            head_dim=8, rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2]},
        ),
        vision_config=dict(
            depth=1, hidden_size=16, intermediate_size=32, num_heads=2,
            patch_size=2, temporal_patch_size=1, spatial_merge_size=2,
            out_hidden_size=32, num_position_embeddings=16, deepstack_visual_indexes=[0],
        ),
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    return Qwen3VLForConditionalGeneration(config)


def make_native_trainer(model, output_dir, projector_lr, vision_lr, weight_decay=0.1):
    args = TrainingArguments(
        output_dir=output_dir, report_to="none", use_cpu=True, optim="sgd",
        learning_rate=0.05, weight_decay=weight_decay,
    )
    args.mm_projector_lr = projector_lr
    args.vision_tower_lr = vision_lr
    return Trainer(model=model, args=args)


class NativeComponentLearningRateTest(unittest.TestCase):
    def test_native_qwen_rates_and_deepstack_mergers(self):
        with tempfile.TemporaryDirectory() as directory:
            for projector_lr, vision_lr in itertools.product((None, 0.0, 0.02), (None, 0.0, 0.005)):
                with self.subTest(projector_lr=projector_lr, vision_lr=vision_lr):
                    model = make_native_model()
                    owner = make_native_trainer(model, directory, projector_lr, vision_lr)
                    optimizer = trainer.create_optimizer(owner)
                    groups = {id(p): g for g in optimizer.param_groups for p in g["params"]}
                    parameters = [p for group in optimizer.param_groups for p in group["params"]]
                    self.assertEqual(len(groups), len(parameters))
                    decay = {name for name in owner.get_decay_parameter_names(model) if "bias" not in name}
                    self.assertTrue(any("deepstack_merger" in name for name, _ in model.named_parameters()))
                    for name, parameter in model.named_parameters():
                        rate = 0.05
                        if "merger" in name and projector_lr:
                            rate = projector_lr
                        elif "visual" in name and "merger" not in name and vision_lr:
                            rate = vision_lr
                        self.assertEqual(groups[id(parameter)]["lr"], rate, name)
                        self.assertEqual(groups[id(parameter)]["weight_decay"], 0.1 if name in decay else 0.0, name)

    def test_native_sgd_step_follows_requested_vision_rate(self):
        with tempfile.TemporaryDirectory() as directory:
            for projector_lr in (None, 0.0, 0.02):
                with self.subTest(projector_lr=projector_lr):
                    model = make_native_model()
                    owner = make_native_trainer(model, directory, projector_lr, 0.005, weight_decay=0.0)
                    optimizer = trainer.create_optimizer(owner)
                    parameter = model.visual.blocks[0].attn.qkv.weight
                    before = parameter.detach().clone()
                    parameter.grad = torch.ones_like(parameter)
                    optimizer.step()
                    torch.testing.assert_close(parameter, before - 0.005)

    def test_native_group_partition_for_all_trainability_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            for flags in itertools.product((False, True), repeat=3):
                for projector_lr, vision_lr in ((None, 0.005), (0.02, None), (0.02, 0.005)):
                    with self.subTest(flags=flags, projector_lr=projector_lr, vision_lr=vision_lr):
                        model = make_native_model()
                        vision, projector, language = flags
                        for name, parameter in model.named_parameters():
                            if "merger" in name:
                                parameter.requires_grad_(projector)
                            elif "visual" in name:
                                parameter.requires_grad_(vision)
                            else:
                                parameter.requires_grad_(language)
                        owner = make_native_trainer(model, directory, projector_lr, vision_lr)
                        optimizer = trainer.create_optimizer(owner)
                        parameters = [p for group in optimizer.param_groups for p in group["params"]]
                        self.assertEqual(len({id(p) for p in parameters}), len(parameters))
                        self.assertEqual(
                            {id(p) for p in parameters},
                            {id(p) for p in model.parameters() if p.requires_grad},
                        )

    def test_native_optimizer_reuse_preserves_existing_instance(self):
        with tempfile.TemporaryDirectory() as directory:
            owner = make_native_trainer(make_native_model(), directory, None, 0.005)
            first = trainer.create_optimizer(owner)
            owner.args.vision_tower_lr = 0.03
            self.assertIs(trainer.create_optimizer(owner), first)


if __name__ == "__main__":
    unittest.main()


