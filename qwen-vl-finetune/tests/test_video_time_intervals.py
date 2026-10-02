"""Native processor/position regressions for video training samples.

Use a local synthetic vocabulary and in-memory videos with the real processor.
Only the file-loading/label-preprocessing boundary is replaced with its native
processor output; the complete dataset and RoPE modules are imported normally.
"""

import importlib
import os
import sys
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import (
    Qwen2TokenizerFast,
    Qwen2VLImageProcessor,
    Qwen2VLVideoProcessor,
    Qwen2_5_VLProcessor,
)

ROOT = Path(
    os.environ.get("QWEN_VIDEO_SOURCE_ROOT", Path(__file__).resolve().parents[2])
)
sys.path.insert(0, str(ROOT / "qwen-vl-finetune"))
data_processor = importlib.import_module("qwenvl.data.data_processor")
get_rope_index_25 = importlib.import_module("qwenvl.data.rope2d").get_rope_index_25


@pytest.fixture(scope="module")
def processor():
    vocab = {f"unused_{i}": i for i in range(151657)}
    tokens = {
        "<pad>": 0,
        "<unk>": 1,
        "text": 2,
        "<|vision_start|>": 151652,
        "<|vision_end|>": 151653,
        "<|image_pad|>": 151655,
        "<|video_pad|>": 151656,
    }
    for token, index in tokens.items():
        del vocab[f"unused_{index}"]
        vocab[token] = index
    tokenizer = Qwen2TokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(vocab, unk_token="<unk>")),
        pad_token="<pad>",
        unk_token="<unk>",
    )
    tokenizer.add_special_tokens({"additional_special_tokens": list(tokens)[3:]})
    return Qwen2_5_VLProcessor(
        image_processor=Qwen2VLImageProcessor(
            patch_size=2,
            temporal_patch_size=2,
            merge_size=2,
            min_pixels=16,
            max_pixels=256,
        ),
        video_processor=Qwen2VLVideoProcessor(
            patch_size=2,
            temporal_patch_size=2,
            merge_size=2,
            min_pixels=16,
            max_pixels=256,
            fps=2.0,
        ),
        tokenizer=tokenizer,
    )


def native_batch(processor, fps):
    text = "<|vision_start|><|video_pad|><|vision_end|>text" * len(fps)
    batch = dict(
        processor(
            text=text,
            videos=[torch.zeros(4, 3, 5, 5, dtype=torch.uint8) for _ in fps],
            fps=fps,
            return_tensors="pt",
            input_data_format="channels_first",
        )
    )
    batch["labels"] = torch.full_like(batch["input_ids"], -100)
    assert batch["video_grid_thw"].shape == (len(fps), 3)
    return batch


def dataset_item(monkeypatch, processor, batch):
    monkeypatch.setattr(
        data_processor, "preprocess_qwen_visual", lambda sources, proc: dict(batch)
    )
    dataset = data_processor.LazySupervisedDataset.__new__(
        data_processor.LazySupervisedDataset
    )
    dataset.processor = processor
    dataset.merge_size = 2
    dataset.get_rope_index = get_rope_index_25
    return dataset._get_item([{}])


def assert_intervals(result, intervals):
    # A 4-frame, 2x2 spatial grid produces two temporal tokens per video.
    # This repository's Qwen2.5 time coordinate uses 2 positions per second.
    ids = result["input_ids"][0]
    positions = result["position_ids"][:, 0]
    starts = torch.nonzero(ids == 151652).flatten()
    assert len(starts) == len(intervals)
    for start, interval in zip(starts, intervals):
        first, second = int(start) + 1, int(start) + 2
        assert ids[first] == ids[second] == 151656
        assert int(positions[0, second] - positions[0, first]) == int(interval * 2)
        assert int(positions[1, second] - positions[1, first]) == 0
        assert int(positions[2, second] - positions[2, first]) == 0
    assert result["position_ids"].shape == (3, 1, ids.numel())
    assert result["attention_mask"] == [ids.numel()]


@pytest.mark.parametrize(
    "fps", [[1.0], [2.0], [4.0], [1.0, 2.0], [4.0, 1.0], [1.0, 2.0, 4.0]]
)
def test_processor_intervals_are_preserved(processor, fps, monkeypatch):
    batch = native_batch(processor, fps)
    intervals = [2.0 / rate for rate in fps]
    assert list(batch["second_per_grid_ts"]) == intervals
    result = dataset_item(monkeypatch, processor, batch)
    assert_intervals(result, intervals)


@pytest.mark.parametrize("count", [1, 2, 3])
@pytest.mark.parametrize("representation", ["tensor", "chunks"])
def test_legacy_interval_fallback_counts_video_rows(
    processor, count, representation, monkeypatch
):
    batch = native_batch(processor, [2.0] * count)
    batch.pop("second_per_grid_ts")
    if representation == "chunks":
        # Deliberately allow a chunk to contain multiple video rows.
        batch["video_grid_thw"] = list(torch.split(batch["video_grid_thw"], 2))
    result = dataset_item(monkeypatch, processor, batch)
    assert_intervals(result, [1.0] * count)


@pytest.mark.parametrize("metadata_type", ["tensor", "list"])
def test_interval_metadata_representations_are_supported(
    processor, metadata_type, monkeypatch
):
    batch = native_batch(processor, [1.0, 4.0])
    values = batch["second_per_grid_ts"]
    batch["second_per_grid_ts"] = (
        torch.as_tensor(values).clone() if metadata_type == "tensor" else list(values)
    )
    result = dataset_item(monkeypatch, processor, batch)
    assert_intervals(result, [2.0, 0.5])


def test_text_only_sample_does_not_need_video_configuration(processor, monkeypatch):
    batch = dict(processor(text="text", return_tensors="pt"))
    batch["labels"] = torch.full_like(batch["input_ids"], -100)
    result = dataset_item(monkeypatch, processor, batch)
    torch.testing.assert_close(
        result["position_ids"], torch.zeros(3, 1, 1, dtype=torch.long)
    )


def test_processor_metadata_does_not_require_a_default_fps(processor, monkeypatch):
    batch = native_batch(processor, [1.0])
    monkeypatch.setattr(processor.video_processor, "fps", None)
    result = dataset_item(monkeypatch, processor, batch)
    assert_intervals(result, [2.0])


def test_getitem_returns_requested_multivideo_sample(processor, monkeypatch):
    videos = native_batch(processor, [1.0, 4.0])
    text = dict(processor(text="text", return_tensors="pt"))
    text["labels"] = torch.full_like(text["input_ids"], -100)
    batches = {"videos": videos, "text": text}
    monkeypatch.setattr(
        data_processor,
        "preprocess_qwen_visual",
        lambda sources, proc: dict(batches[sources[0]["kind"]]),
    )
    # Avoid the dataset's retry delay when exercising the original failure.
    monkeypatch.setattr(data_processor.time, "sleep", lambda seconds: None)
    dataset = data_processor.LazySupervisedDataset.__new__(
        data_processor.LazySupervisedDataset
    )
    dataset.processor = processor
    dataset.merge_size = 2
    dataset.get_rope_index = get_rope_index_25
    dataset.item_fn = dataset._get_item
    dataset.list_data_dict = [{"kind": "videos"}, {"kind": "text"}]
    result = dataset[0]
    torch.testing.assert_close(result["input_ids"], videos["input_ids"])
    assert_intervals(result, [2.0, 0.5])
