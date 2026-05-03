"""Regression test: torchvision backend frames_indices must be absolute.

See https://github.com/QwenLM/Qwen3-VL/issues/2085
"""

import types

import pytest
import torch


def _make_fake_torchvision_io(total_frames=100, fps=25.0):
    """Return a fake torchvision.io module whose read_video returns zeros."""

    def fake_read_video(path, start_pts=0.0, end_pts=None, pts_unit="sec", output_format="TCHW"):
        s = int(start_pts * fps) if start_pts else 0
        e = int(end_pts * fps) if end_pts is not None else total_frames
        e = min(e, total_frames)
        clip = max(e - s, 0)
        video = torch.zeros(clip, 3, 8, 8, dtype=torch.uint8)
        info = {"video_fps": fps}
        return video, None, info

    mod = types.ModuleType("torchvision.io")
    mod.read_video = fake_read_video
    return mod


@pytest.fixture()
def _patch_io(monkeypatch):
    """Patch vision_process.io with the fake module for every test."""
    import qwen_vl_utils.vision_process as vp

    fake = _make_fake_torchvision_io()
    monkeypatch.setattr(vp, "io", fake)


def _run(ele):
    import qwen_vl_utils.vision_process as vp

    return vp._read_video_torchvision(ele)


def test_no_trim_indices_start_at_zero(_patch_io):
    """Without video_start, indices should start at 0."""
    _, meta, _ = _run({"video": "fake.mp4"})
    assert meta["frames_indices"][0] == 0


def test_video_start_offset_indices(_patch_io):
    """video_start=2.0 at 25fps -> first index is 50."""
    _, meta, _ = _run({"video": "fake.mp4", "video_start": 2.0})
    assert meta["frames_indices"][0] == 50


def test_video_start_end_offset_indices(_patch_io):
    """video_start=1.0 video_end=3.0 at 25fps -> first index 25, total_num_frames 50."""
    _, meta, _ = _run({"video": "fake.mp4", "video_start": 1.0, "video_end": 3.0})
    assert meta["frames_indices"][0] == 25
    assert meta["total_num_frames"] == 50
