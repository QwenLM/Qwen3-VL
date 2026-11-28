import av
import numpy as np
import pytest

from qwen_vl_utils import vision_process


@pytest.fixture
def sample_video(tmp_path):
    path = tmp_path / "sample.mp4"
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=4)
        stream.width = stream.height = 64
        stream.pix_fmt = "yuv420p"
        for index in range(8):
            pixels = np.full((64, 64, 3), 32 + index * 24, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            container.mux(stream.encode(frame))
        container.mux(stream.encode())
    return path


def test_video_fallback_without_torchvision_read_video(monkeypatch, sample_video):
    monkeypatch.delattr(vision_process.io, "read_video", raising=False)
    monkeypatch.setattr(vision_process, "get_video_reader_backend", lambda: "torchvision")
    ((video, metadata), sample_fps) = vision_process.fetch_video(
        {
            "video": str(sample_video),
            "video_start": 0.25,
            "video_end": 1.0,
            "nframes": 2,
            "min_pixels": 56 * 56,
        },
        return_video_sample_fps=True,
        return_video_metadata=True,
    )

    assert video.shape == (2, 3, 56, 56)
    assert video[0].mean().item() == pytest.approx(56, abs=5)
    assert video[1].mean().item() == pytest.approx(128, abs=5)
    assert metadata["video_backend"] == "pyav"
    assert metadata["fps"] == 4
    assert metadata["total_num_frames"] == 4
    assert sample_fps == 2
