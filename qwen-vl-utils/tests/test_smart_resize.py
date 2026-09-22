import pytest
from PIL import Image

from qwen_vl_utils.vision_process import fetch_image, smart_resize


# Regression test for https://github.com/QwenLM/Qwen3-VL/pull/1145 : in the
# downscale branch of smart_resize, a tall/narrow image combined with a small
# `max_pixels` budget floors the short side to 0, which makes fetch_image()
# fail with "ValueError: height and width must be > 0" inside PIL.Image.resize.
@pytest.mark.parametrize(
    "height, width, factor, max_pixels",
    [
        (639, 31, 28, 20 * 28 * 28),  # Qwen2-VL / Qwen2.5-VL, image_patch_size=14
        (4104, 64, 32, 64 * 32 * 32),  # Qwen3-VL, image_patch_size=16
    ],
)
def test_smart_resize_never_returns_a_zero_dimension(height, width, factor, max_pixels):
    h_bar, w_bar = smart_resize(height, width, factor=factor, max_pixels=max_pixels)
    assert h_bar >= factor and w_bar >= factor
    assert h_bar % factor == 0 and w_bar % factor == 0


def test_fetch_image_accepts_a_narrow_image_with_a_small_max_pixels():
    image = Image.new("RGB", (31, 639), color=(255, 255, 255))
    resized = fetch_image({"image": image, "max_pixels": 20 * 28 * 28})
    assert min(resized.size) > 0
