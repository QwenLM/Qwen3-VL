import base64
from io import BytesIO

import pytest
import requests
from PIL import Image

from qwen_vl_utils import fetch_image


ORIENTATION_TRANSFORMS = {
    1: None,
    2: Image.Transpose.FLIP_LEFT_RIGHT,
    3: Image.Transpose.ROTATE_180,
    4: Image.Transpose.FLIP_TOP_BOTTOM,
    5: Image.Transpose.TRANSPOSE,
    6: Image.Transpose.ROTATE_270,
    7: Image.Transpose.TRANSVERSE,
    8: Image.Transpose.ROTATE_90,
}


@pytest.fixture
def make_input(tmp_path, monkeypatch):
    def make(orientation, input_kind):
        image = Image.new("RGB", (112, 56), "red")
        image.paste("green", (56, 0, 112, 28))
        image.paste("blue", (0, 28, 56, 56))
        image.paste("yellow", (56, 28, 112, 56))
        exif = image.getexif()
        if orientation is not None:
            exif[274] = orientation
        buffer = BytesIO()
        image.save(buffer, format="JPEG", exif=exif)
        data = buffer.getvalue()
        decoded = Image.open(BytesIO(data))

        if input_kind == "pil":
            source = decoded
        elif input_kind == "base64":
            source = "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")
        elif input_kind == "url":
            response = requests.Response()
            response.status_code = 200
            response._content = data
            monkeypatch.setattr(requests, "get", lambda *args, **kwargs: response)
            source = "https://example.com/oriented.jpg"
        else:
            path = tmp_path / "oriented.jpg"
            path.write_bytes(data)
            source = str(path) if input_kind == "path" else "file://" + str(path)
        return source, decoded

    return make


@pytest.mark.parametrize("input_kind", ["pil", "path", "file_uri", "base64", "url"])
@pytest.mark.parametrize("orientation", list(ORIENTATION_TRANSFORMS) + [None])
def test_fetch_image_exif_orientation(make_input, orientation, input_kind):
    source, decoded = make_input(orientation, input_kind)
    transform = ORIENTATION_TRANSFORMS.get(orientation)
    expected = decoded.transpose(transform) if transform is not None else decoded.copy()

    result = fetch_image({"image": source})

    assert result.mode == "RGB"
    assert result.size == expected.size
    assert result.tobytes() == expected.tobytes()
    assert result.getexif().get(274) in (None, 1)
    assert decoded.getexif().get(274) == orientation


@pytest.mark.parametrize("orientation", ORIENTATION_TRANSFORMS)
def test_fetch_image_exif_before_explicit_resize(make_input, orientation):
    source, decoded = make_input(orientation, "pil")
    transform = ORIENTATION_TRANSFORMS[orientation]
    expected = decoded.transpose(transform) if transform is not None else decoded.copy()
    expected = expected.resize((84, 56))

    result = fetch_image({"image": source, "resized_width": 84, "resized_height": 56})

    assert result.size == (84, 56)
    assert result.tobytes() == expected.tobytes()


def test_fetch_image_preserves_rgba_white_background():
    image = Image.new("RGBA", (56, 56), (255, 0, 0, 0))
    result = fetch_image({"image": image})
    assert result.mode == "RGB"
    assert result.getpixel((0, 0)) == (255, 255, 255)
