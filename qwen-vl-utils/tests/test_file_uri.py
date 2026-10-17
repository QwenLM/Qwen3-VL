import base64
import nturl2path
from io import BytesIO
from unittest.mock import Mock
from urllib.parse import unquote

import pytest
import requests
from PIL import Image

from qwen_vl_utils import fetch_image, process_vision_info, vision_process


FILENAMES = ["plain.png", "space name.png", "中文.png", "100%.png", "literal%20.png", "hash#name.png"]


@pytest.fixture
def image_path(tmp_path, request):
    path = tmp_path / request.param
    Image.new("RGB", (56, 56), "red").save(path)
    return path


def assert_image(image):
    assert image.mode == "RGB"
    assert image.size == (56, 56)
    assert image.getpixel((0, 0)) == (255, 0, 0)


@pytest.mark.parametrize("image_path", FILENAMES, indirect=True)
@pytest.mark.parametrize("key", ["image", "image_url"])
def test_fetch_image_standard_file_uri(image_path, key):
    assert_image(fetch_image({key: image_path.as_uri()}))


@pytest.mark.parametrize("image_path", FILENAMES, indirect=True)
def test_process_vision_info_standard_file_uri(image_path):
    messages = [{"role": "user", "content": [{"type": "image", "image": image_path.as_uri()}]}]
    images, videos = process_vision_info(messages)
    assert len(images) == 1
    assert videos is None
    assert_image(images[0])


@pytest.mark.parametrize("image_path", FILENAMES, indirect=True)
def test_fetch_image_plain_path_unchanged(image_path):
    assert_image(fetch_image({"image": str(image_path)}))


@pytest.mark.parametrize("image_path", ["plain.png", "literal%20.png", "hash#name.png"], indirect=True)
@pytest.mark.skipif(vision_process.os.name != "nt", reason="legacy drive path is Windows-specific")
def test_fetch_image_legacy_windows_path_unchanged(image_path):
    assert_image(fetch_image({"image": "file://" + str(image_path)}))


@pytest.mark.parametrize("image_path", ["space name.png"], indirect=True)
def test_fetch_image_localhost_uri(image_path):
    uri = image_path.as_uri().replace("file:///", "file://localhost/", 1)
    assert_image(fetch_image({"image": uri}))


@pytest.mark.parametrize("scheme", ["http", "https"])
def test_fetch_image_http_unchanged(monkeypatch, scheme):
    buffer = BytesIO()
    Image.new("RGB", (56, 56), "red").save(buffer, format="PNG")
    response = requests.Response()
    response.status_code = 200
    response._content = buffer.getvalue()
    get = Mock(return_value=response)
    monkeypatch.setattr(vision_process.requests, "get", get)
    uri = scheme + "://example.com/image%20name.png"
    assert_image(fetch_image({"image": uri}))
    get.assert_called_once_with(uri, stream=True)


@pytest.mark.parametrize("kind", ["pil", "base64"])
def test_fetch_image_other_inputs_unchanged(kind):
    image = Image.new("RGB", (56, 56), "red")
    if kind == "base64":
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        image = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    assert_image(fetch_image({"image": image}))


@pytest.mark.parametrize(
    "uri,converter,expected",
    [
        ("file:///tmp/space%20name.png", unquote, "/tmp/space name.png"),
        ("file:///tmp/%E4%B8%AD%E6%96%87.png", unquote, "/tmp/中文.png"),
        ("file:///tmp/literal%2520.png", unquote, "/tmp/literal%20.png"),
        ("file:///tmp/hash%23name.png", unquote, "/tmp/hash#name.png"),
        ("file://localhost/tmp/plain.png", unquote, "/tmp/plain.png"),
        ("file:///C:/images/space%20name.png", nturl2path.url2pathname, "C:\\images\\space name.png"),
        ("file:///C:/images/%E4%B8%AD%E6%96%87.png", nturl2path.url2pathname, "C:\\images\\中文.png"),
        ("file:///C:/images/literal%2520.png", nturl2path.url2pathname, "C:\\images\\literal%20.png"),
        ("file://localhost/C:/images/plain.png", nturl2path.url2pathname, "C:\\images\\plain.png"),
        ("file://LOCALHOST/C:/images/plain.png", nturl2path.url2pathname, "C:\\images\\plain.png"),
        ("file:///tmp/question%3Fname.png", unquote, "/tmp/question?name.png"),
        ("file:///C:/images/question%3Fname.png", nturl2path.url2pathname, "C:\\images\\question?name.png"),
        ("file:///tmp/plain.png?query=ignored", unquote, "/tmp/plain.png"),
        ("file:///tmp/plain.png#fragment", unquote, "/tmp/plain.png"),
    ],
)
def test_file_uri_conversion_without_filesystem_access(monkeypatch, uri, converter, expected):
    # Exercise the real function with both platform converters; do not access UNC shares.
    monkeypatch.setattr(vision_process, "url2pathname", converter, raising=False)
    image = Image.new("RGB", (56, 56), "red")
    open_image = Mock(return_value=image)
    monkeypatch.setattr(vision_process.Image, "open", open_image)
    assert_image(fetch_image({"image": uri}))
    open_image.assert_called_once_with(expected)


@pytest.mark.parametrize("uri", [
    "file://server/share/image.png",
    "file://server/C:/images/a.png",
    "file://localhost.evil/share/image.png",
    "file://127.0.0.1/share/image.png",
    "file://user@localhost/share/image.png",
    "file://\\\\server\\share\\image.png",
])
def test_remote_file_authority_rejected_before_open(monkeypatch, uri):
    open_image = Mock(return_value=Image.new("RGB", (56, 56), "red"))
    monkeypatch.setattr(vision_process.Image, "open", open_image)
    with pytest.raises(ValueError, match="Only local file URIs are supported"):
        fetch_image({"image": uri})
    open_image.assert_not_called()


@pytest.mark.parametrize("uri", [
    "file:////server/share/image.png",
    "file:///%2Fserver/share/image.png",
    "file:///%5Cserver/share/image.png",
])
def test_windows_network_uri_rejected_after_decoding(monkeypatch, uri):
    monkeypatch.setattr(vision_process, "url2pathname", nturl2path.url2pathname)
    open_image = Mock(return_value=Image.new("RGB", (56, 56), "red"))
    monkeypatch.setattr(vision_process.Image, "open", open_image)
    with pytest.raises(ValueError, match="Only local file URIs are supported"):
        fetch_image({"image": uri})
    open_image.assert_not_called()


def test_plain_unc_path_unchanged_without_network_access(monkeypatch):
    path = "\\\\server\\share\\image.png"
    open_image = Mock(return_value=Image.new("RGB", (56, 56), "red"))
    monkeypatch.setattr(vision_process.Image, "open", open_image)
    assert_image(fetch_image({"image": path}))
    open_image.assert_called_once_with(path)
