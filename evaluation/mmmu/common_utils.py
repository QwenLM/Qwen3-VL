import base64
import hashlib
import io

import requests
from PIL import Image


def encode_image_to_base64(image, target_size=None):
    """Encode an image to base64 string."""
    if target_size is not None:
        width, height = image.size
        # Resize the image while maintaining the aspect ratio
        if width > height:
            new_width = target_size
            new_height = int(height * target_size / width)
        else:
            new_height = target_size
            new_width = int(width * target_size / height)
        image = image.resize((new_width, new_height))

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def decode_base64_to_image(base64_string):
    """Decode a base64 string to an image."""
    image_data = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(image_data))


def decode_base64_to_image_file(base64_string, output_path):
    """Decode a base64 string and save it to a file."""
    image = decode_base64_to_image(base64_string)
    image.save(output_path)


def download_file(url, local_path):
    """Download a file from a URL to a local path."""
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(local_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)


def md5(file_path):
    """Calculate the MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def toliststr(s):
    if isinstance(s, str) and (s[0] == "[") and (s[-1] == "]"):
        return [str(x) for x in eval(s)]
    elif isinstance(s, str):
        return [s]
    elif isinstance(s, list):
        return [str(x) for x in s]
    raise NotImplementedError
