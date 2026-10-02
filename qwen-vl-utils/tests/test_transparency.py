"""Transparent image encodings must produce the same white RGB background."""
import base64
from io import BytesIO
import unittest

from PIL import Image

from qwen_vl_utils.vision_process import fetch_image, to_rgb


class TransparencyTest(unittest.TestCase):
    def test_luminance_alpha_is_composited(self):
        image = Image.new("LA", (2, 1))
        image.putdata([(0, 0), (0, 128)])
        result = to_rgb(image)
        self.assertEqual(list(result.getdata()), [(255, 255, 255), (127, 127, 127)])
        self.assertEqual(image.mode, "LA")

    def test_palette_alpha_table_is_composited(self):
        image = Image.new("P", (3, 1))
        image.putpalette([0, 0, 0, 255, 0, 0, 0, 0, 255] + [0] * (768 - 9))
        image.putdata([0, 1, 2])
        image.info["transparency"] = bytes([0, 128, 255])
        result = to_rgb(image)
        self.assertEqual(list(result.getdata()), [(255, 255, 255), (255, 127, 127), (0, 0, 255)])
        self.assertEqual(image.mode, "P")

    def test_color_key_transparency_and_opaque_controls(self):
        transparent = Image.new("RGB", (1, 1), (0, 0, 0))
        transparent.info["transparency"] = (0, 0, 0)
        self.assertEqual(to_rgb(transparent).getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(to_rgb(Image.new("RGB", (1, 1), (0, 0, 0))).getpixel((0, 0)), (0, 0, 0))
        self.assertEqual(to_rgb(Image.new("RGBA", (1, 1), (255, 0, 0, 128))).getpixel((0, 0)), (255, 127, 127))

    def test_png_data_uri_matches_equivalent_rgba_image(self):
        palette = Image.new("P", (4, 4), 0)
        palette.putpalette([0] * 768)
        palette.info["transparency"] = 0
        buffer = BytesIO()
        palette.save(buffer, format="PNG")
        uri = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
        from_png = fetch_image({"image": uri}, image_patch_size=1)
        from_rgba = fetch_image({"image": palette.convert("RGBA")}, image_patch_size=1)
        self.assertEqual(from_png.mode, "RGB")
        self.assertEqual(from_png.tobytes(), from_rgba.tobytes())
        self.assertTrue(all(pixel == (255, 255, 255) for pixel in from_png.getdata()))


if __name__ == "__main__":
    unittest.main()

