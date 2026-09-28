import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path


class _Sub:
    def __init__(self, start, end, text):
        self.start = start
        self.end = end
        self.text = text


class _Pysubs2:
    @staticmethod
    def load(path, encoding="utf-8"):
        return [
            _Sub(0, 2000, "Hello"),
            _Sub(1000, 3000, "There"),
        ]


def _load():
    datasets = types.ModuleType("datasets")
    datasets.load_dataset = lambda *args, **kwargs: None
    sys.modules["datasets"] = datasets
    sys.modules["pysubs2"] = _Pysubs2
    path = (
        Path(__file__).resolve().parents[1]
        / "evaluation"
        / "VideoMME"
        / "dataset_utils.py"
    )
    spec = importlib.util.spec_from_file_location("videomme_dataset_utils", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mod = _load()


class VideoMMESubtitleTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".srt")
        self.tmp.write(b"1\n")
        self.tmp.flush()

    def tearDown(self):
        self.tmp.close()

    def test_a_subtitle_that_starts_at_zero_is_kept(self):
        text = mod.load_subtitles(self.tmp.name, [0.0])
        self.assertEqual(text, "Hello")

    def test_a_frame_inside_a_cue_still_matches(self):
        text = mod.load_subtitles(self.tmp.name, [1.5])
        self.assertEqual(text, "Hello There")


if __name__ == "__main__":
    unittest.main()
