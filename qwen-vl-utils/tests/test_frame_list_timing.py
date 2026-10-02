"""Frame-list metadata uses source-frame coordinates and repeats padding time."""
import unittest

from PIL import Image
import torch

from qwen_vl_utils.vision_process import fetch_video


def sampled_frames(count, raw_fps=None):
    item = {
        "video": [Image.new("RGB", (4, 4), (i, i, i)) for i in range(count)],
        "sample_fps": 2.0, "min_pixels": 16, "max_pixels": 16,
    }
    if raw_fps is not None:
        item["raw_fps"] = raw_fps
    return item


class FrameListTimingTest(unittest.TestCase):
    def test_source_rate_does_not_compress_sampling_timestamps(self):
        (video, metadata), sampled_fps = fetch_video(
            sampled_frames(4, 30.0), image_patch_size=2,
            return_video_metadata=True, return_video_sample_fps=True,
        )
        self.assertEqual(metadata["frames_indices"], [0, 15, 30, 45])
        self.assertEqual([index / metadata["fps"] for index in metadata["frames_indices"]], [0.0, 0.5, 1.0, 1.5])
        self.assertEqual(metadata["total_num_frames"], 60.0)
        self.assertEqual(sampled_fps, 2.0)
        self.assertEqual(video.shape[0], 4)

    def test_padding_duplicates_last_frame_timestamp(self):
        video, metadata = fetch_video(sampled_frames(3, 30.0), image_patch_size=2, return_video_metadata=True)
        self.assertEqual(metadata["frames_indices"], [0, 15, 30, 30])
        self.assertEqual(metadata["total_num_frames"], 45.0)
        torch.testing.assert_close(video[-1], video[-2])

    def test_one_frame_padding_keeps_original_time_and_duration(self):
        video, metadata = fetch_video(sampled_frames(1), image_patch_size=2, return_video_metadata=True)
        self.assertEqual(metadata["frames_indices"], [0, 0])
        self.assertEqual(metadata["total_num_frames"], 1.0)
        self.assertEqual(video.shape[0], 2)

    def test_even_default_frames_retain_sequential_metadata(self):
        item = sampled_frames(4)
        video, metadata = fetch_video(item, image_patch_size=2, return_video_metadata=True)
        self.assertEqual(metadata["frames_indices"], [0, 1, 2, 3])
        self.assertEqual(metadata["fps"], 2.0)
        self.assertEqual(metadata["total_num_frames"], 4.0)
        self.assertEqual(len(item["video"]), 4)
        self.assertEqual(video.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()

