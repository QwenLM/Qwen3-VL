"""Regression tests for zero-detection ODinW evaluations."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
from pycocotools.coco import COCO


MODULE_PATH = Path(__file__).resolve().parents[1] / "eval_utils.py"
spec = importlib.util.spec_from_file_location("odinw_eval_utils", MODULE_PATH)
eval_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_utils)


def ground_truth():
    coco = COCO()
    coco.dataset = {
        "info": {},
        "images": [{"id": 7, "height": 300, "width": 300}],
        "categories": [{"id": 3, "name": "object"}],
        "annotations": [
            {"id": i + 1, "image_id": 7, "category_id": 3, "bbox": box,
             "area": box[2] * box[3], "iscrowd": 0}
            for i, box in enumerate([[0, 0, 10, 10], [20, 20, 50, 50], [80, 80, 120, 120]])
        ],
    }
    coco.createIndex()
    return coco


def predictions(boxes):
    return [({}, {
        "img_id": 7, "labels": np.zeros(len(boxes), dtype=np.int64),
        "bboxes": np.asarray(boxes, dtype=float).reshape(-1, 4),
        "scores": np.ones(len(boxes)),
    })]


class EmptyDetectionTest(unittest.TestCase):
    def test_empty_detection_arrays_score_zero(self):
        coco = ground_truth()
        metrics = eval_utils.compute_metrics(predictions([]), _coco_api=coco)
        self.assertEqual(set(metrics), {"mAP", "mAP_50", "mAP_75", "mAP_s", "mAP_m", "mAP_l"})
        self.assertTrue(all(value == 0.0 for value in metrics.values()))
        self.assertEqual(len(coco.dataset["annotations"]), 3)

    def test_no_prediction_rows_score_zero(self):
        metrics = eval_utils.compute_metrics([], _coco_api=ground_truth())
        self.assertEqual(len(metrics), 6)
        self.assertTrue(all(value == 0.0 for value in metrics.values()))

    def test_perfect_detections_still_score_one(self):
        metrics = eval_utils.compute_metrics(
            predictions([[0, 0, 10, 10], [20, 20, 70, 70], [80, 80, 200, 200]]),
            _coco_api=ground_truth(),
        )
        self.assertTrue(all(value == 1.0 for value in metrics.values()))


if __name__ == "__main__":
    unittest.main()

