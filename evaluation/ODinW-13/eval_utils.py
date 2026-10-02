"""
ODinW evaluation utilities.
"""
import os
import copy
import json
import tempfile
import numpy as np
from typing import List, Dict, Sequence
from collections import OrderedDict
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def xyxy2xywh(bbox: np.ndarray) -> list:
    """Convert bbox format from xyxy to xywh.
    
    Args:
        bbox: Bounding box in [x1, y1, x2, y2] format
    
    Returns:
        Bounding box in [x, y, w, h] format
    """
    _bbox = bbox.tolist()
    return [
        _bbox[0],
        _bbox[1],
        _bbox[2] - _bbox[0],
        _bbox[3] - _bbox[1],
    ]


def results2json(results: Sequence[dict], outfile_prefix: str, cat_ids: dict) -> dict:
    """Convert results to COCO JSON format.
    
    Args:
        results: List of prediction results
        outfile_prefix: Output file prefix
        cat_ids: Category ID mapping
    
    Returns:
        result_files: Dictionary of result file paths
    """
    bbox_json_results = []
    for idx, result in enumerate(results):
        image_id = result.get('img_id', idx)
        labels = result['labels']
        bboxes = result['bboxes']
        scores = result['scores']
        
        for i, label in enumerate(labels):
            data = dict()
            data['image_id'] = image_id
            data['bbox'] = xyxy2xywh(bboxes[i])
            data['score'] = float(scores[i])
            data['category_id'] = cat_ids[label]
            bbox_json_results.append(data)
    
    result_files = dict()
    result_files['bbox'] = f'{outfile_prefix}.bbox.json'
    with open(result_files['bbox'], 'w') as f:
        json.dump(bbox_json_results, f)
    
    return result_files


def compute_metrics(results: list, outfile_prefix: str = None, _coco_api: COCO = None) -> Dict[str, float]:
    """Compute mAP and other metrics using COCO API.
    
    Args:
        results: List of evaluation results, each element is a (gt, pred) tuple
        outfile_prefix: Output file prefix (optional)
        _coco_api: COCO API instance
    
    Returns:
        eval_results: Dictionary of evaluation metrics
    """
    proposal_nums = (100, 300, 1000)
    iou_thrs = np.linspace(
        .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
    
    # Separate ground truth and predictions
    if len(results) == 0:
        gts, preds = [], []
    else:
        gts, preds = zip(*results)
    
    tmp_dir = None
    if outfile_prefix is None:
        tmp_dir = tempfile.TemporaryDirectory()
        outfile_prefix = os.path.join(tmp_dir.name, 'results')
    
    cat_ids = _coco_api.getCatIds()
    img_ids = _coco_api.getImgIds()
    
    # Convert to COCO format and save
    result_files = results2json(preds, outfile_prefix, cat_ids)
    
    eval_results = OrderedDict()
    
    for metric in ["bbox"]:
        iou_type = metric
        if metric not in result_files:
            raise KeyError(f'{metric} is not in results')
        try:
            with open(result_files[metric], 'r') as f:
                predictions = json.load(f)
            if predictions:
                coco_dt = _coco_api.loadRes(predictions)
            else:
                # COCO.loadRes indexes the first detection; an empty result is
                # still a valid evaluation and must include all ground-truth images.
                coco_dt = COCO()
                coco_dt.dataset = copy.deepcopy(_coco_api.dataset)
                coco_dt.dataset['annotations'] = []
                coco_dt.createIndex()
        except IndexError:
            print('The testing results of the whole dataset is empty.')
            break
        
        coco_eval = COCOeval(_coco_api, coco_dt, iou_type)
        
        coco_eval.params.catIds = cat_ids
        coco_eval.params.imgIds = img_ids
        coco_eval.params.maxDets = list(proposal_nums)
        coco_eval.params.iouThrs = iou_thrs
        
        # mapping of cocoEval.stats
        coco_metric_names = {
            'mAP': 0,
            'mAP_50': 1,
            'mAP_75': 2,
            'mAP_s': 3,
            'mAP_m': 4,
            'mAP_l': 5,
            'AR@100': 6,
            'AR@300': 7,
            'AR@1000': 8,
            'AR_s@1000': 9,
            'AR_m@1000': 10,
            'AR_l@1000': 11
        }
        
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        metric_items = [
            'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
        ]
        
        for metric_item in metric_items:
            val = coco_eval.stats[coco_metric_names[metric_item]]
            eval_results[metric_item] = float(f'{round(val, 3)}')
    
    if tmp_dir is not None:
        tmp_dir.cleanup()
    
    return eval_results


