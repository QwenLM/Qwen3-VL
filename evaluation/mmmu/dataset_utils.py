import os
import pandas as pd
import numpy as np
from typing import Dict, Any
from common_utils import download_file, md5, toliststr, decode_base64_to_image_file

MMMU_DATASET_URL = 'https://opencompass.openxlab.space/utils/VLMEval/MMMU_DEV_VAL.tsv'
MMMU_DATASET_MD5 = '521afc0f3bf341e6654327792781644d'

def load_dataset(dataset_name='MMMU_DEV_VAL'):
    """Load the MMMU dataset."""
    data_root = os.path.join(os.environ['LMUData'])
    os.makedirs(data_root, exist_ok=True)
    
    file_name = f"{dataset_name}.tsv"
    data_path = os.path.join(data_root, file_name)
    
    # Download if not exists or MD5 doesn't match
    if not os.path.exists(data_path) or md5(data_path) != MMMU_DATASET_MD5:
        print(f"Downloading {dataset_name} dataset...")
        download_file(MMMU_DATASET_URL, data_path)
    
    # Load the dataset
    data = pd.read_csv(data_path, sep='\t').iloc[:8]
    
    # Process the dataset
    data['index'] = [str(x) for x in data['index']]
    
    # Handle image data
    if 'image' in data:
        data['image'] = [str(x) for x in data['image']]
        image_map = {x: y for x, y in zip(data['index'], data['image'])}
        for k in image_map:
            if len(image_map[k]) <= 64:
                idx = image_map[k]
                assert idx in image_map and len(image_map[idx]) > 64
                image_map[k] = image_map[idx]

        images = [toliststr(image_map[k]) for k in data['index']]
        data['image'] = [x[0] if len(x) == 1 else x for x in images]
    
    # Handle image paths
    if 'image_path' in data:
        paths = [toliststr(x) for x in data['image_path']]
        data['image_path'] = [x[0] if len(x) == 1 else x for x in paths]
    
    # Convert index to int if possible
    if np.all([isinstance(x, int) or x.isdigit() for x in data['index']]):
        data['index'] = [int(x) for x in data['index']]
    
    return data

def dump_image(line, img_root):
    """Save image data to disk and return the path."""
    os.makedirs(img_root, exist_ok=True)
    
    if 'image' in line:
        if isinstance(line['image'], list):
            tgt_path = []
            assert 'image_path' in line
            for img, im_name in zip(line['image'], line['image_path']):
                path = os.path.join(img_root, im_name)
                if not os.path.exists(path):
                    decode_base64_to_image_file(img, path)
                tgt_path.append(path)
        else:
            tgt_path = os.path.join(img_root, f"{line['index']}.jpg")
            if not os.path.exists(tgt_path):
                decode_base64_to_image_file(line['image'], tgt_path)
            tgt_path = [tgt_path]
    else:
        assert 'image_path' in line
        tgt_path = toliststr(line['image_path'])
    
    return tgt_path

def MMMU_preproc(data):
    """
    Preprocess MMMU dataset to reformulate open questions to multi-choice ones.
    This aligns with the implementation in multiple_choice.py
    """
    print("Preprocessing MMMU dataset...")
    cnt = 0
    As, Bs, Ans = list(data['A']), list(data['B']), list(data['answer'])
    lt = len(data)
    for i in range(lt):
        if pd.isna(As[i]):
            As[i] = Ans[i]
            Bs[i] = 'Other Answers'
            cnt += 1
    print(f'During MMMU_preproc in Evaluation, {cnt} open questions are re-formulated to multi-choice ones.')
    data['A'] = As
    data['B'] = Bs
    return data