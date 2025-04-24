from collections import defaultdict
import os
import string
from typing import Optional, Any, Dict

import numpy as np
import pandas as pd

from io_utils import load
from utils import track_progress_rich
from evaluation import eval_vanilla 

MMB_abbrs = {
    'coarse_perception': 'CP',
    'finegrained_perception (instance-level)': 'FP-S',
    'finegrained_perception (cross-instance)': 'FP-C',
    'logic_reasoning': 'LR',
    'relation_reasoning': 'RR',
    'attribute_reasoning': 'AR'
}

def report_acc(df):
    # assert group in [None, 'category', 'l2-category']
    res = defaultdict(list)

    if 'split' in df:
        splits = list(set(df['split']))
        res['split'] = splits
    else:
        df['split'] = ['none'] * len(df)
        res['split'] = ['none']

    for group in [None, 'l2-category', 'category']:
        if group is None:
            res['Overall'] = [np.mean(df[df['split'] == sp]['hit']) for sp in res['split']]
        elif group not in df:
            continue
        else:
            abilities = list(set(df[group]))
            abilities.sort()
            for ab in abilities:
                ab_name = MMB_abbrs[ab] if ab in MMB_abbrs else ab
                sub_df = df[df[group] == ab]
                res[ab_name] = [np.mean(sub_df[sub_df['split'] == sp]['hit']) for sp in res['split']]

    # **Add extract_model statistics**
    if 'extract_model' in df.columns:
        models = list(set(df['extract_model']))
        for model in models:
            for sp in res['split']:
                model_df = df[(df['extract_model'] == model) & (df['split'] == sp)]
                total_count = len(model_df)
                success_count = np.sum(model_df['extract_flag'])
                res[model+'_success'].append(success_count)
                res[model+'_all'].append(total_count)

    return pd.DataFrame(res)

def MMMU_preproc(data):
    cnt = 0
    As, Bs, Ans = list(data['A']), list(data['B']), list(data['answer'])
    lt = len(data)
    for i in range(lt):
        if pd.isna(As[i]):
            As[i] = Ans[i]
            Bs[i] = 'Other Answers'
            cnt += 1
    data['A'] = As
    data['B'] = Bs
    return data

def mcq_vanilla_eval(model, data, meta, nproc, result_file, dataset_name=None):
    result = {}
    if os.path.exists(result_file):
        result = load(result_file)
    answer_map = {i: c for i, c in zip(meta['index'], meta['answer'])}

    data = MMMU_preproc(data)
    answer_map = {k: (v if v in list(string.ascii_uppercase) else 'A') for k, v in answer_map.items()}

    data = data[data['index'].isin(answer_map)]
    data['GT'] = [answer_map[idx] for idx in data['index']]
    items = []

    for i in range(len(data)):
        # Dealing with the normal part
        item = data.iloc[i]
        if item['index'] not in result:
            items.append(item)

    tups = [dict(model=model, item=x, dataset_name=dataset_name) for x in items]
    keys = [x['index'] for x in items]
    if len(tups):
        res = track_progress_rich(eval_vanilla, tups, nproc=nproc, chunksize=nproc, save=result_file, keys=keys)
        result = load(result_file)
        for k, v in zip(keys, res):
            if k not in result:
                result[k] = v
    data['hit'] = [result[i]['hit'] for i in data['index']]
    data['log'] = [result[i]['log'] for i in data['index']]
    data['extract_model'] = [result[i].get('extract_model', None) for i in data['index']] 
    data['extract_flag'] = [result[i].get('extract_flag', 0) for i in data['index']]
    if 'GT' in data:
        data.pop('GT')
    return data