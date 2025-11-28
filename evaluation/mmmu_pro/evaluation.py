import os
import copy
import string
import random
from typing import Dict, Any, Optional
import copy as cp

from utils import cn_string
from io_utils import (
    build_choices,
    build_option_str
)

def build_prompt(question, options, prediction):
    tmpl = (
        'You are an AI assistant who will help me to match '
        'an answer with several options of a single-choice question. '
        'You are provided with a question, several options, and an answer, '
        'and you need to find which option is most similar to the answer. '
        'If the meaning of all options are significantly different from the answer, output Z. '
        'Your should output a single uppercase character in A, B, C, D (if they are valid options), and Z. \n'
        'Example 1: \n'
        'Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n'
        'Answer: a cute teddy bear\nYour output: A\n'
        'Example 2: \n'
        'Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n'
        'Answer: Spider\nYour output: Z\n'
        'Example 3: \n'
        'Question: {}?\nOptions: {}\nAnswer: {}\nYour output: '
    )
    return tmpl.format(question, options, prediction)

def build_prompt_cn(question, options, prediction):
    tmpl = (
        '你是一个帮助我匹配答案与单选题中多个选项的 AI 助手。'
        '你会被提供：一个问题，多个选项，一个答案。你的任务是找到与答案意义最相近的选项。'
        '如果所有选项的意义都与答案显著不同，则输出 Z。'
        '你应该输出一个单个的大写字母，例如 A, B, C, D（如果它们是有效选项），或 Z。'
        '例 1:'
        '问题: 图中最主要的物体是什么?\n选项: A. 泰迪熊 B. 兔子 C. 猫 D. 狗\n答案: 一只可爱的泰迪熊\n输出: A\n'
        '例 2: \n'
        '问题: 图中最主要的物体是什么?\n选项: A. 泰迪熊 B. 兔子 C. 猫 D. 狗\n答案: 蜘蛛\n输出: Z\n'
        '例 3: \n'
        '问题: {}?\n选项: {}\n答案: {}\n输出: '
    )
    return tmpl.format(question, options, prediction)

def can_infer_option(answer, choices):
    verbose = os.environ.get('VERBOSE', 0)
    # Choices is a dictionary
    if 'Failed to obtain answer via API' in answer:
        return False

    reject_to_answer = [
        "Sorry, I can't help with images of people yet.",
        "I can't process this file.",
        "I'm sorry, but without the image provided",
        'Cannot determine the answer'
    ]
    for err in reject_to_answer:
        if err in answer:
            return 'Z'

    def count_choice(splits, choices, prefix='', suffix=''):
        cnt = 0
        for c in choices:
            if prefix + c + suffix in splits:
                cnt += 1
        return cnt

    answer_mod = cp.copy(answer)
    chars = '.()[],:;!*#{}'
    for c in chars:
        answer_mod = answer_mod.replace(c, ' ')

    splits = [x.strip() for x in answer_mod.split()]
    count = count_choice(splits, choices)

    if count == 1:
        for ch in choices:
            if 'A' in splits and len(splits) > 3 and verbose:
                return False
            if ch in splits:
                return ch
    elif count == 0 and count_choice(splits, {'Z', ''}) == 1:
        return 'Z'
    return False

def can_infer_text(answer, choices):
    answer = answer.lower()
    assert isinstance(choices, dict)
    for k in choices:
        assert k in string.ascii_uppercase
        choices[k] = str(choices[k]).lower()
    cands = []
    for k in choices:
        if choices[k] in answer:
            cands.append(k)
    if len(cands) == 1:
        return cands[0]
    return False

def can_infer(answer, choices):
    answer = str(answer)
    copt = can_infer_option(answer, choices)
    return copt if copt else can_infer_text(answer, choices)

def extract_answer_from_item(model, item, dataset_name=None):
    # It will return: (pred, raw, llm_time)
    choices = build_choices(item)
    option_str = build_option_str(choices)

    if cn_string(item['question']):
        prompt = build_prompt_cn(item['question'], option_str, item['prediction'])
    else:
        prompt = build_prompt(item['question'], option_str, item['prediction'])
    retry = 5

    ret = can_infer(item['prediction'], choices)
    if ret:
        if ret == 'Z':
            extract_flag = False
            log = f"Rule extract failed with rule result: {ret} prediction: {item['prediction']}"
        else:
            extract_flag = True
            log = f"Rule extract success with rule result: {ret} prediction: {item['prediction']}"
        return dict(opt=ret, log=log, extract_model='rule', extract_flag=extract_flag)

    if model is None and 'MMMU' in dataset_name:
        assert model is not None, 'Judge model is None for MMMU_DEV_VAL !!!'

    while retry:
        ans = model.generate([{"type": "text", "value": prompt}])
        if 'Failed to obtain answer via API' in ans:
            print('GPT API failed to answer. ')
        else:
            ret = can_infer(ans, choices)
            if ret and ret != 'Z':
                log = f'{model.model} extract Succeed. {model.model}:{ans}\n'
                return dict(opt=ret, log=log, extract_model=model.model, extract_flag=True)
            else:
                print(f'Output includes 0 / > 1 letter among candidates {set(choices)} and Z: {ans}')
        retry -= 1

        if retry == 0:
            options = list(choices) + ['Z'] if 'Z' not in choices else []
            log = f'{model.model} extract failed. randomly generate one. {model.model} response:{ans}\n'
            return dict(opt=random.choice(options), log=log, extract_model=model.model, extract_flag=False)

def eval_vanilla(model, item, dataset_name=None):
    res = extract_answer_from_item(model, item, dataset_name=dataset_name)
    opt, match_log, extract_model, extract_flag = res['opt'], res['log'], res['extract_model'], res['extract_flag']
    if opt == item['GT']:
        return dict(hit=1, log=f'Match Log: {match_log}. ', extract_model=extract_model, extract_flag=extract_flag)
    else:
        return dict(hit=0, log=f'Match Log: {match_log}. ', extract_model=extract_model, extract_flag=extract_flag)
