import os
import ast
import operator
import requests
import time
import random
import copy
import traceback
import pandas as pd
from PIL import Image
from typing import List, Dict, Tuple, Any
from common_utils import encode_image_to_base64
from collections import defaultdict

try:
    from latex2sympy2 import latex2sympy
except ImportError:
    print('Warning: latex2sympy2 not installed. Install with: pip install latex2sympy2')
    latex2sympy = None

FAIL_MSG = 'Failed to obtain answer via API.'


def _numeric_expression(text):
    """Evaluate only numeric constants and arithmetic, never Python code."""
    binary_ops = {
        ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow,
        ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv,
    }
    unary_ops = {ast.UAdd: operator.pos, ast.USub: operator.neg}

    def evaluate(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in binary_ops:
            return binary_ops[type(node.op)](evaluate(node.left), evaluate(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in unary_ops:
            return unary_ops[type(node.op)](evaluate(node.operand))
        raise ValueError("Answer is not a numeric expression")

    return evaluate(ast.parse(text, mode="eval").body)


def is_equal(asw: str, gt_asw: str) -> bool:
    """Check if two answers are equal."""
    if not isinstance(asw, str) or not isinstance(gt_asw, str):
        print('Warning: input is not string')
        print(asw, gt_asw)
    asw = str(asw).lower().strip()
    gt_asw = str(gt_asw).lower().strip()
    if gt_asw == asw:
        return True
    try:
        a = _numeric_expression(gt_asw)
        b = _numeric_expression(asw)
        if abs(a - b) < 1e-6:
            return True
    except:
        pass
    if latex2sympy is not None:
        try:
            a = latex2sympy(gt_asw)
            b = latex2sympy(asw)
            if abs(a - b) < 1e-6:
                return True
        except:
            pass
    return False


def get_gpt4_ICE():
    """Get in-context examples for GPT-4 answer extraction."""
    example_1 = """
Hint: Please answer the question and provide the final answer at the end.\n
Question: Which number is missing?\n
Model response: The number missing in the sequence is 14.\n
Extracted answer: 14
"""

    example_2 = """
Hint: Please answer the question and provide the final answer at the end.\n
Question: What is the fraction of females facing the camera?\n
Model response: The fraction of females facing the camera is 0.6,
which means that six out of ten females in the group are facing the camera.\n
Extracted answer: 0.6
"""

    example_3 = """
Hint: Please answer the question and provide the final answer at the end.\n
Question: How much money does Luca need to buy a sour apple candy and a butter-scotch candy? (Unit: $)\n
Model response: Luca needs $1.45 to buy a sour apple candy and a butterscotch candy.\n
Extracted answer: 1.45
"""

    example_4 = """
Hint: Please answer the question and provide the final answer at the end.\n
Question: Between which two years does the line graph saw its maximum peak?\n
Model response: The line graph saw its maximum peak between 2007 and 2008.\n
Extracted answer: [2007, 2008]
"""

    example_5 = """
Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.\n
Question: What fraction of the shape is blue?\n
Choices: (A) 3/11 (B) 8/11 (C) 6/11 (D) 3/5\n
Model response: The correct answer is (B) 8/11.\n
Extracted answer: B
"""

    return [example_1, example_2, example_3, example_4, example_5]


def build_mathv_gpt4_prompt(line):
    """Build the prompt for GPT-4 to extract answer from model response."""
    task_description = """
Please read the following example.
Then extract the answer from the model response and type it at the end of the prompt.\n
"""
    question = line['question']
    prediction = str(line['prediction'])
    prompt = task_description
    examples = get_gpt4_ICE()
    for example in examples:
        prompt += example + '\n'
    prompt += question + '\n'
    prompt += 'Model response: ' + prediction + '\n'
    prompt += 'Extracted answer: '
    return prompt


def list_to_dict(lst):
    """Convert list to dictionary with uppercase letters as keys."""
    return {chr(65 + i): val for i, val in enumerate(lst)}


def can_infer_option(answer, choices):
    """Rule-based extraction of answer option."""
    if FAIL_MSG in answer:
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

    answer_mod = copy.copy(answer)
    chars = '.()[],:;!*#{}'
    for c in chars:
        answer_mod = answer_mod.replace(c, ' ')

    splits = [x.strip() for x in answer_mod.split()]
    count = count_choice(splits, choices)

    if count == 1:
        for ch in choices:
            if 'A' in splits and len(splits) > 3:
                return False
            if ch in splits:
                return ch
    elif count == 0 and count_choice(splits, {'Z', ''}) == 1:
        return 'Z'
    return False


def can_infer_text(answer, choices):
    """Extract answer by matching text content."""
    answer = answer.lower()
    assert isinstance(choices, dict)
    for k in choices:
        choices[k] = str(choices[k]).lower()
    cands = []
    for k in choices:
        if choices[k] in answer:
            cands.append(k)
    if len(cands) == 1:
        return cands[0]
    return False


def can_infer(answer, choices):
    """Combined approach to infer answer choice."""
    answer = str(answer)
    copt = can_infer_option(answer, choices)
    return copt if copt else can_infer_text(answer, choices)


def post_check(line, prefetch=False):
    """Check if the prediction matches the answer."""
    res = None
    ans = line['answer']
    response = line['prediction'] if prefetch else line['res']
    try:
        parsed_choices = ast.literal_eval(line['choices'])
        if len(parsed_choices) > 0:
            ans = line['answer']
            choices = list_to_dict(parsed_choices)
            res = can_infer(response, choices)
            if prefetch:
                return res
        else:
            res = str(response)
            ans = str(ans)
    except (ValueError, SyntaxError, TypeError):
        pass

    if is_equal(res, ans):
        return res if prefetch else True
    else:
        return False


class OpenAIWrapper:
    """Wrapper for OpenAI API."""
    
    def __init__(self, model, api_base, api_key, timeout=60, retry=5, wait=5):
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.timeout = timeout
        self.retry = retry
        self.wait = wait
        self.fail_msg = FAIL_MSG
    
    def generate(self, prompt, temperature=0):
        """Generate a response from the API."""
        headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {self.api_key}'}
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 4096,
            "temperature": temperature
        }
        
        for i in range(self.retry):
            try:
                response = requests.post(
                    self.api_base,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    resp_json = response.json()
                    return resp_json['choices'][0]['message']['content'].strip()
                
                time.sleep(self.wait)
            except Exception as e:
                print(f"API error: {e}")
                time.sleep(self.wait)
        
        return self.fail_msg


class DashScopeWrapper:
    """Wrapper for DashScope API."""
    
    def __init__(self, model, api_base, api_key, timeout=60, retry=5, wait=5):
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.timeout = timeout
        self.retry = retry
        self.wait = wait
        self.fail_msg = FAIL_MSG
    
    def generate(self, prompt, temperature=0):
        """Generate a response from the API."""
        headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {self.api_key}'}
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": 4096,
            "n": 1,
            "temperature": temperature,
            "stream": False
        }

        for i in range(self.retry):
            try:
                response = requests.post(
                    self.api_base,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    resp_json = response.json()
                    
                    # Check finish reason
                    for output in resp_json['choices']:
                        if output['finish_reason'] not in ['stop', 'function_call']:
                            print(f"DashScope finished with error: {resp_json}")
                            time.sleep(self.wait)
                            continue
                    
                    return resp_json['choices'][0]['message']['content']
                else:
                    print(f"DashScope API error: HTTP {response.status_code}")
                    try:
                        error_content = response.json()
                        print(f"Error details: {error_content}")
                    except:
                        print(f"Raw error content: {response.content.decode('utf-8', errors='replace')}")
                
                time.sleep(self.wait)
            except Exception as e:
                print(f"DashScope error: {e}")
                time.sleep(self.wait)
        
        return self.fail_msg


def build_judge(model, api_type):
    """Build a judge model for evaluation."""
    if api_type == 'mit':
        api_key = os.environ.get('MIT_SPIDER_TOKEN', '')
        api_base = os.environ.get('MIT_SPIDER_URL', '')
        return OpenAIWrapper(model, api_base, api_key)
    elif api_type == 'dash':
        api_key = os.environ.get('CHATGPT_DASHSCOPE_API_KEY', '')
        api_base = os.environ.get('DASHSCOPE_API_BASE', '')
        return DashScopeWrapper(model, api_base, api_key)
    else:
        raise ValueError(f"Unsupported API type: {api_type}")


def MATH_V_auxeval(args):
    """Auxiliary evaluation for MathVision - extract answer from model response."""
    model, line = args
    prompt = build_mathv_gpt4_prompt(line)
    log = ''
    retry = 5
    
    # Try rule-based extraction first
    if post_check(line, prefetch=True):
        res = post_check(line, prefetch=True)
        log += 'Prefetch succeed.\n'
        extract_flag = True
        if not res or res == 'Z':
            extract_flag = False
            log += f'Rule extract failed with ans: {res}'
        else:
            log += f'Rule extract success with ans: {res}'
        return dict(log=log, res=res, extract_model='rule', extract_flag=extract_flag)
    
    # Use model-based extraction
    for i in range(retry):
        prediction = line['prediction']
        res = model.generate(prompt, temperature=i * 0.5)

        if FAIL_MSG in res:
            log += f'Try {i}: output is {prediction}, failed to parse.\n'
        else:
            log += f'{model.model} extract Succeed.\n'
            return dict(log=log, res=res, extract_model=model.model, extract_flag=True)
    
    log += f'All {retry} retries failed.\n {model.model} response:{res}'
    return dict(log=log, res='', extract_model=model.model, extract_flag=False)


def MATH_V_acc(result_file):
    """Calculate accuracy for MathVision results."""
    data = pd.read_excel(result_file) if result_file.endswith('.xlsx') else pd.read_csv(result_file)
    
    tot = defaultdict(lambda: 0)
    fetch = defaultdict(lambda: 0)
    hit = defaultdict(lambda: 0)
    lt = len(data)
    extract_counts = {}

    for i in range(lt):
        item = data.iloc[i]
        cate = item['category']
        tot['Overall'] += 1
        tot[cate] += 1
        if 'Prefetch succeed' in item['log']:
            fetch['Overall'] += 1
            fetch[cate] += 1
        if post_check(item, prefetch=False):
            hit['Overall'] += 1
            hit[cate] += 1
        # Statistics of answers extracted by rule and gpt
        extract_model = item['extract_model']
        extract_flag = item['extract_flag']
        if extract_model in extract_counts:
            extract_counts[extract_model][1] += 1
        else:
            extract_counts[extract_model] = [0, 1]  # succeed, total
        if extract_flag:
            extract_counts[extract_model][0] += 1

    res = defaultdict(list)
    for k in tot.keys():
        res['Subject'].append(k)
        res['tot'].append(tot[k])
        res['prefetch'].append(fetch[k])
        res['hit'].append(hit[k])
        res['prefetch_rate'].append(fetch[k] / tot[k] * 100)
        res['acc'].append(hit[k] / tot[k] * 100)
        if k == 'Overall':
            for model_key in extract_counts:
                res[model_key+'_success'].append(extract_counts[model_key][0])
                res[model_key+'_all'].append(extract_counts[model_key][1])
        else:
            for model_key in extract_counts:
                res[model_key+'_success'].append(0)
                res[model_key+'_all'].append(0)
    res = pd.DataFrame(res).sort_values('Subject', ignore_index=True)
    return res


def eval_single_sample(args):
    """Evaluate a single sample."""
    return MATH_V_auxeval(args)
