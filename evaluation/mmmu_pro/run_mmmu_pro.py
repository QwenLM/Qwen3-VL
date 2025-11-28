from __future__ import annotations
import os
import argparse
import warnings
import json
import string
import pandas as pd
from typing import Dict, Any, Optional
from tqdm import tqdm

from api_clients import build_judge, gpt_key_set
from io_utils import load, dump
from metrics import mcq_vanilla_eval, report_acc
from utils import _flatten_dict
from dataset import load_mmmupro_dataset

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
from qwen2_vl.model import Qwen2VLChat

# --- MMMUPro inference pipeline ---
def mmmupro_inference_pipeline(args):
    # Initialize model
    print(f"Initializing model from: {args.model_path}")
    model = Qwen2VLChat(
        model_path=args.model_path,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        use_custom_prompt=True
    )
    
    # Load dataset
    print(f"Loading MMMUPro dataset from: {args.data_dir}, subset: {args.subset}")
    try:
        dataset = load_mmmupro_dataset(
            data_dir=args.data_dir,
            subset=args.subset
        )
    except FileNotFoundError as e:
        print(f"Error loading dataset: {e}")
        return
    print(f"Dataset loaded with {len(dataset)} samples.")
    
    results = []
    for seq_id, data in enumerate(tqdm(dataset, desc="Processing MMMUPro"), start=0):
        try:
            # Build message structure
            messages = []
            
            if args.sys_prompt:
                messages.append({"role": "system", "content": args.sys_prompt})
            
            if args.subset == 'vision':
                messages.append({
                    "type": "image",
                    "value": data['image'],
                    "min_pixels": args.min_pixels or 1280*28*28,
                    "max_pixels": args.max_pixels or 10000*28*28
                })
            else:
                for img_path in data['images']:
                    messages.append({
                        "type": "image",
                        "value": img_path,
                        "min_pixels": args.min_pixels or 1280*28*28,
                        "max_pixels": args.max_pixels or 10000*28*28
                    })
            
            if data['subset'] == 'vision':
                prompt_text = data.get('prompt', 'Identify the problem and solve it. Think step by step before answering.')
            else:
                options_text = "\n".join(
                    [f"{k}. {v}" for k, v in data['choices'].items()]
                )
                prompt_text = f"{data['question']}\n{options_text}\n{data.get('prompt', 'Please select the correct answer from the options.')}"
            
            messages.append({
                "type": "text",
                "value": prompt_text
            })

            response = model.generate(messages)
            
            if data['subset'] == 'vision':
                question_id = seq_id + 2000
            else:
                question_id = seq_id
            # Build result record (preserve all original data)
            result = {
                "question_id": question_id,
                "annotation": data,
                "task": f"MMMUPro_{args.subset.capitalize()}",
                "result": {
                    "gen": response,
                    "success": True,
                    "error_message": ""
                },
                "messages": messages
            }
            results.append(result)
            
        except Exception as e:
            if data['subset'] == 'vision':
                question_id = seq_id + 2000
            else:
                question_id = seq_id
            print(f"Error processing {question_id}: {str(e)}")
            result = {
                "question_id": question_id,
                "annotation": data,
                "task": f"MMMUPro_{args.subset.capitalize()}",
                "result": {
                    "gen": "",
                    "success": False,
                    "error_message": str(e)
                },
                "messages": messages
            }
            results.append(result)
    
    # Save results
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'a', encoding='utf-8') as f:
        for item in results:
            json_str = json.dumps(item, ensure_ascii=False)
            f.write(json_str + '\n')
    
    print(f"Inference completed. Results saved to {args.output_path}")
    return results

# --- Core Evaluation Logic ---

def evaluate(eval_file, args, dataset_name='default'):
    """Orchestrate evaluation process with judge model"""
    dataset = dataset_name
    nproc = args.nproc
    circular = False
    suffix = eval_file.split('.')[-1]

    # Configuration and model setup
    model = args.eval_model
    assert model in ['chatgpt-0125', 'exact_matching', 'gpt-4-0125', 'qwen-plus', 'gpt-4-turbo', 'gpt-4o', 'gpt-4', 'gpt-4-0125-preview']
    name_str_map = {'chatgpt-0125': 'openai', 'gpt-4-0125': 'gpt4', 'qwen-plus': 'qwen-plus'}
    name_str = name_str_map[model] if model in name_str_map else model

    if gpt_key_set():
        model = build_judge(args.eval_model, args.api_type)
    else:
        warnings.warn('OPENAI_API_KEY is not set properly, will use exact matching for evaluation')
        model = None
    
    if model == None and dataset == 'MMMUPro_Standard':
        assert model is not None, 'Judge model is None for MMMUPro_Standard !!!'
    
    result_file = eval_file.replace(f'.{suffix}', f'_{name_str}_result.pkl')

    # Data loading and preprocessing
    data = load(eval_file)
    data = data.sort_values(by='index')
    data['prediction'] = [str(x) for x in data['prediction']]
    # If not choice label, then use lower case
    for k in data.keys():
        data[k.lower() if k not in list(string.ascii_uppercase) else k] = data.pop(k)

    meta = load(eval_file)
    assert 'index' in meta and 'answer' in meta, 'Essentail columns missing in the eval_file.'

    meta_q_map = {x: y for x, y in zip(meta['index'], meta['question'])}
    data_map = {x: y for x, y in zip(data['index'], data['question'])}
    for k in data_map:
        assert k in meta_q_map, (
            f'eval_file should be the same as or a subset of dataset {dataset_name}'
        )
    
    # Core evaluation execution
    data = mcq_vanilla_eval(model, data, meta, nproc, result_file, dataset_name)

    # Post-processing and result saving
    dump(data, eval_file.replace(f'.{suffix}', f'_{name_str}_result.{suffix}'))
    data = load(eval_file.replace(f'.{suffix}', f'_{name_str}_result.{suffix}'))
    acc = report_acc(data)
    score_file = eval_file.replace(f'.{suffix}', '_acc.csv')
    dump(acc, score_file)

    return acc

def run_evaluation(args):
    """Run evaluation on inference results."""
    
    results = []
    with open(args.input_file, 'r') as f:
        for line in f:
            job = json.loads(line)
            annotation = job["annotation"]
            result = {
                "index": job["question_id"],
                "question": annotation["question"],
                "answer": annotation["answer"],
                "category": annotation["subset"],
                "l2-category": annotation["subject"],
                "prediction": job["result"]["gen"],
            }
            result.update(annotation["choices"])
            results.append(result)

    result_filename = args.output_file.split("/")[-1].replace("_acc.csv", ".tsv")
    score_filename = args.output_file.split("/")[-1]
    save_dir = args.output_file.split("/")[0]
    os.makedirs(save_dir, exist_ok=True)

    results = pd.DataFrame(results)
    result_file = os.path.join(save_dir, result_filename)

    dump(results, result_file)

    evaluate(result_file, args, dataset_name='MMMUPro_Standard')
    acc = load(os.path.join(save_dir, score_filename))
    if isinstance(acc, pd.DataFrame):
        if len(acc) == 1:
            acc_dict = acc.iloc[0].to_dict()
        else:
            acc_dict = acc.to_dict()
    elif isinstance(acc, dict):
        acc_dict = acc
    else:
        raise ValueError(f"Invalid return type from eval_func: {type(acc)}")

    print ({"task_samples": len(results), **_flatten_dict(acc_dict), "eval_model": args.eval_model})


def main():
    parser = argparse.ArgumentParser(description="MMMUPro Inference and Evaluation Script")
    subparsers = parser.add_subparsers(dest="mode", required=True, help="Mode to run: 'infer' or 'eval'")

    # --- Inference Arguments ---
    infer_parser = subparsers.add_parser("infer", help="Run inference on MMMUPro dataset")
    infer_parser.add_argument("--model-path", type=str, required=True, help="Path to the Qwen2-VL model")
    infer_parser.add_argument("--data-dir", type=str, default="MMMU_Pro", help="Directory to save MMMUPro data")
    infer_parser.add_argument("--subset", type=str, default="original", choices=["original", "vision"], help="MMMUPro subset to process ('original' or 'vision')")
    infer_parser.add_argument("--output-path", type=str, required=True, help="Output JSONL file path for inference results")
    infer_parser.add_argument("--min-pixels", type=int, default=None, help="Minimum pixels for image processing (default: 1280*28*28)")
    infer_parser.add_argument("--max-pixels", type=int, default=None, help="Maximum pixels for image processing (default: 10000*28*28)")
    infer_parser.add_argument("--sys-prompt", type=str, default=None, help="Path to a file containing the system prompt or the prompt string itself")
    # Add model generation parameters if needed
    infer_parser.add_argument("--temperature", type=float, default=0.01, help="Sampling temperature")
    infer_parser.add_argument("--top-p", type=float, default=0.001, help="Top-p sampling")
    infer_parser.add_argument("--top-k", type=int, default=1, help="Top-k sampling")


    # --- Evaluation Arguments ---
    eval_parser = subparsers.add_parser("eval", help="Run evaluation on inference results")
    eval_parser.add_argument("--input-file", type=str, required=True, help="Input JSONL file with inference results (output from 'infer' mode)")
    eval_parser.add_argument("--output-file", type=str, required=True, help="Output CSV file path for accuracy scores")
    # eval_parser.add_argument("--dataset", type=str, default="MMMUPro_Standard", help="Dataset name for evaluation context (used internally)") # Kept internal for now
    eval_parser.add_argument("--eval-model", type=str, default="exact_matching",
                            choices=['chatgpt-0125', 'exact_matching', 'gpt-4-0125', 'qwen-plus', 'gpt-4-turbo', 'gpt-4o', 'gpt-4', 'gpt-4-0125-preview'],
                            help="Model/method to use for judging answers (e.g., 'exact_matching', 'gpt-4o')")
    eval_parser.add_argument("--api-type", type=str, default="dash", choices=["dash", "mit"], # Kept from original, usage depends on your build_judge
                            help="API type to use for evaluation judge model (if applicable)")
    eval_parser.add_argument("--nproc", type=int, default=4, help="Number of processes to use for evaluation scoring")

    args = parser.parse_args()

    if args.mode == "infer":
        # Handle system prompt loading if it's a file path
        if args.sys_prompt and os.path.isfile(args.sys_prompt):
            print(f"Loading system prompt from file: {args.sys_prompt}")
            try:
                with open(args.sys_prompt, 'r', encoding='utf-8') as f:
                    args.sys_prompt = f.read().strip()
            except Exception as e:
                print(f"Warning: Failed to load system prompt from file: {e}. Using the path as prompt string.")
        elif args.sys_prompt:
             print("Using provided string as system prompt.")

        # Run Inference
        mmmupro_inference_pipeline(args)

    elif args.mode == "eval":
        if not all([build_judge, gpt_key_set, load, dump, mcq_vanilla_eval, report_acc, _flatten_dict]):
             print("Error: Evaluation dependencies (api_clients, io_utils, metrics, utils) seem missing. Cannot proceed with evaluation.")
        else:
             # Run Evaluation
             run_evaluation(args) # Use the wrapper function
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
