import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from typing import List, Dict, Any
import torch
import warnings
from concurrent.futures import ThreadPoolExecutor
import string
import traceback

# Local imports from refactored files
from dataset_utils import load_dataset, dump_image, MMMU_preproc
from eval_utils import build_judge, eval_single_sample

from qwen2_vl.model import Qwen2VLChat
from qwen_vl_utils import process_vision_info

def run_inference(args):
    """Run inference on the MMMU dataset."""
    # Load dataset
    data = load_dataset(args.dataset)
    
    # Set up image root directory
    img_root = os.path.join(os.environ['LMUData'], 'images', 'MMMU')
    os.makedirs(img_root, exist_ok=True)
    
    # Set up dump_image function
    def dump_image_func(line):
        return dump_image(line, img_root)
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # Set up CoT prompt if enabled
    cot_prompt = ""
    if args.use_cot:
        cot_prompt = args.cot_prompt if args.cot_prompt else " If you are uncertain or the problem is too complex, make a reasoned guess based on the information provided. Avoid repeating steps indefinitely—provide your best guess even if unsure. Determine whether to think step by step based on the difficulty of the question, considering all relevant information before answering."
        print(f"Using CoT prompt: {cot_prompt}")

    # Initialize HuggingFace model
    print(f"Loading HuggingFace model from {args.model_path}")
    model = Qwen2VLChat(
        model_path=args.model_path,
        temperature=0.01,
        top_p=0.001,
        top_k=1,
        use_custom_prompt=True,
        min_pixels=1280*28*28,
        max_pixels=5120*28*28
    )
    model.set_dump_image(dump_image_func)

    # Run inference
    results = []
    for i in tqdm(range(len(data)), desc="Running inference"):
        # line = data.iloc[i].to_dict()
        line = data.iloc[i]
        index = line['index']
        
        # Convert line to dict and ensure all values are JSON serializable
        line_dict = line.to_dict()
        for k, v in line_dict.items():
            if isinstance(v, np.integer):
                line_dict[k] = int(v)
            elif isinstance(v, np.floating):
                line_dict[k] = float(v)
        
        # Generate response using HuggingFace
        messages = model.build_prompt(line, args.dataset)
        
        # Add CoT prompt if enabled
        if args.use_cot and len(messages) > 0 and messages[-1]['type'] == 'text':
            messages[-1]['value'] += cot_prompt
            
        response = model.generate(messages)
            
        print(f"response: {response}")
        print(f"annotation answer: {line['answer']}")
        print('-' * 50)
        
        # Save result
        result = {
            "question_id": int(index) if isinstance(index, np.integer) else index,
            "annotation": line_dict,
            "task": args.dataset,
            "result": {"gen": response},
            "messages": messages
        }
        results.append(result)
        
        # Write intermediate results
        if i % 10 == 0:
            with open(args.output_file, 'w') as f:
                for res in results:
                    f.write(json.dumps(res) + '\n')
            
    # Write final results
    with open(args.output_file, 'w') as f:
        for res in results:
            f.write(json.dumps(res) + '\n')
    
    print(f"Inference completed. Results saved to {args.output_file}")

def run_evaluation(args):
    """Run evaluation on inference results."""
    # Load results
    results = []
    with open(args.input_file, 'r') as f:
        for line in f:
            job = json.loads(line)
            annotation = job["annotation"]
            annotation["prediction"] = job["result"]["gen"]
            results.append(annotation)
            
    data = pd.DataFrame.from_records(results)
    data = data.sort_values(by='index')
    data['prediction'] = [str(x) for x in data['prediction']]
    # If not choice label, then use lower case
    for k in data.keys():
        data[k.lower() if k not in list(string.ascii_uppercase) else k] = data.pop(k)

    # Load dataset
    meta = load_dataset(args.dataset)

    # data中data.iloc[i]中的index必须在results中存在，对应results中的results[i]['id']，并且data中data.iloc[i]中的question必须和results中results[i]['annotation']中的question完全一致
    print(f"len(data): {len(data)}")
    print(f"len(meta): {len(meta)}")
    meta_q_map = {x: y for x, y in zip(meta['index'], meta['question'])}
    data_map = {x: y for x, y in zip(data['index'], data['question'])}
    for k in data_map:
        assert k in meta_q_map, (
            f'eval_file should be the same as or a subset of dataset MMMU_DEV_VAL'
        )

    answer_map = {i: c for i, c in zip(meta['index'], meta['answer'])}
    data = MMMU_preproc(data)
    answer_map = {k: (v if v in list(string.ascii_uppercase) else 'A') for k, v in answer_map.items()}
    data = data[data['index'].isin(answer_map)]
    data['GT'] = [answer_map[idx] for idx in data['index']]
    items = []
    for i in range(len(data)):
        item = data.iloc[i]
        items.append(item)

    # Build judge model if needed
    model = None
    model = build_judge(args.eval_model, args.api_type)
    
    # Prepare evaluation tasks
    eval_tasks = []
    for item in items:
        eval_tasks.append((model, item))
    
    # Run evaluation
    eval_results = []
    
    # Debug mode: process single-threaded with first few samples
    debug = os.environ.get('DEBUG', '').lower() == 'true'
    if debug:
        print("Running in debug mode with first 5 samples...")
        # for task in tqdm(eval_tasks[:5], desc="Evaluating"):
        for task in eval_tasks[:5]:
            try:
                result = eval_single_sample(task)
                eval_results.append(result)
            except Exception as e:
                print(f"Error processing task: {e}")
                print(f"Task details: {task}")
                raise
    else:
        # Normal mode: process all samples with threading
        with ThreadPoolExecutor(max_workers=args.nproc) as executor:
            for result in tqdm(executor.map(eval_single_sample, eval_tasks), 
                             total=len(eval_tasks), desc="Evaluating"):
                eval_results.append(result)
    
    # Calculate overall accuracy
    accuracy = sum(r['hit'] for r in eval_results) / len(eval_results)
    
    # Calculate accuracy by split
    results_by_split = {}
    for result in eval_results:
        split = result.get('split', 'unknown')
        if split not in results_by_split:
            results_by_split[split] = []
        results_by_split[split].append(result)
    
    accuracy_by_split = {}
    for split, split_results in results_by_split.items():
        split_accuracy = sum(r['hit'] for r in split_results) / len(split_results)
        accuracy_by_split[split] = split_accuracy
        print(f"Accuracy for {split} split: {split_accuracy:.4f} ({sum(r['hit'] for r in split_results)}/{len(split_results)})")
    
    # Save results
    output_df = pd.DataFrame(eval_results)
    output_df.to_csv(args.output_file, index=False)
    
    # Save accuracy
    with open(args.output_file.replace('.csv', '_acc.json'), 'w') as f:
        json.dump({
            "overall_accuracy": accuracy,
            "accuracy_by_split": accuracy_by_split
        }, f, indent=2)
    
    # print(f"Evaluation completed. Overall accuracy: {accuracy:.4f}")
    print(f"Results saved to {args.output_file}")

def main():
    parser = argparse.ArgumentParser(description="MMMU Evaluation Script")
    subparsers = parser.add_subparsers(dest="mode", help="Mode to run")
    
    # Inference parser
    infer_parser = subparsers.add_parser("infer", help="Run inference")
    infer_parser.add_argument("--model-path", type=str, required=True, help="Path to the model")
    infer_parser.add_argument("--dataset", type=str, default="MMMU_DEV_VAL", help="Dataset name")
    infer_parser.add_argument("--data-dir", type=str, help="The absolute path of MMMU_DEV_VAL.tsv")
    infer_parser.add_argument("--output-file", type=str, required=True, help="Output file path")
    infer_parser.add_argument("--use-cot", action="store_true", help="Use Chain-of-Thought prompting")
    infer_parser.add_argument("--cot-prompt", type=str, default="", help="Custom Chain-of-Thought prompt")
    
    # Evaluation parser
    eval_parser = subparsers.add_parser("eval", help="Run evaluation")
    eval_parser.add_argument("--data-dir", type=str, help="The absolute path of MMMU_DEV_VAL.tsv")
    eval_parser.add_argument("--input-file", type=str, required=True, help="Input file with inference results")
    eval_parser.add_argument("--output-file", type=str, required=True, help="Output file path")
    eval_parser.add_argument("--dataset", type=str, default="MMMU_DEV_VAL", help="Dataset name")
    eval_parser.add_argument("--eval-model", type=str, default="gpt-3.5-turbo-0125", 
                            choices=["gpt-3.5-turbo-0125","gpt-4-0125-preview"],
                            help="Model to use for evaluation")
    eval_parser.add_argument("--api-type", type=str, default="dash", choices=["dash", "mit"],
                            help="API type to use for evaluation")
    eval_parser.add_argument("--nproc", type=int, default=4, help="Number of processes to use")
    
    args = parser.parse_args()

    os.environ['LMUData'] = args.data_dir
    
    if args.mode == "infer":
        run_inference(args)
    elif args.mode == "eval":
        run_evaluation(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 
