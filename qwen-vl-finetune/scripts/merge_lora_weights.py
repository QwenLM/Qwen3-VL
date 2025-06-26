import os
import torch
import argparse
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, required=True,
                        help="Path to the base model to merge LoRA weights into")
    parser.add_argument("--lora_model_path", type=str, required=True, 
                        help="Path to the LoRA adapter weights")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Where to save the merged model")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", 
                        choices=["float16", "bfloat16", "float32"],
                        help="Data type for model weights")
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"Loading base model: {args.base_model_path}")
    
    # Map dtype string to torch data type
    dtype_mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32
    }
    torch_dtype = dtype_mapping[args.torch_dtype]
    
    # Load base model
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_model_path,
        torch_dtype=torch_dtype,
        device_map="auto"
    )
    
    # Load LoRA weights
    print(f"Loading LoRA adapter: {args.lora_model_path}")
    model = PeftModel.from_pretrained(model, args.lora_model_path)
    
    # Merge weights
    print("Merging LoRA weights into base model...")
    model = model.merge_and_unload()
    
    # Save the merged model
    print(f"Saving merged model to: {args.output_path}")
    model.save_pretrained(args.output_path)
    
    # Save processor/tokenizer
    processor = AutoProcessor.from_pretrained(args.base_model_path)
    processor.save_pretrained(args.output_path)
    
    print("Done!")

if __name__ == "__main__":
    main()