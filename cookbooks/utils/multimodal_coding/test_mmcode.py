import os
import re
import subprocess
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_path):
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype="auto", device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer


def generate_code(model, tokenizer, prompt):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(**model_inputs, max_new_tokens=512)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

def extract_code(text):
    pattern = r"```python\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1) if match else None

def test_code(code):
    try:
        # Execute the code in a separate process for security
        result = subprocess.run(
            ['python3', '-c', code],
            capture_output=True,
            text=True,
            timeout=60
        )
        # Return True if exit code is 0 (success)
        if result.returncode == 0:
            return True
        else:
            print(f"Error executing code (exit code {result.returncode}):")
            if result.stderr:
                print(result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print("Error: Code execution timed out after 60 seconds.")
        return False
    except FileNotFoundError:
        print("Error: 'python3' interpreter not found.")
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False


def main():
    model_path = os.getenv("MODEL_PATH", "Qwen/Qwen2.5-Coder-7B-Instruct")
    model, tokenizer = load_model(model_path)
    
    prompt = "Write a Python function to calculate the factorial of a number."
    print(f"Prompt: {prompt}")
    
    response = generate_code(model, tokenizer, prompt)
    print(f"\nGenerated Response:\n{response}")
    
    code = extract_code(response)
    if code:
        print(f"\nExtracted Code:\n{code}")
        if test_code(code):
            print("\nCode executed successfully!")
        else:
            print("\nCode execution failed.")
    else:
        print("\nNo code block found in the response.")


if __name__ == "__main__":
    main()
