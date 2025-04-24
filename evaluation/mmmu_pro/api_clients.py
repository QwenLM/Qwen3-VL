import os
from PIL import Image
import requests
import time
import traceback
from typing import Optional

class OpenAIWrapper:
    """Wrapper for OpenAI API."""
    
    def __init__(self, model, api_base, api_key, timeout=60, retry=5, wait=5):
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.timeout = timeout
        self.retry = retry
        self.wait = wait
        self.fail_msg = 'Failed to obtain answer via API.'
    
    def generate(self, messages):
        """Generate a response from the API."""
        headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {self.api_key}'}
        
        # Format messages for API
        formatted_messages = []
        for msg in messages:
            if msg['type'] == 'text':
                formatted_messages.append({"role": "user", "content": msg['value']})
            elif msg['type'] == 'image':
                # Load and encode the image
                image = Image.open(msg['value'])
                image_data = encode_image_to_base64(image)
                formatted_messages.append({
                    "role": "user", 
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                    ]
                })
        
        payload = {
            "model": self.model,
            "messages": formatted_messages,
            "max_tokens": 1024,
            "temperature": 0
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
        self.fail_msg = 'Failed to obtain answer via API.'
    
    def generate(self, messages):
        """Generate a response from the API."""
        headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {self.api_key}'}
        
        # Format messages for API
        formatted_messages = []
        for msg in messages:
            # print (msg)
            if msg['type'] == 'text':
                formatted_messages.append({"role": "user", "content": [{"type": "text", "text": msg['value']}]})
            elif msg['type'] == 'image':
                # Load and encode the image
                image = Image.open(msg['value'])
                image_data = encode_image_to_base64(image)
                formatted_messages.append({
                    "role": "user", 
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                    ]
                })
        
        payload = {
            "model": self.model,
            "messages": formatted_messages,
            "max_completion_tokens": 1024,
            "temperature": 0,
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
            except requests.exceptions.ConnectionError as conn_err:
                print(f"DashScope: Connection error occurred: {conn_err}")
                time.sleep(self.wait)
            except requests.exceptions.Timeout as timeout_err:
                print(f"DashScope: Timeout error occurred: {timeout_err}")
                time.sleep(self.wait)
            except requests.exceptions.RequestException as req_err:
                print(f"DashScope: Request exception occurred: {req_err}")
                time.sleep(self.wait)
            except Exception as e:
                print(f"DashScope: An error occurred: {e}")
                print(traceback.format_exc())
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

def gpt_key_set():
    openai_key = os.environ.get('OPENAI_API_KEY', None)
    dashscope_key = os.environ.get('CHATGPT_DASHSCOPE_API_KEY', None)

    is_openai_key_valid = isinstance(openai_key, str) and len(openai_key.strip()) > 0
    is_dashscope_key_valid = isinstance(dashscope_key, str) and len(dashscope_key.strip()) > 0

    config_valid = is_openai_key_valid or is_dashscope_key_valid

    print("OPENAI_API_KEY set: ", is_openai_key_valid)
    print("CHATGPT_DASHSCOPE_API_KEY set: ", is_dashscope_key_valid)

    return config_valid