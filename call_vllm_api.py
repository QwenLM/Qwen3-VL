from openai import OpenAI
 
# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"
 
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://ofasys-multimodal-wlcb-3-toshanghai.oss-accelerate.aliyuncs.com/wpf272043/keepme/image/receipt.png"
                }
            },
            {
                "type": "text",
                "text": "Read all the text in the image."
            }
        ]
    }
]
chat_response = client.chat.completions.create(
    model="./model/Qwen/Qwen3-VL-4B-Instruct",
    messages=messages,
)
print("Chat response:", chat_response.model_dump_json())

