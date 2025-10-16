import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

# default: Load the model on the available device(s)
# model_path = "./model/Qwen/Qwen3-VL-30B-A3B-Instruct"
# model_path = "./model/Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"
model_path = "./model/Qwen/Qwen3-VL-4B-Instruct"

# model = AutoModelForImageTextToText.from_pretrained(
#     model_path,
#     dtype="auto",
#     # device_map="cuda:0",
# )

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

model = model.to("cuda")
processor = AutoProcessor.from_pretrained(model_path)
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "./0199c890-3a3e-7352-8b97-2747334f3f3c.jpg",
            },
            # {   
            #     "type": "text", 
            #     "text": "图像宽1920，高1080，对以[x1,y1,x2,y2]矩形坐标顺序的[1249,547,1532,755]区域进行详细的描述"
            # },
            {   
                "type": "text", 
                "text": "图像宽1920，高1080，图片中有没有人？"
            },
        ],
    }
]

import time
start = time.time()
for _ in range(50):
    # Preparation for inference
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=1024,)
    # generated_ids = model.generate(**inputs, max_new_tokens=128,
    #     do_sample=True,
    #     temperature=0.7,
    #     top_p=0.9,
    #     repetition_penalty=1.1)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text)

end = time.time()
print("avg time:", (end - start) / 50)