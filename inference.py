from transformers import AutoModelForImageTextToText, AutoProcessor

# default: Load the model on the available device(s)
# model_path = "./model/Qwen/Qwen3-VL-30B-A3B-Instruct"
model_path = "./model/Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"
model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    dtype="auto",
    # device_map="cuda:0",
)
model = model.to("cuda")

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# import torch
# model = AutoModelForImageTextToText.from_pretrained(
#     model_path,
#     dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )
processor = AutoProcessor.from_pretrained(model_path)
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "./demo.jpeg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

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
generated_ids = model.generate(**inputs, max_new_tokens=128,)
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
