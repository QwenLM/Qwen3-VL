from starlette.formparsers import MultiPartParser
MultiPartParser.max_file_size = 20 * 1024 * 1024

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
import ast
from PIL import Image
from pathlib import Path
import io

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

model_path = "./model/Qwen/Qwen3-VL-4B-Instruct"

model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    dtype="auto",
)
# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = AutoModelForImageTextToText.from_pretrained(
#     model_path,
#     dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
# )
model = model.to("cuda").eval()
processor = AutoProcessor.from_pretrained(model_path)

torch.cuda.empty_cache()

app = FastAPI()

@torch.inference_mode()
def inference(question: str, image: str):
    print("Question: ", question)
    print("Image path: ", image)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {   
                    "type": "text", 
                    "text": question,
                },
            ],
        }
    ]

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
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    print("Prediction: ", output_text)
    return output_text


@app.post("/understand_image_and_question")
async def understand_image_and_question(
    question: str = Form(...),
    file: UploadFile = File(...),
):
    # 统计执行时间
    import time
    start_time = time.time()
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))
    image_path = Path(f'./tmp/{file.filename}')
    image.save(image_path)
    response = inference(question, image_path.absolute().as_posix())
    print("执行输出: ", response)
    end_time = time.time()
    print("执行时间: ", end_time - start_time)
    return JSONResponse({"response": response})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10058)