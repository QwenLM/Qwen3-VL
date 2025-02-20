# Qwen2.5-VL


<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-VL/qwen2.5vl_logo.png" width="400"/>
<p>

<p align="center">
        💜 <a href="https://chat.qwenlm.ai/"><b>Qwen Chat</b></a>&nbsp&nbsp | &nbsp&nbsp🤗 <a href="https://huggingface.co/collections/Qwen/qwen25-vl-6795ffac22b334a837c0f9a5">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/organization/qwen">ModelScope</a>&nbsp&nbsp | &nbsp&nbsp📑 <a href="https://qwenlm.github.io/blog/qwen2.5-vl/">Blog</a>&nbsp&nbsp | &nbsp&nbsp📚 <a href="https://github.com/QwenLM/Qwen2.5-VL/tree/main/cookbooks">Cookbooks</a>&nbsp&nbsp | &nbsp&nbsp📑 <a href="https://arxiv.org/abs/2502.13923">Paper</a>&nbsp&nbsp
<br>
🖥️ <a href="https://huggingface.co/spaces/Qwen/Qwen2.5-VL">Demo</a>&nbsp&nbsp | &nbsp&nbsp💬 <a href="https://github.com/QwenLM/Qwen/blob/main/assets/wechat.png">WeChat (微信)</a>&nbsp&nbsp | &nbsp&nbsp🫨 <a href="https://discord.gg/CV4E9rpNSD">Discord</a>&nbsp&nbsp | &nbsp&nbsp📑 <a href="https://help.aliyun.com/zh/model-studio/developer-reference/qwen-vl-api">API</a>&nbsp&nbsp | &nbsp&nbsp🖥️ <a href="https://gallery.pai-ml.com/#/preview/deepLearning/cv/qwen2.5-vl">PAI-DSW</a>
</p>




## Introduction
In the past five months since Qwen2-VL's release, numerous developers have built new models on the Qwen2-VL vision-language models, providing us with valuable feedback. During this period, we focused on building more useful vision-language models. Today, we are excited to introduce the latest addition to the Qwen family: Qwen2.5-VL.


#### Key Enhancements:

* **Powerful Document Parsing Capabilities**: Upgrade text recognition to omnidocument parsing, excelling in processing multi-scene, multilingual, and various built-in (handwriting, tables, charts, chemical formulas, and music sheets) documents.

* **Precise Object Grounding Across Formats**: Unlock improved accuracy in detecting, pointing, and counting objects, accommodating absolute coordinate and JSON formats for advanced spatial reasoning.

* **Ultra-long Video Understanding and Fine-grained Video Grounding**: Extend native dynamic resolution to the temporal dimension, enhancing the ability to understand videos lasting hours while extracting event segments in seconds.

* **Enhanced Agent Functionality for Computer and Mobile Devices**: Leverage advanced grounding, reasoning, and decision-making abilities, boosting the model with superior agent functionality on smartphones and computers.


#### Model Architecture Updates:

* **Dynamic Resolution and Frame Rate Training for Video Understanding**:

We extend dynamic resolution to the temporal dimension by adopting dynamic FPS sampling, enabling the model to comprehend videos at various sampling rates. Accordingly, we update mRoPE in the time dimension with IDs and absolute time alignment, enabling the model to learn temporal sequence and speed, and ultimately acquire the ability to pinpoint specific moments.

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-VL/qwen2.5vl_arc.jpeg" width="80%"/>
<p>


* **Streamlined and Efficient Vision Encoder**

We enhance both training and inference speeds by strategically implementing window attention into the ViT. The ViT architecture is further optimized with SwiGLU and RMSNorm, aligning it with the structure of the Qwen2.5 LLM.


## News
* 2025.02.20: we have released the [Qwen2.5-VL Technical Report](https://arxiv.org/abs/2502.13923). Alongside the report, we have also released AWQ-quantized models for Qwen2.5-VL in three different sizes: [3B](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct-AWQ), [7B](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct-AWQ) , and [72B](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct-AWQ) parameters.
* 2025.01.28: We have released the [Qwen2.5-VL series](https://huggingface.co/Qwen). For more details, please check our [blog](https://qwenlm.github.io/blog/qwen2.5-vl/)!
* 2024.12.25: We have released the [QvQ-72B-Preview](https://huggingface.co/Qwen/QVQ-72B-Preview). QvQ-72B-Preview is an experimental research model, focusing on enhancing visual reasoning capabilities. For more details, please check our [blog](https://qwenlm.github.io/blog/qvq-72b-preview/)!
* 2024.09.19: The instruction-tuned [Qwen2-VL-72B model](https://huggingface.co/Qwen/Qwen2-VL-72B-Instruct) and its quantized version [[AWQ](https://huggingface.co/Qwen/Qwen2-VL-72B-Instruct-AWQ), [GPTQ-Int4](https://huggingface.co/Qwen/Qwen2-VL-72B-Instruct-GPTQ-Int4), [GPTQ-Int8](https://huggingface.co/Qwen/Qwen2-VL-72B-Instruct-GPTQ-Int8)] are now available. We have also released the [Qwen2-VL paper](https://arxiv.org/pdf/2409.12191) simultaneously.
* 2024.08.30: We have released the [Qwen2-VL series](https://huggingface.co/collections/Qwen/qwen2-vl-66cee7455501d7126940800d). The 2B and 7B models are now available, and the 72B model for open source is coming soon. For more details, please check our [blog](https://qwenlm.github.io/blog/qwen2-vl/)!


## Performance



| Dataset            | Qwen2.5-VL-3B<br><sup>([🤗](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)[🤖](https://modelscope.cn/models/qwen/Qwen2.5-VL-3B-Instruct))     | Qwen2.5-VL-7B<br><sup>([🤗](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)[🤖](https://modelscope.cn/models/qwen/Qwen2.5-VL-7B-Instruct))    | Qwen2.5-VL-72B<br><sup>([🤗](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct)[🤖](https://modelscope.cn/models/qwen/Qwen2.5-VL-72B-Instruct)) | Gemini-2 Flash | GPT-4o | Claude3.5 Sonnet | Qwen2-VL 72B | 
|--------------------|--------|--------|----------------|----------------|--------|------------------|--------------|
| MMMU               | 53.1  | 58.6 |**70.2**           | **70.7**           | 70.3   | 70.4             | 64.5         | 
| MMMU Pro           | 31.6  | 38.3 |**51.1**           | **57**             | 54.5   | 54.7             | 46.2         | 
| DocVQA             | 93.9 | 95.7 |**96.4**           | 92.1           | 91.1   | 95.2             | **96.5**         | 
| InfoVQA            | 77.1 | 82.6 | **87.3**          | 77.8           | 80.7   | 74.3             | 84.5         | 
| CC-OCR             | 74.5 |  77.8 | **79.8**          | 73.0           | 66.6   | 62.7             | 68.7         | 
| OCRBenchV2         | 54.3/52.1 | 56.3/57.2 | **61.5/63.7**      | -              | 46.5/32.3 | 45.2/39.6       | 47.8/46.1    | 
| MegaBench          | 28.9 |36.8 | **51.3**          | **55.2**           | 54.2   | 52.1             | 46.8         | 
| MMStar             | 55.8|63.9|**70.8**           | 69.4           | 64.7   | 65.1             | 68.3         | 
| MMBench1.1         |  81.5  | 84.3 |**88.0**           | 83.0           | 82.1   | 83.4             | 86.6         | 
| MathVista          | 62.3  | 68.2  | **74.8**           | 73.1           | 63.8   | 65.4             | 70.5         | 
| MathVision         | 21.2  | 25.1 |**38.1**           | **41.3**           | 30.4   | 38.3             | 25.9         | 
| VideoMME           | 61.5/67.6 | 65.1/71.6 |**73.3/79.1**      | -/-            | 71.9/77.2 | 60/62.9         | 71.2/77.8    | 
| MMBench-Video      | 1.63           | 1.79    |**2.02**           | -              | 1.68   | 1.38             | 1.7          | 
| LVBench            | 43.3        | 45.3 |**47.3**           | -              | 30.8   | -                | -            | 
| CharadesSTA        | 38.8        | 43.6        |**50.9**           | -              | 35.7   | -                | -            | 
| AITZ               | 76.9|81.9 |**83.2**           | -              | 35.3   | -                | -            | 
| Android Control    | 63.7/90.8 | 60.1/91.4 |**67.36/93.7**     | -              | -      | -                | 66.4/84.4    | 
| ScreenSpot         | 55.5|84.7|**87.1**           | 84.0           | 18.1   | 83.0             | -            | 
| ScreenSpot Pro     | 23.9|29.0 |**43.6**           | -              | -      | 17.1             | -            | 
| AndroidWorld       | -  | -  |**35**             | -              | 34.5(SoM) | 27.9            | -            | 
| OSWorld            | -  | -  |**8.83**           | -              | 5.03   | **14.9**             | -            | 





## Quickstart

Below, we provide simple examples to show how to use Qwen2.5-VL with 🤖 ModelScope and 🤗 Transformers.

The code of Qwen2.5-VL has been in the latest Hugging face transformers and we advise you to build from source with command:
```
pip install git+https://github.com/huggingface/transformers accelerate
```
or you might encounter the following error:
```
KeyError: 'qwen2_5_vl'
```


We offer a toolkit to help you handle various types of visual input more conveniently, as if you were using an API. This includes base64, URLs, and interleaved images and videos. You can install it using the following command:

```bash
# It's highly recommended to use `[decord]` feature for faster video loading.
pip install qwen-vl-utils[decord]
```

If you are not using Linux, you might not be able to install `decord` from PyPI. In that case, you can use `pip install qwen-vl-utils` which will fall back to using torchvision for video processing. However, you can still [install decord from source](https://github.com/dmlc/decord?tab=readme-ov-file#install-from-source) to get decord used when loading video.

We are preparing [cookbooks](https://github.com/QwenLM/Qwen2.5-VL/tree/main/cookbooks) for many capabilities, including recognition, localization, document parsing, video understanding, key information extraction, and more. Welcome to learn more!

## Cookbooks

| Cookbook | Description | Open |
| -------- | ----------- | ---- |
| [Universal Recognition](https://github.com/QwenLM/Qwen2.5-VL/blob/main/cookbooks/universal_recognition.ipynb) | Not only identify animals, plants, people, and scenic spots but also recognize various objects such as cars and merchandise. | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/QwenLM/Qwen2.5-VL/blob/main/cookbooks/universal_recognition.ipynb) |
| [Powerful Document Parsing Capabilities](https://github.com/QwenLM/Qwen2.5-VL/blob/main/cookbooks/document_parsing.ipynb) | The parsing of documents has reached a higher level, including not only text but also layout position information and our Qwen HTML format. | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/QwenLM/Qwen2.5-VL/blob/main/cookbooks/document_parsing.ipynb) |
| [Precise Object Grounding Across Formats](https://github.com/QwenLM/Qwen2.5-VL/blob/main/cookbooks/spatial_understanding.ipynb) | Using absolute position coordinates, it supports both boxes and points, allowing for diverse combinations of positioning and labeling tasks. | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/QwenLM/Qwen2.5-VL/blob/main/cookbooks/spatial_understanding.ipynb) |
| [General OCR and Key Information Extraction](https://github.com/QwenLM/Qwen2.5-VL/blob/main/cookbooks/ocr.ipynb) | Stronger text recognition capabilities in natural scenes and multiple languages, supporting diverse key information extraction needs. | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/QwenLM/Qwen2.5-VL/blob/main/cookbooks/ocr.ipynb) |
| [Video Understanding](https://github.com/QwenLM/Qwen2.5-VL/blob/main/cookbooks/video_understanding.ipynb) | Better video OCR, long video understanding, and video grounding. | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/QwenLM/Qwen2.5-VL/blob/main/cookbooks/video_understanding.ipynb) |
| [Mobile Agent](https://github.com/QwenLM/Qwen2.5-VL/blob/main/cookbooks/mobile_agent.ipynb) | Locate and think for mobile phone control. | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/QwenLM/Qwen2.5-VL/blob/main/cookbooks/mobile_agent.ipynb) |
| [Computer-Use Agent](https://github.com/QwenLM/Qwen2.5-VL/blob/main/cookbooks/computer_use.ipynb) | Locate and think for controlling computers and Web. | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/QwenLM/Qwen2.5-VL/blob/main/cookbooks/computer_use.ipynb) |


### Using 🤗  Transformers to Chat

Here we show a code snippet to show you how to use the chat model with `transformers` and `qwen_vl_utils`:

```python
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-7B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processor
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to(model.device)

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
```
<details>
<summary>Multi image inference</summary>

```python
# Messages containing multiple images and a text query
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "file:///path/to/image1.jpg"},
            {"type": "image", "image": "file:///path/to/image2.jpg"},
            {"type": "text", "text": "Identify the similarities between these images."},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Inference
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
```
</details>

<details>
<summary>Video inference</summary>

```python
# Messages containing a images list as a video and a text query
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": [
                    "file:///path/to/frame1.jpg",
                    "file:///path/to/frame2.jpg",
                    "file:///path/to/frame3.jpg",
                    "file:///path/to/frame4.jpg",
                ],
            },
            {"type": "text", "text": "Describe this video."},
        ],
    }
]

# Messages containing a local video path and a text query
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": "file:///path/to/video1.mp4",
                "max_pixels": 360 * 420,
                "fps": 1.0,
            },
            {"type": "text", "text": "Describe this video."},
        ],
    }
]

# Messages containing a video url and a text query
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-VL/space_woaudio.mp4",
            },
            {"type": "text", "text": "Describe this video."},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    fps=fps,
    padding=True,
    return_tensors="pt",
    **video_kwargs,
)
inputs = inputs.to("cuda")

# Inference
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
```

Video URL compatibility largely depends on the third-party library version. The details are in the table below. change the backend by `FORCE_QWENVL_VIDEO_READER=torchvision` or `FORCE_QWENVL_VIDEO_READER=decord` if you prefer not to use the default one.

| Backend     | HTTP | HTTPS |
|-------------|------|-------|
| torchvision >= 0.19.0 | ✅  | ✅   |
| torchvision < 0.19.0  | ❌  | ❌   |
| decord      | ✅  | ❌   |
</details>

<details>
<summary>Batch inference</summary>

```python
# Sample messages for batch inference
messages1 = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "file:///path/to/image1.jpg"},
            {"type": "image", "image": "file:///path/to/image2.jpg"},
            {"type": "text", "text": "What are the common elements in these pictures?"},
        ],
    }
]
messages2 = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Who are you?"},
]
# Combine messages for batch processing
messages = [messages1, messages2]

# Preparation for batch inference
texts = [
    processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    for msg in messages
]
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=texts,
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Batch Inference
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_texts = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_texts)
```
</details>

### 🤖 ModelScope
We strongly advise users especially those in mainland China to use ModelScope. `snapshot_download` can help you solve issues concerning downloading checkpoints.

### More Usage Tips

For input images, we support local files, base64, and URLs. For videos, we currently only support local files.

```python
# You can directly insert a local file path, a URL, or a base64-encoded image into the position where you want in the text.
## Local file path
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "file:///path/to/your/image.jpg"},
            {"type": "text", "text": "Describe this image."},
        ],
    }
]
## Image URL
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "http://path/to/your/image.jpg"},
            {"type": "text", "text": "Describe this image."},
        ],
    }
]
## Base64 encoded image
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "data:image;base64,/9j/..."},
            {"type": "text", "text": "Describe this image."},
        ],
    }
]
```
#### Image Resolution for performance boost

The model supports a wide range of resolution inputs. By default, it uses the native resolution for input, but higher resolutions can enhance performance at the cost of more computation. Users can set the minimum and maximum number of pixels to achieve an optimal configuration for their needs, such as a token count range of 256-1280, to balance speed and memory usage.

```python
min_pixels = 256 * 28 * 28
max_pixels = 1280 * 28 * 28
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels
)
```

Besides, We provide two methods for fine-grained control over the image size input to the model:

1. Specify exact dimensions: Directly set `resized_height` and `resized_width`. These values will be rounded to the nearest multiple of 28.

2. Define min_pixels and max_pixels: Images will be resized to maintain their aspect ratio within the range of min_pixels and max_pixels.

```python
# resized_height and resized_width
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "file:///path/to/your/image.jpg",
                "resized_height": 280,
                "resized_width": 420,
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]
# min_pixels and max_pixels
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "file:///path/to/your/image.jpg",
                "min_pixels": 50176,
                "max_pixels": 50176,
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]
```

#### Add ids for Multiple Image Inputs
By default, images and video content are directly included in the conversation. When handling multiple images, it's helpful to add labels to the images and videos for better reference. Users can control this behavior with the following settings:
<details>
<summary>Add vision ids</summary>

```python
conversation = [
    {
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": "Hello, how are you?"}],
    },
    {
        "role": "assistant",
        "content": "I'm doing well, thank you for asking. How can I assist you today?",
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Can you describe these images and video?"},
            {"type": "image"},
            {"type": "image"},
            {"type": "video"},
            {"type": "text", "text": "These are from my vacation."},
        ],
    },
    {
        "role": "assistant",
        "content": "I'd be happy to describe the images and video for you. Could you please provide more context about your vacation?",
    },
    {
        "role": "user",
        "content": "It was a trip to the mountains. Can you see the details in the images and video?",
    },
]

# default:
prompt_without_id = processor.apply_chat_template(
    conversation, add_generation_prompt=True
)
# Excepted output: '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Hello, how are you?<|im_end|>\n<|im_start|>assistant\nI'm doing well, thank you for asking. How can I assist you today?<|im_end|>\n<|im_start|>user\nCan you describe these images and video?<|vision_start|><|image_pad|><|vision_end|><|vision_start|><|image_pad|><|vision_end|><|vision_start|><|video_pad|><|vision_end|>These are from my vacation.<|im_end|>\n<|im_start|>assistant\nI'd be happy to describe the images and video for you. Could you please provide more context about your vacation?<|im_end|>\n<|im_start|>user\nIt was a trip to the mountains. Can you see the details in the images and video?<|im_end|>\n<|im_start|>assistant\n'


# add ids
prompt_with_id = processor.apply_chat_template(
    conversation, add_generation_prompt=True, add_vision_id=True
)
# Excepted output: '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nPicture 1: <|vision_start|><|image_pad|><|vision_end|>Hello, how are you?<|im_end|>\n<|im_start|>assistant\nI'm doing well, thank you for asking. How can I assist you today?<|im_end|>\n<|im_start|>user\nCan you describe these images and video?Picture 2: <|vision_start|><|image_pad|><|vision_end|>Picture 3: <|vision_start|><|image_pad|><|vision_end|>Video 1: <|vision_start|><|video_pad|><|vision_end|>These are from my vacation.<|im_end|>\n<|im_start|>assistant\nI'd be happy to describe the images and video for you. Could you please provide more context about your vacation?<|im_end|>\n<|im_start|>user\nIt was a trip to the mountains. Can you see the details in the images and video?<|im_end|>\n<|im_start|>assistant\n'
```
</details>

#### Flash-Attention 2 to speed up generation

First, make sure to install the latest version of Flash Attention 2:

```bash
pip install -U flash-attn --no-build-isolation
```

Also, you should have a hardware that is compatible with Flash-Attention 2. Read more about it in the official documentation of the [flash attention repository](https://github.com/Dao-AILab/flash-attention). FlashAttention-2 can only be used when a model is loaded in `torch.float16` or `torch.bfloat16`.

To load and run a model using Flash Attention-2, simply add `attn_implementation="flash_attention_2"` when loading the model as follows:

```python
from transformers import Qwen2_5_VLForConditionalGeneration

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", 
    torch_dtype=torch.bfloat16, 
    attn_implementation="flash_attention_2",
)
```

### Processing Long Texts

The current `config.json` is set for context length up to 32,768 tokens.
To handle extensive inputs exceeding 32,768 tokens, we utilize [YaRN](https://arxiv.org/abs/2309.00071), a technique for enhancing model length extrapolation, ensuring optimal performance on lengthy texts.

For supported frameworks, you could add the following to `config.json` to enable YaRN:

```
{
	...,
    "type": "yarn",
    "mrope_section": [
        16,
        24,
        24
    ],
    "factor": 4,
    "original_max_position_embeddings": 32768
}
```

However, it should be noted that this method has a significant impact on the performance of temporal and spatial localization tasks, and is therefore not recommended for use.

At the same time, for long video inputs, since MRoPE itself is more economical with ids, the max_position_embeddings can be directly modified to a larger value, such as 64k.


### Try Qwen2.5-VL-72B with API!

To explore Qwen2.5-VL-72B, a more fascinating multimodal model, we encourage you to test our cutting-edge API service. Let's start the exciting journey right now!

#### Installation
```bash
pip install dashscope
```

#### Examples
```python
import dashscope


dashscope.api_key = "your_api_key"

messages = [{
    'role': 'user',
    'content': [
        {
            'image': "https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg"
        },
        {
            'text': 'What are in the image?'
        },
    ]
}]

response = dashscope.MultiModalConversation.call(model='qwen2.5-vl-72b-instruct', messages=messages)
print(response)
```

For more usage, please refer to the tutorial at [aliyun](https://help.aliyun.com/zh/model-studio/developer-reference/qwen-vl-api).




## Demo
### Web UI Example

In this section, we provide instructions for users to build a web-based user interface (UI) demo. This UI demo allows users to interact with a predefined model or application through a web browser. Follow the steps below to get started.

#### Installation

Before you begin, ensure that you have the required dependencies installed on your system. You can install them by running the following command:

```bash
pip install -r requirements_web_demo.txt
```

#### Running the Demo with FlashAttention-2

Once the required packages are installed, you can launch the web demo using the following command. This command will start a web server and provide you with a link to access the UI in your web browser.

**Recommended**: For enhanced performance and efficiency, especially in multi-image and video processing scenarios, we strongly recommend using [FlashAttention-2](https://github.com/Dao-AILab/flash-attention). FlashAttention-2 provides significant improvements in memory usage and speed, making it ideal for handling large-scale models and data processing.

To enable FlashAttention-2, use the following command:

```bash
python web_demo_mm.py --flash-attn2
```

This will load the model with FlashAttention-2 enabled.

**Default Usage**: If you prefer to run the demo without FlashAttention-2 or if you do not specify the `--flash-attn2` option, the demo will load the model using the standard attention implementation:

```bash
python web_demo_mm.py
```

After running the command, you’ll see a link generated in the terminal similar to this:

```
Running on local: http://127.0.0.1:7860/
```

Copy this link and paste it into your browser to access the web UI, where you can interact with the model by inputting text, uploading images, or using any other provided functionalities.

##### Running the Streaming Video Chat Demo
An experimental streaming video chat demo is also available in the ``web_demo_streaming`` directory.

To run the streaming video chat demo, use the following command:

```bash
cd web_demo_streaming/
python app.py --flash-attn2
```

If you prefer to run the demo without FlashAttention-2, use the following command:
```bash
cd web_demo_streaming/
python app.py
```

This demo supports webcam/screen capture as its video input source. To support screen capture video input, we use code snippet from the following hugginface space: [gstaff/gradio-screen-recorder](https://huggingface.co/spaces/gstaff/gradio-screen-recorder/tree/main).


## Deployment

We recommend using vLLM for fast Qwen2.5-VL deployment and inference. You need to install `vllm>0.7.2` to enable Qwen2.5-VL support. You can also use our [official docker image](#-docker).

You can also check [vLLM official documentation](https://docs.vllm.ai/en/latest/serving/multimodal_inputs.html) for more details about online serving and offline inference.

### Installation
```bash
pip install git+https://github.com/huggingface/transformers@f3f6c86582611976e72be054675e2bf0abb5f775
pip install accelerate
pip install qwen-vl-utils
pip install 'vllm>0.7.2'

```
### Start an OpenAI API Service

Run the command below to start an OpenAI-compatible API service:

```bash
vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8000 --host 0.0.0.0 --dtype bfloat16 --limit-mm-per-prompt image=5,video=5
```

Then you can use the chat API as below (via curl or Python API):

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
    "model": "Qwen/Qwen2.5-VL-7B-Instruct",
    "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "https://modelscope.oss-cn-beijing.aliyuncs.com/resource/qwen.png"}},
        {"type": "text", "text": "What is the text in the illustrate?"}
    ]}
    ]
    }'
```

```python
from openai import OpenAI

# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

chat_response = client.chat.completions.create(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://modelscope.oss-cn-beijing.aliyuncs.com/resource/qwen.png"
                    },
                },
                {"type": "text", "text": "What is the text in the illustrate?"},
            ],
        },
    ],
)
print("Chat response:", chat_response)
```

You can also upload base64-encoded local images (see [OpenAI API protocol document](https://platform.openai.com/docs/guides/vision/uploading-base-64-encoded-images) for more details):
```python
import base64
from openai import OpenAI
# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
image_path = "/path/to/local/image.png"
with open(image_path, "rb") as f:
    encoded_image = base64.b64encode(f.read())
encoded_image_text = encoded_image.decode("utf-8")
base64_qwen = f"data:image;base64,{encoded_image_text}"
chat_response = client.chat.completions.create(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": base64_qwen
                    },
                },
                {"type": "text", "text": "What is the text in the illustrate?"},
            ],
        },
    ],
)
print("Chat response:", chat_response)
```

For videos, you can use the chat API  as follows：
```python
import base64
import numpy as np
from PIL import Image
from io import BytesIO
from openai import OpenAI
from qwen_vl_utils import process_vision_info


# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)


video_messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": [
        {"type": "text", "text": "请用表格总结一下视频中的商品特点"},
        {
            "type": "video",
            "video": "https://duguang-labelling.oss-cn-shanghai.aliyuncs.com/qiansun/video_ocr/videos/50221078283.mp4",
            "total_pixels": 20480 * 28 * 28, "min_pixels": 16 * 28 * 2, 
            'fps': 3.0  # The default value is 2.0, but for demonstration purposes, we set it to 3.0.
        }]
    },
]


def prepare_message_for_vllm(content_messages):
    """
    The frame extraction logic for videos in `vLLM` differs from that of `qwen_vl_utils`.
    Here, we utilize `qwen_vl_utils` to extract video frames, with the `media_typ`e of the video explicitly set to `video/jpeg`.
    By doing so, vLLM will no longer attempt to extract frames from the input base64-encoded images.
    """
    vllm_messages, fps_list = [], []
    for message in content_messages:
        message_content_list = message["content"]
        if not isinstance(message_content_list, list):
            vllm_messages.append(message)
            continue

        new_content_list = []
        for part_message in message_content_list:
            if 'video' in part_message:
                video_message = [{'content': [part_message]}]
                image_inputs, video_inputs, video_kwargs = process_vision_info(video_message, return_video_kwargs=True)
                assert video_inputs is not None, "video_inputs should not be None"
                video_input = (video_inputs.pop()).permute(0, 2, 3, 1).numpy().astype(np.uint8)
                fps_list.extend(video_kwargs.get('fps', []))

                # encode image with base64
                base64_frames = []
                for frame in video_input:
                    img = Image.fromarray(frame)
                    output_buffer = BytesIO()
                    img.save(output_buffer, format="jpeg")
                    byte_data = output_buffer.getvalue()
                    base64_str = base64.b64encode(byte_data).decode("utf-8")
                    base64_frames.append(base64_str)

                part_message = {
                    "type": "video_url",
                    "video_url": {"url": f"data:video/jpeg;base64,{','.join(base64_frames)}"}
                }
            new_content_list.append(part_message)
        message["content"] = new_content_list
        vllm_messages.append(message)
    return vllm_messages, {'fps': fps_list}


video_messages, video_kwargs = prepare_message_for_vllm(video_messages)
chat_response = client.chat.completions.create(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    messages=video_messages,
    extra_body={
        "mm_processor_kwargs": video_kwargs
    }
)
print("Chat response:", chat_response)
```

### Inference Locally

You can also use vLLM to inference Qwen2.5-VL locally:

```python
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info

MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"

llm = LLM(
    model=MODEL_PATH,
    limit_mm_per_prompt={"image": 10, "video": 10},
)

sampling_params = SamplingParams(
    temperature=0.1,
    top_p=0.001,
    repetition_penalty=1.05,
    max_tokens=256,
    stop_token_ids=[],
)

image_messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://modelscope.oss-cn-beijing.aliyuncs.com/resource/qwen.png",
                "min_pixels": 224 * 224,
                "max_pixels": 1280 * 28 * 28,
            },
            {"type": "text", "text": "What is the text in the illustrate?"},
        ],
    },
]


# For video input, you can pass following values instead:
# "type": "video",
# "video": "<video URL>",
video_messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": [
            {"type": "text", "text": "请用表格总结一下视频中的商品特点"},
            {
                "type": "video", 
                "video": "https://duguang-labelling.oss-cn-shanghai.aliyuncs.com/qiansun/video_ocr/videos/50221078283.mp4",
                "total_pixels": 20480 * 28 * 28, "min_pixels": 16 * 28 * 28
            }
        ]
    },
]

# Here we use video messages as a demonstration
messages = video_messages

processor = AutoProcessor.from_pretrained(MODEL_PATH)
prompt = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

mm_data = {}
if image_inputs is not None:
    mm_data["image"] = image_inputs
if video_inputs is not None:
    mm_data["video"] = video_inputs

llm_inputs = {
    "prompt": prompt,
    "multi_modal_data": mm_data,

    # FPS will be returned in video_kwargs
    "mm_processor_kwargs": video_kwargs,
}

outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
generated_text = outputs[0].outputs[0].text

print(generated_text)
```


## 🐳 Docker

To simplify the deploy process, we provide docker images with pre-build environments: [qwenllm/qwenvl](https://hub.docker.com/r/qwenllm/qwenvl). You only need to install the driver and download model files to launch demos.

```bash
docker run --gpus all --ipc=host --network=host --rm --name qwen2.5 -it qwenllm/qwenvl:2.5-cu121 bash
```

## Citation

If you find our paper and code useful in your research, please consider giving a star :star: and citation :pencil: :)




```BibTeX

@article{Qwen2.5-VL,
  title={Qwen2.5-VL Technical Report},
  author={Bai, Shuai and Chen, Keqin and Liu, Xuejing and Wang, Jialin and Ge, Wenbin and Song, Sibo and Dang, Kai and Wang, Peng and Wang, Shijie and Tang, Jun and Zhong, Humen and Zhu, Yuanzhi and Yang, Mingkun and Li, Zhaohai and Wan, Jianqiang and Wang, Pengfei and Ding, Wei and Fu, Zheren and Xu, Yiheng and Ye, Jiabo and Zhang, Xi and Xie, Tianbao and Cheng, Zesen and Zhang, Hang and Yang, Zhibo and Xu, Haiyang and Lin, Junyang},
  journal={arXiv preprint arXiv:2502.13923},
  year={2025}
}

@article{Qwen2-VL,
  title={Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution},
  author={Wang, Peng and Bai, Shuai and Tan, Sinan and Wang, Shijie and Fan, Zhihao and Bai, Jinze and Chen, Keqin and Liu, Xuejing and Wang, Jialin and Ge, Wenbin and Fan, Yang and Dang, Kai and Du, Mengfei and Ren, Xuancheng and Men, Rui and Liu, Dayiheng and Zhou, Chang and Zhou, Jingren and Lin, Junyang},
  journal={arXiv preprint arXiv:2409.12191},
  year={2024}
}

@article{Qwen-VL,
  title={Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond},
  author={Bai, Jinze and Bai, Shuai and Yang, Shusheng and Wang, Shijie and Tan, Sinan and Wang, Peng and Lin, Junyang and Zhou, Chang and Zhou, Jingren},
  journal={arXiv preprint arXiv:2308.12966},
  year={2023}
}
```

<br>
