# Qwen2-VL


<p align="center">
    <img src="https://qianwen-res.oss-accelerate-overseas.aliyuncs.com/Qwen2-VL/qwen2VL_logo.png" width="400"/>
<p>

<p align="center">
        💜 <a href="https://chat.qwenlm.ai/"><b>Qwen Chat</b></a>&nbsp&nbsp | &nbsp&nbsp🤗 <a href="https://huggingface.co/collections/Qwen/qwen2-vl-66cee7455501d7126940800d">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/organization/qwen">ModelScope</a>&nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://qwenlm.github.io/blog/qwen2-vl/">Blog</a> &nbsp&nbsp| &nbsp&nbsp 📑 <a href="https://arxiv.org/pdf/2409.12191">Paper</a> &nbsp&nbsp  </a>
<br>
🖥️ <a href="https://huggingface.co/spaces/Qwen/Qwen2-VL">Demo</a>&nbsp&nbsp | &nbsp&nbsp💬 <a href="https://github.com/QwenLM/Qwen/blob/main/assets/wechat.png">WeChat (微信)</a>&nbsp&nbsp | &nbsp&nbsp🫨 <a href="https://discord.gg/CV4E9rpNSD">Discord</a>&nbsp&nbsp | &nbsp&nbsp<a href="https://help.aliyun.com/zh/model-studio/developer-reference/qwen-vl-api"> 📑 API</a>&nbsp&nbsp | &nbsp&nbsp🖥️ <a href="https://gallery.pai-ml.com/#/preview/deepLearning/cv/qwen2-vl">PAI-DSW</a>&nbsp&nbsp
</p>




## Introduction

After a year's relentless efforts, today we are thrilled to release **Qwen2-VL**! Qwen2-VL is the latest version of the vision language models in the Qwen model families. 

#### Key Enhancements:

* **SoTA understanding of images of various resolution & ratio**: Qwen2-VL achieves state-of-the-art performance on visual understanding benchmarks, including MathVista, DocVQA, RealWorldQA, MTVQA, etc.

* **Understanding videos of 20min+**: with the online streaming capabilities, Qwen2-VL can understand videos over 20 minutes by high-quality video-based question answering, dialog, content creation, etc.

* **Agent that can operate your mobiles, robots, etc.**: with the abilities of complex reasoning and decision making, Qwen2-VL can be integrated with devices like mobile phones, robots, etc., for automatic operation based on visual environment and text instructions.

* **Multilingual Support**: to serve global users, besides English and Chinese, Qwen2-VL now supports the understanding of texts in different languages inside images, including most European languages, Japanese, Korean, Arabic, Vietnamese, etc.

#### Model Architecture Updates:

* **Naive Dynamic Resolution**: Unlike before, Qwen2-VL can handle arbitrary image resolutions, mapping them into a dynamic number of visual tokens, offering a more human-like visual processing experience.
<p align="center">
    <img src="https://qianwen-res.oss-accelerate-overseas.aliyuncs.com/Qwen2-VL/qwen2_vl_framework.jpg" width="80%"/>
<p>

* **Multimodal Rotary Position Embedding (M-ROPE)**: Decomposes positional embedding into parts to capture 1D textual, 2D visual, and 3D video positional information, enhancing its multimodal processing capabilities.

<p align="center">
    <img src="http://qianwen-res.oss-accelerate-overseas.aliyuncs.com/Qwen2-VL/mrope.png" width="80%"/>
<p>


We have open-sourced Qwen2-VL models, including Qwen2-VL-2B and Qwen2-VL-7B under the Apache 2.0 license, as well as Qwen2-VL-72B under the Qwen license. These models are now integrated with Hugging Face Transformers, vLLM, and other third-party frameworks. We hope you enjoy using them!


## News
* 2024.09.19: The instruction-tuned [Qwen2-VL-72B model](https://huggingface.co/Qwen/Qwen2-VL-72B-Instruct) and its quantized version [[AWQ](https://huggingface.co/Qwen/Qwen2-VL-72B-Instruct-AWQ), [GPTQ-Int4](https://huggingface.co/Qwen/Qwen2-VL-72B-Instruct-GPTQ-Int4), [GPTQ-Int8](https://huggingface.co/Qwen/Qwen2-VL-72B-Instruct-GPTQ-Int8)] are now available. We have also released the [Qwen2-VL paper](https://arxiv.org/pdf/2409.12191) simultaneously.
* 2024.08.30: We have released the [Qwen2-VL series]("https://huggingface.co/collections/Qwen/qwen2-vl-66cee7455501d7126940800d). The 2B and 7B models are now available, and the 72B model for opensource is coming soon. For more details, please check our [blog](https://qwenlm.github.io/blog/qwen2-vl/)!


## Performance
### Image Benchmarks

| Benchmark | Previous SoTA<br><sup>(Open-source LVLM)<sup> | Claude-3.5 Sonnet | GPT-4o | **Qwen2-VL-72B**<br><sup>([🤗](https://huggingface.co/Qwen/Qwen2-VL-72B-Instruct) [🤖](https://modelscope.cn/models/qwen/Qwen2-VL-72B-Instruct) |**Qwen2-VL-7B**<br><sup>([🤗](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) [🤖](https://modelscope.cn/models/qwen/Qwen2-VL-7B-Instruct)) |**Qwen2-VL-2B**<br><sup>([🤗](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)[🤖](https://modelscope.cn/models/qwen/Qwen2-VL-2B-Instruct)) 
| :--- | :---: | :---: | :---: | :---: |:---: |:---: |
| MMMU<sub>val</sub>  | 58.3 | 68.3 | **69.1** | 64.5 | 54.1|41.1
| MMMU-Pro | 46.9 | 51.5 | **51.9** | 46.2 | 30.5 | 21.2 
| DocVQA<sub>test</sub>  | 94.1 | 95.2 | 92.8 | **96.5** | 94.5| 90.1
| InfoVQA<sub>test</sub>  | 82.0 | - | - | **84.5** | 76.5|65.5
| ChartQA<sub>test</sub>  | 88.4 | **90.8** | 85.7 | 88.3 |83.0| 73.5
| TextVQA<sub>val</sub>  | 84.4 | - | - | **85.5** |84.3|79.7
| OCRBench | 852 | 788 | 736 | **877** |845| 794
| MTVQA | 17.3 | 25.7 | 27.8 | **30.9** |25.6| 18.1
| VCR<sub>en easy</sub>  | 84.67 | 63.85 | 91.55 | **91.93** | 89.70| 81.45
| VCR<sub>zh easy</sub>  | 22.09 | 1.0| 14.87 | **65.37** | 59.94| 46.16
| RealWorldQA | 72.2 | 60.1 | 75.4 | **77.8** | 70.1| 62.9
| MME<sub>sum</sub>   | 2414.7 | 1920.0 | 2328.7 | **2482.7** | 2326.8 | 1872.0
| MMBench-EN<sub>test</sub>  | **86.5** | 79.7 | 83.4 | **86.5** | 83.0 | 74.9
| MMBench-CN<sub>test</sub>  | 86.3 | 80.7 | 82.1 | **86.6** | 80.5| 73.5
| MMBench-V1.1<sub>test</sub>  | 85.5 | 78.5 | 82.2 | **85.9** |80.7| 72.2
| MMT-Bench<sub>test</sub> | 63.4 | - | 65.5 | **71.7** |63.7| 54.5
| MMStar | 67.1 | 62.2 | 63.9 | **68.3** |60.7| 48.0
| MMVet<sub>GPT-4-Turbo</sub>  | 65.7 | 66.0 | 69.1 | **74.0** |62.0| 49.5
| HallBench<sub>avg</sub>  | 55.2 | 49.9 | 55.0 | **58.1** | 50.6 | 41.7
| MathVista<sub>testmini</sub>  | 67.5 | 67.7 | 63.8 | **70.5** |58.2| 43.0
| MathVision  | 16.97 | - | **30.4** | 25.9 | 16.3| 12.4

### Video Benchmarks

| Benchmark |  Previous SoTA<br><sup>(Open-source LVLM)<sup> | Gemini 1.5-Pro | GPT-4o | **Qwen2-VL-72B**<br><sup>([🤗](https://huggingface.co/Qwen/Qwen2-VL-72B-Instruct) [🤖](https://modelscope.cn/models/qwen/Qwen2-VL-72B-Instruct)) |**Qwen2-VL-7B**<br><sup>([🤗](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) [🤖](https://modelscope.cn/models/qwen/Qwen2-VL-7B-Instruct)) |**Qwen2-VL-2B**<br><sup>([🤗](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)[🤖](https://modelscope.cn/models/qwen/Qwen2-VL-2B-Instruct)) 
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| MVBench | 69.6 | - | - | **73.6** | 67.0| 63.2 
| PerceptionTest<sub>test</sub> |  66.9 | - | - | **68.0** | 62.3 |53.9
| EgoSchema<sub>test</sub>  | 62.0 | 63.2 | 72.2 | **77.9** | 66.7 |54.9
| Video-MME<br><sub>(wo/w subs)</sub>  | 66.3/69.6  | **75.0**/**81.3** | 71.9/77.2 | 71.2/77.8 | 63.3/69.0 |55.6/60.4

### Agent Benchmarks
|     |Benchmark | Metric | Previous SoTA | GPT-4o | **Qwen2-VL-72B** |
| :-- | :-- | :--: | :--: | :--: | :--: |
|   General  | FnCall<sup>[1]</sup> | TM | - | 90.2 | **93.1** |
|     |  | EM | - | 50.0 | **53.2** |
|   Game  | Number Line | SR | 89.4<sup>[2]</sup> | 91.5 | **100.0** |
|     | BlackJack | SR | 40.2<sup>[2]</sup> | 34.5 | **42.6** |
|     | EZPoint | SR | 50.0<sup>[2]</sup> | 85.5 | **100.0** |
|     | Point24 | SR | 2.6<sup>[2]</sup> | 3.0 | **4.5** |
| Android | AITZ  | TM | 83.0<sup>[3]</sup> | 70.0 | **89.6** |
|     |  | EM | 47.7<sup>[3]</sup> | 35.3 | **72.1** |
| AI2THOR | ALFRED<sub>valid-unseen</sub> | SR | 67.7<sup>[4]</sup> | - | **67.8** |
|     |  | GC | 75.3<sup>[4]</sup> | - | **75.8** | 
|  VLN   | R2R<sub>valid-unseen</sub>  | SR | **79.0** | 43.7<sup>[5]</sup> | 51.7 | 
|     | REVERIE<sub>valid-unseen</sub> | SR | **61.0** | 31.6<sup>[5]</sup> | 31.0 | 

SR, GC, TM and EM are short for success rate, goal-condition success, type match and exact match. ALFRED is supported by SAM<sup>[6]</sup>.
1. Self-Curated Function Call Benchmark by Qwen Team
2. Fine-Tuning Large Vision-Language Models as Decision-Making Agents via Reinforcement Learning
3. Android in the Zoo: Chain-of-Action-Thought for GUI Agents
4. ThinkBot: Embodied Instruction Following with Thought Chain Reasoning
5. MapGPT: Map-Guided Prompting with Adaptive Path Planning for Vision-and-Language Navigation
6. Segment Anything.

### Multilingual Benchmarks

<table style="width:75%; text-align:center;">
    <tr>
        <th>Models</th>
        <td>AR </td>
        <td>DE </td>
        <td>FR </td>
        <td>IT </td>
        <td>JA </td>
        <td>KO </td>
        <td>RU </td>
        <td>TH </td>
        <td>VI </td>
        <td>AVG</td>
    </tr>
    <tr>
        <th align="left">Qwen2-VL-72B</th>
        <td>20.7 </td>
        <td>36.5 </td>
        <td>44.1 </td>
        <td>42.8 </td>
        <td>21.6 </td>
        <td>37.4 </td>
        <td>15.6 </td>
        <td>17.7 </td>
        <td>41.6 </td>
        <td><b>30.9</b></td>
    </tr>
    <tr>
        <th align="left">GPT-4o</th>
        <td>20.2 </td>
        <td>34.2 </td>
        <td>41.2 </td>
        <td>32.7 </td>
        <td>20.0 </td>
        <td>33.9 </td>
        <td>11.5 </td>
        <td>22.5 </td>
        <td>34.2 </td>
        <td>27.8</td>
    </tr>
        <tr>
        <th align="left">Claude3 Opus</th>
        <td>15.1 </td>
        <td>33.4 </td>
        <td>40.6 </td>
        <td>34.4 </td>
        <td>19.4 </td>
        <td>27.2 </td>
        <td>13.0 </td>
        <td>19.5 </td>
        <td>29.1 </td>
        <td>25.7 </td>
    </tr>
    <tr>
        <th align="left">Gemini Ultra</th>
        <td>14.7 </td>
        <td>32.3 </td>
        <td>40.0 </td>
        <td>31.8 </td>
        <td>12.3 </td>
        <td>17.2 </td>
        <td>11.8 </td>
        <td>20.3 </td>
        <td>28.6 </td>
        <td>23.2</td>
    </tr>
</table>

These results are evaluated on the benchmark of [MTVQA](https://github.com/bytedance/MTVQA/tree/main).

## Quickstart

Below, we provide simple examples to show how to use Qwen2-VL with 🤖 ModelScope and 🤗 Transformers.

The code of Qwen2-VL has been in the latest Hugging face transformers and we advise you to build from source with command:
```
pip install git+https://github.com/huggingface/transformers@21fac7abba2a37fae86106f87fcf9974fd1e3830 accelerate
```
or you might encounter the following error:
```
KeyError: 'qwen2_vl'
```

- ⚠️**NOTE**: Current latest version of `transformers` have [a bug](https://github.com/huggingface/transformers/issues/33401) when loading Qwen2-VL config, so you need to install a specific version of transformers as above.

We offer a toolkit to help you handle various types of visual input more conveniently, as if you were using an API. This includes base64, URLs, and interleaved images and videos. You can install it using the following command:

```bash
# It's highly recommanded to use `[decord]` feature for faster video loading.
pip install qwen-vl-utils[decord]
```

If you are not using Linux, you might not be able to install `decord` from PyPI. In that case, you can use `pip install qwen-vl-utils` which will fall back to using torchvision for video processing. However, you can still [install decord from source](https://github.com/dmlc/decord?tab=readme-ov-file#install-from-source) to get decord used when loading video.

### Using 🤗  Transformers to Chat

Here we show a code snippet to show you how to use the chat model with `transformers` and `qwen_vl_utils`:

```python
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

# default: Load the model on the available device(s)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2-VL-7B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processer
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

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
inputs = inputs.to("cuda")

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
    "Qwen/Qwen2-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels
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
from transformers import Qwen2VLForConditionalGeneration

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct", 
    torch_dtype=torch.bfloat16, 
    attn_implementation="flash_attention_2",
)
```

### Multi-turn Conversation
There is an easy way to use Qwen2-VL for multi-turn conversations which supports pure text, single image, multi-images. You can use it as follows:

First copy the class `Qwen2VL`.
```python
class Qwen2VL:
    def __init__(self, model_path = None, max_new_tokens = 1024, min_pixels = 256*28*28, max_pixels = 1280*28*28):
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_path, min_pixels=min_pixels, max_pixels=max_pixels)
        self.gen_config = {
            "max_new_tokens": max_new_tokens,
        }
    
    def parse_input(self, query=None, imgs=None):
        if imgs is None:
            messages = [{"role": "user", "content": query}]
            return messages
        
        if isinstance(imgs, str):
            imgs = [imgs]
        content = []
        for img in imgs:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": query})
        messages = [{"role": "user", "content": content}]
        return messages

    def chat(self, query = None, imgs = None, history = None):
        if history is None:
            history = []
            
        user_query = self.parse_input(query, imgs)
        history.extend(user_query)

        text = self.processor.apply_chat_template(history, tokenize=False, add_generation_prompt=True, add_vision_id=True)
        image_inputs, video_inputs = process_vision_info(history)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to("cuda")
        generated_ids = self.model.generate(**inputs, **self.gen_config)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        history.append({"role": "assistant", "content": response})

        del inputs, generated_ids, generated_ids_trimmed
        torch.cuda.empty_cache()
        gc.collect()
        return response, history
```
Then use the `chat` API in `Qwen2VL` class, `query` parameter is the user query in natural language format, `imgs` parameter is the image url or path or base64, `history` parameter is the history of the conversation.
```python
chat_model = Qwen2VL(model_path="local path/repo id")

# First turn
history = None
response, history = chat_model.chat(query="hello", history=history)
print(response, history)

# Second turn
# For image type, (imgae_url, local_image_path, base64)
# For image count, ([image], [image1, image2], ...)
response, history = chat_model.chat(query="please describe the image", imgs=["image_url"], history=history)
print(response, history)
```

### Try Qwen2-VL-72B with API!

To explore Qwen2-VL-72B, a more fascinating multimodal model, we encourage you to test our cutting-edge API service. Let's start the exciting journey right now!

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
# The model name 'qwen-vl-max-0809' is the identity of 'Qwen2-VL-72B'.
response = dashscope.MultiModalConversation.call(model='qwen-vl-max-0809', messages=messages)
print(response)
```

For more usage, please refer to the tutorial at [aliyun](https://help.aliyun.com/zh/model-studio/developer-reference/qwen-vl-api).

## Quantization

For quantized models, we offer two types of quantization: AWQ and GPQ([🤗](https://huggingface.co/collections/Qwen/qwen2-vl-66cee7455501d7126940800d)[🤖](https://modelscope.cn/organization/qwen)).

### AWQ
One of our recommendations is the usage of [AWQ](https://arxiv.org/abs/2306.00978) with [AutoAWQ](https://github.com/casper-hansen/AutoAWQ). AWQ refers to Activation-aware Weight Quantization, a hardware-friendly approach for LLM low-bit weight-only quantization. AutoAWQ is an easy-to-use package for 4-bit quantized models.
#### Usage of AWQ Quantized Models with Transformers
Now, Transformers has officially supported AutoAWQ, which means that you can directly use the quantized model with Transformers. The following is a very simple code snippet showing how to run `Qwen2-VL-7B-Instruct-AWQ` with the quantized model:
```python
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2-VL-7B-Instruct-AWQ",
#     torch_dtype="auto",
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default: Load the model on the available device(s)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct-AWQ", torch_dtype="auto", device_map="auto"
)

# The default range for the number of visual tokens per image in the model is 4-16384. You can set min_pixels and max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
min_pixels = 256 * 28 * 28
max_pixels = 1280 * 28 * 28
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct-AWQ", min_pixels=min_pixels, max_pixels=max_pixels
)

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
#### Quantize Your Own Model with AutoAWQ
If you want to quantize your own model to AWQ quantized models, we advise you to use AutoAWQ. It is suggested installing the forked version of the package by installing from source code:


```bash
git clone https://github.com/kq-chen/AutoAWQ.git
cd AutoAWQ
pip install numpy gekko pandas
pip install -e .
```

Suppose you have finetuned a model based on `Qwen2-VL-7B`. To build your own AWQ quantized model, you need to use the training data for calibration. Below, we provide a simple demonstration for you to run:

```python
from transformers import Qwen2VLProcessor
from awq.models.qwen2vl import Qwen2VLAWQForConditionalGeneration

# Specify paths and hyperparameters for quantization
model_path = "your_model_path"
quant_path = "your_quantized_model_path"
quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}

# Load your processor and model with AutoAWQ
processor = Qwen2VLProcessor.from_pretrained(model_path)
# We recommend enabling flash_attention_2 for better acceleration and memory saving
# model = Qwen2VLAWQForConditionalGeneration.from_pretrained(
#     model_path, model_type="qwen2_vl", use_cache=False, attn_implementation="flash_attention_2"
# )
model = Qwen2VLAWQForConditionalGeneration.from_pretrained(
    model_path, model_type="qwen2_vl", use_cache=False
)
```
Then you need to prepare your data for calibration. What you need to do is just put samples into a list, each of which is a typical chat message as shown below. you can specify `text` and `image` in `content` field, For example:
```python
dataset = [
    # message 0
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me who you are."},
        {"role": "assistant", "content": "I am a large language model named Qwen..."},
    ],
    # message 1
    [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "file:///path/to/your/image.jpg"},
                {"type": "text", "text": "Output all text in the image"},
            ],
        },
        {"role": "assistant", "content": "The text in the image is balabala..."},
    ],
    # other messages...
    ...,
]
```
here, we use a caption dataset **only for demonstration**. You should replace it with your own sft dataset.

```python
def prepare_dataset(n_sample: int = 8) -> list[list[dict]]:
    from datasets import load_dataset

    dataset = load_dataset(
        "laion/220k-GPT4Vision-captions-from-LIVIS", split=f"train[:{n_sample}]"
    )
    return [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": sample["url"]},
                    {"type": "text", "text": "generate a caption for this image"},
                ],
            },
            {"role": "assistant", "content": sample["caption"]},
        ]
        for sample in dataset
    ]


dataset = prepare_dataset()
```

Then process the dataset into tensors:
```python
from qwen_vl_utils import process_vision_info

text = processor.apply_chat_template(
    dataset, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(dataset)
inputs = processor(
    text=text,
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
```

Then just run the calibration process by one line of code:
```python
model.quantize(calib_data=inputs, quant_config=quant_config)
```
Finally, save the quantized model:
```python
model.model.config.use_cache = model.model.generation_config.use_cache = True
model.save_quantized(quant_path, safetensors=True, shard_size="4GB")
processor.save_pretrained(quant_path)
```
Then you can obtain your own AWQ quantized model for deployment. Enjoy!
### GPTQ
#### Usage of GPTQ Models with Transformers
Now, Transformers has officially supported AutoGPTQ, which means that you can directly use the quantized model with Transformers. The following is a very simple code snippet showing how to run `Qwen2-VL-7B-Instruct-GPTQ-Int4` with the quantized model:
```python
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default: Load the model on the available device(s)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4", torch_dtype="auto", device_map="auto"
)

# The default range for the number of visual tokens per image in the model is 4-16384. You can set min_pixels and max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
min_pixels = 256 * 28 * 28
max_pixels = 1280 * 28 * 28
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4", min_pixels=min_pixels, max_pixels=max_pixels
)

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
#### Quantize Your Own Model with AutoGPTQ
If you want to quantize your own model to GPTQ quantized models, we advise you to use AutoGPTQ. It is suggested installing the forked version of the package by installing from source code:

```bash
git clone https://github.com/kq-chen/AutoGPTQ.git
cd AutoGPTQ
pip install numpy gekko pandas
pip install -vvv --no-build-isolation -e .
```
Suppose you have finetuned a model based on `Qwen2-VL-7B`. To build your own GPTQ quantized model, you need to use the training data for calibration. Below, we provide a simple demonstration for you to run:
```python
from transformers import Qwen2VLProcessor
from auto_gptq import BaseQuantizeConfig
from auto_gptq.modeling import Qwen2VLGPTQForConditionalGeneration

# Specify paths and hyperparameters for quantization
model_path = "your_model_path"
quant_path = "your_quantized_model_path"
quantize_config = BaseQuantizeConfig(
    bits=8,  # 4 or 8
    group_size=128,
    damp_percent=0.1,
    desc_act=False,  # set to False can significantly speed up inference but the perplexity may slightly bad
    static_groups=False,
    sym=True,
    true_sequential=True,
)
# Load your processor and model with AutoGPTQ
processor = Qwen2VLProcessor.from_pretrained(model_path)
# We recommend enabling flash_attention_2 for better acceleration and memory saving
# model = Qwen2VLGPTQForConditionalGeneration.from_pretrained(model_path, quantize_config, attn_implementation="flash_attention_2")
model = Qwen2VLGPTQForConditionalGeneration.from_pretrained(model_path, quantize_config)
```
Then you need to prepare your data for calibration. What you need to do is just put samples into a list, each of which is a typical chat message as shown below. you can specify `text` and `image` in `content` field, For example:
```python
dataset = [
    # message 0
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me who you are."},
        {"role": "assistant", "content": "I am a large language model named Qwen..."},
    ],
    # message 1
    [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "file:///path/to/your/image.jpg"},
                {"type": "text", "text": "Output all text in the image"},
            ],
        },
        {"role": "assistant", "content": "The text in the image is balabala..."},
    ],
    # other messages...
    ...,
]
```
Here, we use a caption dataset **only for demonstration**. You should replace it with your own sft dataset.
```python
def prepare_dataset(n_sample: int = 20) -> list[list[dict]]:
    from datasets import load_dataset

    dataset = load_dataset(
        "laion/220k-GPT4Vision-captions-from-LIVIS", split=f"train[:{n_sample}]"
    )
    return [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": sample["url"]},
                    {"type": "text", "text": "generate a caption for this image"},
                ],
            },
            {"role": "assistant", "content": sample["caption"]},
        ]
        for sample in dataset
    ]


dataset = prepare_dataset()
```

Then process the dataset into tensors:
```python
from qwen_vl_utils import process_vision_info


def batched(iterable, n: int):
    # batched('ABCDEFG', 3) → ABC DEF G
    assert n >= 1, "batch size must be at least one"
    from itertools import islice

    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        yield batch


batch_size = 1
calib_data = []
for batch in batched(dataset, batch_size):
    text = processor.apply_chat_template(
        batch, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(batch)
    inputs = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    calib_data.append(inputs)
```
Then just run the calibration process by one line of code:
```python
model.quantize(dataset, cache_examples_on_gpu=False)
```
Finally, save the quantized model:
```python
model.save_quantized(quant_path, use_safetensors=True)
processor.save_pretrained(quant_path)
```
Then you can obtain your own GPTQ quantized model for deployment. Enjoy!
### Benchmark
#### Performance of Quantized Models
This section reports the generation performance of quantized models (including GPTQ and AWQ) of the Qwen2-VL series. Specifically, we report:

- MMMU_VAL (Accuracy)
- DocVQA_VAL (Accuracy)
- MMBench_DEV_EN (Accuracy)
- MathVista_MINI (Accuracy)

We use [VLMEvalkit](https://github.com/open-compass/VLMEvalKit) to evaluate all models.

| Model Size | Quantization | MMMU | DocVQA | MMBench | MathVista  |
| --- | --- | --- | --- | --- | --- |
| Qwen2-VL-72B-Instruct | BF16<br><sup>([🤗](https://huggingface.co/Qwen/Qwen2-VL-72B-Instruct)[🤖](https://modelscope.cn/models/qwen/Qwen2-VL-72B-Instruct)) | 65.44 | 95.79 | 86.94 | 70.19 |
|  | GPTQ-Int8<br><sup>([🤗](https://huggingface.co/Qwen/Qwen2-VL-72B-Instruct-GPTQ-Int8)[🤖](https://modelscope.cn/models/qwen/Qwen2-VL-72B-Instruct-GPTQ-Int8)) | 64.56 | 95.84 | 87.03 | 68.90 |
|  | GPTQ-Int4<br><sup>([🤗](https://huggingface.co/Qwen/Qwen2-VL-72B-Instruct-GPTQ-Int4)[🤖](https://modelscope.cn/models/qwen/Qwen2-VL-72B-Instruct-GPTQ-Int4)) | 64.00 | 95.70 | 86.68 | 69.20 |
|  | AWQ<br><sup>([🤗](https://huggingface.co/Qwen/Qwen2-VL-72B-Instruct-AWQ)[🤖](https://modelscope.cn/models/qwen/Qwen2-VL-72B-Instruct-AWQ)) | 64.22 | 95.72 | 86.43 | 68.40 |
| Qwen2-VL-7B-Instruct | BF16<br><sup>([🤗](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)[🤖](https://modelscope.cn/models/qwen/Qwen2-VL-7B-Instruct)) | 53.77 | 93.89 | 81.78 | 58.20 |
|  | GPTQ-Int8<br><sup>([🤗](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int8)[🤖](https://modelscope.cn/models/qwen/Qwen2-VL-7B-Instruct-GPTQ-Int8)) | 53.00 | 93.94 | 82.38 | 57.90 |
|  | GPTQ-Int4<br><sup>([🤗](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4)[🤖](https://modelscope.cn/models/qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4)) | 52.55 | 93.16 | 81.27 | 60.30 |
|  | AWQ<br><sup>([🤗](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct-AWQ)[🤖](https://modelscope.cn/models/qwen/Qwen2-VL-7B-Instruct-AWQ)) | 53.66 | 93.10 | 81.61 | 56.80 |
| Qwen2-VL-2B-Instruct | BF16<br><sup>([🤗](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)[🤖](https://modelscope.cn/models/qwen/Qwen2-VL-2B-Instruct)) | 41.88 | 88.34 | 72.07 | 44.40 |
|  | GPTQ-Int8<br><sup>([🤗](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int8)[🤖](https://modelscope.cn/models/qwen/Qwen2-VL-2B-Instruct-GPTQ-Int8)) | 41.55 |  88.28 | 71.99 | 44.60 |
|  | GPTQ-Int4<br><sup>([🤗](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int4)[🤖](https://modelscope.cn/models/qwen/Qwen2-VL-2B-Instruct-GPTQ-Int4)) | 39.22 | 87.21 | 70.87 | 41.69 |
|  | AWQ<br><sup>([🤗](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct-AWQ)[🤖](https://modelscope.cn/models/qwen/Qwen2-VL-2B-Instruct-AWQ)) | 41.33 | 86.96 | 71.64 | 39.90 |






#### Speed Benchmark
This section reports the speed performance of bf16 models, quantized models (including GPTQ-Int4, GPTQ-Int8 and AWQ) of the Qwen2-VL series. Specifically, we report the inference speed (tokens/s) as well as memory footprint (GB) under the conditions of different context lengths.

The environment of the evaluation with huggingface transformers is:

- NVIDIA A100 80GB
- CUDA 11.8
- Pytorch 2.2.1+cu118
- Flash Attention 2.6.1
- Transformers 4.38.2
- AutoGPTQ 0.6.0+cu118
- AutoAWQ 0.2.5+cu118 (autoawq_kernels 0.0.6+cu118)

Note:

- We use the batch size of 1 and the least number of GPUs as possible for the evalution.
- We test the speed and memory of generating 2048 tokens with the input lengths of 1, 6144, 14336, 30720, 63488, and 129024 tokens.
- 72B (transformers)

| Model | Input Length | Quantization | GPU Num | Speed(tokens/s) | GPU Memory(GB) |
| --- | --- | --- | --- | --- | --- |
| Qwen2-VL-72B-Instruct | 1 | BF16 | 2 | 8.90 | 138.74 |
|  |  | GPTQ-Int8 | 2 | 9.53 | 75.173 |
|  |  | GPTQ-Int4 | 1 | 11.04 | 42.46 |
|  |  | AWQ | 1 | 12.00 | 41.98 |
|  | 6144 | BF16 | 2 | 6.53 | 148.66 |
|  |  | GPTQ-Int8 | 2 | 6.97 | 85.09 |
|  |  | GPTQ-Int4 | 1 | 7.62 | 49.05 |
|  |  | AWQ | 1 | 8.33 | 48.58 |
|  | 14336 | BF16 | 3 | 4.39 | 165.92 |
|  |  | GPTQ-Int8 | 2 | 5.04 | 99.31 |
|  |  | GPTQ-Int4 | 1 | 5.39 | 58.76 |
|  |  | AWQ | 1 | 5.72 | 58.29 |
|  | 30720 | BF16 | 4 | 2.93 | 204.33 |
|  |  | GPTQ-Int8 | 2 | 3.16 | 127.77 |
|  |  | GPTQ-Int4 | 2 | 3.27 | 85.13 |
|  |  | AWQ | 2 | 3.39 | 94.65 |

- 7B (transformers)

| Model | Input Length | Quantization | GPU Num | Speed(tokens/s) | GPU Memory(GB) |
| --- | --- | --- | --- | --- | --- |
| Qwen2-VL-7B-Instruct | 1 | BF16 | 1 | 39.02 | 16.07 |
|  |  | GPTQ-Int8 | 1 | 31.60 | 10.11 |
|  |  | GPTQ-Int4 | 1 | 42.76 | 7.20 |
|  |  | AWQ | 1 | 32.08 | 7.07 |
|  | 6144 | BF16 | 1 | 38.75 | 21.56 |
|  |  | GPTQ-Int8 | 1 | 31.31 | 15.61 |
|  |  | GPTQ-Int4 | 1 | 39.75 | 12.69 |
|  |  | AWQ | 1 | 32.66 | 12.56 |
|  | 14336 | BF16 | 1 | 30.65 | 29.07 |
|  |  | GPTQ-Int8 | 1 | 27.96 | 23.11 |
|  |  | GPTQ-Int4 | 1 | 29.72 | 20.20 |
|  |  | AWQ | 1 | 31.42 | 20.07 |
|  | 30720 | BF16 | 1 | 19.53 | 44.08 |
|  |  | GPTQ-Int8 | 1 | 18.37 | 38.13 |
|  |  | GPTQ-Int4 | 1 | 19.15 | 35.22 |
|  |  | AWQ | 1 | 19.95 | 35.08 |


- 2B (transformers)

| Model | Input Length | Quantization | GPU Num | Speed(tokens/s) | GPU Memory(GB) |
| --- | --- | --- | --- | --- | --- |
| Qwen2-VL-2B-Instruct | 1 | BF16 | 1 | 35.29 | 4.68 |
|  |  | GPTQ-Int8 | 1 | 28.59 | 3.55 |
|  |  | GPTQ-Int4 | 1 | 39.76 | 2.91 |
|  |  | AWQ | 1 | 29.89 | 2.88 |
|  | 6144 | BF16 | 1 | 36.58 | 10.01 |
|  |  | GPTQ-Int8 | 1 | 29.53  | 8.87 |
|  |  | GPTQ-Int4 | 1 | 39.27 | 8.21 |
|  |  | AWQ | 1 | 33.42 | 8.18 |
|  | 14336 | BF16 | 1 | 36.31 | 17.20 |
|  |  | GPTQ-Int8 | 1 | 31.03 | 16.07 |
|  |  | GPTQ-Int4 | 1 | 39.89 | 15.40 |
|  |  | AWQ | 1 | 32.28 | 15.40 |
|  | 30720 | BF16 | 1 | 32.53 | 31.64 |
|  |  | GPTQ-Int8 | 1 | 27.76 | 30.51 |
|  |  | GPTQ-Int4 | 1 | 30.73 | 29.84 |
|  |  | AWQ | 1 | 31.55 | 29.84 |



## Deployment

We recommend using vLLM for fast Qwen2-VL deployment and inference. You need to use `vllm>=0.6.1` to enable Qwen2-VL support. You can also use our [official docker image](#-docker).

### Installation
```bash
pip install git+https://github.com/huggingface/transformers@21fac7abba2a37fae86106f87fcf9974fd1e3830
pip install accelerate
pip install qwen-vl-utils
# Change to your CUDA version
CUDA_VERSION=cu121
pip install 'vllm==0.6.1' --extra-index-url https://download.pytorch.org/whl/${CUDA_VERSION}

```
### Start an OpenAI API Service

Run the command below to start an OpenAI-compatible API service:

```bash
python -m vllm.entrypoints.openai.api_server --served-model-name Qwen2-VL-7B-Instruct --model Qwen/Qwen2-VL-7B-Instruct
```

Then you can use the chat API as below (via curl or Python API):

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
    "model": "Qwen2-VL-7B-Instruct",
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
    model="Qwen2-VL-7B-Instruct",
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
    model="Qwen2-7B-Instruct",
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

### Notes

- ⚠️**NOTE**: Now `vllm.entrypoints.openai.api_server` does not support set `min_pixels` and `max_pixels` in messages (we are working hard on supporting this feature). If you want to limit the resolution, you can set them in model's `preprocessor_config.json`:

```json
{
  "min_pixels": 50176,
  "max_pixels": 1003520,
  ...
}
```

- ⚠️**NOTE**: Now `vllm.entrypoints.openai.api_server` does not support video input yet. We are actively developing on it.
- ⚠️**NOTE**: If you want to pass multiple images in a single prompt, you need to pass `--limit-mm-per-prompt image=<N>` argument (`N` is max number of images in each prompt) when launching `vllm.entrypoints.openai.api_server`.
### Inference Locally

You can also use vLLM to inference Qwen2-VL locally:

```python
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info

MODEL_PATH = "Qwen/Qwen2-VL-7B-Instruct"

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

messages = [
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

processor = AutoProcessor.from_pretrained(MODEL_PATH)
prompt = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
image_inputs, video_inputs = process_vision_info(messages)

mm_data = {}
if image_inputs is not None:
    mm_data["image"] = image_inputs
if video_inputs is not None:
    mm_data["video"] = video_inputs

llm_inputs = {
    "prompt": prompt,
    "multi_modal_data": mm_data,
}

outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
generated_text = outputs[0].outputs[0].text

print(generated_text)
```


## Training
#### LLaMA-Factory

Here we provide a script for supervised finetuning Qwen2-VL with
`LLaMA-Factory <https://github.com/hiyouga/LLaMA-Factory>`. This
script for supervised finetuning (SFT) has the following features:

-  Support multi-images input;

-  Support single-GPU and multi-GPU training;

-  Support full-parameter tuning, LoRA.

In the following, we introduce more details about the usage of the
script.

#### Installation

Before you start, make sure you have installed the following packages:

1. Follow the instructions of
   `LLaMA-Factory <https://github.com/hiyouga/LLaMA-Factory>`, and build
   the environment.
2. Install these packages (Optional):

```
pip install deepspeed
pip install flash-attn --no-build-isolation
```

3. If you want to use
   `FlashAttention-2 <https://github.com/Dao-AILab/flash-attention>`,
   make sure your CUDA is 11.6 and above.

#### Data Preparation

LLaMA-Factory provides several training datasets in ``data`` folder, you
can use it directly. If you are using a custom dataset, please prepare
your dataset as follows.

1. Organize your data in a **json** file and put your data in ``data``
   folder. LLaMA-Factory supports multimodal dataset in ``sharegpt``
   format.

-  The dataset in ``sharegpt`` format should follow the below format:

```json
[
  {
    "messages": [
      {
        "content": "<image>Who are they?",
        "role": "user"
      },
      {
        "content": "They're Kane and Gretzka from Bayern Munich.",
        "role": "assistant"
      },
      {
        "content": "What are they doing?<image>",
        "role": "user"
      },
      {
        "content": "They are celebrating on the soccer field.",
        "role": "assistant"
      }
    ],
    "images": [
      "mllm_demo_data/1.jpg",
      "mllm_demo_data/1.jpg"
    ]
  },
]
```

1. Provide your dataset definition in ``data/dataset_info.json`` in the
   following format .

-  For ``sharegpt`` format dataset, the columns in ``dataset_info.json``
   should be:

```json
   "dataset_name": {
       "file_name": "dataset_name.json",
       "formatting": "sharegpt",
       "columns": {
          "messages": "messages",
          "images": "images"
        },
      "tags": {
         "role_tag": "role",
         "content_tag": "content",
         "user_tag": "user",
         "assistant_tag": "assistant"
        }
   }
```

#### Training

Lora SFT examples:
```
llamafactory-cli train examples/train_lora/qwen2vl_lora_sft.yaml
llamafactory-cli export examples/merge_lora/qwen2vl_lora_sft.yaml
```

LoRA DPO/ORPO/SimPO examples: (using [RLHF-V Dataset](https://huggingface.co/datasets/llamafactory/RLHF-V))
```
llamafactory-cli train examples/train_lora/qwen2vl_lora_dpo.yaml
```

Full SFT examples:
```
llamafactory-cli train examples/train_full/qwen2vl_full_sft.yaml
```

Inference examples:
```
llamafactory-cli webchat examples/inference/qwen2_vl.yaml
llamafactory-cli api examples/inference/qwen2_vl.yaml
```

Execute the following training command:

```bash
DISTRIBUTED_ARGS="
    --nproc_per_node $NPROC_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
    "

torchrun $DISTRIBUTED_ARGS src/train.py \
    --deepspeed $DS_CONFIG_PATH \
    --stage sft \
    --do_train \
    --model_name_or_path Qwen/Qwen2-VL-7B-Instruct \
    --dataset mllm_demo \
    --template qwen2_vl \
    --finetuning_type lora \
    --output_dir $OUTPUT_PATH \
    --overwrite_cache \
    --overwrite_output_dir \
    --warmup_steps 100 \
    --weight_decay 0.1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --ddp_timeout 9000 \
    --learning_rate 5e-6 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --cutoff_len 4096 \
    --save_steps 1000 \
    --plot_loss \
    --num_train_epochs 3 \
    --bf16 
```

and enjoy the training process. To make changes to your training, you
can modify the arguments in the training command to adjust the
hyperparameters. One argument to note is ``cutoff_len``, which is the
maximum length of the training data. Control this parameter to avoid OOM
error.

## Function Calling

Qwen2-VL supports Function Calling (aka. Tool Calling or Tool Use). For details on how to use this capability, please refer to the Qwen-Agent project for [the function calling example](https://github.com/QwenLM/Qwen-Agent/blob/main/examples/qwen2vl_function_calling.py) and [the agent example](https://github.com/QwenLM/Qwen-Agent/blob/main/examples/qwen2vl_assistant_tooluse.py). 
### Simple Use Case
```python
# pip install qwen_agent
from typing import List, Union
from datetime import datetime
from qwen_agent.agents import FnCallAgent
from qwen_agent.gui import WebUI
from qwen_agent.tools.base import BaseToolWithFileAccess, register_tool

@register_tool("get_date")
class GetDate(BaseToolWithFileAccess):
    description = "call this tool to get the current date"
    parameters = [
        {
            "name": "lang",
            "type": "string",
            "description": "one of ['en', 'zh'], default is en",
            "required": False
        },
    ]

    def call(self, params: Union[str, dict], files: List[str] = None, **kwargs) -> str:
        super().call(params=params, files=files)
        params = self._verify_json_format_args(params)
        lang = "zh" if "zh" in params["lang"] else "en"
        now = datetime.now()
        result = now.strftime("%Y-%m-%d %H:%M:%S") + "\n"
        weekday = now.weekday()
        if lang == "zh":
            days_chinese = ["一", "二", "三", "四", "五", "六", "日"]
            result += "今天是星期" + days_chinese[weekday]
        else:
            days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            result += "Today is " + days[weekday]
        return result


def init_agent_service():
    llm_cfg_vl = {
        # Using Qwen2-VL deployed at any openai-compatible service such as vLLM:
        "model_type": "qwenvl_oai",
        "model": "Qwen/Qwen2-VL-7B-Instruct",
        "model_server": "http://localhost:8000/v1",  # api_base
        "api_key": 'EMPTY",
    }
    tools = [
        "get_date",
        "code_interpreter",
    ]  # code_interpreter is a built-in tool in Qwen-Agent
    bot = FnCallAgent(
        llm=llm_cfg_vl,
        name="Qwen2-VL",
        description="function calling",
        function_list=tools,
    )
    return bot

def app_gui():
    # Define the agent
    bot = init_agent_service()
    WebUI(bot).run()

# Launch gradio app
app_gui()
```


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

#### Selecting Different Models (Qwen2-VL Series Only)

The demo is configured by default to use the `Qwen/Qwen2-VL-7B-Instruct` model, which is part of the Qwen2-VL series and is well-suited for various vision-language tasks. However, if you want to use a different model within the Qwen2-VL series, you can simply update the `DEFAULT_CKPT_PATH` variable in the script:

1. **Locate the `DEFAULT_CKPT_PATH` Variable**:
   Inside `web_demo_mm.py`, find the `DEFAULT_CKPT_PATH` variable that defines the model checkpoint path. It should look like this:

   ```python
   DEFAULT_CKPT_PATH = 'Qwen/Qwen2-VL-7B-Instruct'
   ```

2. **Replace with a Different Qwen2-VL Model Path**:
   Modify `DEFAULT_CKPT_PATH` to point to another checkpoint path within the Qwen2-VL series. For example:

   ```python
   DEFAULT_CKPT_PATH = 'Qwen/Qwen2-VL-2B-Instruct'  # Example for a different model in the series
   ```

3. **Save and Re-run**:
   After modifying the path, save the script and then re-run the demo using the instructions provided in the `Running the Demo` section above.

**Note:** This `DEFAULT_CKPT_PATH` only supports models from the Qwen2-VL series. If you're using a model outside of the Qwen2-VL series, additional changes to the codebase may be necessary.


#### Customization

Further customization of the web demo, including UI layout, interactions, and additional functionalities like handling specialized input, can be done by modifying the `web_demo_mm.py` script. This flexibility allows you to tailor the web interface to better fit specific tasks or workflows.


## Limitations

While Qwen2-VL are applicable to a wide range of visual tasks, it is equally important to understand its limitations. Here are some known restrictions:

1. Lack of Audio Support: The current model does **not comprehend audio information** within videos.
2. Data timeliness: Our image dataset is **updated until June 2023**, and information subsequent to this date may not be covered.
3. Constraints in Individuals and Intellectual Property (IP): The model's capacity to recognize specific individuals or IPs is limited, potentially failing to comprehensively cover all well-known personalities or brands.
4. Limited Capacity for Complex Instruction: When faced with intricate multi-step instructions, the model's understanding and execution capabilities require enhancement.
5. Insufficient Counting Accuracy: Particularly in complex scenes, the accuracy of object counting is not high, necessitating further improvements.
6. Weak Spatial Reasoning Skills: Especially in 3D spaces, the model's inference of object positional relationships is inadequate, making it difficult to precisely judge the relative positions of objects.

These limitations serve as ongoing directions for model optimization and improvement, and we are committed to continually enhancing the model's performance and scope of application.


## 🐳 Docker

To simplify the deploy process, we provide docker images with pre-build environments: [qwenllm/qwenvl](https://hub.docker.com/r/qwenllm/qwenvl). You only need to install the driver and download model files to launch demos.

```bash
docker run --gpus all --ipc=host --network=host --rm --name qwen2 -it qwenllm/qwenvl:2-cu121 bash
```

## Citation

If you find our paper and code useful in your research, please consider giving a star :star: and citation :pencil: :)




```BibTeX
@article{Qwen2VL,
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
