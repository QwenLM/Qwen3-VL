# qwen-vl-utils

Qwen-VL Utils contains a set of helper functions for processing and integrating visual language information with Qwen-VL Series Model.

## Install

```bash
pip install qwen-vl-utils
```

## Usage

### Qwen2VL

```python
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


# You can directly insert a local file path, a URL, or a base64-encoded image into the position where you want in the text.
messages = [
    # Image
    ## Local file path
    [{"role": "user", "content": [{"type": "image", "image": "file:///path/to/your/image.jpg"}, {"type": "text", "text": "Describe this image."}]}],
    ## Image URL
    [{"role": "user", "content": [{"type": "image", "image": "http://path/to/your/image.jpg"}, {"type": "text", "text": "Describe this image."}]}],
    ## Base64 encoded image
    [{"role": "user", "content": [{"type": "image", "image": "data:image;base64,/9j/..."}, {"type": "text", "text": "Describe this image."}]}],
    ## PIL.Image.Image
    [{"role": "user", "content": [{"type": "image", "image": pil_image}, {"type": "text", "text": "Describe this image."}]}],
    ## Model dynamically adjusts image size, specify dimensions if required.
    [{"role": "user", "content": [{"type": "image", "image": "file:///path/to/your/image.jpg", "resized_height": 280, "resized_width": 420}, {"type": "text", "text": "Describe this image."}]}],
    # Video
    ## Local video path
    [{"role": "user", "content": [{"type": "video", "video": "file:///path/to/video1.mp4"}, {"type": "text", "text": "Describe this video."}]}],
    ## Local video frames
    [{"role": "user", "content": [{"type": "video", "video": ["file:///path/to/extracted_frame1.jpg", "file:///path/to/extracted_frame2.jpg", "file:///path/to/extracted_frame3.jpg"],}, {"type": "text", "text": "Describe this video."},],}],
    ## Model dynamically adjusts video nframes, video height and width. specify args if required.
    [{"role": "user", "content": [{"type": "video", "video": "file:///path/to/video1.mp4", "fps": 2.0, "resized_height": 280, "resized_width": 280}, {"type": "text", "text": "Describe this video."}]}],
]

processor = AutoProcessor.from_pretrained(model_path)
model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
images, videos = process_vision_info(messages)
inputs = processor(text=text, images=images, videos=videos, padding=True, return_tensors="pt")
print(inputs)
generated_ids = model.generate(**inputs)
print(generated_ids)
```

### Qwen2.5VL

```python
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


# You can set the maximum tokens for a video through the environment variable VIDEO_MAX_PIXELS
# based on the maximum tokens that the model can accept. 
# export VIDEO_MAX_PIXELS = 32000 * 28 * 28 * 0.9


# You can directly insert a local file path, a URL, or a base64-encoded image into the position where you want in the text.
messages = [
    # Image
    ## Local file path
    [{"role": "user", "content": [{"type": "image", "image": "file:///path/to/your/image.jpg"}, {"type": "text", "text": "Describe this image."}]}],
    ## Image URL
    [{"role": "user", "content": [{"type": "image", "image": "http://path/to/your/image.jpg"}, {"type": "text", "text": "Describe this image."}]}],
    ## Base64 encoded image
    [{"role": "user", "content": [{"type": "image", "image": "data:image;base64,/9j/..."}, {"type": "text", "text": "Describe this image."}]}],
    ## PIL.Image.Image
    [{"role": "user", "content": [{"type": "image", "image": pil_image}, {"type": "text", "text": "Describe this image."}]}],
    ## Model dynamically adjusts image size, specify dimensions if required.
    [{"role": "user", "content": [{"type": "image", "image": "file:///path/to/your/image.jpg", "resized_height": 280, "resized_width": 420}, {"type": "text", "text": "Describe this image."}]}],
    # Video
    ## Local video path
    [{"role": "user", "content": [{"type": "video", "video": "file:///path/to/video1.mp4"}, {"type": "text", "text": "Describe this video."}]}],
    ## Local video frames
    [{"role": "user", "content": [{"type": "video", "video": ["file:///path/to/extracted_frame1.jpg", "file:///path/to/extracted_frame2.jpg", "file:///path/to/extracted_frame3.jpg"],}, {"type": "text", "text": "Describe this video."},],}],
    ## Model dynamically adjusts video nframes, video height and width. specify args if required.
    [{"role": "user", "content": [{"type": "video", "video": "file:///path/to/video1.mp4", "fps": 2.0, "resized_height": 280, "resized_width": 280}, {"type": "text", "text": "Describe this video."}]}],
]

processor = AutoProcessor.from_pretrained(model_path)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
images, videos, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
inputs = processor(text=text, images=images, videos=videos, padding=True, return_tensors="pt", **video_kwargs)
print(inputs)
generated_ids = model.generate(**inputs)
print(generated_ids)
```

### Qwen3VL

```python
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# You can directly insert a local file path, a URL, or a base64-encoded image into the position where you want in the text.
messages = [
    # Image
    ## Local file path
    [{"role": "user", "content": [{"type": "image", "image": "file:///path/to/your/image.jpg"}, {"type": "text", "text": "Describe this image."}]}],
    ## Image URL
    [{"role": "user", "content": [{"type": "image", "image": "http://path/to/your/image.jpg"}, {"type": "text", "text": "Describe this image."}]}],
    ## Base64 encoded image
    [{"role": "user", "content": [{"type": "image", "image": "data:image;base64,/9j/..."}, {"type": "text", "text": "Describe this image."}]}],
    ## PIL.Image.Image
    [{"role": "user", "content": [{"type": "image", "image": pil_image}, {"type": "text", "text": "Describe this image."}]}],
    ## Model dynamically adjusts image size, specify dimensions if required.
    [{"role": "user", "content": [{"type": "image", "image": "file:///path/to/your/image.jpg", "resized_height": 280, "resized_width": 420}, {"type": "text", "text": "Describe this image."}]}],
    # Video
    ## Local video path
    [{"role": "user", "content": [{"type": "video", "video": "file:///path/to/video1.mp4"}, {"type": "text", "text": "Describe this video."}]}],
    ## Local video frames
    [{"role": "user", "content": [{"type": "video", "video": ["file:///path/to/extracted_frame1.jpg", "file:///path/to/extracted_frame2.jpg", "file:///path/to/extracted_frame3.jpg"],}, {"type": "text", "text": "Describe this video."},],}],
    ## Model dynamically adjusts video nframes, video height and width. specify args if required.
    [{"role": "user", "content": [{"type": "video", "video": "file:///path/to/video1.mp4", "fps": 2.0, "resized_height": 280, "resized_width": 280}, {"type": "text", "text": "Describe this video."}]}],
]

processor = AutoProcessor.from_pretrained(model_path)
model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, dtype="auto", device_map="auto")

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
images, videos, video_kwargs = process_vision_info(messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True)

if videos is not None:
    videos, video_metadatas = zip(*videos)
    videos, video_metadatas = list(videos), list(video_metadatas)
else:
    video_metadatas = None

inputs = processor(text=text, images=images, videos=videos, video_metadata=video_metadatas, return_tensors="pt", do_resize=False, **video_kwargs)
inputs = inputs.to(model.device)

generated_ids = model.generate(**inputs)
print(generated_ids)
```