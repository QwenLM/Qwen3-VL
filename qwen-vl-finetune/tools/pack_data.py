import json
import os
import numpy as np
from PIL import Image
from copy import deepcopy
from transformers import AutoTokenizer, Qwen2VLImageProcessor
from torchcodec.decoders import VideoDecoder
import binpacking
from tqdm import tqdm
import concurrent.futures
import time


def read_data(file_path):
    """Read JSON or JSONL file"""
    if file_path.endswith(('.json', '.jsonl')):
        with open(file_path, 'r') as f:
            if file_path.endswith('.json'):
                return json.load(f)
            return [json.loads(line) for line in f]
    raise ValueError('Please provide a .json or .jsonl file')


def write_data(file_path, data):
    """Write data to JSON or JSONL file"""
    with open(file_path, 'w') as f:
        if file_path.endswith('.json'):
            json.dump(data, f, indent=4)
        elif file_path.endswith('.jsonl'):
            for item in data:
                f.write(json.dumps(item) + '\n')


class DataArguments:
    def __init__(self):
        self.max_pixels = 2048 * 28 * 28
        self.min_pixels = 32 * 28 * 28
        self.video_max_frame_pixels = 576 * 28 * 28
        self.video_min_frame_pixels = 144 * 28 * 28
        self.base_interval = 4
        self.video_min_frames = 4
        self.video_max_frames = 8
        self.data_path = ''


class MultimodalProcessor:
    def __init__(self, data_args, base_processor, device='cpu'):
        self.data_args = data_args
        self.base_processor = base_processor
        self.device = device

    def _configure_processor(self, max_val, min_val):
        processor = deepcopy(self.base_processor)
        processor.max_pixels = max_val
        processor.min_pixels = min_val
        processor.size = {'longest_edge': max_val, 'shortest_edge': min_val}
        return processor

    def process_image(self, image_file):
        image_path = os.path.join(self.data_args.data_path, image_file)
        if not os.path.exists(image_path):
            print(f'Image file does not exist: {image_path}')
            return 0
        processor = self._configure_processor(self.data_args.max_pixels, self.data_args.min_pixels)
        image = Image.open(image_path).convert('RGB')
        visual_processed = processor.preprocess(images=image, return_tensors='pt')
        return visual_processed['image_grid_thw'].prod() // 4

    def process_video(self, video_file):
        video_path = os.path.join(self.data_args.data_path, video_file)
        processor = self._configure_processor(self.data_args.video_max_frame_pixels, self.data_args.video_min_frame_pixels)
        decoder = VideoDecoder(video_path, device=self.device)
        total_frames = decoder.metadata.num_frames
        avg_fps = decoder.metadata.average_fps
        video_length = total_frames / avg_fps
        interval = self.data_args.base_interval
        num_frames_to_sample = round(video_length / interval)
        target_frames = min(max(num_frames_to_sample, self.data_args.video_min_frames), self.data_args.video_max_frames)
        frame_idx = np.unique(np.linspace(0, total_frames - 1, target_frames, dtype=int)).tolist()
        frame_batch = decoder.get_frames_at(indices=frame_idx)
        video_frames_numpy = frame_batch.data.cpu().numpy()
        visual_processed = processor.preprocess(images=None, videos=video_frames_numpy, return_tensors='pt')
        return visual_processed['video_grid_thw'].prod() // 4


def calculate_tokens(conversation, processor, tokenizer):
    total_tokens = 21
    roles = {'human': 'user', 'gpt': 'assistant'}
    for message in conversation['conversations']:
        role = message['from']
        text = message['value']
        conv = [{'role': roles[role], 'content': text}]
        encode_id = tokenizer.apply_chat_template(conv, return_tensors='pt', add_generation_prompt=False)[0]
        total_tokens += len(encode_id)
    if 'image' in conversation:
        images = conversation['image'] if isinstance(conversation['image'], list) else [conversation['image']]
        for image_file in images:
            total_tokens += processor.process_image(image_file)
    elif 'video' in conversation:
        videos = conversation['video'] if isinstance(conversation['video'], list) else [conversation['video']]
        for video_file in videos:
            total_tokens += processor.process_video(video_file)
    return total_tokens


def pack_data(data_list, pack_length):
    # Extract the length of each data item
    lengths = [data["num_tokens"] for data in data_list]
    grouped_indices = binpacking.to_constant_volume(
        list(enumerate(lengths)),  # Explicitly convert to list
        pack_length,
        weight_pos=1
    )
    packed_data = []
    for group in grouped_indices:
        group_data = []
        for index, _ in group:
            new_data = data_list[index].copy()
            new_data.pop("num_tokens", None)
            group_data.append(new_data)
        packed_data.append(group_data)
    return packed_data


datasets = {
    'dummy_dataset': {
        'data_path': '',
        'annotation_path': 'path/to/your/annotation.json'
    }
}

data_args = DataArguments()
model_path = 'path/to/your/model'
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
base_image_processor = Qwen2VLImageProcessor.from_pretrained(model_path)
print(f'Successfully loaded model components from {model_path}')

processor = MultimodalProcessor(data_args, base_image_processor, device='cpu')

for dataset_name, config in datasets.items():
    processor.data_args.data_path = config['data_path']
    annotation_path = os.path.join(processor.data_args.data_path, config['annotation_path'])
    print(f'\n--- Processing dataset: {dataset_name} ---')
    print(f'Annotation file path: {annotation_path}')
    print(f'Image configuration: max_pixels={data_args.max_pixels}, min_pixels={data_args.min_pixels}')
    print(f'Video frame configuration: video_max_frame_pixels={data_args.video_max_frame_pixels}, video_min_frame_pixels={data_args.video_min_frame_pixels}')
    if not os.path.exists(annotation_path):
        print(f'Annotation file not found: {annotation_path}')
        continue
    data = read_data(annotation_path)

    count_file_path = annotation_path.replace('.jsonl', '_count.json').replace('.json', '_count.json')
    if os.path.exists(count_file_path):
        print(f"Found pre - calculated token counts, loading data from {count_file_path}.")
        data_with_tokens = read_data(count_file_path)
    else:
        def calculate_and_update(item):
            item['num_tokens'] = calculate_tokens(item, processor, tokenizer)
            return item

        with concurrent.futures.ThreadPoolExecutor() as executor:
            data_with_tokens = list(tqdm(executor.map(calculate_and_update, data), total=len(data), desc=f"Processing {dataset_name} data"))

        # Save the token count results
        write_data(count_file_path, data_with_tokens)
        print(f"Token counts saved to: {count_file_path}")

    # Assume the packing length is 4096
    pack_length = 4096
    # Define the batch size
    batch_size = 256
    all_packed_results = []

    # Record the start time of binpacking
    start_time = time.time()
    for i in range(0, len(data_with_tokens), batch_size):
        batch_data = data_with_tokens[i: i + batch_size]
        batch_packed_result = pack_data(batch_data, pack_length)
        all_packed_results.extend(batch_packed_result)
    # Record the end time of binpacking
    end_time = time.time()

    # Calculate the time spent on binpacking
    binpack_time = end_time - start_time
    print(f"Time spent on binpacking: {binpack_time:.4f} seconds")

    # Save the packed results as a JSON file
    pack_output_path = annotation_path.replace('.jsonl', '_pack.json').replace('.json', '_pack.json')
    with open(pack_output_path, 'w', encoding='utf-8') as file:
        json.dump(all_packed_results, file, indent=2)
    print(f"Packed results saved to: {pack_output_path}")