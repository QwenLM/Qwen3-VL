# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import copy
import re
from argparse import ArgumentParser
from threading import Thread

import gradio as gr
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, TextIteratorStreamer

try:
    from vllm import SamplingParams, LLM
    from qwen_vl_utils import process_vision_info
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("Warning: vLLM not available. Install vllm and qwen-vl-utils to use vLLM backend.")


def _get_args():
    parser = ArgumentParser()

    parser.add_argument('-c',
                        '--checkpoint-path',
                        type=str,
                        default='Qwen/Qwen3-VL-235B-A22B-Instruct',
                        help='Checkpoint name or path, default to %(default)r')
    parser.add_argument('--cpu-only', action='store_true', help='Run demo with CPU only')

    parser.add_argument('--flash-attn2',
                        action='store_true',
                        default=False,
                        help='Enable flash_attention_2 when loading the model.')
    parser.add_argument('--share',
                        action='store_true',
                        default=False,
                        help='Create a publicly shareable link for the interface.')
    parser.add_argument('--inbrowser',
                        action='store_true',
                        default=False,
                        help='Automatically launch the interface in a new tab on the default browser.')
    parser.add_argument('--server-port', type=int, default=7860, help='Demo server port.')
    parser.add_argument('--server-name', type=str, default='127.0.0.1', help='Demo server name.')
    parser.add_argument('--backend',
                        type=str,
                        choices=['hf', 'vllm'],
                        default='vllm',
                        help='Backend to use: hf (HuggingFace) or vllm (vLLM)')
    parser.add_argument('--gpu-memory-utilization',
                        type=float,
                        default=0.70,
                        help='GPU memory utilization for vLLM (default: 0.70)')
    parser.add_argument('--tensor-parallel-size',
                        type=int,
                        default=None,
                        help='Tensor parallel size for vLLM (default: auto)')

    args = parser.parse_args()
    return args


def _load_model_processor(args):
    if args.backend == 'vllm':
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM is not available. Please install vllm and qwen-vl-utils.")

        os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
        tensor_parallel_size = args.tensor_parallel_size
        if tensor_parallel_size is None:
            tensor_parallel_size = torch.cuda.device_count()

        # Initialize vLLM sync engine
        model = LLM(
            model=args.checkpoint_path,
            trust_remote_code=True,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enforce_eager=False,
            tensor_parallel_size=tensor_parallel_size,
            seed=0
        )

        # Load processor for vLLM
        processor = AutoProcessor.from_pretrained(args.checkpoint_path)
        return model, processor, 'vllm'
    else:
        if args.cpu_only:
            device_map = 'cpu'
        else:
            device_map = 'auto'

        # Check if flash-attn2 flag is enabled and load model accordingly
        if args.flash_attn2:
            model = AutoModelForImageTextToText.from_pretrained(args.checkpoint_path,
                                                                    torch_dtype='auto',
                                                                    attn_implementation='flash_attention_2',
                                                                    device_map=device_map)
        else:
            model = AutoModelForImageTextToText.from_pretrained(args.checkpoint_path, device_map=device_map)

        processor = AutoProcessor.from_pretrained(args.checkpoint_path)
        return model, processor, 'hf'


def _parse_text(text):
    lines = text.split('\n')
    lines = [line for line in lines if line != '']
    count = 0
    for i, line in enumerate(lines):
        if '```' in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = '<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace('`', r'\`')
                    line = line.replace('<', '&lt;')
                    line = line.replace('>', '&gt;')
                    line = line.replace(' ', '&nbsp;')
                    line = line.replace('*', '&ast;')
                    line = line.replace('_', '&lowbar;')
                    line = line.replace('-', '&#45;')
                    line = line.replace('.', '&#46;')
                    line = line.replace('!', '&#33;')
                    line = line.replace('(', '&#40;')
                    line = line.replace(')', '&#41;')
                    line = line.replace('$', '&#36;')
                lines[i] = '<br>' + line
    text = ''.join(lines)
    return text


def _remove_image_special(text):
    text = text.replace('<ref>', '').replace('</ref>', '')
    return re.sub(r'<box>.*?(</box>|$)', '', text)


def _is_video_file(filename):
    video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.mpeg']
    return any(filename.lower().endswith(ext) for ext in video_extensions)


def _gc():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _transform_messages(original_messages):
    transformed_messages = []
    for message in original_messages:
        new_content = []
        for item in message['content']:
            if 'image' in item:
                new_item = {'type': 'image', 'image': item['image']}
            elif 'text' in item:
                new_item = {'type': 'text', 'text': item['text']}
            elif 'video' in item:
                new_item = {'type': 'video', 'video': item['video']}
            else:
                continue
            new_content.append(new_item)

        new_message = {'role': message['role'], 'content': new_content}
        transformed_messages.append(new_message)

    return transformed_messages


def _prepare_inputs_for_vllm(messages, processor):
    """Prepare inputs for vLLM inference"""
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True
    )

    mm_data = {}
    if image_inputs is not None:
        mm_data['image'] = image_inputs
    if video_inputs is not None:
        mm_data['video'] = video_inputs

    return {
        'prompt': text,
        'multi_modal_data': mm_data,
        'mm_processor_kwargs': video_kwargs
    }


def _launch_demo(args, model, processor, backend):

    def call_local_model(model, processor, messages, backend):
        messages = _transform_messages(messages)

        if backend == 'vllm':
            # vLLM inference
            inputs = _prepare_inputs_for_vllm(messages, processor)
            sampling_params = SamplingParams(max_tokens=1024)

            accumulated_text = ''
            for output in model.generate(inputs, sampling_params=sampling_params):
                for completion in output.outputs:
                    new_text = completion.text
                    if new_text:
                        accumulated_text += new_text
                        yield accumulated_text
        else:
            # HuggingFace inference
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )

            tokenizer = processor.tokenizer
            streamer = TextIteratorStreamer(tokenizer, timeout=20.0, skip_prompt=True, skip_special_tokens=True)

            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            gen_kwargs = {'max_new_tokens': 1024, 'streamer': streamer, **inputs}
            thread = Thread(target=model.generate, kwargs=gen_kwargs)
            thread.start()

            generated_text = ''
            for new_text in streamer:
                generated_text += new_text
                yield generated_text

    def create_predict_fn():

        def predict(_chatbot, task_history):
            nonlocal model, processor, backend
            chat_query = _chatbot[-1][0]
            query = task_history[-1][0]
            if len(chat_query) == 0:
                _chatbot.pop()
                task_history.pop()
                return _chatbot
            print('User: ' + _parse_text(query))
            history_cp = copy.deepcopy(task_history)
            full_response = ''
            messages = []
            content = []
            for q, a in history_cp:
                if isinstance(q, (tuple, list)):
                    if _is_video_file(q[0]):
                        content.append({'video': f'{os.path.abspath(q[0])}'})
                    else:
                        content.append({'image': f'{os.path.abspath(q[0])}'})
                else:
                    content.append({'text': q})
                    messages.append({'role': 'user', 'content': content})
                    messages.append({'role': 'assistant', 'content': [{'text': a}]})
                    content = []
            messages.pop()

            for response in call_local_model(model, processor, messages, backend):
                _chatbot[-1] = (_parse_text(chat_query), _remove_image_special(_parse_text(response)))

                yield _chatbot
                full_response = _parse_text(response)

            task_history[-1] = (query, full_response)
            print('Qwen-VL-Chat: ' + _parse_text(full_response))
            yield _chatbot

        return predict


    def create_regenerate_fn():

        def regenerate(_chatbot, task_history):
            nonlocal model, processor, backend
            if not task_history:
                return _chatbot
            item = task_history[-1]
            if item[1] is None:
                return _chatbot
            task_history[-1] = (item[0], None)
            chatbot_item = _chatbot.pop(-1)
            if chatbot_item[0] is None:
                _chatbot[-1] = (_chatbot[-1][0], None)
            else:
                _chatbot.append((chatbot_item[0], None))
            _chatbot_gen = predict(_chatbot, task_history)
            for _chatbot in _chatbot_gen:
                yield _chatbot

        return regenerate

    predict = create_predict_fn()
    regenerate = create_regenerate_fn()

    def add_text(history, task_history, text):
        task_text = text
        history = history if history is not None else []
        task_history = task_history if task_history is not None else []
        history = history + [(_parse_text(text), None)]
        task_history = task_history + [(task_text, None)]
        return history, task_history, ''

    def add_file(history, task_history, file):
        history = history if history is not None else []
        task_history = task_history if task_history is not None else []
        history = history + [((file.name,), None)]
        task_history = task_history + [((file.name,), None)]
        return history, task_history

    def reset_user_input():
        return gr.update(value='')

    def reset_state(_chatbot, task_history):
        task_history.clear()
        _chatbot.clear()
        _gc()
        return []

    with gr.Blocks() as demo:
        gr.Markdown("""\
<p align="center"><img src="https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3-VL/qwen3vllogo.png" style="height: 80px"/><p>"""
                   )
        gr.Markdown("""<center><font size=8>Qwen3-VL</center>""")
        gr.Markdown(f"""\
<center><font size=3>This WebUI is based on Qwen3-VL, developed by Alibaba Cloud. Backend: {backend.upper()}</center>""")
        gr.Markdown(f"""<center><font size=3>本 WebUI 基于 Qwen3-VL。</center>""")

        chatbot = gr.Chatbot(label='Qwen3-VL', elem_classes='control-height', height=500)
        query = gr.Textbox(lines=2, label='Input')
        task_history = gr.State([])

        with gr.Row():
            addfile_btn = gr.UploadButton('📁 Upload (上传文件)', file_types=['image', 'video'])
            submit_btn = gr.Button('🚀 Submit (发送)')
            regen_btn = gr.Button('🤔️ Regenerate (重试)')
            empty_bin = gr.Button('🧹 Clear History (清除历史)')

        submit_btn.click(add_text, [chatbot, task_history, query],
                         [chatbot, task_history]).then(predict, [chatbot, task_history], [chatbot], show_progress=True)
        submit_btn.click(reset_user_input, [], [query])
        empty_bin.click(reset_state, [chatbot, task_history], [chatbot], show_progress=True)
        regen_btn.click(regenerate, [chatbot, task_history], [chatbot], show_progress=True)
        addfile_btn.upload(add_file, [chatbot, task_history, addfile_btn], [chatbot, task_history], show_progress=True)

        gr.Markdown("""\
<font size=2>Note: This demo is governed by the original license of Qwen3-VL. \
We strongly advise users not to knowingly generate or allow others to knowingly generate harmful content, \
including hate speech, violence, pornography, deception, etc. \
(注：本演示受 Qwen3-VL 的许可协议限制。我们强烈建议，用户不应传播及不应允许他人传播以下内容，\
包括但不限于仇恨言论、暴力、色情、欺诈相关的有害信息。)""")

    demo.queue().launch(
        share=args.share,
        inbrowser=args.inbrowser,
        server_port=args.server_port,
        server_name=args.server_name,
    )


def main():
    args = _get_args()
    model, processor, backend = _load_model_processor(args)
    _launch_demo(args, model, processor, backend)


if __name__ == '__main__':
    main()
