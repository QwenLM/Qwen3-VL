# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import re
from argparse import ArgumentParser
from threading import Thread

import gradio as gr
import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

DEFAULT_CKPT_PATH = 'Qwen/Qwen2.5-VL-7B-Instruct'


def _get_args():
    parser = ArgumentParser()

    parser.add_argument('-c',
                        '--checkpoint-path',
                        type=str,
                        default=DEFAULT_CKPT_PATH,
                        help='Checkpoint name or path, default to %(default)r')
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

    # vLLM specific arguments
    parser.add_argument('--tensor-parallel-size', type=int, default=1,
                        help='Number of GPUs to use for tensor parallelism.')
    parser.add_argument('--max-model-len', type=int, default=4096, help='Maximum sequence length for the model.')
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.9, help='GPU memory utilization for vLLM.')

    args = parser.parse_args()
    return args


def _load_model_processor(args):
    """Load model and processor using vLLM"""

    # Set vLLM parameters
    llm_kwargs = {
        'model': args.checkpoint_path,
        'tensor_parallel_size': args.tensor_parallel_size,
        'max_model_len': args.max_model_len,
        'gpu_memory_utilization': args.gpu_memory_utilization,
        'trust_remote_code': True,
    }
    try:
        # Load vLLM model
        model = LLM(**llm_kwargs)

        # Load processor for preprocessing
        processor = AutoProcessor.from_pretrained(args.checkpoint_path)

        # Check vLLM version
        import vllm
        print(f"vLLM version: {vllm.__version__}")

        return model, processor
    except Exception as e:
        print(f"Failed to load model: {e}")
        raise


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


def _launch_demo(args, model, processor):
    def call_vllm_model(model, processor, messages):
        """Use vLLM for inference"""

        messages = _transform_messages(messages)

        # Use processor to prepare input
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        # Set sampling parameters
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.8,
            repetition_penalty=1.05,
            max_tokens=512,
            stop_token_ids=None
        )

        try:
            # Multimodal input format for vLLM 0.8.0
            if image_inputs or video_inputs:
                # Build input list, format expected by vLLM 0.8.0
                inputs = []

                # Create multimodal input dictionary
                mm_input = {
                    "prompt": text,
                }

                # Add image data
                if image_inputs:
                    # vLLM 0.8.0 expects images in multi_modal_data
                    mm_input["multi_modal_data"] = {"image": image_inputs}

                # Add video data
                if video_inputs:
                    if "multi_modal_data" not in mm_input:
                        mm_input["multi_modal_data"] = {}
                    mm_input["multi_modal_data"]["video"] = video_inputs

                inputs.append(mm_input)

                try:
                    outputs = model.generate(
                        inputs,
                        sampling_params=sampling_params,
                        use_tqdm=False,
                    )
                except Exception as api_error:
                    print(f"Multimodal API call failed: {api_error}")
                    print("Trying fallback method...")

                    # Fallback: revert to text-only processing
                    # Extract image/video description info and add to text
                    if image_inputs:
                        text = f"[User uploaded {len(image_inputs)} image(s)] " + text
                    if video_inputs:
                        text = f"[User uploaded {len(video_inputs)} video(s)] " + text

                    outputs = model.generate(
                        [text],
                        sampling_params=sampling_params,
                        use_tqdm=False
                    )

            else:
                # Text-only input
                outputs = model.generate(
                    [text],
                    sampling_params=sampling_params,
                    use_tqdm=False
                )

            if outputs and len(outputs) > 0:
                generated_text = outputs[0].outputs[0].text
                return generated_text
            else:
                return "Generation failed, please retry."

        except Exception as e:
            print(f"vLLM generation error: {e}")
            return f"Error occurred during generation: {str(e)}"

    def create_predict_fn():

        def predict(_chatbot, task_history):
            nonlocal model, processor
            chat_query = _chatbot[-1][0]
            query = task_history[-1][0]
            if len(chat_query) == 0:
                _chatbot.pop()
                task_history.pop()
                return _chatbot
            print('User: ' + _parse_text(query))
            history_cp = copy.deepcopy(task_history)
            messages = []
            content = []
            for q, a in history_cp:
                if isinstance(q, (tuple, list)):
                    if _is_video_file(q[0]):
                        content.append({'video': f'file://{q[0]}'})
                    else:
                        content.append({'image': f'file://{q[0]}'})
                else:
                    content.append({'text': q})
                    messages.append({'role': 'user', 'content': content})
                    messages.append({'role': 'assistant', 'content': [{'text': a}]})
                    content = []
            messages.pop()

            # Use vLLM to generate response
            try:
                response = call_vllm_model(model, processor, messages)
                response = _remove_image_special(_parse_text(response))
                _chatbot[-1] = (_parse_text(chat_query), response)
                task_history[-1] = (query, response)
                print('Qwen-VL-Chat: ' + response)
                yield _chatbot
            except Exception as e:
                error_msg = f"Error occurred while generating response: {str(e)}"
                print(error_msg)
                _chatbot[-1] = (_parse_text(chat_query), error_msg)
                yield _chatbot

        return predict

    def create_regenerate_fn():

        def regenerate(_chatbot, task_history):
            nonlocal model, processor
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
        return history, task_history

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
<p align="center"><img src="https://modelscope.oss-cn-beijing.aliyuncs.com/resource/qwen.png" style="height: 80px"/><p>"""
                    )
        gr.Markdown("""<center><font size=8>Qwen2.5-VL (vLLM)</center>""")
        gr.Markdown("""\
<center><font size=3>This WebUI is based on Qwen2.5-VL with vLLM acceleration, developed by Alibaba Cloud.</center>""")
        gr.Markdown("""<center><font size=3>本WebUI基于Qwen2.5-VL，使用vLLM加速推理。</center>""")

        chatbot = gr.Chatbot(label='Qwen2.5-VL (vLLM)', elem_classes='control-height', height=500)
        query = gr.Textbox(lines=2, label='Input')
        task_history = gr.State([])

        with gr.Row():
            addfile_btn = gr.UploadButton('📁 Upload (上传文件)', file_types=['image', 'video'])
            submit_btn = gr.Button('🚀 Submit (发送)')
            regen_btn = gr.Button('🤔️ Regenerate (重试)')
            empty_bin = gr.Button('🧹 Clear History (清除历史)')

        submit_btn.click(add_text, [chatbot, task_history, query],
                         [chatbot, task_history]).then(predict, [chatbot, task_history], [chatbot],
                                                       show_progress=True).then(reset_user_input, [], [query])
        empty_bin.click(reset_state, [chatbot, task_history], [chatbot], show_progress=True)
        regen_btn.click(regenerate, [chatbot, task_history], [chatbot], show_progress=True)
        addfile_btn.upload(add_file, [chatbot, task_history, addfile_btn], [chatbot, task_history], show_progress=True)

        gr.Markdown("""\
<font size=2>Note: This demo is governed by the original license of Qwen2.5-VL. \
We strongly advise users not to knowingly generate or allow others to knowingly generate harmful content, \
including hate speech, violence, pornography, deception, etc. \
(注：本演示受Qwen2.5-VL的许可协议限制。我们强烈建议，用户不应传播及不应允许他人传播以下内容，\
包括但不限于仇恨言论、暴力、色情、欺诈相关的有害信息。)""")

    demo.queue().launch(
        share=args.share,
        inbrowser=args.inbrowser,
        server_port=args.server_port,
        server_name=args.server_name,
    )


def main():
    args = _get_args()
    try:
        model, processor = _load_model_processor(args)
        print(f"Starting Web UI, access at: http://{args.server_name}:{args.server_port}")
        _launch_demo(args, model, processor)
    except Exception as e:
        print(f"Got an err: {e}")


if __name__ == '__main__':
    main()