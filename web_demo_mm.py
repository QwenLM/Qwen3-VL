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
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, TextIteratorStreamer

DEFAULT_CKPT_PATH = 'Qwen/Qwen2.5-VL-7B-Instruct'


def get_args():
    parser = ArgumentParser()

    parser.add_argument('-c', '--checkpoint-path', type=str, default=DEFAULT_CKPT_PATH,
                        help='Checkpoint name or path, default to %(default)r')
    parser.add_argument('--cpu-only', action='store_true', help='Run demo with CPU only')
    parser.add_argument('--flash-attn2', action='store_true', default=False,
                        help='Enable flash_attention_2 when loading the model.')
    parser.add_argument('--share', action='store_true', default=False,
                        help='Create a publicly shareable link for the interface.')
    parser.add_argument('--inbrowser', action='store_true', default=False,
                        help='Automatically launch the interface in a new tab on the default browser.')
    parser.add_argument('--server-port', type=int, default=7860, help='Demo server port.')
    parser.add_argument('--server-name', type=str, default='127.0.0.1', help='Demo server name.')

    return parser.parse_args()


def load_model_processor(args):
    device_map = 'cpu' if args.cpu_only else 'auto'

    model_kwargs = {'torch_dtype': 'auto', 'device_map': device_map}
    if args.flash_attn2:
        model_kwargs['attn_implementation'] = 'flash_attention_2'

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.checkpoint_path, **model_kwargs)
    processor = AutoProcessor.from_pretrained(args.checkpoint_path)
    return model, processor


def parse_text(text):
    lines = text.split('\n')
    lines = [line for line in lines if line]
    count = 0
    for i, line in enumerate(lines):
        if '```' in line:
            count += 1
            items = line.split('`')
            lines[i] = f'<pre><code class="language-{items[-1]}">' if count % 2 == 1 else '<br></code></pre>'
        else:
            if count % 2 == 1:
                line = (line.replace('`', r'\`').replace('<', '&lt;').replace('>', '&gt;')
                           .replace(' ', '&nbsp;').replace('*', '&ast;').replace('_', '&lowbar;')
                           .replace('-', '&#45;').replace('.', '&#46;').replace('!', '&#33;')
                           .replace('(', '&#40;').replace(')', '&#41;').replace('$', '&#36;'))
            lines[i] = '<br>' + line if i > 0 else line
    return ''.join(lines)


def remove_image_special(text):
    return re.sub(r'<box>.*?(</box>|$)', '', text.replace('<ref>', '').replace('</ref>', ''))


def is_video_file(filename):
    video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.mpeg']
    return any(filename.lower().endswith(ext) for ext in video_extensions)


def gc_collect():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def transform_messages(original_messages):
    transformed_messages = []
    for message in original_messages:
        new_content = []
        for item in message['content']:
            if 'image' in item:
                new_content.append({'type': 'image', 'image': item['image']})
            elif 'text' in item:
                new_content.append({'type': 'text', 'text': item['text']})
            elif 'video' in item:
                new_content.append({'type': 'video', 'video': item['video']})
        transformed_messages.append({'role': message['role'], 'content': new_content})
    return transformed_messages


def launch_demo(args, model, processor):
    def call_local_model(model, processor, messages):
        messages = transform_messages(messages)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors='pt')
        inputs = inputs.to(model.device)

        streamer = TextIteratorStreamer(processor.tokenizer, timeout=20.0, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = {'max_new_tokens': 512, 'streamer': streamer, **inputs}
        thread = Thread(target=model.generate, kwargs=gen_kwargs)
        thread.start()

        generated_text = ''
        for new_text in streamer:
            generated_text += new_text
            yield generated_text

    def create_predict_fn():
        def predict(chatbot, task_history):
            nonlocal model, processor
            chat_query = chatbot[-1][0]
            query = task_history[-1][0]
            if not chat_query:
                chatbot.pop()
                task_history.pop()
                return chatbot

            print('User:', parse_text(query))
            history_cp = copy.deepcopy(task_history)
            messages = []
            content = []
            for q, a in history_cp:
                if isinstance(q, (tuple, list)):
                    content.append({'video': f'file://{q[0]}' if is_video_file(q[0]) else {'image': f'file://{q[0]}'}})
                else:
                    content.append({'text': q})
                    messages.append({'role': 'user', 'content': content})
                    messages.append({'role': 'assistant', 'content': [{'text': a}]})
                    content = []
            messages.pop()

            full_response = ''
            for response in call_local_model(model, processor, messages):
                chatbot[-1] = (parse_text(chat_query), remove_image_special(parse_text(response)))
                yield chatbot
                full_response = parse_text(response)

            task_history[-1] = (query, full_response)
            print('Qwen-VL-Chat:', parse_text(full_response))
            yield chatbot

        return predict

    def create_regenerate_fn():
        def regenerate(chatbot, task_history):
            nonlocal model, processor
            if not task_history:
                return chatbot
            item = task_history[-1]
            if item[1] is None:
                return chatbot
            task_history[-1] = (item[0], None)
            chatbot_item = chatbot.pop(-1)
            chatbot.append((chatbot_item[0], None) if chatbot_item[0] else (chatbot[-1][0], None))
            chatbot_gen = predict(chatbot, task_history)
            for chatbot in chatbot_gen:
                yield chatbot

        return regenerate

    predict = create_predict_fn()
    regenerate = create_regenerate_fn()

    def add_text(history, task_history, text):
        history = history or []
        task_history = task_history or []
        history.append((parse_text(text), None))
        task_history.append((text, None))
        return history, task_history, ''

    def add_file(history, task_history, file):
        history = history or []
        task_history = task_history or []
        history.append(((file.name,), None))
        task_history.append(((file.name,), None))
        return history, task_history

    def reset_user_input():
        return gr.update(value='')

    def reset_state(chatbot, task_history):
        task_history.clear()
        chatbot.clear()
        gc_collect()
        return []

    with gr.Blocks() as demo:
        gr.Markdown(
            """<p align="center"><img src="https://modelscope.oss-cn-beijing.aliyuncs.com/resource/qwen.png" style="height: 80px"/></p>"""
        )
        gr.Markdown("<center><font size=8>Qwen2.5-VL</center>")
        gr.Markdown("<center><font size=3>This WebUI is based on Qwen2.5-VL, developed by Alibaba Cloud.</center>")
        gr.Markdown("<center><font size=3>本WebUI基于Qwen2.5-VL。</center>")

        chatbot = gr.Chatbot(label='Qwen2.5-VL', elem_classes='control-height', height=500)
        query = gr.Textbox(lines=2, label='Input')
        task_history = gr.State([])

        with gr.Row():
            addfile_btn = gr.UploadButton('📁 Upload (上传文件)', file_types=['image', 'video'])
            submit_btn = gr.Button('🚀 Submit (发送)')
            regen_btn = gr.Button('🤔️ Regenerate (重试)')
            empty_bin = gr.Button('🧹 Clear History (清除历史)')

        submit_btn.click(add_text, [chatbot, task_history, query], [chatbot, task_history]).then(
            predict, [chatbot, task_history], [chatbot], show_progress=True)
        submit_btn.click(reset_user_input, [], [query])
        empty_bin.click(reset_state, [chatbot, task_history], [chatbot], show_progress=True)
        regen_btn.click(regenerate, [chatbot, task_history], [chatbot], show_progress=True)
        addfile_btn.upload(add_file, [chatbot, task_history, addfile_btn], [chatbot, task_history], show_progress=True)

        gr.Markdown(
            """<font size=2>Note: This demo is governed by the original license of Qwen2.5-VL. \
We strongly advise users not to knowingly generate or allow others to knowingly generate harmful content, \
including hate speech, violence, pornography, deception, etc. \
(注：本演示受Qwen2.5-VL的许可协议限制。我们强烈建议，用户不应传播及不应允许他人传播以下内容，\
包括但不限于仇恨言论、暴力、色情、欺诈相关的有害信息。)"""
        )

    demo.queue().launch(
        share=args.share,
        inbrowser=args.inbrowser,
        server_port=args.server_port,
        server_name=args.server_name,
    )


def main():
    args = get_args()
    model, processor = load_model_processor(args)
    launch_demo(args, model, processor)


if __name__ == '__main__':
    main()
