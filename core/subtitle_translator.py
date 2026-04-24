#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
subtitle_translator.py - 双语字幕翻译工具 (llama.cpp 后端) v2.5.4
修复：推理模型思考内容被误判为异常、停止时保留了错误输出
Copyright 2026 光影的故事2018
"""

import os
import re
import json
import time
import asyncio
import aiohttp
import requests
import tempfile
import webbrowser
import threading
import uuid
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from collections import OrderedDict
from pathlib import Path
import gradio as gr


class BilingualTranslator:
    def __init__(self):
        self.llama_api_url = "http://127.0.0.1:8080/v1/chat/completions"
        self.models_url = "http://127.0.0.1:8080/v1/models"
        self.output_dir = self._get_output_dir()
        self.available_models = []
        self._load_models()
        self.translation_cache = OrderedDict()
        self.cache_max_size = 1000
        self.cache_file = os.path.join(self.output_dir, "translation_cache.json")
        self._load_cache_from_disk()
        self.max_concurrent = 3
        self.semaphore = asyncio.Semaphore(self.max_concurrent)
        self._stop_flags = {}
        self._stop_lock = threading.Lock()

    def _get_output_dir(self) -> str:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(script_dir)
        output_dir = os.path.join(root_dir, "双语字幕输出")
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def _load_models(self):
        try:
            resp = requests.get(self.models_url, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                models = [item["id"] for item in data.get("data", [])]
                self.available_models = models
                print(f"找到 {len(self.available_models)} 个模型: {self.available_models[:5]}...")
        except:
            pass

    def _load_cache_from_disk(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for k_str, v in data.items():
                        parts = k_str.split('|||')
                        if len(parts) == 4:
                            key = (parts[0], parts[1], parts[2], float(parts[3]))
                            self.translation_cache[key] = v
            except:
                pass

    def _save_cache_to_disk(self):
        def _save():
            try:
                cache_dict = {'|||'.join(str(i) for i in k): v for k, v in self.translation_cache.items()}
                with open(self.cache_file, 'w', encoding='utf-8') as f:
                    json.dump(cache_dict, f, ensure_ascii=False, indent=2)
            except:
                pass
        try:
            loop = asyncio.get_running_loop()
            loop.run_in_executor(None, _save)
        except RuntimeError:
            _save()

    def _add_to_cache(self, key, value):
        if key in self.translation_cache:
            self.translation_cache.move_to_end(key)
        self.translation_cache[key] = value
        if len(self.translation_cache) > self.cache_max_size:
            self.translation_cache.popitem(last=False)
        self._save_cache_to_disk()

    def _get_cache_key(self, text, target_lang, model, temperature):
        return (text.strip().lower(), target_lang, model, temperature)

    def set_stop_flag(self, session_id):
        with self._stop_lock:
            self._stop_flags[session_id] = True

    def clear_stop_flag(self, session_id):
        with self._stop_lock:
            self._stop_flags.pop(session_id, None)

    def is_stopped(self, session_id):
        with self._stop_lock:
            return self._stop_flags.get(session_id, False)

    # ==================== 异步翻译核心 ====================
    async def _async_translate_one(self, session, text, target_lang, model, temperature,
                                   gpu_layers, max_tokens=1024):
        if not text.strip():
            return ""
        cache_key = self._get_cache_key(text, target_lang, model, temperature)
        if cache_key in self.translation_cache:
            return self.translation_cache[cache_key]

        language_names = {
            "英语": "English", "中文": "Chinese", "日语": "Japanese",
            "韩语": "Korean", "法语": "French", "德语": "German",
            "西班牙语": "Spanish", "俄语": "Russian", "意大利语": "Italian",
            "葡萄牙语": "Portuguese", "荷兰语": "Dutch", "波兰语": "Polish",
            "土耳其语": "Turkish", "阿拉伯语": "Arabic", "泰语": "Thai",
            "越南语": "Vietnamese", "印尼语": "Indonesian", "希伯来语": "Hebrew",
            "瑞典语": "Swedish", "芬兰语": "Finnish"
        }
        target_lang_en = language_names.get(target_lang, target_lang)
        system_prompt = "你是一个专业的字幕翻译助手。请将用户提供的文本翻译成指定语言，只输出翻译结果，不要包含任何额外解释。"
        user_prompt = f"""请将以下文本翻译成{target_lang_en}，要求：
1. 翻译准确、自然，适合字幕显示
2. 保持原意和语气，不添加额外解释
3. 如果原文有特殊格式（如标点、换行），尽量保留
4. 只输出翻译结果，不要包含任何额外内容

原文：{text}
翻译："""
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        payload = {
            "model": model, "messages": messages, "stream": False,
            "temperature": temperature, "max_tokens": int(max_tokens),
            "ngl": gpu_layers, "n_gpu_layers": gpu_layers
        }

        for _ in range(3):
            try:
                async with self.semaphore:
                    async with session.post(self.llama_api_url, json=payload,
                                            timeout=aiohttp.ClientTimeout(total=120)) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            translated = data["choices"][0]["message"]["content"].strip()
                            # 清洗
                            translated = re.sub(r'^(翻译[:：]?|["\“\”]{1,})', '', translated)
                            translated = re.sub(r'["\“\”]{1,}$', '', translated)
                            translated = re.sub(r'<\|.*?\|>', '', translated)
                            translated = re.sub(r'</?s>', '', translated)
                            translated = re.sub(r'<\s*think\s*>.*?<\s*/think\s*>', '', translated, flags=re.DOTALL)
                            translated = translated.strip()
                            self._add_to_cache(cache_key, translated)
                            return translated
            except:
                await asyncio.sleep(1)
        return f"[翻译失败] {text}"

    # ==================== 流式翻译（简化版，处理推理模型） ====================
    async def _async_stream_translate_one(self, session, text, target_lang, model, temperature,
                                          gpu_layers, max_tokens=1024, session_id=None):
        if not text.strip():
            yield "", "", True
            return

        language_names = {
            "英语": "English", "中文": "Chinese", "日语": "Japanese",
            "韩语": "Korean", "法语": "French", "德语": "German",
            "西班牙语": "Spanish", "俄语": "Russian", "意大利语": "Italian",
            "葡萄牙语": "Portuguese", "荷兰语": "Dutch", "波兰语": "Polish",
            "土耳其语": "Turkish", "阿拉伯语": "Arabic", "泰语": "Thai",
            "越南语": "Vietnamese", "印尼语": "Indonesian", "希伯来语": "Hebrew",
            "瑞典语": "Swedish", "芬兰语": "Finnish"
        }
        target_lang_en = language_names.get(target_lang, target_lang)
        system_prompt = "你是一个专业的字幕翻译助手。请将用户提供的文本翻译成指定语言，只输出翻译结果，不要包含任何额外解释。"
        user_prompt = f"""请将以下文本翻译成{target_lang_en}，要求：
1. 翻译准确、自然，适合字幕显示
2. 保持原意和语气，不添加额外解释
3. 如果原文有特殊格式（如标点、换行），尽量保留
4. 只输出翻译结果，不要包含任何额外内容

原文：{text}
翻译："""
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        payload = {
            "model": model, "messages": messages, "stream": True,
            "temperature": temperature, "max_tokens": int(max_tokens),
            "ngl": gpu_layers, "n_gpu_layers": gpu_layers
        }

        thought = ""
        answer = ""

        try:
            async with session.post(self.llama_api_url, json=payload,
                                    timeout=aiohttp.ClientTimeout(total=120)) as resp:
                if resp.status != 200:
                    yield "", f"[错误] 服务返回状态码 {resp.status}", True
                    return

                async for line in resp.content:
                    if self.is_stopped(session_id):
                        # 停止时保留当前内容，优先返回 answer，否则返回 thought
                        if not answer and thought:
                            answer = thought.strip()
                        if not answer:
                            answer = "[已停止]"
                        else:
                            answer += "\n\n[已停止]"
                        yield thought, answer, True
                        return

                    line = line.decode('utf-8', errors='replace').strip()
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        continue
                    try:
                        chunk = json.loads(data_str)
                        delta = chunk["choices"][0].get("delta", {})
                        reasoning = delta.get("reasoning_content", "")
                        if reasoning:
                            thought += reasoning
                            yield thought, answer, False
                            continue
                        content = delta.get("content", "")
                        if content:
                            answer += content
                            yield thought, answer, False
                    except:
                        continue

                # 最终处理
                final_answer = answer.strip()
                # 如果只有推理内容，直接使用推理内容作为答案
                if not final_answer and thought:
                    final_answer = thought.strip()
                    thought = ""
                # 清洗
                final_answer = re.sub(r'^(翻译[:：]?|["\“\”]{1,})', '', final_answer)
                final_answer = re.sub(r'["\“\”]{1,}$', '', final_answer)
                final_answer = re.sub(r'<\|.*?\|>', '', final_answer)
                final_answer = re.sub(r'</?s>', '', final_answer)
                final_answer = re.sub(r'<\s*think\s*>.*?<\s*/think\s*>', '', final_answer, flags=re.DOTALL)
                final_answer = final_answer.strip()
                if not final_answer:
                    final_answer = "[翻译失败] 模型未返回有效翻译内容"
                yield thought, final_answer, True

        except Exception as e:
            if not answer and thought:
                answer = thought.strip()
            if not answer:
                answer = f"[翻译异常] {str(e)}"
            yield thought, answer, True

    # ==================== 批量翻译 ====================
    async def translate_subtitles_async(self, subtitles, target_lang, model, bilingual_style,
                                        temperature, gpu_layers, max_tokens=1024):
        translated = []
        async with aiohttp.ClientSession() as session:
            tasks = []
            for sub in subtitles:
                t = sub.get('original_text', '')
                if t.strip():
                    tasks.append(self._async_translate_one(session, t, target_lang, model, temperature, gpu_layers, max_tokens))
                else:
                    tasks.append(asyncio.sleep(0, result=""))
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for sub, res in zip(subtitles, results):
                if isinstance(res, Exception):
                    res = f"[翻译异常] {str(res)}"
                sub_copy = sub.copy()
                sub_copy['translated_text'] = res
                sub_copy['bilingual_text'] = self.create_bilingual_text(sub.get('original_text', ''), res, bilingual_style)
                translated.append(sub_copy)
        return translated

    # ==================== 字幕解析与生成（不变） ====================
    def parse_srt(self, srt_content: str) -> List[Dict]:
        subtitles = []
        srt_content = srt_content.replace('\r\n', '\n').replace('\r', '\n')
        blocks = re.split(r'\n\s*\n', srt_content.strip())
        for block in blocks:
            lines = block.strip().split('\n')
            if len(lines) >= 3:
                try:
                    if lines[0].strip().isdigit():
                        index = int(lines[0])
                        timecode_line = lines[1]
                        text_lines = lines[2:]
                    else:
                        index = len(subtitles) + 1
                        timecode_line = lines[0]
                        text_lines = lines[1:]
                    time_match = re.search(
                        r'(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})',
                        timecode_line
                    )
                    if time_match:
                        start_time = time_match.group(1)
                        end_time = time_match.group(2)
                    else:
                        continue
                    text = '\n'.join(text_lines).strip()
                    if text:
                        subtitles.append({
                            'index': index,
                            'start_time': start_time,
                            'end_time': end_time,
                            'original_text': text,
                            'translated_text': '',
                            'bilingual_text': ''
                        })
                except Exception as e:
                    print(f"解析SRT块失败: {e}")
                    continue
        return subtitles

    def parse_txt(self, txt_content: str) -> List[Dict]:
        subtitles = []
        lines = txt_content.strip().split('\n')
        for i, line in enumerate(lines, 1):
            line = line.strip()
            if line:
                subtitles.append({
                    'index': i,
                    'original_text': line,
                    'translated_text': '',
                    'bilingual_text': ''
                })
        return subtitles

    def create_bilingual_text(self, original: str, translated: str, style: str = "上下对照") -> str:
        if not original.strip() or not translated.strip():
            return original if original.strip() else translated
        if style == "上下对照":
            return f"{original}\n{translated}"
        elif style == "括号对照":
            return f"{original} ({translated})"
        elif style == "斜杠分隔":
            return f"{original} / {translated}"
        else:
            return f"{original}\n({translated})"

    def generate_bilingual_srt(self, subtitles: List[Dict]) -> str:
        srt_lines = []
        for sub in subtitles:
            srt_lines.append(str(sub['index']))
            if 'start_time' in sub and 'end_time' in sub and sub['start_time'] and sub['end_time']:
                srt_lines.append(f"{sub['start_time']} --> {sub['end_time']}")
            else:
                start_sec = (sub['index'] - 1) * 5
                end_sec = start_sec + 5
                start_time = f"{start_sec//3600:02d}:{(start_sec%3600)//60:02d}:{start_sec%60:02d},000"
                end_time = f"{end_sec//3600:02d}:{(end_sec%3600)//60:02d}:{end_sec%60:02d},000"
                srt_lines.append(f"{start_time} --> {end_time}")
            srt_lines.append(sub.get('bilingual_text') or sub.get('original_text', ''))
            srt_lines.append("")
        return '\n'.join(srt_lines)

    def generate_bilingual_txt(self, subtitles: List[Dict], include_original: bool = True) -> str:
        txt_lines = []
        for sub in subtitles:
            if include_original:
                txt_lines.append(sub.get('bilingual_text') or sub.get('original_text', ''))
            else:
                txt_lines.append(sub.get('translated_text') or sub.get('original_text', ''))
        return '\n'.join(txt_lines)

    def generate_comparison_txt(self, subtitles: List[Dict]) -> str:
        txt_lines = ["=" * 60, "双语字幕对照表", "=" * 60, ""]
        for sub in subtitles:
            txt_lines.append(f"【第 {sub['index']} 条】")
            txt_lines.append(f"原文: {sub.get('original_text', '')}")
            txt_lines.append(f"译文: {sub.get('translated_text', '')}")
            txt_lines.append("-" * 40)
        return '\n'.join(txt_lines)

    def save_results(self, content: str, filename: str) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(filename)[0]
        ext = os.path.splitext(filename)[1]
        filepath = os.path.join(self.output_dir, f"{base_name}_{timestamp}{ext}")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return filepath


class TranslatorUI:
    """翻译工具界面 - 异步适配 + 聊天翻译面板 + 思考过程嵌入 + tokens控制 + 停止生成"""

    def __init__(self):
        self.translator = BilingualTranslator()
        self.current_subtitles = []
        print(f"输出目录: {self.translator.output_dir}")
        print(f"可用模型: {self.translator.available_models}")

    def create_interface(self):
        with gr.Blocks(title="双语字幕翻译工具 (llama.cpp 异步版)") as demo:
            gr.Markdown("""# 双语字幕翻译工具 (llama.cpp 异步版)
**使用本地 llama.cpp 服务翻译字幕，支持 GPU 层数调整、缓存持久化、异步并发，聊天面板支持思考过程显示**
""")

            with gr.Tabs():
                # ==================== Tab 1: 聊天翻译 ====================
                with gr.Tab("聊天翻译"):
                    with gr.Row():
                        with gr.Column(scale=4):
                            chat_history_box = gr.Chatbot(
                                value=[{"role": "assistant", "content": "你好！我是翻译助手。输入文本，我将为你翻译成目标语言。"}],
                                label="对话历史",
                                height=800
                            )
                            with gr.Row():
                                chat_input = gr.Textbox(
                                    label="输入翻译文本",
                                    placeholder="输入文本后按回车发送...",
                                    lines=8,
                                    scale=4
                                )
                                with gr.Column(scale=1, min_width=100):
                                    chat_send_btn = gr.Button("发送", variant="primary")
                                    chat_stop_btn = gr.Button("停止", variant="stop")
                                    chat_clear_btn = gr.Button("清空对话", variant="secondary")

                            with gr.Row():
                                chat_export_format = gr.Dropdown(
                                    choices=["纯文本", "Markdown"],
                                    label="导出格式",
                                    value="纯文本",
                                    scale=2
                                )
                                chat_export_btn = gr.Button("导出对话", variant="primary", scale=1)
                                chat_open_dir_btn = gr.Button("打开输出目录", variant="secondary", scale=1)
                            chat_export_file = gr.File(label="下载导出的文件", visible=True, height=35)

                        with gr.Column(scale=1, min_width=280):
                            gr.Markdown("### 翻译设置")
                            chat_model = gr.Dropdown(
                                label="翻译模型",
                                choices=self._get_model_choices(),
                                value=self._get_default_model(),
                                interactive=True
                            )
                            refresh_models_btn = gr.Button("🔄 刷新模型列表", size="sm")
                            chat_target_lang = gr.Dropdown(
                                label="目标语言",
                                choices=["英语", "中文", "日语", "韩语", "法语", "德语",
                                         "西班牙语", "俄语", "意大利语", "葡萄牙语",
                                         "荷兰语", "波兰语", "土耳其语", "阿拉伯语",
                                         "泰语", "越南语", "印尼语", "希伯来语",
                                         "瑞典语", "芬兰语"],
                                value="英语"
                            )
                            chat_temperature = gr.Slider(0.1, 1.0, value=0.3, step=0.1, label="温度参数")
                            chat_max_tokens = gr.Slider(128, 8192, value=1024, step=128, label="最大长度 (tokens)")
                            chat_gpu_layers = gr.Slider(0, 99, value=0, step=1, label="GPU 层数 (0=仅CPU, 99=全GPU)")
                            gr.Markdown("---")
                            chat_status = gr.Textbox(label="状态", value="就绪", interactive=False, lines=2)

                    chat_session_state = gr.State("")

                    refresh_models_btn.click(
                        self.refresh_models,
                        outputs=[chat_model]
                    )

                    async def chat_translate(message, history, model, target_lang, temperature, max_tokens, gpu_layers, session_id):
                        if not session_id:
                            session_id = uuid.uuid4().hex
                        self.translator.clear_stop_flag(session_id)

                        if not message.strip():
                            yield history, "请输入文本", session_id
                            return
                        if not model:
                            yield history, "请先选择翻译模型", session_id
                            return

                        history.append({"role": "user", "content": message})
                        history.append({"role": "assistant", "content": "..."})
                        yield history, "正在翻译...", session_id

                        full_thought = ""
                        full_answer = ""
                        try:
                            async with aiohttp.ClientSession() as session:
                                async for thought, answer, is_final in self.translator._async_stream_translate_one(
                                    session, message, target_lang, model, temperature, int(gpu_layers),
                                    int(max_tokens), session_id
                                ):
                                    full_thought = thought
                                    full_answer = answer

                                    if is_final:
                                        if "[已停止]" in answer:
                                            if full_thought:
                                                final_content = f"""
                                                <details class="thoughts-details">
                                                    <summary><strong>思考过程</strong> (点击展开/折叠)</summary>
                                                    <div class="thoughts-content">{full_thought.replace(chr(10), '<br>')}</div>
                                                </details>
                                                {full_answer}
                                                """
                                            else:
                                                final_content = full_answer
                                            history[-1]["content"] = final_content
                                            yield history, "翻译已停止", session_id
                                            return

                                        if full_thought:
                                            final_content = f"""
                                            <details class="thoughts-details">
                                                <summary><strong>思考过程</strong> (点击展开/折叠)</summary>
                                                <div class="thoughts-content">{full_thought.replace(chr(10), '<br>')}</div>
                                            </details>
                                            {full_answer}
                                            """
                                        else:
                                            final_content = full_answer

                                        history[-1]["content"] = final_content
                                        yield history, f"翻译完成", session_id
                                    else:
                                        current_content = ""
                                        if full_thought:
                                            current_content += f"""
                                            <details class="thoughts-details" open>
                                                <summary><strong>思考过程</strong> (点击折叠)</summary>
                                                <div class="thoughts-content">{full_thought.replace(chr(10), '<br>')}</div>
                                            </details>
                                            """
                                        current_content += full_answer if full_answer else "▌"
                                        history[-1]["content"] = current_content
                                        yield history, f"翻译中... (已生成 {len(full_answer)} 字符)", session_id

                        except Exception as e:
                            history[-1]["content"] = f"翻译失败: {str(e)}"
                            yield history, f"错误: {str(e)}", session_id
                        finally:
                            self.translator.clear_stop_flag(session_id)

                    chat_send_btn.click(
                        fn=chat_translate,
                        inputs=[chat_input, chat_history_box, chat_model, chat_target_lang,
                                chat_temperature, chat_max_tokens, chat_gpu_layers, chat_session_state],
                        outputs=[chat_history_box, chat_status, chat_session_state]
                    ).then(lambda: "", None, [chat_input])

                    chat_input.submit(
                        fn=chat_translate,
                        inputs=[chat_input, chat_history_box, chat_model, chat_target_lang,
                                chat_temperature, chat_max_tokens, chat_gpu_layers, chat_session_state],
                        outputs=[chat_history_box, chat_status, chat_session_state]
                    ).then(lambda: "", None, [chat_input])

                    def stop_chat_translation(session_id):
                        if session_id:
                            self.translator.set_stop_flag(session_id)
                        return session_id, "已请求停止"

                    chat_stop_btn.click(
                        fn=stop_chat_translation,
                        inputs=[chat_session_state],
                        outputs=[chat_session_state, chat_status]
                    )

                    def clear_chat():
                        return (
                            [{"role": "assistant", "content": "对话已清空。输入文本开始翻译。"}],
                            "对话已清空",
                            ""
                        )

                    chat_clear_btn.click(
                        fn=clear_chat,
                        outputs=[chat_history_box, chat_status, chat_session_state]
                    )

                    def export_chat(history, fmt):
                        if not history:
                            return None, "没有对话内容可导出"
                        lines = []
                        for msg in history:
                            role = msg.get("role", "unknown")
                            content = msg.get("content", "")
                            clean_content = self._strip_html(content)
                            if role == "user":
                                lines.append(f"用户: {clean_content}")
                            elif role == "assistant":
                                lines.append(f"助手: {clean_content}")
                            lines.append("")
                        content = "\n".join(lines)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"chat_translation_{timestamp}.txt"
                        filepath = self.translator.save_results(content, filename)
                        return gr.update(value=filepath, visible=True), f"导出成功: {filepath}"

                    chat_export_btn.click(
                        fn=export_chat,
                        inputs=[chat_history_box, chat_export_format],
                        outputs=[chat_export_file, chat_status]
                    )

                    def open_output_dir():
                        try:
                            os.startfile(self.translator.output_dir)
                        except:
                            webbrowser.open(self.translator.output_dir)
                        return "已打开输出目录"

                    chat_open_dir_btn.click(
                        fn=open_output_dir,
                        outputs=[chat_status]
                    )

                # ==================== Tab 2: 单文件翻译 ====================
                with gr.Tab("单文件翻译"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            input_file = gr.File(label="上传字幕文件", file_types=[".srt", ".txt"], type="filepath")
                            input_text = gr.Textbox(label="或直接粘贴字幕内容", placeholder="粘贴SRT或TXT格式的字幕内容...", lines=10)

                            with gr.Group():
                                gr.Markdown("### 翻译设置")
                                model_select = gr.Dropdown(
                                    label="翻译模型",
                                    choices=self._get_model_choices(),
                                    value=self._get_default_model(),
                                    interactive=True
                                )
                                refresh_models_btn2 = gr.Button("🔄 刷新模型列表", size="sm")
                                target_lang = gr.Dropdown(
                                    label="目标语言",
                                    choices=["英语", "中文", "日语", "韩语", "法语", "德语",
                                             "西班牙语", "俄语", "意大利语", "葡萄牙语",
                                             "荷兰语", "波兰语", "土耳其语", "阿拉伯语",
                                             "泰语", "越南语", "印尼语", "希伯来语",
                                             "瑞典语", "芬兰语"],
                                    value="英语"
                                )
                                bilingual_style = gr.Radio(
                                    label="双语样式",
                                    choices=["上下对照", "括号对照", "斜杠分隔", "原文优先"],
                                    value="上下对照"
                                )

                            with gr.Group():
                                gr.Markdown("### 翻译参数")
                                temperature = gr.Slider(0.1, 1.0, value=0.3, step=0.1, label="温度参数")
                                max_tokens_single = gr.Slider(128, 8192, value=1024, step=128, label="最大长度 (tokens)")
                                gpu_layers = gr.Slider(0, 99, value=0, step=1, label="GPU 层数 (0=仅CPU, 99=全GPU)", info="需服务器支持请求级设置")
                                max_concurrent = gr.Slider(1, 10, value=3, step=1, label="最大并发数", info="同时翻译的句子数")

                            translate_btn = gr.Button("开始翻译", variant="primary")
                            preview_btn = gr.Button("预览前5条", variant="secondary")

                        with gr.Column(scale=1):
                            output_preview = gr.Textbox(label="翻译预览", lines=15, placeholder="翻译结果将显示在这里...")
                            with gr.Group():
                                gr.Markdown("### 输出格式")
                                output_format = gr.Radio(
                                    label="选择输出格式",
                                    choices=["双语SRT", "双语TXT", "纯译文TXT", "对照表TXT"],
                                    value="双语SRT"
                                )
                                download_btn = gr.Button("下载结果", variant="primary")
                            output_stats = gr.Textbox(label="统计信息", interactive=False, lines=3)
                            output_file = gr.File(label="下载文件", visible=False)

                    input_file.change(self.load_file_content, inputs=[input_file], outputs=[input_text])
                    refresh_models_btn2.click(self.refresh_models, outputs=[model_select])

                    translate_btn.click(
                        fn=self.translate_subtitles_async_wrapper,
                        inputs=[input_text, model_select, target_lang, bilingual_style,
                                temperature, max_tokens_single, gpu_layers, max_concurrent],
                        outputs=[output_preview, output_stats]
                    ).then(
                        lambda: (gr.update(visible=False), None),
                        outputs=[output_file, output_stats],
                        queue=False
                    )

                    preview_btn.click(
                        fn=self.preview_translation_async_wrapper,
                        inputs=[input_text, model_select, target_lang, bilingual_style,
                                temperature, max_tokens_single, gpu_layers],
                        outputs=[output_preview]
                    )

                    download_btn.click(
                        self.download_results,
                        inputs=[output_format],
                        outputs=[output_file, output_stats]
                    )

                # ==================== Tab 3: 批量翻译 ====================
                with gr.Tab("批量翻译"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            batch_files = gr.Files(label="选择多个字幕文件", file_count="multiple", file_types=[".srt", ".txt"], type="filepath")
                            with gr.Group():
                                gr.Markdown("### 批量设置")
                                batch_model = gr.Dropdown(
                                    label="翻译模型",
                                    choices=self._get_model_choices(),
                                    value=self._get_default_model()
                                )
                                batch_target_lang = gr.Dropdown(
                                    label="目标语言",
                                    choices=["英语", "中文", "日语", "韩语", "法语", "德语",
                                             "西班牙语", "俄语", "意大利语", "葡萄牙语",
                                             "荷兰语", "波兰语", "土耳其语", "阿拉伯语",
                                             "泰语", "越南语", "印尼语", "希伯来语",
                                             "瑞典语", "芬兰语"],
                                    value="英语"
                                )
                                batch_temperature = gr.Slider(0.1, 1.0, value=0.3, step=0.1, label="温度参数")
                                batch_max_tokens = gr.Slider(128, 8192, value=1024, step=128, label="最大长度 (tokens)")
                                batch_gpu_layers = gr.Slider(0, 99, value=0, step=1, label="GPU 层数")
                                batch_style = gr.Radio(
                                    label="双语样式",
                                    choices=["上下对照", "括号对照", "斜杠分隔", "原文优先"],
                                    value="上下对照"
                                )
                                batch_format = gr.Radio(
                                    label="输出格式",
                                    choices=["双语SRT", "双语TXT"],
                                    value="双语SRT"
                                )
                                batch_translate_btn = gr.Button("批量翻译", variant="primary")

                        with gr.Column(scale=1):
                            batch_output = gr.Textbox(label="批量处理结果", lines=20, placeholder="批量处理结果将显示在这里...")

                    batch_translate_btn.click(
                        self.batch_translate_async_wrapper,
                        inputs=[batch_files, batch_model, batch_target_lang, batch_format,
                                batch_temperature, batch_max_tokens, batch_gpu_layers, batch_style],
                        outputs=[batch_output]
                    )

                # ==================== Tab 4: 设置与状态 ====================
                with gr.Tab("设置与状态"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### llama.cpp 服务状态")
                            llama_status = gr.Textbox(
                                label="llama.cpp 服务状态",
                                value=self.check_llama_status(),
                                interactive=False,
                                lines=3
                            )
                            refresh_status_btn = gr.Button("刷新状态", variant="secondary")
                            gr.Markdown("### 自定义 API 地址")
                            api_url_input = gr.Textbox(
                                label="API 基础地址 (例如 http://127.0.0.1:8080)",
                                value=self.translator.llama_api_url.replace("/v1/chat/completions", "")
                            )
                            apply_api_btn = gr.Button("应用 API 地址")
                            api_status = gr.Textbox(label="设置结果", interactive=False)
                            gr.Markdown("### 模型管理")
                            model_info = gr.Textbox(
                                label="可用模型列表",
                                value=self.get_model_list(),
                                interactive=False,
                                lines=10
                            )
                        with gr.Column(scale=1):
                            gr.Markdown("### 输出目录")
                            output_dir_info = gr.Textbox(
                                label="当前输出目录",
                                value=self.translator.output_dir,
                                interactive=False,
                                lines=2
                            )
                            open_dir_btn = gr.Button("打开输出目录", variant="secondary")
                            gr.Markdown("### 使用说明")
                            gr.Markdown("""
**使用步骤：**
1. 确保 llama.cpp 服务正在运行
2. 点击"刷新模型列表"获取可用模型
3. 上传字幕文件或粘贴内容（或使用聊天翻译）
4. 选择翻译模型、目标语言、GPU 层数等参数
5. 点击"开始翻译"（异步并发，界面不卡顿）
6. 下载生成的双语字幕
""")
                    refresh_status_btn.click(
                        self.update_status,
                        outputs=[llama_status, model_info]
                    )
                    open_dir_btn.click(self.open_output_dir)

                    def apply_api_url(url):
                        base_url = url.rstrip('/')
                        self.translator.llama_api_url = f"{base_url}/v1/chat/completions"
                        self.translator.models_url = f"{base_url}/v1/models"
                        self.translator._load_models()
                        return f"API 地址已更新为 {base_url}，模型列表已刷新"
                    apply_api_btn.click(apply_api_url, inputs=[api_url_input], outputs=[api_status])

            # 页脚
            gr.HTML("""
<div style="text-align: center; color: #666; font-size: 0.9em;">
 <p>本工具仅用于个人学习与视频剪辑使用，禁止商业用途。</p>
 <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 15px; border-radius: 8px; margin: 15px auto; max-width: 600px;">
  <p style="color: white; font-weight: bold; margin: 5px 0;">🎬 更新请关注B站up主：光影的故事2018</p>
  <p style="color: white; margin: 5px 0;">
   🔗 <strong>B站主页</strong>: <a href="https://space.bilibili.com/381518712" target="_blank" style="color: #ffdd40; text-decoration: none;"> space.bilibili.com/381518712 </a>
  </p>
 </div>
 <p>© 原创 WebUI 代码 © 2026 光影紐扣 版权所有</p>
</div>
""")
        return demo

    # ==================== UI 辅助方法 ====================
    def _get_model_choices(self):
        models = self.translator.available_models
        if models:
            return [(m, m) for m in models]
        else:
            return [("(无可用模型，请刷新)", None)]

    def _get_default_model(self):
        models = self.translator.available_models
        if models:
            return models[0]
        return None

    def load_file_content(self, file):
        if file is None:
            return ""
        try:
            if isinstance(file, str) and os.path.exists(file):
                with open(file, 'r', encoding='utf-8') as f:
                    return f.read()
            return ""
        except Exception as e:
            return f"读取文件失败: {str(e)}"

    def refresh_models(self):
        self.translator._load_models()
        choices = self._get_model_choices()
        value = self._get_default_model()
        return gr.update(choices=choices, value=value)

    @staticmethod
    def _strip_html(text: str) -> str:
        if not text:
            return ""
        clean = re.sub(r'<[^>]*>', '', text)
        clean = clean.replace('\n', ' ').strip()
        return clean

    # ==================== 异步包装器 ====================
    async def translate_subtitles_async_wrapper(self, content, model, target_lang, bilingual_style,
                                                temperature, max_tokens, gpu_layers, max_concurrent,
                                                progress=gr.Progress()):
        if not content.strip():
            return "请输入或上传字幕内容", "请提供字幕内容"
        if not model:
            return "请先选择翻译模型（点击刷新模型列表）", "无可用模型"

        self.translator.max_concurrent = int(max_concurrent)
        self.translator.semaphore = asyncio.Semaphore(self.translator.max_concurrent)

        try:
            if self._is_srt_format(content):
                subtitles = self.translator.parse_srt(content)
            else:
                subtitles = self.translator.parse_txt(content)

            if not subtitles:
                return "未找到有效字幕", "字幕解析失败"

            total = len(subtitles)
            progress(0, desc="开始翻译...")

            start_time = time.time()
            translated_subtitles = await self.translator.translate_subtitles_async(
                subtitles, target_lang, model, bilingual_style, temperature, int(gpu_layers), int(max_tokens)
            )
            self.current_subtitles = translated_subtitles

            elapsed = time.time() - start_time
            stats = f"翻译完成\n处理条数: {total}\n用时: {elapsed:.1f}秒\n平均: {elapsed/total:.1f}秒/条"

            preview_lines = []
            for sub in translated_subtitles[:10]:
                preview_lines.append(f"【{sub['index']}】 {sub.get('bilingual_text', sub.get('original_text', ''))}")
            if len(translated_subtitles) > 10:
                preview_lines.append(f"... 还有 {len(translated_subtitles) - 10} 条")

            return '\n'.join(preview_lines), stats

        except Exception as e:
            return f"翻译失败: {str(e)}", "翻译失败"

    async def preview_translation_async_wrapper(self, content, model, target_lang,
                                                bilingual_style, temperature, max_tokens, gpu_layers):
        if not content.strip():
            return "请输入或上传字幕内容"
        if not model:
            return "请先选择翻译模型（点击刷新模型列表）"

        try:
            if self._is_srt_format(content):
                subtitles = self.translator.parse_srt(content)
            else:
                subtitles = self.translator.parse_txt(content)

            if not subtitles:
                return "未找到有效字幕"

            preview_subtitles = subtitles[:5]
            translated_preview = await self.translator.translate_subtitles_async(
                preview_subtitles, target_lang, model, bilingual_style, temperature, int(gpu_layers), int(max_tokens)
            )

            preview_lines = ["**预览前5条翻译结果：**", ""]
            for sub in translated_preview:
                preview_lines.append(f"【原文】 {sub.get('original_text', '')}")
                preview_lines.append(f"【译文】 {sub.get('translated_text', '')}")
                preview_lines.append(f"【双语】 {sub.get('bilingual_text', '')}")
                preview_lines.append("")
            preview_lines.append("**提示：** 如果预览效果满意，可以点击'开始翻译'进行完整翻译")
            return '\n'.join(preview_lines)

        except Exception as e:
            return f"预览失败: {str(e)}"

    def download_results(self, output_format):
        if not self.current_subtitles:
            return gr.update(visible=False), "请先进行翻译"

        try:
            if output_format == "双语SRT":
                content = self.translator.generate_bilingual_srt(self.current_subtitles)
                filename = "bilingual_subtitles.srt"
            elif output_format == "双语TXT":
                content = self.translator.generate_bilingual_txt(self.current_subtitles, True)
                filename = "bilingual_subtitles.txt"
            elif output_format == "纯译文TXT":
                content = self.translator.generate_bilingual_txt(self.current_subtitles, False)
                filename = "translated_subtitles.txt"
            else:
                content = self.translator.generate_comparison_txt(self.current_subtitles)
                filename = "subtitle_comparison.txt"

            filepath = self.translator.save_results(content, filename)
            stats = f"下载完成\n文件已保存到: {filepath}\n包含 {len(self.current_subtitles)} 条字幕"
            return gr.update(value=filepath, visible=True), stats

        except Exception as e:
            return gr.update(visible=False), f"下载失败: {str(e)}"

    async def batch_translate_async_wrapper(self, files, model, target_lang, output_format,
                                            temperature, max_tokens, gpu_layers, bilingual_style):
        if not files:
            return "请选择要翻译的文件"
        if not model:
            return "请先选择翻译模型"

        results = []
        async with aiohttp.ClientSession() as session:
            for file in files:
                if isinstance(file, str):
                    filepath = file
                elif hasattr(file, 'name'):
                    filepath = file.name
                elif hasattr(file, 'path'):
                    filepath = file.path
                else:
                    results.append("  ⚠ 无法获取文件路径，跳过")
                    continue

                filename = os.path.basename(filepath)
                results.append(f"处理文件: {filename}")

                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()

                    if self._is_srt_format(content):
                        subtitles = self.translator.parse_srt(content)
                    else:
                        subtitles = self.translator.parse_txt(content)

                    if not subtitles:
                        results.append("  未找到有效字幕")
                        continue

                    translated_subtitles = await self.translator.translate_subtitles_async(
                        subtitles, target_lang, model, bilingual_style, temperature, int(gpu_layers), int(max_tokens)
                    )

                    if output_format == "双语SRT":
                        output_content = self.translator.generate_bilingual_srt(translated_subtitles)
                        output_filename = f"双语_{os.path.splitext(filename)[0]}.srt"
                    else:
                        output_content = self.translator.generate_bilingual_txt(translated_subtitles, True)
                        output_filename = f"双语_{os.path.splitext(filename)[0]}.txt"

                    save_path = self.translator.save_results(output_content, output_filename)
                    results.append(f"  ✅ 翻译完成: {len(translated_subtitles)} 条 -> {save_path}")

                except Exception as e:
                    results.append(f"  ❌ 处理失败: {str(e)}")

        results.append("")
        return '\n'.join(results)

    def _is_srt_format(self, content: str) -> bool:
        content = content.replace('\r\n', '\n').replace('\r', '\n')
        blocks = re.split(r'\n\s*\n', content.strip())
        if not blocks:
            return False
        first_block_lines = blocks[0].strip().split('\n')
        if len(first_block_lines) < 3:
            return False
        if not first_block_lines[0].strip().isdigit():
            return False
        if not re.search(r'\d{2}:\d{2}:\d{2}[,.]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[,.]\d{3}', first_block_lines[1]):
            return False
        if not first_block_lines[2].strip():
            return False
        return True

    def check_llama_status(self):
        try:
            resp = requests.get(self.translator.models_url, timeout=3)
            if resp.status_code == 200:
                data = resp.json()
                model_count = len(data.get("data", []))
                return f"llama.cpp 服务正常\n已加载模型: {model_count} 个"
            else:
                return f"llama.cpp 服务异常，状态码: {resp.status_code}"
        except Exception as e:
            return f"无法连接 llama.cpp 服务: {str(e)}"

    def get_model_list(self):
        if self.translator.available_models:
            return "\n".join([f"• {model}" for model in self.translator.available_models])
        else:
            return "未检测到模型，请确保 llama-server 已运行且已加载模型"

    def update_status(self):
        self.translator._load_models()
        status = self.check_llama_status()
        model_list = self.get_model_list()
        return status, model_list

    def open_output_dir(self):
        try:
            os.startfile(self.translator.output_dir)
        except Exception:
            pass


def main():
    print("=" * 70)
    print("双语字幕翻译工具 v2.5.3 (异步版 + 流式简化 + 缓存优化)")
    print("=" * 70)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=18005)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    ui = TranslatorUI()
    try:
        demo = ui.create_interface()
        print(f"\n启动地址: http://{args.host}:{args.port}")
        print(f"输出目录: {ui.translator.output_dir}")
        demo.queue().launch(
            server_name=args.host,
            server_port=args.port,
            share=args.share,
            inbrowser=True,
            theme=gr.themes.Default()
        )
    except Exception as e:
        print(f"启动失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()