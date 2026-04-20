#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
轻舟 AI・LightShip AI - 无记忆聊天 (llama.cpp 后端)
支持流式输出、思考过程、多模态图片、系统提示词、导出对话、一键唤起转换工具箱
Copyright 2026 光影的故事2018
"""

import sys
import os
import socket
import subprocess
import tempfile
import webbrowser
import threading
import time
import re
import base64
import imghdr
import json
from datetime import datetime
from pathlib import Path

import gradio as gr
import requests

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

# ==================== 全局停止标志 ====================
_stop_flags = {}
_stop_lock = threading.Lock()

# ==================== llama.cpp 配置 ====================
LLAMA_API_URL = "http://127.0.0.1:8080/v1/chat/completions"
MODELS_URL = "http://127.0.0.1:8080/v1/models"

def is_llama_available():
    try:
        resp = requests.get(MODELS_URL, timeout=3)
        if resp.status_code == 200:
            return True, "服务正常"
        else:
            return False, f"服务异常，状态码：{resp.status_code}"
    except Exception as e:
        return False, f"服务未启动或连接失败：{str(e)}"

def get_llama_models():
    try:
        resp = requests.get(MODELS_URL, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            models = [item["id"] for item in data.get("data", [])]
            return models
    except Exception as e:
        print(f"获取模型列表失败: {e}")
    return []

def is_multimodal(model_name: str) -> bool:
    multimodal_keywords = ["qwen", "llava", "bakllava", "gemini", "cogvlm", "minicpm", "gemma"]
    return any(kw in model_name.lower() for kw in multimodal_keywords)

def get_model_display_list(models):
    display_list = []
    for m in models:
        if is_multimodal(m):
            display_list.append((f"{m} (多模态)", m))
        elif "deepseek" in m.lower() or "r1" in m.lower():
            display_list.append((f"{m} (深度推理)", m))
        else:
            display_list.append((m, m))
    return display_list

# ==================== 图片编码 ====================
def encode_image_to_base64(image_path):
    try:
        img_format = imghdr.what(image_path)
        if img_format is None:
            ext = os.path.splitext(image_path)[1].lower()
            if ext in ['.jpg', '.jpeg']:
                img_format = 'jpeg'
            elif ext == '.png':
                img_format = 'png'
            elif ext == '.gif':
                img_format = 'gif'
            else:
                img_format = 'jpeg'
        mime_type = f"image/{img_format}"
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime_type};base64,{b64}"
    except Exception as e:
        print(f"图片编码失败: {e}")
        return None

# ==================== 流式解析器（含UTF-8清洗） ====================
class StreamResponseParser:
    def __init__(self):
        self.thought = ""
        self.answer = ""
        self.buffer = ""
        self.in_think_tag = False
        self.has_think = False
        self.start_time = time.time()
        self.total_tokens = 0
        self.char_count = 0

    def _clean_text(self, text: str) -> str:
        if not text:
            return text
        return text.encode('utf-8', errors='replace').decode('utf-8')

    def parse_chunk(self, chunk_data: dict) -> dict:
        result = {"thought": "", "answer": "", "status": "answering"}
        delta = chunk_data.get("choices", [{}])[0].get("delta", {})

        reasoning = delta.get("reasoning_content", "")
        if reasoning:
            clean_reasoning = self._clean_text(reasoning)
            self.thought += clean_reasoning
            self.has_think = True
            result["thought"] = clean_reasoning
            result["status"] = "thinking"
            return result

        content = delta.get("content", "")
        if not content:
            return result

        content = self._clean_text(content)
        self.char_count += len(content)
        self.buffer += content

        while True:
            if not self.in_think_tag:
                start_idx = self.buffer.find('<think>')
                if start_idx != -1:
                    before_think = self.buffer[:start_idx]
                    if before_think:
                        before_think = self._clean_text(before_think)
                        self.answer += before_think
                        result["answer"] += before_think
                    self.buffer = self.buffer[start_idx + 7:]
                    self.in_think_tag = True
                    self.has_think = True
                    result["status"] = "thinking"
                    continue
                else:
                    if self.buffer.rstrip().endswith('<') or '<' in self.buffer[-5:]:
                        last_lt = self.buffer.rfind('<')
                        if last_lt != -1:
                            safe_part = self.buffer[:last_lt]
                            self.buffer = self.buffer[last_lt:]
                            if safe_part:
                                safe_part = self._clean_text(safe_part)
                                self.answer += safe_part
                                result["answer"] += safe_part
                        else:
                            clean_buf = self._clean_text(self.buffer)
                            self.answer += clean_buf
                            result["answer"] += clean_buf
                            self.buffer = ""
                    else:
                        clean_buf = self._clean_text(self.buffer)
                        self.answer += clean_buf
                        result["answer"] += clean_buf
                        self.buffer = ""
                    break
            else:
                end_idx = self.buffer.find('</think>')
                if end_idx != -1:
                    think_part = self.buffer[:end_idx]
                    if think_part:
                        think_part = self._clean_text(think_part)
                        self.thought += think_part
                        result["thought"] += think_part
                    self.buffer = self.buffer[end_idx + 8:]
                    self.in_think_tag = False
                    result["status"] = "answering"
                    continue
                else:
                    clean_buf = self._clean_text(self.buffer)
                    self.thought += clean_buf
                    result["thought"] += clean_buf
                    self.buffer = ""
                    break
        return result

    def finalize(self, usage=None):
        if self.buffer:
            clean_buffer = self._clean_text(self.buffer)
            if self.in_think_tag:
                self.thought += clean_buffer
            else:
                self.answer += clean_buffer
            self.buffer = ""

        if not self.has_think and self.thought:
            self.answer = self.thought + self.answer
            self.thought = ""

        if usage:
            self.total_tokens = usage.get("completion_tokens", 0)
        else:
            self.total_tokens = self.char_count // 2

        return self.answer, self.thought

# ==================== 流式响应（无记忆） ====================
def stream_response_llama(message, image_path, model_name, temperature, max_tokens, gpu_layers,
                          system_prompt, vision_mode, history, request: gr.Request):
    global _stop_flags, _stop_lock

    available, status_msg = is_llama_available()
    if not available:
        error_content = f"❌ {status_msg}，请先启动 llama-server。"
        updated_history = history + [{"role": "assistant", "content": error_content}]
        yield updated_history, status_msg
        return

    models = get_llama_models()
    if model_name not in models:
        error_content = f"❌ 模型 {model_name} 不在 llama-server 可用列表中。请检查模型名称或刷新列表。"
        updated_history = history + [{"role": "assistant", "content": error_content}]
        yield updated_history, "模型不可用"
        return

    session_id = request.session_hash
    with _stop_lock:
        _stop_flags[session_id] = False

    enable_vision = True
    force_cpu_vision = False
    if vision_mode == "禁用多模态":
        enable_vision = False
        image_path = None
    elif vision_mode == "仅 CPU（节省显存）":
        force_cpu_vision = True

    image_data_url = None
    if enable_vision and image_path is not None:
        if is_multimodal(model_name):
            image_data_url = encode_image_to_base64(image_path)
            if not image_data_url:
                image_path = None
        else:
            print(f"警告：模型 {model_name} 不支持图片，已忽略。")
            image_path = None

    messages = [{"role": "system", "content": system_prompt}]
    if image_data_url and enable_vision:
        user_content = [
            {"type": "text", "text": message},
            {"type": "image_url", "image_url": {"url": image_data_url}}
        ]
        messages.append({"role": "user", "content": user_content})
    else:
        messages.append({"role": "user", "content": message})

    payload = {
        "model": model_name,
        "messages": messages,
        "stream": True,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if gpu_layers != -1:
        payload["n_gpu_layers"] = gpu_layers
    if force_cpu_vision:
        payload["mmproj_cpu"] = True

    timestamp = time.strftime('%H:%M:%S')
    user_content = f"[{timestamp}] 用户：{message}" + (" [附图片]" if image_path else "")
    updated_history = history + [{"role": "user", "content": user_content}]
    updated_history.append({"role": "assistant", "content": f"[{timestamp}] {model_name}："})
    gpu_display = "服务器默认" if gpu_layers == -1 else str(gpu_layers)
    yield updated_history, f"模型 [{model_name}] 正在生成... (GPU层数: {gpu_display})"

    parser = StreamResponseParser()
    full_answer = ""
    full_thought = ""

    try:
        response = requests.post(LLAMA_API_URL, json=payload, stream=True, timeout=180)
        response.raise_for_status()
        response.encoding = 'utf-8'

        for line in response.iter_lines(decode_unicode=True):
            with _stop_lock:
                if _stop_flags.get(session_id, False):
                    updated_history[-1]["content"] = f"[{timestamp}] {model_name}：{full_answer}\n\n[已手动停止]"
                    yield updated_history, "生成已停止"
                    return

            if line:
                line = line.strip()
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        continue
                    try:
                        chunk = json.loads(data_str)
                        parsed = parser.parse_chunk(chunk)

                        if parsed["thought"]:
                            full_thought += parsed["thought"]
                        if parsed["answer"]:
                            full_answer += parsed["answer"]

                        display = f"[{timestamp}] {model_name}："
                        if full_thought:
                            display += f"\n\n💭 {full_thought}\n\n"
                        if full_answer:
                            display += full_answer
                        else:
                            display += "█"

                        updated_history[-1]["content"] = display
                        yield updated_history, f"生成中... ({len(full_answer)} 字符)"

                        if "usage" in chunk:
                            parser.total_tokens = chunk["usage"].get("completion_tokens", 0)
                    except json.JSONDecodeError:
                        continue

        end_time = time.time()
        time_cost = end_time - parser.start_time
        final_answer, final_thought = parser.finalize()
        total_tokens = parser.total_tokens if parser.total_tokens > 0 else parser.char_count // 2
        speed = total_tokens / time_cost if time_cost > 0 else 0
        stat_str = f"{total_tokens} tokens, {time_cost:.1f}s, {speed:.2f}t/s"

        thought_html = ""
        if final_thought:
            safe_thought = final_thought.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            thought_html = f'''
            <details class="thoughts-details">
                <summary><strong>💭 思考过程</strong> (点击展开/折叠)</summary>
                <div class="thoughts-content"><em>{safe_thought.replace(chr(10), '<br>')}</em></div>
            </details>
            '''
        final_content = f"[{timestamp}] {model_name}："
        if thought_html:
            final_content += f"\n\n{thought_html}\n\n"
        final_content += f"{final_answer}\n\n[统计] {stat_str} (GPU层数: {gpu_display})"
        updated_history[-1]["content"] = final_content
        yield updated_history, f"生成完成，{stat_str}"

    except requests.exceptions.ConnectionError:
        updated_history[-1]["content"] = f"[{timestamp}] {model_name}：连接失败，请确保已运行 llama-server"
        yield updated_history, "连接失败"
    except Exception as e:
        updated_history[-1]["content"] = f"[{timestamp}] {model_name}：错误：{str(e)}"
        yield updated_history, f"错误：{str(e)}"
    finally:
        with _stop_lock:
            _stop_flags.pop(session_id, None)

def stop_generation(request: gr.Request):
    session_id = request.session_hash
    with _stop_lock:
        _stop_flags[session_id] = True
    return "正在停止生成..."

def clear_all():
    initial_history = [{
        "role": "assistant",
        "content": """AI本地小助手 (llama.cpp 后端)

使用指南：
1. 选择支持多模态的模型（如 qwen3.5 系列、gemma4 系列）以使用图片识别
2. 可在右侧自定义系统提示词
3. 温度控制创意度，最大长度控制回复长度
4. 对话支持思考过程显示
5. 可随时停止生成
6. 多模态模式可选「自动/仅CPU/禁用」以控制显存

开始对话："""
    }]
    return initial_history, "对话历史已清空"

def check_llama_status():
    available, msg = is_llama_available()
    if available:
        models = get_llama_models()
        if models:
            return f"✅ llama.cpp 服务正常，可用模型：{', '.join(models[:5])}{'...' if len(models)>5 else ''}"
        else:
            return "✅ llama.cpp 服务正常，但未检测到任何模型"
    else:
        return f"❌ {msg}"

def refresh_models():
    models = get_llama_models()
    if not models:
        return gr.Dropdown(choices=[("请先启动 llama-server", "none")], value="none")
    choices = get_model_display_list(models)
    return gr.Dropdown(choices=choices, value=choices[0][1] if choices else None)

# ==================== 导出功能（含 YAML 容错） ====================
PANDOC_PATH = SCRIPT_DIR.parent / "pandoc" / "pandoc.exe"
OUTPUT_DIR = SCRIPT_DIR.parent / "output"
CHAT_EXPORT_DIR = OUTPUT_DIR / "chat_exports"
FORMAT_ALIASES = {".docx": "docx", ".pdf": "pdf", ".html": "html", ".txt": "plain", ".md": "markdown"}

def strip_html_tags(text) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r'<details[^>]*>', '', text)
    text = re.sub(r'</details>', '', text)
    text = re.sub(r'<summary[^>]*>', '', text)
    text = re.sub(r'</summary>', '', text)
    text = re.sub(r'<div[^>]*>', '', text)
    text = re.sub(r'</div>', '', text)
    text = re.sub(r'<em>', '', text)
    text = re.sub(r'</em>', '', text)
    text = re.sub(r'<strong>', '', text)
    text = re.sub(r'</strong>', '', text)
    text = re.sub(r'<br\s*/?>', '\n', text)
    text = re.sub(r'<[^>]+>', '', text)
    return text.strip()

def export_full_chat(history, target_format):
    if not history:
        return None, "对话为空，无法导出。"
    lines = []
    for msg in history:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        clean = strip_html_tags(content)
        if not clean:
            continue
        if role == "user":
            lines.append(f"## 用户\n\n{clean}\n\n")
        else:
            lines.append(f"## 助手\n\n{clean}\n\n")
    if not lines:
        return None, "无有效对话内容。"
    full_md = "".join(lines)

    CHAT_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    ext_map = {
        "Microsoft Word (docx)": ".docx",
        "PDF": ".pdf",
        "HTML": ".html",
        "Plain Text": ".txt",
        "Markdown": ".md",
    }
    tgt_ext = ext_map.get(target_format, ".docx")
    writer = FORMAT_ALIASES.get(tgt_ext, "docx")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = CHAT_EXPORT_DIR / f"chat_export_{timestamp}{tgt_ext}"

    def run_pandoc(md_text, reader):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
            f.write(md_text)
            temp_md = f.name
        extra_args = []
        if target_format == "PDF":
            extra_args = ["--pdf-engine", "xelatex"]
        cmd = [str(PANDOC_PATH), temp_md, "-f", reader, "-t", writer, "-o", str(output_path)] + extra_args
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=True)
            return True, None
        except subprocess.CalledProcessError as e:
            return False, e.stderr
        finally:
            os.unlink(temp_md)

    # 尝试1：禁用YAML
    ok, err = run_pandoc(full_md, "markdown-yaml_metadata_block")
    if ok:
        return str(output_path), f"导出成功：{output_path.name}"

    # 尝试2：剥离Front Matter
    stripped = re.sub(r'^\s*---\s*\n.*?\n---\s*\n', '', full_md, flags=re.DOTALL)
    if stripped != full_md and stripped.strip():
        ok, err = run_pandoc(stripped, "markdown-yaml_metadata_block")
        if ok:
            return str(output_path), f"导出成功：{output_path.name} (已忽略YAML头部)"

    # 尝试3：纯文本兜底
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_md)
        return str(output_path), f"导出成功：{output_path.name} (纯文本模式)"
    except Exception as e:
        return None, f"导出失败：{str(e)}"

def open_chat_export_dir():
    CHAT_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        os.startfile(str(CHAT_EXPORT_DIR))
    else:
        webbrowser.open(str(CHAT_EXPORT_DIR))
    return f"已打开导出目录：{CHAT_EXPORT_DIR}"

# ==================== 一键启动转换工具箱 ====================
def is_port_open(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.connect(("127.0.0.1", port))
            return True
        except:
            return False

def is_format_converter_running(port):
    """检查转换工具箱是否真正在运行（通过 HTTP 请求验证）"""
    try:
        r = requests.get(f"http://127.0.0.1:{port}", timeout=1)
        return r.status_code == 200 and "轻舟 AI 工具箱" in r.text
    except:
        return False

def launch_format_converter():
    port = 7969
    url = f"http://127.0.0.1:{port}"
    if is_format_converter_running(port):
        webbrowser.open(url)
        return f"转换工具箱已在运行，浏览器已打开 {url}"
    script_path = Path(SCRIPT_DIR).parent / "pandoc" / "format_converter.py"
    if not script_path.exists():
        return f"❌ 未找到转换器脚本：{script_path}"
    try:
        subprocess.Popen(
            [sys.executable, str(script_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        )
        time.sleep(2)
        webbrowser.open(url)
        return f"✅ 转换工具箱已启动，浏览器已打开 {url}"
    except Exception as e:
        return f"❌ 启动失败：{str(e)}"
    
# ==================== Gradio 界面 ====================
logo_path = SCRIPT_DIR / "ai_logo.png"
ico_path = SCRIPT_DIR / "ai_logo.ico"

initial_models = get_llama_models()
if not initial_models:
    initial_models = ["请先启动 llama-server"]
model_choices = get_model_display_list(initial_models)

css = """
.thoughts-details { border: 1px solid var(--border-color-primary); border-radius: 6px; padding: 10px; margin: 10px 0; background-color: var(--color-background-tertiary); }
.thoughts-details summary { cursor: pointer; padding: 8px; font-weight: bold; border-radius: 4px; background-color: var(--color-background-primary); }
.thoughts-content { padding: 10px; background-color: var(--color-background-secondary); border-radius: 4px; font-style: italic; }
"""

with gr.Blocks(title="轻舟 AI・无记忆聊天 (llama.cpp)", css=css) as demo:
    gr.Markdown("# 轻舟 AI・无记忆聊天 (llama.cpp)")
    gr.Markdown("###### 轻舟渡万境，一智载千寻。")

    with gr.Row():
        # 左侧：聊天主区域
        with gr.Column(scale=4):
            history_box = gr.Chatbot(
                value=[{
                    "role": "assistant",
                    "content": """AI本地小助手 (llama.cpp 后端)

使用指南：
1. 选择支持多模态的模型（如 qwen3.5 系列、gemma4 系列）以使用图片识别
2. 可在右侧自定义系统提示词
3. 温度控制创意度，最大长度控制回复长度
4. 对话支持思考过程显示
5. 可随时停止生成
6. 多模态模式可选「自动/仅CPU/禁用」以控制显存

开始对话："""
                }],
                height=750,
                sanitize_html=False
            )
            with gr.Row():
                input_box = gr.Textbox(
                    label="输入文字",
                    placeholder="请输入问题... (按回车发送)",
                    lines=3,
                    scale=4
                )
                send_btn = gr.Button("发送", variant="primary", scale=1)
                stop_btn = gr.Button("停止", variant="stop", scale=1)

            # 导出控件
            with gr.Row():
                export_format = gr.Dropdown(
                    choices=["Microsoft Word (docx)", "PDF", "HTML", "Markdown", "Plain Text"],
                    label="导出格式",
                    value="Microsoft Word (docx)",
                    scale=2
                )
                export_btn = gr.Button("导出对话", variant="primary", scale=1)
                open_export_dir_btn = gr.Button("打开导出目录", variant="secondary", scale=1)
            export_file = gr.File(label="下载导出的文件", visible=True, height=50)

        # 右侧：设置边栏
        with gr.Column(scale=1, min_width=280):
            gr.Markdown("### 系统提示词")
            system_prompt_box = gr.Textbox(
                label="系统提示词",
                placeholder="输入系统指令...",
                lines=3,
                value="你是一个乐于助人的助手，请用中文回答用户的问题。"
            )
            gr.Markdown("---")
            gr.Markdown("### 模型设置")
            model_select = gr.Dropdown(
                choices=model_choices,
                value=model_choices[0][1] if model_choices else None,
                label="选择模型"
            )
            refresh_btn = gr.Button("🔄 刷新模型列表", size="sm")
            temp_slider = gr.Slider(0.1, 2.0, value=0.7, step=0.1, label="温度")
            token_slider = gr.Slider(512, 8192, value=2048, step=128, label="最大长度")
            gpu_layers_slider = gr.Slider(
                -1, 99, value=-1, step=1,
                label="GPU 层数 (-1=服务器默认, 0=仅CPU, 1~99=指定层数)",
                info="需 llama-server 支持请求级设置"
            )
            gr.Markdown("---")
            gr.Markdown("### 多模态设置")
            vision_mode = gr.Dropdown(
                choices=["自动（跟随 GPU 层数）", "仅 CPU（节省显存）", "禁用多模态"],
                value="自动（跟随 GPU 层数）",
                label="视觉模型模式"
            )
            gr.Markdown("---")
            gr.Markdown("### 图片上传")
            image_input = gr.Image(
                label="仅多模态模型支持",
                type="filepath",
                height=180
            )
            gr.Markdown("---")
            check_btn = gr.Button("检查服务", variant="secondary")
            clear_btn = gr.Button("清空对话", variant="secondary")
            gr.Markdown("---")
            launch_toolbox_btn = gr.Button("🧰 打开转换工具箱", variant="secondary")

    # 底部状态栏
    with gr.Row():
        status_box = gr.Textbox(label="状态", value="就绪", interactive=False)

    # 事件绑定
    refresh_btn.click(fn=refresh_models, outputs=[model_select])
    send_btn.click(
        fn=stream_response_llama,
        inputs=[input_box, image_input, model_select, temp_slider, token_slider,
                gpu_layers_slider, system_prompt_box, vision_mode, history_box],
        outputs=[history_box, status_box]
    ).then(lambda: ("", None), None, [input_box, image_input])
    input_box.submit(
        fn=stream_response_llama,
        inputs=[input_box, image_input, model_select, temp_slider, token_slider,
                gpu_layers_slider, system_prompt_box, vision_mode, history_box],
        outputs=[history_box, status_box]
    ).then(lambda: ("", None), None, [input_box, image_input])
    stop_btn.click(fn=stop_generation, outputs=[status_box])
    clear_btn.click(fn=clear_all, outputs=[history_box, status_box])
    check_btn.click(fn=check_llama_status, outputs=[status_box])
    launch_toolbox_btn.click(fn=launch_format_converter, outputs=[status_box])
    demo.load(fn=check_llama_status, outputs=[status_box])

    def handle_export(history, fmt):
        path, msg = export_full_chat(history, fmt)
        if path:
            return gr.update(value=path, visible=True), msg
        else:
            return gr.update(visible=False), msg

    export_btn.click(
        fn=handle_export,
        inputs=[history_box, export_format],
        outputs=[export_file, status_box]
    )
    open_export_dir_btn.click(fn=open_chat_export_dir, outputs=[status_box])

    # 页脚
    gr.Markdown("---")
    gr.HTML("""
    <div style="text-align: center; color: #666; font-size: 0.9em;">
        <p>本工具仅用于个人学习与文档处理，禁止商业用途。</p>
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 15px; border-radius: 8px; margin: 15px auto; max-width: 600px;">
            <p style="color: white; font-weight: bold; margin: 5px 0;">🎬 更新请关注B站up主：光影的故事2018</p>
            <p style="color: white; margin: 5px 0;">
                🔗 <strong>B站主页</strong>: <a href="https://space.bilibili.com/381518712" target="_blank" style="color: #ffdd40; text-decoration: none;">space.bilibili.com/381518712</a>
            </p>
        </div>
        <p>© 原创 WebUI 代码 © 2026 光影紐扣 版权所有</p>
    </div>
    """)

if __name__ == "__main__":
    print("=" * 60)
    print("启动 轻舟 AI・无记忆聊天 (llama.cpp 后端)")
    print("请确保已运行 llama-server")
    print("=" * 60)
    status = check_llama_status()
    print(status)
    demo.launch(server_name="127.0.0.1", server_port=7861, share=False, inbrowser=True)