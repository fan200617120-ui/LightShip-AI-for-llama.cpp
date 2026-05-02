#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
轻舟 AI・LightShip AI - chat_Ai_no_memory.py (llama.cpp 后端)
无记忆聊天，外挂 file_converter + llama_params_controller
支持流式输出、思考过程、多模态图片、系统提示词、导出对话、在线AI入口、参数预设
Copyright 2026 光影的故事2018
"""

import sys, os, socket, subprocess, tempfile, webbrowser, threading, time, re, base64, json, shutil
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageOps
import gradio as gr
import requests

# 当前脚本所在目录 (core/)
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

# ---------- 外挂模块 ----------
from file_converter import (
    convert_docs as fc_convert_docs,
    convert_images as fc_convert_images,
    batch_copy_rename as fc_batch_copy_rename,
    export_content_to_format,
    get_output_dir, get_chat_export_dir, get_image_output_dir, get_rename_output_dir,
    SUPPORTED_FORMATS, FORMAT_ALIASES, IMAGE_FORMATS, IMAGE_EXT_MAP,
)
from llama_params_controller import (
    create_param_controls,
    bind_param_events,
    get_preset_choices,
    on_preset_select,
    save_current_preset,
    delete_preset,
    export_presets,
    import_presets,
)

# ==================== 全局配置 ====================
LLAMA_API_URL = "http://127.0.0.1:8080/v1/chat/completions"
MODELS_URL   = "http://127.0.0.1:8080/v1/models"

_stop_flags = {}
_stop_lock = threading.Lock()
_generating_locks = {}
_generating_lock_dict_lock = threading.Lock()

# 路径
PANDOC_PATH = SCRIPT_DIR.parent / "pandoc" / "pandoc.exe"
OUTPUT_DIR = SCRIPT_DIR.parent / "output"
CHAT_EXPORT_DIR = OUTPUT_DIR / "chat_exports"
IMAGE_OUTPUT_DIR = OUTPUT_DIR / "images"
RENAME_OUTPUT_DIR = OUTPUT_DIR / "renamed"

# ==================== llama.cpp 基础 ====================
def is_llama_available():
    try:
        resp = requests.get(MODELS_URL, timeout=3)
        if resp.status_code == 200: return True, "服务正常"
        else: return False, f"服务异常，状态码：{resp.status_code}"
    except Exception as e: return False, f"连接失败：{str(e)}"

def get_llama_models():
    try:
        resp = requests.get(MODELS_URL, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            return [item["id"] for item in data.get("data", [])]
    except Exception as e: print(f"获取模型列表失败: {e}")
    return []

def is_multimodal(model_name: str) -> bool:
    if not model_name: return False
    multimodal_keywords = ["Qwen3","Qwen3.5","qwen3.5","llava","bakllava","gemini","cogvlm","minicpm","deepseek-vl","gemma4","gemma 4"]
    return any(kw in model_name.lower() for kw in multimodal_keywords)

def get_model_display_list(models):
    display_list = []
    for m in models:
        if is_multimodal(m): display_list.append((f"{m} (多模态)", m))
        elif "deepseek" in m.lower() or "r1" in m.lower(): display_list.append((f"{m} (深度推理)", m))
        else: display_list.append((m, m))
    return display_list

# ==================== 图片编码 ====================
def encode_image_to_base64(image_path):
    try:
        with Image.open(image_path) as img:
            img_format = img.format.lower() if img.format else None
            if img_format == "jpg": img_format = "jpeg"
        if img_format is None:
            ext = os.path.splitext(image_path)[1].lower()
            img_format = 'jpeg' if ext in ['.jpg','.jpeg'] else ext[1:]
        mime_type = f"image/{img_format}"
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime_type};base64,{b64}"
    except Exception as e:
        print(f"图片编码失败: {e}")
        return None

# ==================== 思考过程格式化 (升级版) ====================
def format_thoughts_streaming(thoughts: str) -> str:
    """流式输出过程中，给思考文字加上标题和简单的分点样式"""
    if not thoughts:
        return ""
    thought_content = re.sub(r'<think>|</think>', '', thoughts).strip()
    if not thought_content:
        return ""
    lines = thought_content.split('\n')
    formatted = "<strong>💭 思考过程：</strong><br><br>"
    for line in lines:
        line = line.strip()
        if line:
            # 自动检测引导句，赋予强调样式
            if re.match(r'^(首先|第一|1\.|其次|第二|然后|最后|第三|最终|总结|所以|因此)', line):
                formatted += f"<em>🔹 {line}</em><br>"
            else:
                formatted += f"<em>• {line}</em><br>"
    return formatted

def format_thoughts_collapsible(thoughts: str) -> str:
    """生成结束后，把思考过程放入折叠块，保留原始换行结构"""
    if not thoughts:
        return ""
    thought_content = re.sub(r'<think>|</think>', '', thoughts).strip()
    if not thought_content:
        return ""
    lines = thought_content.split('\n')
    italic_lines = []
    for line in lines:
        if line.strip():
            # 引导句加粗，其他行正常斜体
            if re.match(r'^(首先|第一|1\.|其次|第二|然后|最后|第三|最终|总结|所以|因此)', line):
                italic_lines.append(f"🔹 {line}")
            else:
                italic_lines.append(f"• {line}")
        else:
            # 空行保留一个 <br> 保持段落间距
            italic_lines.append("<br>")
    content_html = "<br>".join(italic_lines)
    return f'''<details>
        <summary><strong>💭 思考过程</strong>（点击展开/折叠）</summary>
        <div style="font-style: italic; padding-left: 1em; margin-top: 0.5em;">
            {content_html}
        </div>
    </details>'''

# ==================== 流式解析器 ====================
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
        if not text: return text
        return text.encode('utf-8', errors='replace').decode('utf-8')

    def parse_chunk(self, chunk_data: dict) -> dict:
        result = {"thought": "", "answer": "", "status": "answering"}
        choices = chunk_data.get("choices", [])
        if not choices: return result
        delta = choices[0].get("delta", {})

        reasoning = delta.get("reasoning_content", "")
        if reasoning:
            clean_reasoning = self._clean_text(reasoning)
            self.thought += clean_reasoning
            self.has_think = True
            result["thought"] = clean_reasoning
            result["status"] = "thinking"
            return result

        content = delta.get("content", "")
        if not content: return result
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
                    if self.buffer.endswith('<') or self.buffer.endswith('</t') or self.buffer.endswith('</th'):
                        keep_len = min(6, len(self.buffer))
                        safe_part = self.buffer[:-keep_len] if len(self.buffer) > keep_len else ""
                        self.buffer = self.buffer[-keep_len:] if len(self.buffer) >= keep_len else self.buffer
                        if safe_part:
                            safe_part = self._clean_text(safe_part)
                            self.answer += safe_part
                            result["answer"] += safe_part
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
                    if len(self.buffer) > 200:
                        clean_buf = self._clean_text(self.buffer)
                        self.thought += clean_buf
                        result["thought"] += clean_buf
                        self.buffer = ""
                    break
        return result

    def finalize(self, usage=None):
        if self.buffer:
            clean_buffer = self._clean_text(self.buffer)
            if self.in_think_tag or self.has_think:
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

# ==================== 无记忆流式响应 ====================
def stream_response_llama(message, image_path, model_name, temperature, max_tokens, gpu_layers,
                          system_prompt, vision_mode, thinking_mode, history, request: gr.Request):
    global _stop_flags, _generating_locks

    session_id = request.session_hash
    with _generating_lock_dict_lock:
        if session_id not in _generating_locks:
            _generating_locks[session_id] = threading.Lock()
        lock = _generating_locks[session_id]
    if not lock.acquire(blocking=False):
        updated_history = history + [{"role": "assistant", "content": "⏳ 上一个生成任务仍在进行，请稍后或点击「停止对话」再试。"}]
        yield updated_history, "生成任务进行中"
        return

    try:
        available, status_msg = is_llama_available()
        if not available:
            yield history + [{"role": "assistant", "content": f"❌ {status_msg}，请先启动 llama-server。"}], status_msg
            return

        models = get_llama_models()
        if model_name not in models:
            yield history + [{"role": "assistant", "content": f"❌ 模型 {model_name} 不在列表中。"}], "模型不可用"
            return

        with _stop_lock:
            _stop_flags[session_id] = False

        # 多模态处理
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
                image_path = None  # 忽略图片

        messages = [{"role": "system", "content": system_prompt}]
        if image_data_url:
            user_content = [{"type": "text", "text": message}, {"type": "image_url", "image_url": {"url": image_data_url}}]
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

        # 思考模式开关
        payload["chat_template_kwargs"] = {"enable_thinking": thinking_mode}
        payload["extra_body"] = {"override_kv": {"llama.enable_thinking": thinking_mode}}

        timestamp = time.strftime('%H:%M:%S')
        user_content = f"[{timestamp}] 用户：{message}" + (" [附图片]" if image_path else "")
        updated_history = history + [{"role": "user", "content": user_content}]
        updated_history.append({"role": "assistant", "content": f"[{timestamp}] {model_name}："})
        gpu_display = "服务器默认" if gpu_layers == -1 else str(gpu_layers)
        yield updated_history, f"模型 [{model_name}] 正在生成... (GPU层数: {gpu_display}, 思考: {'开' if thinking_mode else '关'})"

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
                        if data_str == "[DONE]": continue
                        try:
                            chunk = json.loads(data_str)
                            parsed = parser.parse_chunk(chunk)

                            if parsed["thought"]: full_thought += parsed["thought"]
                            if parsed["answer"]: full_answer += parsed["answer"]

                            display = f"[{timestamp}] {model_name}："
                            thought_formatted = format_thoughts_streaming(full_thought)
                            if thought_formatted:
                                display += f"\n\n{thought_formatted}\n\n"
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

            thought_html = format_thoughts_collapsible(final_thought) if final_thought else ""
            final_content = f"[{timestamp}] {model_name}："
            if thought_html:
                final_content += f"\n{thought_html}\n"
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
    finally:
        lock.release()

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
1. 选择支持多模态的模型（如 qwen3.5、gemma4）以使用图片识别
2. 右侧可自定义系统提示词，或使用角色预设快速切换专业提示词
3. 温度控制创意度，最大长度控制回复长度
4. 支持思考过程显示（深度推理模型有效）
5. 可随时停止生成
6. 多模态模式可选「自动/仅CPU/禁用」以控制显存
7. 开启/关闭思考模式以控制深度推理

开始对话："""
    }]
    return initial_history, "对话历史已清空"

def check_llama_status():
    available, msg = is_llama_available()
    if available:
        models = get_llama_models()
        if models: return f"✅ llama.cpp 服务正常，可用模型：{', '.join(models[:5])}{'...' if len(models)>5 else ''}"
        else: return "✅ llama.cpp 服务正常，但未检测到任何模型"
    else: return f"❌ {msg}"

def refresh_models():
    models = get_llama_models()
    if not models: return gr.Dropdown(choices=[("请先启动 llama-server", "none")], value="none")
    choices = get_model_display_list(models)
    return gr.Dropdown(choices=choices, value=choices[0][1] if choices else None)

# ==================== 角色预设 (外挂 chat_prompts.json) ====================
def load_prompts():
    json_path = SCRIPT_DIR / "chat_prompts.json"
    if not json_path.exists(): return {}
    try:
        with open(json_path, 'r', encoding='utf-8') as f: return json.load(f)
    except Exception as e: print(f"加载角色配置失败: {e}"); return {}

def get_prompt_options():
    prompts_data = load_prompts()
    role_lib = prompts_data.get("角色库", {})
    options = []
    for category, roles in role_lib.items():
        for role_id, info in roles.items():
            options.append((f"{info['角色名称']} ({category})", role_id))
    return options

def apply_preset(role_id):
    prompts_data = load_prompts()
    role_lib = prompts_data.get("角色库", {})
    for category, roles in role_lib.items():
        if role_id in roles:
            info = roles[role_id]
            return info["系统提示词"], info.get("输入占位符", "请输入内容...")
    return "", "请输入内容..."

def get_role_info(category, role_id):
    prompts_data = load_prompts()
    role_lib = prompts_data.get("角色库", {})
    roles = role_lib.get(category, {})
    return roles.get(role_id, {})

def build_category_roles_map():
    mapping = {}
    prompts_data = load_prompts()
    cat_list = prompts_data.get("分类目录", {})
    role_lib = prompts_data.get("角色库", {})
    for cat_name in cat_list.keys():
        roles = role_lib.get(cat_name, {})
        pairs = [(role_id, info["角色名称"]) for role_id, info in roles.items()]
        mapping[cat_name] = pairs
    return mapping

CATEGORY_ROLES_MAP = build_category_roles_map()
CATEGORY_NAMES = list(CATEGORY_ROLES_MAP.keys())
if CATEGORY_NAMES:
    initial_category = CATEGORY_NAMES[0]
    initial_roles = CATEGORY_ROLES_MAP[initial_category]
    first_role_id = initial_roles[0][0] if initial_roles else ""
    first_role_info = get_role_info(initial_category, first_role_id)
else:
    initial_category = ""; initial_roles = []; first_role_id = ""; first_role_info = {}

def open_url(url):
    webbrowser.open(url)
    return f"已打开 {url}"

# ==================== 导出辅助 (复用 file_converter 部分) ====================
def open_chat_export_dir():
    CHAT_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    if os.name == "nt": os.startfile(str(CHAT_EXPORT_DIR))
    else: webbrowser.open(str(CHAT_EXPORT_DIR))
    return f"已打开导出目录：{CHAT_EXPORT_DIR}"

def strip_html_tags(text):
    if not text: return ""
    text = re.sub(r'<details[^>]*>', '', text)
    text = re.sub(r'</details>', '', text)
    text = re.sub(r'<summary[^>]*>', '', text)
    text = re.sub(r'</summary>', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    return text.strip()

def get_last_assistant_markdown(history):
    if not history: return ""
    for msg in reversed(history):
        if msg.get("role") == "assistant":
            return strip_html_tags(msg.get("content", ""))
    return ""

def export_full_chat(history, target_format):
    if not history: return None, "对话为空"
    lines = []
    for msg in history:
        role = msg.get("role", "unknown")
        content = strip_html_tags(msg.get("content", ""))
        if not content: continue
        if role == "user": lines.append(f"## 用户\n\n{content}\n\n")
        else: lines.append(f"## 助手\n\n{content}\n\n")
    if not lines: return None, "无有效对话内容"
    full_md = "".join(lines)
    return export_content_to_format(full_md, target_format, SCRIPT_DIR.parent)

# ==================== Gradio 界面 ====================
logo_path = SCRIPT_DIR / "ai_logo.png"
ico_path = SCRIPT_DIR / "ai_logo.ico"
model_choices = []

css = """
.thoughts-details { border:1px solid var(--border-color-primary); border-radius:6px; padding:10px; margin:10px 0; background:var(--color-background-tertiary); }
.thoughts-details summary { cursor:pointer; padding:8px; font-weight:bold; border-radius:4px; background:var(--color-background-primary); }
"""

with gr.Blocks(title="轻舟 AI・无记忆聊天 (llama.cpp, 外挂模块)", css=css) as demo:
    with gr.Row():
        if logo_path.exists():
            gr.Image(str(logo_path), height=50, show_label=False, container=False, scale=0)
        with gr.Column(scale=1):
            gr.Markdown("# 轻舟 AI・LightShip AI (llama.cpp)  \n###### 轻舟渡万境，一智载千寻。")

    with gr.Tabs():
        # ==================== Tab1: 聊天 ====================
        with gr.Tab("聊天"):
            with gr.Row():
                with gr.Column(scale=4):
                    history_box = gr.Chatbot(
                        value=[{
                            "role": "assistant",
                            "content": """AI本地小助手 (llama.cpp 后端)

使用指南：
1. 选择支持多模态的模型以使用图片识别
2. 右侧可自定义系统提示词，或使用角色预设快速切换
3. 温度控制创意度，最大长度控制回复长度
4. 支持思考过程显示
5. 可随时停止生成
6. 多模态模式可选「自动/仅CPU/禁用」
7. 开启/关闭思考模式

开始对话："""
                        }],
                        height=880, sanitize_html=False
                    )

                    thinking_mode_cb = gr.Checkbox(label="开启思考过程（仅深度推理模型生效）", value=True, info="关闭后模型直接输出答案")

                    with gr.Row():
                        with gr.Column(scale=3):
                            input_box = gr.Textbox(label="输入文字", placeholder="请输入内容...", lines=14)
                        with gr.Column(scale=1, min_width=150):
                            send_btn = gr.Button("发送消息", variant="primary")
                            stop_btn = gr.Button("停止对话", variant="stop")
                            export_btn = gr.Button("导出对话", variant="primary")
                            open_export_dir_btn = gr.Button("打开目录", variant="secondary")
                            export_format = gr.Dropdown(
                                choices=["Microsoft Word (docx)", "PDF", "HTML", "Markdown", "Plain Text"],
                                label="导出格式", value="Microsoft Word (docx)"
                            )

                with gr.Column(scale=1, min_width=280):
                    gr.Markdown("### 角色预设")
                    preset_options = get_prompt_options()
                    role_dropdown = gr.Dropdown(choices=preset_options, label="选择一个专业角色", value=None, interactive=True)
                    gr.Markdown("### 系统提示词")
                    system_prompt_box = gr.Textbox(label="系统提示词", placeholder="输入系统指令...", lines=3,
                                                   value="你是一个乐于助人的助手，请用中文回答用户的问题。")
                    gr.Markdown("---")
                    gr.Markdown("### 模型设置")
                    model_select = gr.Dropdown(choices=model_choices, value=None, label="选择模型")
                    refresh_btn = gr.Button("🔄 刷新模型列表", size="sm")

                    # 参数预设控件（来自 llama_params_controller）
                    param_controls = create_param_controls()
                    # 将参数预设事件与控件绑定
                    bind_param_events(param_controls, additional_outputs=[
                        param_controls["temperature"],
                        param_controls["max_tokens"],
                        param_controls["gpu_layers"],
                        param_controls["vision_mode"],
                        param_controls["thinking_mode"]
                    ])

                    gr.Markdown("---")
                    image_input = gr.Image(label="仅多模态模型支持", type="filepath", height=180)
                    gr.Markdown("---")
                    check_btn = gr.Button("检查服务", variant="secondary")
                    clear_btn = gr.Button("清空对话", variant="secondary")

            status_box = gr.Textbox(label="状态", value="就绪", interactive=False, lines=2)

            # 事件绑定
            refresh_btn.click(fn=refresh_models, outputs=[model_select])
            demo.load(fn=lambda: (check_llama_status(), refresh_models()), outputs=[status_box, model_select])

            send_btn.click(
                fn=stream_response_llama,
                inputs=[input_box, image_input, model_select,
                        param_controls["temperature"], param_controls["max_tokens"], param_controls["gpu_layers"],
                        system_prompt_box, param_controls["vision_mode"], thinking_mode_cb, history_box],
                outputs=[history_box, status_box]
            ).then(lambda: ("", None), None, [input_box, image_input])

            input_box.submit(
                fn=stream_response_llama,
                inputs=[input_box, image_input, model_select,
                        param_controls["temperature"], param_controls["max_tokens"], param_controls["gpu_layers"],
                        system_prompt_box, param_controls["vision_mode"], thinking_mode_cb, history_box],
                outputs=[history_box, status_box]
            ).then(lambda: ("", None), None, [input_box, image_input])

            stop_btn.click(fn=stop_generation, outputs=[status_box])
            clear_btn.click(fn=clear_all, outputs=[history_box, status_box])
            check_btn.click(fn=check_llama_status, outputs=[status_box])

            def on_preset_change(role_id):
                if not role_id:
                    return "你是一个乐于助人的助手，请用中文回答用户的问题。", gr.Textbox(placeholder="请输入内容...")
                prompt, placeholder = apply_preset(role_id)
                return prompt, gr.Textbox(placeholder=placeholder)

            role_dropdown.change(fn=on_preset_change, inputs=[role_dropdown], outputs=[system_prompt_box, input_box])

            def handle_export(history, fmt):
                path, msg = export_full_chat(history, fmt)
                if path: return f"✅ {msg}\n文件位置：{path}"
                return f"❌ {msg}"

            export_btn.click(fn=handle_export, inputs=[history_box, export_format], outputs=[status_box])
            open_export_dir_btn.click(fn=open_chat_export_dir, outputs=[status_box])

        # ==================== Tab2: 转换工具 ====================
        with gr.Tab("转换工具"):
            with gr.Tabs():
                with gr.Tab("聊天记录排版"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            extract_btn = gr.Button("提取最近助手回复", variant="secondary")
                            markdown_input = gr.Textbox(label="Markdown 内容 (可编辑)", lines=15,
                                                        placeholder="点击上方按钮提取聊天记录，或直接粘贴 Markdown 文本")
                            with gr.Accordion("格式预览", open=False):
                                preview_md = gr.Markdown(value="*预览将在这里显示...*")
                            target_export_format = gr.Dropdown(
                                choices=["Microsoft Word (docx)", "PDF", "HTML", "Plain Text", "Markdown"],
                                label="导出格式", value="Microsoft Word (docx)"
                            )
                            with gr.Row():
                                export_md_btn = gr.Button("开始导出", variant="primary")
                                open_md_dir_btn = gr.Button("打开导出目录", variant="secondary")
                        with gr.Column(scale=1):
                            export_status = gr.Textbox(label="导出状态", interactive=False, lines=3)
                            export_file_download = gr.File(label="下载导出的文件", visible=True)

                    markdown_input.change(fn=lambda txt: txt if txt.strip() else "*内容为空*", inputs=markdown_input, outputs=preview_md)
                    extract_btn.click(fn=get_last_assistant_markdown, inputs=[history_box], outputs=[markdown_input])

                    def handle_export_md(md_text, fmt):
                        if not md_text.strip(): return None, "请先提取或输入 Markdown 内容。"
                        path, msg = export_content_to_format(md_text, fmt, SCRIPT_DIR.parent)
                        if path: return (gr.update(value=path, visible=True), msg)
                        return (gr.update(visible=False), msg)

                    export_md_btn.click(fn=handle_export_md, inputs=[markdown_input, target_export_format],
                                        outputs=[export_file_download, export_status])
                    open_md_dir_btn.click(fn=open_chat_export_dir, outputs=[export_status])

                with gr.Tab("文档格式转换"):
                    status_doc = gr.Textbox(label="Pandoc 状态", value="Pandoc 可用" if PANDOC_PATH.exists() else f"未找到 Pandoc：{PANDOC_PATH}", interactive=False)
                    with gr.Row():
                        with gr.Column(scale=2):
                            file_input = gr.File(label="上传文件（可多选）", file_count="multiple")
                            src_format = gr.Dropdown(choices=list(SUPPORTED_FORMATS.keys()), label="源格式", value="Markdown")
                            tgt_format = gr.Dropdown(choices=list(SUPPORTED_FORMATS.keys()), label="目标格式", value="Microsoft Word (docx)")
                            with gr.Accordion("高级选项", open=False):
                                enable_toc = gr.Checkbox(label="为 Word 添加目录", value=False)
                                reference_doc = gr.File(label="参考样式模板 (仅 Word，可选 .docx)", file_types=[".docx"])
                            with gr.Row():
                                convert_btn = gr.Button("开始批量转换", variant="primary", scale=3)
                                open_folder_btn = gr.Button("打开输出目录", variant="secondary", scale=1)
                        with gr.Column(scale=1):
                            output_msg = gr.Textbox(label="转换结果", interactive=False, lines=8)
                    convert_btn.click(
                        fn=lambda files, src, tgt, toc, ref: fc_convert_docs(files, src, tgt, toc, ref, get_output_dir(SCRIPT_DIR.parent), PANDOC_PATH),
                        inputs=[file_input, src_format, tgt_format, enable_toc, reference_doc],
                        outputs=[output_msg]
                    )
                    open_folder_btn.click(fn=lambda: webbrowser.open(str(get_output_dir(SCRIPT_DIR.parent))), outputs=[output_msg])

                with gr.Tab("图片格式转换"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            img_input = gr.File(label="上传图片（可多选）", file_count="multiple")
                            img_format = gr.Dropdown(choices=IMAGE_FORMATS, label="输出格式", value="PNG")
                            img_quality = gr.Slider(1, 100, 85, step=1, label="图片质量 (仅对 JPEG/WebP 有效)")
                            with gr.Row():
                                img_convert_btn = gr.Button("开始转换图片", variant="primary", scale=3)
                                img_open_folder_btn = gr.Button("打开图片输出目录", variant="secondary", scale=1)
                        with gr.Column(scale=1):
                            img_output_msg = gr.Textbox(label="转换结果", interactive=False, lines=8)
                    img_convert_btn.click(
                        fn=lambda files, fmt, q: fc_convert_images(files, fmt, q, get_image_output_dir(SCRIPT_DIR.parent)),
                        inputs=[img_input, img_format, img_quality],
                        outputs=[img_output_msg]
                    )
                    img_open_folder_btn.click(fn=lambda: webbrowser.open(str(get_image_output_dir(SCRIPT_DIR.parent))), outputs=[img_output_msg])

                with gr.Tab("批量复制并重命名"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            rename_files = gr.File(label="选择文件（可多选）", file_count="multiple")
                            new_extension = gr.Textbox(label="新扩展名（例如 .txt 或 txt）", placeholder=".md")
                            with gr.Row():
                                rename_btn = gr.Button("开始复制并重命名", variant="primary", scale=3)
                                rename_open_folder_btn = gr.Button("打开输出目录", variant="secondary", scale=1)
                            gr.Markdown("注意：此操作会将文件复制到 `output/renamed/` 目录并修改扩展名，原始文件保持不变。")
                        with gr.Column(scale=1):
                            rename_result = gr.Textbox(label="操作结果", interactive=False, lines=8)
                    rename_btn.click(
                        fn=lambda files, ext: fc_batch_copy_rename(files, ext, get_rename_output_dir(SCRIPT_DIR.parent)),
                        inputs=[rename_files, new_extension],
                        outputs=[rename_result]
                    )
                    rename_open_folder_btn.click(fn=lambda: webbrowser.open(str(get_rename_output_dir(SCRIPT_DIR.parent))), outputs=[rename_result])

        # ==================== Tab3: 在线AI入口 ====================
        with gr.Tab("在线AI入口"):
            AI_URLS = {
                "DeepL": "https://www.deepl.com/zh",
                "有道翻译": "https://fanyi.youdao.com/",
                "豆包": "https://www.doubao.com/",
                "通义千问": "https://www.qianwen.com/",
                "DeepSeek": "https://www.deepseek.com/",
                "ChatGLM": "https://chatglm.cn",
                "Kimi": "https://kimi.moonshot.cn/",
                "腾讯元宝": "https://yuanbao.tencent.com/",
            }
            gr.Markdown("### 快速入口")
            with gr.Row(equal_height=True):
                btn_deepl = gr.Button("DeepL", variant="secondary")
                btn_youdao = gr.Button("有道翻译", variant="secondary")
                btn_deepseek = gr.Button("DeepSeek", variant="secondary")
                btn_doubao = gr.Button("豆包", variant="secondary")
            with gr.Row(equal_height=True):
                btn_qianwen = gr.Button("通义千问", variant="secondary")
                btn_kimi = gr.Button("Kimi", variant="secondary")
                btn_chatglm = gr.Button("ChatGLM", variant="secondary")
                btn_yuanbao = gr.Button("腾讯元宝", variant="secondary")
            status_trans = gr.Textbox(label="", value="点击按钮打开对应网站", interactive=False)

            gr.Markdown("---")
            gr.Markdown("### 智能提示词模板（动态加载）")
            with gr.Row():
                category_dd = gr.Dropdown(label="选择分类", choices=CATEGORY_NAMES, value=initial_category, interactive=True)
                role_dd = gr.Dropdown(label="选择角色",
                                      choices=[(name, role_id) for (role_id, name) in initial_roles] if initial_roles else [],
                                      value=first_role_id, interactive=True)
            current_prompt = gr.Textbox(label="系统提示词", value=first_role_info.get("系统提示词", ""), lines=8, interactive=True)
            placeholder_box = gr.Textbox(label="输入占位符（可复制到需要输入的地方）", value=first_role_info.get("输入占位符", ""), interactive=False)

            gr.Markdown("使用方法：选择分类和角色 → 复制提示词/占位符 → 点击上方按钮打开 AI 网站 → 粘贴使用。")

            def on_category_change(cat):
                roles = CATEGORY_ROLES_MAP.get(cat, [])
                if roles:
                    choices = [(name, role_id) for (role_id, name) in roles]
                    default_role = roles[0][0]
                    return (gr.Dropdown(choices=choices, value=default_role), gr.Textbox(value=""), gr.Textbox(value=""))
                else:
                    return (gr.Dropdown(choices=[], value=None), gr.Textbox(value=""), gr.Textbox(value=""))

            def on_role_change(cat, role):
                info = get_role_info(cat, role)
                return (gr.Textbox(value=info.get("系统提示词", "")), gr.Textbox(value=info.get("输入占位符", "")))

            category_dd.change(fn=on_category_change, inputs=[category_dd], outputs=[role_dd, current_prompt, placeholder_box])
            role_dd.change(fn=on_role_change, inputs=[category_dd, role_dd], outputs=[current_prompt, placeholder_box])

            btn_deepl.click(fn=lambda: open_url(AI_URLS["DeepL"]), outputs=status_trans)
            btn_youdao.click(fn=lambda: open_url(AI_URLS["有道翻译"]), outputs=status_trans)
            btn_deepseek.click(fn=lambda: open_url(AI_URLS["DeepSeek"]), outputs=status_trans)
            btn_doubao.click(fn=lambda: open_url(AI_URLS["豆包"]), outputs=status_trans)
            btn_qianwen.click(fn=lambda: open_url(AI_URLS["通义千问"]), outputs=status_trans)
            btn_kimi.click(fn=lambda: open_url(AI_URLS["Kimi"]), outputs=status_trans)
            btn_chatglm.click(fn=lambda: open_url(AI_URLS["ChatGLM"]), outputs=status_trans)
            btn_yuanbao.click(fn=lambda: open_url(AI_URLS["腾讯元宝"]), outputs=status_trans)

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
    print("启动 轻舟 AI・无记忆聊天 (llama.cpp 后端, 外挂模块)")
    print("请确保已运行 llama-server")
    print("=" * 60)
    port = 7961
    demo.launch(
        server_name="127.0.0.1",
        server_port=port,
        share=False,
        inbrowser=True,
        css=css,
        favicon_path=str(ico_path) if ico_path.exists() else None
    )