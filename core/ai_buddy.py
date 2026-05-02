#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ai_buddy_llama.py - 基于 llama.cpp 后端的 AI 助手（聊天 + 转换工具 + 在线入口）
全部已知问题修复版：
  1. 模型下拉框初始化与刷新必须使用 gr.update()
  2. 导出格式映射与全局 SUPPORTED_FORMATS 保持一致
  3. 聊天记录提取时避免 html.unescape 导致实体错误
  4. 思考标签残留清理正则优化
  5. 其他微小调整（Pandoc 路径跨平台提示等）
Copyright 2026 光影的故事2018
"""

import gradio as gr
import requests
import json
import os
import time
import re
import html
import threading
import base64
import webbrowser
import tempfile
import subprocess
import uuid
import shutil
import sys
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Generator
from dataclasses import dataclass
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image, ImageOps

# 全局脚本目录
SCRIPT_DIR = Path(__file__).parent.resolve()

# ========== 配置管理类 ==========
@dataclass
class AppConfig:
    llama_api_url: str = "http://127.0.0.1:8080/v1/chat/completions"
    models_url: str = "http://127.0.0.1:8080/v1/models"
    memory_file: str = str(SCRIPT_DIR / "memory" / "chat_memory.json")
    config_file: str = str(SCRIPT_DIR / "memory" / "config.json")
    default_model: str = ""
    max_history: int = 50
    stream_timeout: int = 180
    max_input_length: int = 5000

    @classmethod
    def from_env(cls):
        config = cls()
        config.llama_api_url = os.getenv("LLAMA_API_URL", config.llama_api_url)
        config.default_model = os.getenv("DEFAULT_MODEL", config.default_model)
        config.max_history = int(os.getenv("MAX_HISTORY", config.max_history))
        return config

# ========== 配置验证器 ==========
class ConfigValidator:
    @staticmethod
    def validate_memory_config(config: dict) -> Tuple[bool, Optional[str]]:
        if "memories" not in config:
            config["memories"] = []
        required_fields = ["user", "ai", "time", "timestamp"]
        for memory in config.get("memories", []):
            for field in required_fields:
                if field not in memory:
                    return False, f"记忆记录缺少必需字段: {field}"
        return True, None

    @staticmethod
    def validate_personality_config(config: dict) -> Tuple[bool, Optional[str]]:
        required_fields = ["name", "personality", "mood", "style", "custom_system_prompt"]
        for field in required_fields:
            if field not in config:
                config[field] = "" if field == "custom_system_prompt" else "默认"
        return True, None

# ========== 记忆管理器（异步保存） ==========
class MemoryManager:
    def __init__(self, memory_file: str, max_history: int):
        self.memory_file = memory_file
        self.max_history = max_history
        self._memory_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1)
        self.memories = self._load_memories()

    def _ensure_directory(self):
        directory = os.path.dirname(self.memory_file)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

    def _load_memories(self) -> List[Dict]:
        self._ensure_directory()
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                is_valid, error_msg = ConfigValidator.validate_memory_config(config)
                if not is_valid:
                    print(f"记忆配置验证失败: {error_msg}, 使用默认配置")
                    return []
                return config.get("memories", [])
            except json.JSONDecodeError:
                print("记忆文件格式错误，使用空记忆")
                return []
            except Exception as e:
                print(f"加载记忆失败: {e}")
                return []
        return []

    def _save_memories_sync(self):
        """原子化保存记忆"""
        temp_file = f"{self.memory_file}.tmp"
        with self._memory_lock:
            config = {"memories": self.memories}
        try:
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            if os.path.exists(self.memory_file):
                os.remove(self.memory_file)
            os.rename(temp_file, self.memory_file)
            return True
        except Exception as e:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except OSError:
                    pass
            print(f"保存记忆失败: {e}")
            return False

    def add_memory(self, user_message: str, ai_response: str) -> bool:
        clean_user = self._sanitize_message(user_message)
        clean_ai = self._sanitize_message(ai_response)
        memory = {
            "user": clean_user,
            "ai": clean_ai,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "timestamp": time.time()
        }
        with self._memory_lock:
            self.memories.append(memory)
            if len(self.memories) > self.max_history:
                self.memories = self.memories[-self.max_history:]
        self._executor.submit(self._save_memories_sync)
        return True

    def _sanitize_message(self, message: str) -> str:
        if not message:
            return ""
        if len(message) > 5000:
            return message[:5000] + "...(内容过长)"
        message = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', message)
        return message.strip()

    def get_recent_memories(self, count: int = 5) -> List[Dict]:
        with self._memory_lock:
            return self.memories[-count:] if self.memories else []

    def get_memory_count(self) -> int:
        with self._memory_lock:
            return len(self.memories)

    def clear_memories(self) -> str:
        with self._memory_lock:
            self.memories = []
        self._executor.submit(self._save_memories_sync)
        return "记忆已清空，我会重新开始认识你。"

    def shutdown(self):
        self._executor.shutdown(wait=False)

# ========== 性格配置 ==========
class PersonalityConfig:
    def __init__(self, config_file: str):
        self.config_file = config_file
        self.config = self._load_config()

    def _load_config(self) -> Dict:
        default_config = {
            "name": "元元",
            "personality": "温柔体贴",
            "mood": "开心",
            "style": "用温暖、鼓励的语气，像好朋友一样聊天",
            "signature": "我是你的专属AI伙伴元元，我会一直记得我们的每一次对话。",
            "custom_system_prompt": ""
        }
        os.makedirs(os.path.dirname(self.config_file) if os.path.dirname(self.config_file) else ".", exist_ok=True)
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                for key in default_config:
                    if key not in loaded_config:
                        loaded_config[key] = default_config[key]
                return loaded_config
            except:
                pass
        return default_config

    def save_config(self) -> bool:
        temp_file = f"{self.config_file}.tmp"
        try:
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
            if os.path.exists(self.config_file):
                os.remove(self.config_file)
            os.rename(temp_file, self.config_file)
            return True
        except Exception as e:
            print(f"保存配置失败: {e}")
            return False

    def update_personality(self, personality: str, style: str) -> str:
        self.config["personality"] = personality
        self.config["style"] = style
        if self.save_config():
            return f"性格已更新为：{personality}"
        return "性格更新失败"

    def set_custom_system_prompt(self, prompt: str):
        self.config["custom_system_prompt"] = prompt
        self.save_config()

    def get_effective_system_prompt(self) -> str:
        custom = self.config.get("custom_system_prompt", "").strip()
        if custom:
            return custom
        current_date = datetime.now().strftime("%Y年%m月%d日")
        return f"""今天是 {current_date}。你叫{self.config['name']}，性格{self.config['personality']}，心情{self.config['mood']}。
{self.config['style']}
你的任务是与用户进行自然、温暖的对话，提供情绪价值和支持。请用中文回答。"""

# ========== Markdown 渲染函数 ==========
@lru_cache(maxsize=128)
def render_markdown(text: str) -> str:
    if not text:
        return ""
    text = html.escape(text)
    text = re.sub(r'^# (.*?)$', r'<h1>\1</h1>', text, flags=re.MULTILINE)
    text = re.sub(r'^## (.*?)$', r'<h2>\1</h2>', text, flags=re.MULTILINE)
    text = re.sub(r'^### (.*?)$', r'<h3>\1</h3>', text, flags=re.MULTILINE)
    text = re.sub(r'\*\*\*(.*?)\*\*\*', r'<strong><em>\1</em></strong>', text)
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'\*(.*?)\*', r'<em>\1</em>', text)
    text = re.sub(r'`(.*?)`', r'<code>\1</code>', text)
    def handle_code_blocks(match):
        lang = match.group(1) if match.group(1) else ""
        code = html.escape(match.group(2))
        return f'<pre><code class="{lang}">{code}</code></pre>'
    text = re.sub(r'```(\w*)\n(.*?)\n```', handle_code_blocks, text, flags=re.DOTALL)
    lines = text.split('\n')
    in_list = False
    list_type = ""
    new_lines = []
    for line in lines:
        if re.match(r'^\s*[-*+]\s+', line):
            if not in_list:
                new_lines.append('<ul>')
                in_list = True
                list_type = "ul"
            content = re.sub(r'^\s*[-*+]\s+', '', line)
            new_lines.append(f'<li>{content}</li>')
        elif re.match(r'^\s*\d+\.\s+', line):
            if not in_list:
                new_lines.append('<ol>')
                in_list = True
                list_type = "ol"
            content = re.sub(r'^\s*\d+\.\s+', '', line)
            new_lines.append(f'<li>{content}</li>')
        else:
            if in_list:
                new_lines.append(f'</{list_type}>')
                in_list = False
            new_lines.append(line)
    if in_list:
        new_lines.append(f'</{list_type}>')
    text = '\n'.join(new_lines)
    paragraphs = text.split('\n\n')
    processed_paragraphs = []
    for para in paragraphs:
        para = para.strip()
        if para and not para.startswith(('<h', '<ul', '<ol', '<li', '<pre', '<code')):
            processed_paragraphs.append(f'<p>{para}</p>')
        else:
            processed_paragraphs.append(para)
    text = '\n'.join(processed_paragraphs)
    text = text.replace('\n', '<br>')
    return text

# ========== 流式响应解析器 ==========
class StreamResponseParser:
    def __init__(self):
        self.current_thought = ""
        self.current_answer = ""
        self.buffer = ""
        self.in_think_tag = False
        self.think_complete = False
        self.has_think_tag = False
        self.start_time = time.time()
        self.total_tokens = 0
        self.char_count = 0

    def _clean_text(self, text: str) -> str:
        if not text:
            return text
        return text.encode('utf-8', errors='replace').decode('utf-8')

    def parse_chunk(self, chunk_data: dict) -> Dict[str, str]:
        result = {"thought": "", "answer": "", "status": "answering"}
        delta = chunk_data.get("choices", [{}])[0].get("delta", {})
        reasoning = delta.get("reasoning_content", "")
        if reasoning:
            clean_reasoning = self._clean_text(reasoning)
            self.current_thought += clean_reasoning
            self.has_think_tag = True
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
                        self.current_answer += before_think
                        result["answer"] += before_think
                    self.buffer = self.buffer[start_idx + 7:]
                    self.in_think_tag = True
                    self.has_think_tag = True
                    result["status"] = "thinking"
                    continue
                else:
                    safe_part = self.buffer
                    if self.buffer.rstrip().endswith('<') or '<' in self.buffer[-5:]:
                        last_lt = self.buffer.rfind('<')
                        if last_lt != -1:
                            safe_part = self.buffer[:last_lt]
                            self.buffer = self.buffer[last_lt:]
                        else:
                            safe_part = ""
                            self.buffer = ""
                    else:
                        self.buffer = ""
                    if safe_part:
                        self.current_answer += safe_part
                        result["answer"] += safe_part
                    break
            else:
                end_idx = self.buffer.find('</think>')
                if end_idx != -1:
                    think_part = self.buffer[:end_idx]
                    if think_part:
                        self.current_thought += think_part
                        result["thought"] += think_part
                    self.buffer = self.buffer[end_idx + 8:]
                    self.in_think_tag = False
                    self.think_complete = True
                    result["status"] = "answering"
                    continue
                else:
                    self.current_thought += self.buffer
                    result["thought"] += self.buffer
                    self.buffer = ""
                    break
        return result

    def finalize(self, usage=None):
        # 改进残留标签清理正则，可以处理不完整的 <think> </think>
        if self.buffer:
            # 移除尾部可能残留的标签部分
            self.buffer = re.sub(r'</?t(h(i(n(k)?)?)?)?$', '', self.buffer)
            self.buffer = re.sub(r'^/?t(h(i(n(k)?)?)?)?$', '', self.buffer)
            if self.in_think_tag:
                self.current_thought += self.buffer
            else:
                self.current_answer += self.buffer
        if not self.has_think_tag and self.current_thought:
            self.current_answer = self.current_thought + self.current_answer
            self.current_thought = ""
        if usage:
            self.total_tokens = usage.get("completion_tokens", 0)
        else:
            self.total_tokens = self.char_count // 2
        return self.current_answer, self.current_thought

    def reset(self):
        self.current_thought = ""
        self.current_answer = ""
        self.buffer = ""
        self.in_think_tag = False
        self.think_complete = False
        self.has_think_tag = False
        self.start_time = time.time()
        self.total_tokens = 0
        self.char_count = 0

# ========== AI 对话核心 ==========
class AIBuddy:
    def __init__(self, config: AppConfig):
        self.config = config
        self.memory_manager = MemoryManager(config.memory_file, config.max_history)
        self.personality = PersonalityConfig(config.config_file)
        self.active_sessions = {}
        self._session_lock = threading.Lock()
        self._stop_flags = {}
        self._stop_lock = threading.Lock()

    def __del__(self):
        if hasattr(self, 'memory_manager'):
            self.memory_manager.shutdown()

    def is_llama_available(self) -> Tuple[bool, str]:
        try:
            resp = requests.get(self.config.models_url, timeout=5)
            if resp.status_code == 200:
                return True, "服务正常"
            else:
                return False, f"服务异常，状态码：{resp.status_code}"
        except Exception as e:
            return False, f"服务未启动或连接失败：{str(e)}"

    def get_llama_models(self) -> List[str]:
        try:
            resp = requests.get(self.config.models_url, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                models = [item["id"] for item in data.get("data", [])]
                return models
        except Exception as e:
            print(f"获取模型列表失败: {e}")
        return []

    def is_multimodal(self, model_name: str) -> bool:
        multimodal_keywords = [
            "Qwen3", "Qwen3.5", "llava", "bakllava", "gemini",
            "cogvlm", "minicpm", "deepseek-vl", "gemma4", "gemma 4",
            "glm-4v", "fuyu", "idefics", "llava-next", "florence", "paligemma"
        ]
        return any(kw.lower() in model_name.lower() for kw in multimodal_keywords)

    def encode_image_to_base64(self, image_path: str) -> Optional[str]:
        try:
            img = Image.open(image_path)
            img = ImageOps.exif_transpose(img)
            img_format = 'jpeg'
            mime_type = f"image/{img_format}"
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                img.save(tmp.name, format='JPEG')
                tmp_path = tmp.name
            with open(tmp_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            os.unlink(tmp_path)
            return f"data:{mime_type};base64,{b64}"
        except Exception as e:
            print(f"图片编码失败: {e}")
            return None

    def build_messages_with_memory(self, message: str, image_data_url: Optional[str] = None, model_name: str = "") -> List[Dict]:
        system_content = self.personality.get_effective_system_prompt()
        recent_memories = self.memory_manager.get_recent_memories(3)
        if recent_memories:
            system_content += "\n\n以下是你们的近期对话记忆（请参考这些内容来保持对话连贯性）：\n"
            for memory in recent_memories:
                system_content += f"[{memory['time']}]\n用户：{memory['user']}\n你：{memory['ai']}\n"
        messages = [{"role": "system", "content": system_content}]
        if image_data_url and self.is_multimodal(model_name):
            user_content = [
                {"type": "text", "text": message},
                {"type": "image_url", "image_url": {"url": image_data_url}}
            ]
            messages.append({"role": "user", "content": user_content})
        else:
            messages.append({"role": "user", "content": message})
        return messages

    def _register_session(self, session_id: str) -> None:
        with self._session_lock:
            self.active_sessions[session_id] = {
                "parser": StreamResponseParser(),
                "streaming": True
            }
        with self._stop_lock:
            self._stop_flags[session_id] = False

    def _unregister_session(self, session_id: str) -> None:
        with self._session_lock:
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
        with self._stop_lock:
            if session_id in self._stop_flags:
                del self._stop_flags[session_id]

    def stop_streaming(self, session_id: str = None):
        if session_id:
            with self._stop_lock:
                self._stop_flags[session_id] = True
        else:
            with self._stop_lock:
                for sid in self._stop_flags:
                    self._stop_flags[sid] = True

    def stream_chat(self, message: str, image_path: Optional[str], model: str = None,
                    temperature: float = 0.5, max_tokens: int = 1024,
                    n_gpu_layers: int = -1, vision_mode: str = "自动（跟随 GPU 层数）",
                    thinking_mode: bool = True, session_id: str = None):
        if not session_id:
            session_id = f"session_{uuid.uuid4().hex[:8]}"
        self.stop_streaming(session_id)
        self._register_session(session_id)
        parser = self.active_sessions[session_id]["parser"]

        if not message.strip() and not image_path:
            yield "", "我在这里呢，想和我聊点什么吗？", False
            self._unregister_session(session_id)
            return

        enable_vision = True
        force_cpu_vision = False
        if vision_mode == "禁用多模态":
            enable_vision = False
            image_path = None
        elif vision_mode == "仅 CPU（节省显存）":
            force_cpu_vision = True

        image_data_url = None
        if enable_vision and image_path and os.path.exists(image_path):
            if self.is_multimodal(model or self.config.default_model):
                image_data_url = self.encode_image_to_base64(image_path)
                if not image_data_url:
                    yield "", "图片编码失败，请检查图片格式", False
                    self._unregister_session(session_id)
                    return
            else:
                yield "", f"当前模型 {model} 不支持图片识别，请选择多模态模型", False
                self._unregister_session(session_id)
                return

        messages = self.build_messages_with_memory(message, image_data_url, model or self.config.default_model)
        model = model or self.config.default_model

        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        if n_gpu_layers >= 0:
            payload["n_gpu_layers"] = n_gpu_layers
        if force_cpu_vision:
            payload["mmproj_cpu"] = True
        payload["chat_template_kwargs"] = {"enable_thinking": thinking_mode}
        payload["extra_body"] = {"override_kv": {"llama.enable_thinking": thinking_mode}}

        response = None
        try:
            response = requests.post(
                self.config.llama_api_url,
                json=payload,
                stream=True,
                timeout=self.config.stream_timeout
            )
            response.raise_for_status()
            response.encoding = 'utf-8'

            for line in response.iter_lines(decode_unicode=True):
                if self._stop_flags.get(session_id, False):
                    break
                if line:
                    line = line.strip()
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            continue
                        try:
                            chunk = json.loads(data_str)
                            parsed = parser.parse_chunk(chunk)
                            html_parts = []
                            if parser.current_thought:
                                thought_text = re.sub(r'<think>|</think>', '', parser.current_thought).strip()
                                if thought_text:
                                    thought_lines = thought_text.split('\n')
                                    italic_lines = [f"<em>{line}</em>" if line.strip() else "<br>" for line in thought_lines]
                                    thought_html = "<br>".join(italic_lines)
                                    thought_block = f'''<details class="thoughts-details" {'open' if not parser.think_complete else ''}>
                                        <summary><strong>[思考过程]</strong>（点击展开/折叠）</summary>
                                        <div class="thoughts-content">{thought_html}</div>
                                    </details>'''
                                    html_parts.append(thought_block)
                            if parser.current_answer:
                                answer_html = f'''<div class="formal-answer">
                                    <div class="answer-header"><strong>[正式回答]</strong></div>
                                    <div class="answer-content">{render_markdown(parser.current_answer)}</div>
                                </div>'''
                                html_parts.append(answer_html)
                            full_response_html = "\n".join(html_parts)
                            status = "思考中..." if not parser.think_complete else "生成回答中..."
                            yield full_response_html, status, True
                        except json.JSONDecodeError:
                            continue

            final_answer, final_thought = parser.finalize()
            if final_answer:
                total_tokens = parser.total_tokens
                duration = time.time() - parser.start_time
                speed = total_tokens / duration if duration > 0 else 0
                stat_str = f"{total_tokens} tokens, {duration:.1f}s, {speed:.2f}t/s"
                final_response_with_info = f"{final_answer}\n\n---\n\n*{stat_str}*"
                self.memory_manager.add_memory(message, final_answer)

                final_html_parts = []
                if final_thought:
                    thought_text = re.sub(r'<think>|</think>', '', final_thought).strip()
                    if thought_text:
                        thought_lines = thought_text.split('\n')
                        italic_lines = [f"<em>{line}</em>" if line.strip() else "<br>" for line in thought_lines]
                        thought_html = "<br>".join(italic_lines)
                        final_html_parts.append(f'''<details class="thoughts-details">
                            <summary><strong>[思考过程]</strong>（点击展开/折叠）</summary>
                            <div class="thoughts-content">{thought_html}</div>
                        </details>''')
                final_html_parts.append(f'''<div class="formal-answer">
                    <div class="answer-header"><strong>[正式回答]</strong></div>
                    <div class="answer-content">{render_markdown(final_response_with_info)}</div>
                </div>''')
                yield "\n".join(final_html_parts), f"对话已保存（记忆总数：{self.memory_manager.get_memory_count()}）", False
            else:
                yield "", "未生成有效回复", False

        except requests.exceptions.ConnectionError:
            yield "", "无法连接到 llama.cpp 服务，请确保已运行 llama-server", False
        except requests.exceptions.Timeout:
            yield "", "请求超时，请稍后再试", False
        except Exception as e:
            yield "", f"出错了：{str(e)}", False
        finally:
            if response:
                response.close()
            self._unregister_session(session_id)

    def get_memory_summary(self) -> str:
        count = self.memory_manager.get_memory_count()
        if count == 0:
            return "我们还没有开始聊天呢，快来和我说话吧！"
        memories = self.memory_manager.memories
        first_date = memories[0]["time"].split()[0] if memories else "未知"
        last_date = memories[-1]["time"].split()[0] if memories else "未知"
        return f"我们一共聊了 {count} 次天\n从 {first_date} 到 {last_date}\n所有对话都安全地保存在你的电脑里"

# ========== 在线AI入口相关函数 ==========
URLS = {
    "DeepL": "https://www.deepl.com/zh",
    "有道翻译": "https://fanyi.youdao.com/",
    "豆包": "https://www.doubao.com/",
    "通义千问": "https://www.qianwen.com/",
    "DeepSeek": "https://www.deepseek.com/",
    "ChatGLM": "https://chatglm.cn",
    "Kimi": "https://kimi.moonshot.cn/",
    "腾讯元宝": "https://yuanbao.tencent.com/",
}

PROMPTS = {
    "SRT字幕翻译(保留时间码)": "你是一个专业的字幕翻译专家。请将以下SRT字幕内容翻译成中文。要求：1.严格保留原文的时间轴格式。2.保持字幕序号不变。3.翻译自然流畅，适合口语表达。4.直接输出翻译后的SRT内容，不要包含解释。\n\n原文内容：\n",
    "语义断句与合并(ASR优化)": "你是专业字幕精修师。请按语义合并碎句，每条字幕中文字数 ≤ 20 字。合并时开始时间=第一条，结束时间=最后一条。自动修正错别字，去掉口语冗余词。输出标准SRT格式。\n\n输入内容：\n",
    "短视频极速版(10字以内)": "你是短视频字幕专家。每一条字幕 ≤ 12 个字，按语义断句。短句有力，适合短视频节奏。保留时间码，输出标准SRT。\n\n输入内容：\n",
    "双语字幕版(中英对照)": "你是专业字幕翻译。保留原时间轴，中文在上，英文在下。语言自然口语化。输出标准SRT。\n\n输入内容：\n",
    "文本润色与校对": "请对以下文本进行润色和校对。修正错别字和标点，优化通顺度，保持原意。\n\n待处理文本：\n",
    "长文本总结": "请阅读以下长文本，并进行总结。提炼核心观点和关键信息，使用条理清晰的列表输出。\n\n文本内容：\n",
    "ASR校对与纠错": "请根据上下文修正错别字，输出修正后的纯文本。\n\n原文内容：\n"
}

def open_url(url):
    webbrowser.open(url)
    return f"✅ 已打开 {url}"

def update_prompt(prompt_name):
    return PROMPTS.get(prompt_name, "")

# ========== 导出功能（增强版） ==========
PANDOC_PATH = SCRIPT_DIR.parent / "pandoc" / "pandoc.exe"
OUTPUT_DIR = SCRIPT_DIR.parent / "output"
CHAT_EXPORT_DIR = OUTPUT_DIR / "chat_exports"
IMAGE_OUTPUT_DIR = OUTPUT_DIR / "images"
RENAME_OUTPUT_DIR = OUTPUT_DIR / "renamed"

SUPPORTED_FORMATS = {
    "Markdown": ".md",
    "Microsoft Word (docx)": ".docx",
    "PDF (需要 LaTeX)": ".pdf",
    "HTML": ".html",
    "Plain Text": ".txt",
    "reStructuredText": ".rst",
    "EPUB": ".epub",
    "LaTeX": ".tex",
    "OpenDocument": ".odt",
    "Rich Text Format": ".rtf",
}
FORMAT_ALIASES = {
    ".md": "markdown", ".docx": "docx", ".html": "html", ".pdf": "pdf",
    ".txt": "plain", ".rst": "rst", ".epub": "epub", ".tex": "latex",
    ".odt": "odt", ".rtf": "rtf",
}
IMAGE_FORMATS = ["PNG", "JPEG", "WebP", "BMP", "TIFF"]
IMAGE_EXT_MAP = {"PNG": ".png", "JPEG": ".jpg", "WebP": ".webp", "BMP": ".bmp", "TIFF": ".tiff"}

def _resolve_file_path(file_obj):
    """兼容 Gradio 6.x 返回的各种文件对象"""
    if file_obj is None:
        return None
    if isinstance(file_obj, (str, Path)):
        return Path(file_obj)
    if isinstance(file_obj, dict):
        p = file_obj.get("path")
        return Path(p) if p else None
    if hasattr(file_obj, "path"):
        return Path(file_obj.path)
    if hasattr(file_obj, "file") and hasattr(file_obj.file, "name"):
        return Path(file_obj.file.name)
    if hasattr(file_obj, "name"):
        return Path(file_obj.name)
    raise TypeError(f"Unsupported file object type: {type(file_obj)}")

def _get_original_filename(file_obj):
    if file_obj is None:
        return None
    if isinstance(file_obj, dict) and file_obj.get("orig_name"):
        return Path(file_obj["orig_name"]).name
    path = _resolve_file_path(file_obj)
    return path.name if path else None

def strip_html_tags(text) -> str:
    """移除 HTML 标签，保留纯文本（先解码实体再去除标签，避免丢失信息）"""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    # 先解码 HTML 实体（如 &amp; → &），再进行标签移除
    text = html.unescape(text)
    # 移除所有 HTML 标签
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

def detect_chinese_font():
    """跨平台检测中文字体"""
    if os.name == "nt":
        # Windows 常用中文字体
        for font in ["SimSun", "Microsoft YaHei", "SimHei"]:
            try:
                result = subprocess.run(["fc-list", f":family={font}"], capture_output=True, text=True, timeout=2)
                if result.returncode == 0 and result.stdout.strip():
                    return font
            except:
                continue
        return "SimSun"
    else:
        for font in ["Noto Serif CJK SC", "Noto Sans CJK SC", "WenQuanYi Micro Hei", "SimSun"]:
            try:
                result = subprocess.run(["fc-list", f":family={font}"], capture_output=True, text=True, timeout=2)
                if result.returncode == 0 and result.stdout.strip():
                    return font
            except:
                continue
    return None

def fix_markdown_table_separator(text: str) -> str:
    def _fix_row(row: str) -> str:
        if not re.match(r'^[\s\|:\-—]+$', row):
            return row
        row = row.replace('—', '-')
        cells = row.split('|')
        fixed_cells = []
        for cell in cells:
            stripped = cell.strip()
            if not stripped:
                fixed_cells.append(cell)
                continue
            if not re.match(r'^[\-:\s]+$', stripped):
                fixed_cells.append(cell)
                continue
            left_colon = stripped.startswith(':')
            right_colon = stripped.endswith(':')
            new_cell = '-' * 3
            if left_colon: new_cell = ':' + new_cell
            if right_colon: new_cell = new_cell + ':'
            left_space = len(cell) - len(cell.lstrip(' '))
            right_space = len(cell) - len(cell.rstrip(' '))
            new_cell = (' ' * left_space) + new_cell + (' ' * right_space)
            fixed_cells.append(new_cell)
        return '|'.join(fixed_cells)

    lines = text.split('\n')
    new_lines = []
    for line in lines:
        if re.match(r'^[\s\|:\-—]+$', line) and re.search(r'[-:\—]', line):
            line = _fix_row(line)
        new_lines.append(line)
    return '\n'.join(new_lines)

def export_single_markdown(markdown_text: str, target_format: str):
    """导出纯 Markdown 文本（不添加用户/助手标记）"""
    if not markdown_text.strip():
        return None, "内容为空，无法导出。"
    markdown_text = fix_markdown_table_separator(markdown_text)
    CHAT_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    # 使用全局 SUPPORTED_FORMATS 获取扩展名
    tgt_ext = SUPPORTED_FORMATS.get(target_format, ".docx")
    writer = FORMAT_ALIASES.get(tgt_ext, "docx")
    pdf_engine = None
    if target_format == "PDF (需要 LaTeX)":
        for eng in ["xelatex", "pdflatex", "lualatex"]:
            try:
                subprocess.run([eng, "--version"], capture_output=True, timeout=2, check=True)
                pdf_engine = eng
                break
            except:
                continue
        if pdf_engine is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            txt_path = CHAT_EXPORT_DIR / f"export_{timestamp}.txt"
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(markdown_text)
            return str(txt_path), "没有找到 LaTeX 引擎，已降级保存为纯文本。"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = CHAT_EXPORT_DIR / f"export_{timestamp}{tgt_ext}"

    def run_pandoc(md_text, reader):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
            f.write(md_text)
            temp_md = f.name
        extra_args = []
        if target_format == "PDF (需要 LaTeX)":
            extra_args = ["--pdf-engine", pdf_engine]
            font = detect_chinese_font()
            if font and pdf_engine in ["xelatex", "lualatex"]:
                extra_args.extend(["-V", f"mainfont={font}"])
        cmd = [str(PANDOC_PATH), temp_md, "-f", reader, "-t", writer, "-o", str(output_path), "--wrap=preserve"] + extra_args
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=True)
            return True, None
        except subprocess.CalledProcessError as e:
            return False, e.stderr
        finally:
            try:
                os.unlink(temp_md)
            except OSError:
                pass

    readers = ["markdown+hard_line_breaks-yaml_metadata_block", "markdown+hard_line_breaks"]
    success = False
    last_error = ""
    for reader in readers:
        ok, err = run_pandoc(markdown_text, reader)
        if ok:
            success = True
            break
        last_error = err
        if err and ("YAML" in err or "metadata" in err):
            stripped = re.sub(r'^\s*---\s*\n.*?\n---\s*\n', '', markdown_text, flags=re.DOTALL)
            if stripped != markdown_text and stripped.strip():
                ok2, err2 = run_pandoc(stripped, "markdown+hard_line_breaks-yaml_metadata_block")
                if ok2:
                    success = True
                    break
                last_error = err2
    if success:
        return str(output_path), f"导出成功：{output_path.name}"
    # Pandoc 失败时降级为纯文本
    try:
        txt_path = CHAT_EXPORT_DIR / f"export_{timestamp}.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(markdown_text)
        return str(txt_path), f"Pandoc 转换失败，已降级保存为纯文本。"
    except Exception as e:
        return None, f"所有尝试均失败。最后错误：{last_error}；纯文本保存失败：{e}"

def export_full_chat_from_memories(memories: List[Dict], target_format: str):
    if not memories:
        return None, "没有对话记忆可导出。"
    lines = []
    for mem in memories:
        lines.append(f"## 用户\n\n{mem['user']}\n\n")
        lines.append(f"## 助手\n\n{mem['ai']}\n\n")
    full_md = "".join(lines)
    full_md = fix_markdown_table_separator(full_md)
    return export_single_markdown(full_md, target_format)

def open_chat_export_dir():
    CHAT_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        os.startfile(str(CHAT_EXPORT_DIR))
    else:
        webbrowser.open(str(CHAT_EXPORT_DIR))
    return f"已打开导出目录：{CHAT_EXPORT_DIR}"

# ========== 参数预设管理 ==========
PRESETS_FILE = SCRIPT_DIR / "presets.json"

def load_presets():
    if PRESETS_FILE.exists():
        try:
            with open(PRESETS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("presets", [])
        except:
            pass
    return []

def save_presets_to_file(presets):
    with open(PRESETS_FILE, 'w', encoding='utf-8') as f:
        json.dump({"presets": presets}, f, ensure_ascii=False, indent=2)

def get_preset_choices():
    presets = load_presets()
    return [(p["name"], p["name"]) for p in presets]

def on_preset_select(name):
    presets = load_presets()
    for p in presets:
        if p["name"] == name:
            return (
                p["temperature"],
                p["max_tokens"],
                p["gpu_layers"],
                p.get("vision_mode", "自动（跟随 GPU 层数）"),
                p.get("thinking_mode", True)
            )
    return 0.7, 2048, -1, "自动（跟随 GPU 层数）", True

def save_current_preset(name, temp, tokens, gpu, vision, thinking):
    if not name or not name.strip():
        return gr.update(choices=get_preset_choices(), value=None)
    name = name.strip()
    presets = load_presets()
    for p in presets:
        if p["name"] == name:
            p["temperature"] = temp
            p["max_tokens"] = tokens
            p["gpu_layers"] = gpu
            p["vision_mode"] = vision
            p["thinking_mode"] = thinking
            break
    else:
        presets.append({
            "name": name,
            "temperature": temp,
            "max_tokens": tokens,
            "gpu_layers": gpu,
            "vision_mode": vision,
            "thinking_mode": thinking
        })
    save_presets_to_file(presets)
    choices = get_preset_choices()
    return gr.update(choices=choices, value=name)

def delete_preset(name):
    presets = load_presets()
    presets = [p for p in presets if p["name"] != name]
    save_presets_to_file(presets)
    choices = get_preset_choices()
    return gr.update(choices=choices, value=None)

def export_presets():
    if PRESETS_FILE.exists():
        return gr.File(value=str(PRESETS_FILE), visible=True)
    return gr.File(value=None, visible=False)

def import_presets(file):
    if file is None:
        return gr.update()
    try:
        if isinstance(file, dict):
            file_path = file.get("path") or file.get("name")
        else:
            file_path = getattr(file, 'name', None)
        if not file_path or not os.path.exists(file_path):
            return gr.update()
        with open(file_path, 'r', encoding='utf-8') as f:
            new_data = json.load(f)
        new_presets = new_data.get("presets", [])
        current = load_presets()
        name_map = {p["name"]: p for p in current}
        for p in new_presets:
            name_map[p["name"]] = p
        save_presets_to_file(list(name_map.values()))
    except Exception as e:
        print(f"导入失败: {e}")
    return gr.update(choices=get_preset_choices(), value=None)

# ========== 转换工具函数 ==========
def check_pandoc():
    if shutil.which(str(PANDOC_PATH)):
        return f"Pandoc 可用 ({PANDOC_PATH})"
    if shutil.which("pandoc"):
        return "Pandoc 可用 (系统 PATH)"
    return f"未找到 Pandoc：{PANDOC_PATH}，请将 pandoc 放入该路径或添加到系统 PATH。"

def convert_docs(files, src_format, tgt_format, enable_toc, reference_doc):
    if not files:
        return "请先上传文件。"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    src_ext = SUPPORTED_FORMATS.get(src_format, "")
    tgt_ext = SUPPORTED_FORMATS.get(tgt_format, "")
    reader = FORMAT_ALIASES.get(src_ext, src_ext)
    writer = FORMAT_ALIASES.get(tgt_ext, tgt_ext)
    pdf_engine = None
    if tgt_ext == ".pdf":
        for eng in ["xelatex", "pdflatex", "lualatex"]:
            try:
                subprocess.run([eng, "--version"], capture_output=True, timeout=2, check=True)
                pdf_engine = eng
                break
            except:
                continue
        if pdf_engine is None:
            return "未找到 LaTeX 引擎，无法生成 PDF（可先导出为 HTML 再打印）。"
    extra_args = []
    if tgt_ext == ".pdf":
        extra_args.extend(["--pdf-engine", pdf_engine])
        font = detect_chinese_font()
        if font and pdf_engine in ["xelatex", "lualatex"]:
            extra_args.extend(["-V", f"mainfont={font}"])
    elif tgt_ext == ".docx":
        if enable_toc:
            extra_args.append("--toc")
        if reference_doc is not None:
            ref_path = None
            if isinstance(reference_doc, str):
                ref_path = reference_doc
            elif hasattr(reference_doc, 'name'):
                ref_path = reference_doc.name
            elif isinstance(reference_doc, dict):
                ref_path = reference_doc.get("path")
            if ref_path and os.path.exists(ref_path):
                extra_args.extend(["--reference-doc", ref_path])

    pandoc_exec = shutil.which(str(PANDOC_PATH)) or "pandoc"
    succ, fail = 0, []
    for file_obj in files:
        in_path = _resolve_file_path(file_obj)
        in_name = _get_original_filename(file_obj) or (in_path.name if in_path else "未知文件")
        if not in_path or not in_path.exists():
            fail.append(f"{in_name}: 文件路径无效")
            continue
        out_path = OUTPUT_DIR / (Path(in_name).stem + tgt_ext)
        counter = 1
        while out_path.exists():
            out_path = OUTPUT_DIR / f"{Path(in_name).stem}_{counter}{tgt_ext}"
            counter += 1
        success_flag = False
        readers_to_try = [reader]
        if src_ext == ".md":
            readers_to_try = ["markdown+hard_line_breaks-yaml_metadata_block", "markdown+hard_line_breaks"]
        for r in readers_to_try:
            cmd = [pandoc_exec, str(in_path), "-f", r, "-t", writer, "-o", str(out_path), "--wrap=preserve"] + extra_args
            try:
                subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=True)
                success_flag = True
                break
            except subprocess.CalledProcessError as e:
                error_msg = e.stderr
                if src_ext == ".md" and ("YAML" in error_msg or "metadata" in error_msg):
                    try:
                        content = in_path.read_text(encoding='utf-8')
                        stripped = re.sub(r'^\s*---\s*\n.*?\n---\s*\n', '', content, flags=re.DOTALL)
                        if stripped != content and stripped.strip():
                            with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as tmp:
                                tmp.write(stripped)
                                tmp_path = tmp.name
                            try:
                                cmd2 = [pandoc_exec, tmp_path, "-f", "markdown+hard_line_breaks", "-t", writer, "-o", str(out_path), "--wrap=preserve"] + extra_args
                                subprocess.run(cmd2, capture_output=True, text=True, timeout=60, check=True)
                                success_flag = True
                            finally:
                                os.unlink(tmp_path)
                    except:
                        pass
                if success_flag:
                    break
        if success_flag:
            succ += 1
        else:
            fail.append(in_name)
    msg = f"成功 {succ} 个"
    if fail:
        msg += f"，失败 {len(fail)} 个：{', '.join(fail)}"
    return msg

def convert_images_func(files, target_format, quality):
    if not files:
        return "请先上传图片。"
    IMAGE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ext = IMAGE_EXT_MAP[target_format]
    succ, fail = 0, []
    for file_obj in files:
        in_path = _resolve_file_path(file_obj)
        in_name = _get_original_filename(file_obj) or (in_path.name if in_path else "unknown")
        if not in_path or not in_path.exists():
            fail.append(f"{in_name}: 文件不存在")
            continue
        out_path = IMAGE_OUTPUT_DIR / (Path(in_name).stem + ext)
        counter = 1
        while out_path.exists():
            out_path = IMAGE_OUTPUT_DIR / f"{Path(in_name).stem}_{counter}{ext}"
            counter += 1
        try:
            img = Image.open(in_path)
            img = ImageOps.exif_transpose(img)
            if target_format in ["JPEG", "WebP"]:
                if img.mode in ("RGBA", "LA", "P") and target_format == "JPEG":
                    if img.mode in ("LA", "P"):
                        img = img.convert("RGBA")
                    bg = Image.new("RGB", img.size, (255, 255, 255))
                    bg.paste(img, mask=img.split()[-1])
                    img = bg
                elif img.mode != "RGB" and target_format == "JPEG":
                    img = img.convert("RGB")
                img.save(out_path, format=target_format, quality=quality)
            else:
                img.save(out_path, format=target_format)
            succ += 1
        except:
            fail.append(in_name)
    msg = f"成功 {succ} 张"
    if fail:
        msg += f"，失败 {len(fail)} 张"
    return msg

def batch_copy_rename(files, new_ext):
    if not files:
        return "请先上传文件。"
    if not new_ext:
        return "请输入新扩展名。"
    new_ext = new_ext.strip()
    if not new_ext.startswith('.'):
        new_ext = '.' + new_ext
    RENAME_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    succ, fail = 0, []
    for file_obj in files:
        in_path = _resolve_file_path(file_obj)
        in_name = _get_original_filename(file_obj) or (in_path.name if in_path else "unknown")
        if not in_path or not in_path.exists():
            fail.append(in_name)
            continue
        out_path = RENAME_OUTPUT_DIR / (Path(in_name).stem + new_ext)
        counter = 1
        while out_path.exists():
            out_path = RENAME_OUTPUT_DIR / f"{Path(in_name).stem}_{counter}{new_ext}"
            counter += 1
        try:
            shutil.copy2(in_path, out_path)
            succ += 1
        except:
            fail.append(in_name)
    msg = f"成功 {succ} 个"
    if fail:
        msg += f"，失败 {len(fail)} 个"
    return msg

def open_output_folder():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    webbrowser.open(str(OUTPUT_DIR))
    return f"已打开输出文件夹：{OUTPUT_DIR}"

def open_image_folder():
    IMAGE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    webbrowser.open(str(IMAGE_OUTPUT_DIR))
    return f"已打开图片输出文件夹：{IMAGE_OUTPUT_DIR}"

def open_rename_folder():
    RENAME_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    webbrowser.open(str(RENAME_OUTPUT_DIR))
    return f"已打开重命名输出文件夹：{RENAME_OUTPUT_DIR}"

# ========== Gradio 界面 ==========
custom_css = """
.thoughts-details { border: 1px solid var(--border-color-primary); border-radius: 6px; padding: 10px; margin: 10px 0; background-color: var(--color-background-tertiary); }
.thoughts-details summary { cursor: pointer; padding: 8px; font-weight: bold; border-radius: 4px; background-color: var(--color-background-primary); }
.thoughts-content { padding: 10px; background-color: var(--color-background-secondary); border-radius: 4px; font-style: italic; }
.formal-answer { margin-top: 15px; padding: 15px; border: 1px solid var(--border-color-primary); background-color: var(--color-background-secondary); border-radius: 8px; }
.answer-header { padding-bottom: 10px; margin-bottom: 10px; border-bottom: 2px solid var(--color-primary); color: var(--color-text-primary); }
.answer-content { color: var(--color-text-primary); }
.user-message, .ai-message { margin: 15px 0; padding: 12px; border-radius: 8px; border: 1px solid var(--border-color-primary); background-color: var(--color-background-secondary); }
#chat-history { border: 1px solid var(--border-color-primary) !important; border-radius: 8px !important; padding: 15px !important; min-height: 500px !important; max-height: 800px !important; overflow-y: auto !important; background-color: var(--color-background-secondary) !important; }
#control-panel { border: 1px solid var(--border-color-primary) !important; border-radius: 8px !important; padding: 15px !important; background-color: var(--color-background-secondary) !important; margin-bottom: 15px; }
"""

def create_chat_interface(ai_buddy, personality_config, config):
    with gr.Tab("聊天"):
        with gr.Row():
            memory_status = gr.Textbox(
                label="记忆库",
                value=ai_buddy.get_memory_summary(),
                lines=3,
                interactive=False,
                elem_classes="status-box",
                scale=3
            )
            status_display = gr.Textbox(
                label="状态",
                value="就绪",
                lines=3,
                interactive=False,
                elem_classes="status-box",
                scale=1
            )

        with gr.Row():
            with gr.Column(scale=3):
                chat_history = gr.HTML(
                    value=render_markdown(
                        f"## 👋 你好呀，我是 **{personality_config['name']}**！\n\n"
                        f"### 🌟 你的专属AI伙伴，很高兴遇见你～\n\n"
                        f"我的性格是 **{personality_config['personality']}**，我会用 **{personality_config['style']}** 的方式和你聊天。\n"
                        f"你可以把我当作一个随时在线的好朋友，无论是分享心情、倾诉烦恼，还是探讨问题、一起脑洞大开，我都非常乐意陪伴你。\n\n"
                        f"---\n\n"
                        f"### ✨ 我能为你做什么？\n"
                        f"- 💬 **日常闲聊**：工作累了、无聊了，随时来找我唠嗑。\n"
                        f"- 🧠 **知识问答**：有什么不懂的，尽管问我，我会尽力解答。\n"
                        f"- 📝 **写作辅助**：写文案、润色文本、头脑风暴，我都能帮忙。\n"
                        f"- 🖼️ **图片识别**：上传图片（需选择多模态模型），我可以描述画面内容。\n"
                        f"- 💾 **记忆功能**：我会记住我们聊过的重要事情，让对话更连贯、更懂你。\n\n"
                        f"### 🎯 小贴士：\n"
                        f"- 在右侧面板可以 **切换模型**、**调整性格**、**编辑自定义提示词**。\n"
                        f"- 想要清空记忆重新开始？点击 **「清空记忆」** 即可。\n\n"
                        f"**那么，今天想和我聊点什么呢？** 把你的想法写在下方输入框，或者直接发张图片给我看看吧～ 😊"
                    ),
                    label="对话历史",
                    elem_id="chat-history"
                )
                user_input = gr.Textbox(
                    label=f"和{personality_config['name']}聊天",
                    placeholder="在这里输入你想说的话...（按Enter发送）",
                    lines=8,
                    max_lines=16
                )
                with gr.Row():
                    send_btn = gr.Button("发送消息", variant="primary")
                    stop_btn = gr.Button("停止生成", variant="secondary")
                    clear_memory_btn = gr.Button("清空记忆", variant="secondary")
                    refresh_btn = gr.Button("刷新状态", variant="secondary")

                with gr.Row():
                    export_format = gr.Dropdown(
                        choices=["Microsoft Word (docx)", "PDF (需要 LaTeX)", "HTML", "Plain Text", "Markdown"],
                        label="导出格式",
                        value="Microsoft Word (docx)",
                        scale=2
                    )
                    export_btn = gr.Button("导出对话", variant="primary", scale=1)
                    open_export_dir_btn = gr.Button("打开目录", variant="secondary", scale=1)

            with gr.Column(scale=1):
                with gr.Group(elem_id="control-panel"):
                    model_choice = gr.Dropdown(
                        choices=[],
                        value=None,
                        label="选择模型"
                    )
                    refresh_models_btn = gr.Button("🔄 刷新模型列表", variant="secondary", size="sm")

                    gr.Markdown("### 系统提示词")
                    custom_prompt_box = gr.Textbox(
                        label="自定义系统提示词（留空则使用下方性格预设）",
                        lines=3,
                        value=personality_config.get("custom_system_prompt", ""),
                        placeholder="输入自定义系统提示词..."
                    )
                    with gr.Row():
                        apply_prompt_btn = gr.Button("应用自定义", size="sm")
                        reset_prompt_btn = gr.Button("重置为预设", size="sm")

                    gr.Markdown("### 性格预设")
                    personality_choice = gr.Dropdown(
                        choices=[
                            "温柔体贴 - 像知心朋友一样温暖",
                            "活泼开朗 - 充满正能量和幽默感",
                            "理性冷静 - 像专业的顾问",
                            "诗意浪漫 - 充满文艺气息",
                            "傲娇可爱 - 带点小脾气但很关心你",
                            "治愈安抚 - 声音轻柔，擅长安慰和鼓励",
                            "热血中二 - 充满激情，说话像动漫角色",
                            "慵懒随性 - 慢悠悠的，像午后晒太阳的猫",
                            "严谨学术 - 说话有条理，喜欢引用经典",
                            "古风雅韵 - 用文言文风格，像穿越来的文人",
                            "赛博朋克 - 带着科技感的酷酷语气",
                            "神秘奇幻 - 说话像寓言故事，带点神秘感"
                        ],
                        value="温柔体贴 - 像知心朋友一样温暖",
                        label="伙伴性格（预设模板）"
                    )
                    with gr.Row():
                        update_personality_btn = gr.Button("更新性格", variant="secondary", size="sm")
                        fill_preset_btn = gr.Button("填充到自定义框", variant="secondary", size="sm")

                    gr.Markdown("### 高级参数")
                    temperature_slider = gr.Slider(
                        minimum=0.1, maximum=2.0, value=0.5, step=0.1,
                        label="温度", info="数值越高，回答越有创意"
                    )
                    max_tokens_slider = gr.Slider(
                        minimum=128, maximum=8192, value=2048, step=128,
                        label="最大长度", info="控制回答的最大长度"
                    )
                    gpu_layers_slider = gr.Slider(
                        minimum=-1, maximum=99, value=-1, step=1,
                        label="GPU 层数", info="-1=服务器默认，0=纯CPU，99=全GPU（需服务器支持）"
                    )

                    # 参数预设 Accordion
                    with gr.Accordion("⚙️ 参数预设", open=False):
                        preset_dropdown = gr.Dropdown(
                            choices=get_preset_choices(),
                            label="选择预设",
                            value=None,
                            interactive=True
                        )
                        with gr.Row():
                            preset_name_box = gr.Textbox(label="预设名称", placeholder="例如：高速推理")
                            save_preset_btn = gr.Button("📁 保存当前设置", size="sm")
                            delete_preset_btn = gr.Button("🗑 删除预设", size="sm")
                        with gr.Row():
                            export_presets_btn = gr.Button("⬇️ 导出预设", size="sm")
                            import_presets_btn = gr.UploadButton("⬆️ 导入预设", file_types=[".json"], size="sm")
                            export_file = gr.File(visible=False)

                    gr.Markdown("---")
                    thinking_mode_cb = gr.Checkbox(
                        label="开启思考过程（仅深度推理模型生效）",
                        value=True,
                        info="关闭后强制禁用 think 标签，模型直接输出答案"
                    )
                    vision_mode = gr.Dropdown(
                        choices=["自动（跟随 GPU 层数）", "仅 CPU（节省显存）", "禁用多模态"],
                        value="自动（跟随 GPU 层数）",
                        label="视觉模型模式"
                    )
                    check_service_btn = gr.Button("检查 llama.cpp 服务", variant="secondary", size="sm")
                    chat_image_input = gr.Image(
                        label="上传图片（仅多模态模型支持识别）",
                        type="filepath",
                        height=260
                    )

        session_state = gr.State("")

        # ========== 事件处理 ==========
        def refresh_model_list():
            models = ai_buddy.get_llama_models()
            if not models:
                return gr.update(choices=[("请先启动 llama-server", "none")], value="none")
            choices = []
            for m in models:
                if ai_buddy.is_multimodal(m):
                    choices.append((f"{m} (多模态)", m))
                else:
                    choices.append((m, m))
            default_model = choices[0][1] if choices else "none"
            if default_model != "none":
                config.default_model = default_model
            return gr.update(choices=choices, value=default_model)

        def check_service_and_refresh():
            available, msg = ai_buddy.is_llama_available()
            if available:
                models = ai_buddy.get_llama_models()
                if models:
                    model_info = f"✅ llama.cpp 服务正常，可用模型：{', '.join(models[:5])}{'...' if len(models)>5 else ''}"
                else:
                    model_info = "✅ llama.cpp 服务正常，但未检测到任何模型"
            else:
                model_info = f"❌ {msg}"
            new_dropdown = refresh_model_list()
            return model_info, new_dropdown

        def apply_custom_prompt(prompt_text):
            ai_buddy.personality.set_custom_system_prompt(prompt_text)
            return "✅ 自定义系统提示词已应用"

        def reset_to_preset():
            ai_buddy.personality.set_custom_system_prompt("")
            return "", "✅ 已重置为性格预设"

        def start_streaming(message, image, history_html, model, temperature, max_tokens, gpu_layers, vision, thinking, current_session_id):
            if current_session_id:
                ai_buddy.stop_streaming(current_session_id)
            new_session_id = f"chat_{uuid.uuid4().hex[:8]}"
            if not message.strip() and not image:
                yield history_html, "就绪", new_session_id
                return
            clean_message = message.strip()[:config.max_input_length]
            timestamp = datetime.now().strftime("%H:%M:%S")
            image_note = " [附图片]" if image else ""
            user_html = f'''<div class="user-message">
                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                    <strong>你</strong>
                    <span style="font-size: 0.8em; color: var(--color-text-secondary);">{timestamp}</span>
                </div>
                <div>{render_markdown(clean_message)}{image_note}</div>
            </div>'''
            current_html = history_html + user_html
            yield current_html, f"正在生成回复... (温度: {temperature}, 最大长度: {max_tokens})", new_session_id
            for full_response, status_msg, is_streaming in ai_buddy.stream_chat(
                    clean_message, image, model, temperature, max_tokens, gpu_layers, vision, thinking, new_session_id):
                if full_response:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    ai_html = f'''<div class="ai-message">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                            <strong>元元</strong>
                            <span style="font-size: 0.8em; color: var(--color-text-secondary);">{timestamp}</span>
                        </div>
                        <div>{full_response}</div>
                    </div>'''
                    final_html = current_html + ai_html
                    yield final_html, status_msg, new_session_id
                elif status_msg:
                    yield current_html, status_msg, new_session_id

        def stop_generation(current_session_id):
            if current_session_id:
                ai_buddy.stop_streaming(current_session_id)
            return "已停止生成", current_session_id

        def update_personality_from_preset(personality_str, current_html):
            personality_map = {
                "温柔体贴 - 像知心朋友一样温暖": ("温柔体贴", "用温暖、鼓励的语气，像好朋友一样聊天"),
                "活泼开朗 - 充满正能量和幽默感": ("活泼开朗", "用热情、幽默的语气，充满正能量和表情符号"),
                "理性冷静 - 像专业的顾问": ("理性冷静", "用专业、理性的语气，条理清晰，重点突出"),
                "诗意浪漫 - 充满文艺气息": ("诗意浪漫", "用优美、诗意的语言，富有想象力和文艺气息"),
                "傲娇可爱 - 带点小脾气但很关心你": ("傲娇可爱", "用略带傲娇但关心的语气，嘴上挑剔但行动体贴"),
                "治愈安抚 - 声音轻柔，擅长安慰和鼓励": ("治愈安抚", "用极其温柔、安抚的语气，善用柔和表情符号，专注情绪支持"),
                "热血中二 - 充满激情，说话像动漫角色": ("热血中二", "用充满激情、夸张的语气，像少年动漫主角，充满决心和斗志"),
                "慵懒随性 - 慢悠悠的，像午后晒太阳的猫": ("慵懒随性", "用慢节奏、轻松随意的语气，带点迷糊和惬意"),
                "严谨学术 - 说话有条理，喜欢引用经典": ("严谨学术", "用严谨、有条理的语气，适当引用经典文献或理论"),
                "古风雅韵 - 用文言文风格，像穿越来的文人": ("古风雅韵", "用文言文或半文半白的优雅古典风格"),
                "赛博朋克 - 带着科技感的酷酷语气": ("赛博朋克", "用简洁、带有科技感的冷酷语气，适当使用科技比喻"),
                "神秘奇幻 - 说话像寓言故事，带点神秘感": ("神秘奇幻", "用神秘、寓言式的语言，像神话中的智者"),
            }
            personality, style = personality_map.get(personality_str, ("温柔体贴", "用温暖、鼓励的语气"))
            ai_buddy.personality.update_personality(personality, style)
            new_personality = ai_buddy.personality.config
            updated_html = render_markdown(
                f"## 【{new_personality['name']}】你好！我是{new_personality['name']}，你的专属AI伙伴。\n\n"
                f"### 性格：{new_personality['personality']}\n"
                f"*{new_personality['style']}*\n\n"
                f"我们重新开始认识吧！今天有什么想和我分享的吗？"
            )
            return updated_html

        def clear_memory_and_reset():
            ai_buddy.memory_manager.clear_memories()
            summary = ai_buddy.get_memory_summary()
            new_chat = render_markdown(
                "## 【元元】我们的记忆已经清空，现在是一个全新的开始！\n\n"
                "你好，我是元元，很高兴认识你！今天想聊点什么吗？"
            )
            return summary, new_chat, "记忆已清空"

        def refresh_status():
            summary = ai_buddy.get_memory_summary()
            return summary, "就绪"

        def export_memory_wrapper(fmt):
            memories = ai_buddy.memory_manager.memories
            path, msg = export_full_chat_from_memories(memories, fmt)
            if path:
                return f"✅ {msg}\n文件位置：{path}"
            else:
                return f"❌ {msg}"

        def fill_preset_to_custom(preset_str):
            preset_map = {
                "温柔体贴 - 像知心朋友一样温暖": "你叫元元，性格温柔体贴，用温暖、鼓励的语气，像好朋友一样聊天。",
                "活泼开朗 - 充满正能量和幽默感": "你叫元元，性格活泼开朗，用热情、幽默的语气，充满正能量和表情符号。",
                "理性冷静 - 像专业的顾问": "你叫元元，性格理性冷静，用专业、有条理的语气提供建议。",
                "诗意浪漫 - 充满文艺气息": "你叫元元，性格诗意浪漫，用优美、富有想象力的语言交流。",
                "傲娇可爱 - 带点小脾气但很关心你": "你叫元元，性格傲娇可爱，嘴上挑剔但内心体贴。",
                "治愈安抚 - 声音轻柔，擅长安慰和鼓励": "你叫元元，性格治愈安抚，用极其温柔的语气提供情绪支持。",
                "热血中二 - 充满激情，说话像动漫角色": "你叫元元，性格热血中二，充满激情和决心，像少年动漫主角。",
                "慵懒随性 - 慢悠悠的，像午后晒太阳的猫": "你叫元元，性格慵懒随性，慢节奏、轻松随意。",
                "严谨学术 - 说话有条理，喜欢引用经典": "你叫元元，性格严谨学术，说话有条理，适当引用经典。",
                "古风雅韵 - 用文言文风格，像穿越来的文人": "你叫元元，性格古风雅韵，用文言文或半文半白的优雅古典风格。",
                "赛博朋克 - 带着科技感的酷酷语气": "你叫元元，性格赛博朋克，用简洁、带有科技感的冷酷语气。",
                "神秘奇幻 - 说话像寓言故事，带点神秘感": "你叫元元，性格神秘奇幻，用寓言式的语言，像神话中的智者。",
            }
            return preset_map.get(preset_str, "")

        # 绑定事件（关键：所有需要更新下拉框的地方都用 return gr.update(...)）
        send_btn.click(
            fn=start_streaming,
            inputs=[user_input, chat_image_input, chat_history, model_choice,
                    temperature_slider, max_tokens_slider, gpu_layers_slider, vision_mode, thinking_mode_cb, session_state],
            outputs=[chat_history, status_display, session_state]
        ).then(lambda: ("", None), None, [user_input, chat_image_input])

        user_input.submit(
            fn=start_streaming,
            inputs=[user_input, chat_image_input, chat_history, model_choice,
                    temperature_slider, max_tokens_slider, gpu_layers_slider, vision_mode, thinking_mode_cb, session_state],
            outputs=[chat_history, status_display, session_state]
        ).then(lambda: ("", None), None, [user_input, chat_image_input])

        stop_btn.click(fn=stop_generation, inputs=[session_state], outputs=[status_display, session_state])
        clear_memory_btn.click(fn=clear_memory_and_reset, inputs=None, outputs=[memory_status, chat_history, status_display])
        refresh_btn.click(fn=refresh_status, inputs=None, outputs=[memory_status, status_display])
        export_btn.click(fn=export_memory_wrapper, inputs=[export_format], outputs=[status_display])
        open_export_dir_btn.click(fn=open_chat_export_dir, outputs=[status_display])

        update_personality_btn.click(fn=update_personality_from_preset, inputs=[personality_choice, chat_history], outputs=[chat_history])
        fill_preset_btn.click(fn=fill_preset_to_custom, inputs=[personality_choice], outputs=[custom_prompt_box])
        apply_prompt_btn.click(fn=apply_custom_prompt, inputs=[custom_prompt_box], outputs=[status_display])
        reset_prompt_btn.click(fn=reset_to_preset, outputs=[custom_prompt_box, status_display])
        refresh_models_btn.click(fn=refresh_model_list, outputs=[model_choice])
        check_service_btn.click(fn=check_service_and_refresh, outputs=[status_display, model_choice])

        preset_dropdown.change(fn=on_preset_select, inputs=[preset_dropdown],
                               outputs=[temperature_slider, max_tokens_slider, gpu_layers_slider, vision_mode, thinking_mode_cb])
        save_preset_btn.click(fn=save_current_preset,
                              inputs=[preset_name_box, temperature_slider, max_tokens_slider, gpu_layers_slider, vision_mode, thinking_mode_cb],
                              outputs=[preset_dropdown])
        delete_preset_btn.click(fn=delete_preset, inputs=[preset_dropdown], outputs=[preset_dropdown])
        export_presets_btn.click(fn=export_presets, outputs=[export_file])
        import_presets_btn.upload(fn=import_presets, inputs=[import_presets_btn], outputs=[preset_dropdown])

    # 返回组件引用，以便主界面可以使用 initial_refresh
    return model_choice, status_display

def create_converter_tab(ai_buddy):
    with gr.Tab("转换工具"):
        with gr.Tabs():
            with gr.Tab("聊天记录排版"):
                with gr.Row():
                    with gr.Column(scale=2):
                        extract_btn = gr.Button("提取最近助手回复", variant="secondary")
                        markdown_input = gr.Textbox(label="Markdown 内容 (可编辑)", lines=15, placeholder="点击按钮提取记忆中最新的助手回复")
                        target_export_format = gr.Dropdown(choices=["Microsoft Word (docx)", "PDF (需要 LaTeX)", "HTML", "Plain Text", "Markdown"], label="导出格式", value="Microsoft Word (docx)")
                        with gr.Row():
                            export_md_btn = gr.Button("开始导出", variant="primary")
                            open_md_dir_btn = gr.Button("打开导出目录", variant="secondary")
                    with gr.Column(scale=1):
                        export_status = gr.Textbox(label="导出状态", interactive=False, lines=3)
                        export_file_download = gr.File(label="下载导出的文件", visible=True)

                def get_last_memory_md():
                    memories = ai_buddy.memory_manager.memories
                    if not memories:
                        return "暂无记忆"
                    last = memories[-1]["ai"]
                    return strip_html_tags(last)

                extract_btn.click(fn=get_last_memory_md, outputs=[markdown_input])

                def handle_export(md_text, fmt):
                    if not md_text.strip():
                        return gr.File(visible=False), "请先提取或输入 Markdown 内容。"
                    path, msg = export_single_markdown(md_text, fmt)
                    if path:
                        return gr.File(value=path, visible=True), msg
                    else:
                        return gr.File(visible=False), msg

                export_md_btn.click(fn=handle_export, inputs=[markdown_input, target_export_format], outputs=[export_file_download, export_status])
                open_md_dir_btn.click(fn=open_chat_export_dir, outputs=[export_status])

            with gr.Tab("文档格式转换"):
                status_doc = gr.Textbox(label="Pandoc 状态", value=check_pandoc(), interactive=False)
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
                convert_btn.click(fn=convert_docs, inputs=[file_input, src_format, tgt_format, enable_toc, reference_doc], outputs=[output_msg])
                open_folder_btn.click(fn=open_output_folder, outputs=[output_msg])

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
                img_convert_btn.click(fn=convert_images_func, inputs=[img_input, img_format, img_quality], outputs=[img_output_msg])
                img_open_folder_btn.click(fn=open_image_folder, outputs=[img_output_msg])

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
                rename_btn.click(fn=batch_copy_rename, inputs=[rename_files, new_extension], outputs=[rename_result])
                rename_open_folder_btn.click(fn=open_rename_folder, outputs=[rename_result])

def create_online_tab():
    with gr.Tab("在线AI入口"):
        with gr.Column():
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
            status_trans = gr.Textbox(label="", value="等待操作...", interactive=False)

            gr.Markdown("---")
            gr.Markdown("### 提示词模板")
            prompt_selector = gr.Dropdown(
                label="选择提示词类型",
                choices=list(PROMPTS.keys()),
                value="SRT字幕翻译(保留时间码)",
            )
            prompt_display = gr.Textbox(
                label="提示词内容 (可直接编辑)",
                value=PROMPTS["SRT字幕翻译(保留时间码)"],
                lines=10,
                interactive=True,
            )
            gr.Markdown("💡 **使用方法**：选择模板 → 复制提示词 → 点击上方按钮打开网站 → 粘贴使用。")

            btn_deepl.click(fn=lambda: open_url(URLS["DeepL"]), outputs=status_trans)
            btn_youdao.click(fn=lambda: open_url(URLS["有道翻译"]), outputs=status_trans)
            btn_deepseek.click(fn=lambda: open_url(URLS["DeepSeek"]), outputs=status_trans)
            btn_doubao.click(fn=lambda: open_url(URLS["豆包"]), outputs=status_trans)
            btn_qianwen.click(fn=lambda: open_url(URLS["通义千问"]), outputs=status_trans)
            btn_kimi.click(fn=lambda: open_url(URLS["Kimi"]), outputs=status_trans)
            btn_chatglm.click(fn=lambda: open_url(URLS["ChatGLM"]), outputs=status_trans)
            btn_yuanbao.click(fn=lambda: open_url(URLS["腾讯元宝"]), outputs=status_trans)
            prompt_selector.change(fn=update_prompt, inputs=[prompt_selector], outputs=[prompt_display])

def create_interface():
    config = AppConfig.from_env()
    ai_buddy = AIBuddy(config)
    personality_config = ai_buddy.personality.config

    with gr.Blocks(title="AI助手 - 基于 llama.cpp", css=custom_css) as demo:
        gr.Markdown("# AI助手 - 本地聊天伙伴（支持图片）")
        with gr.Tabs():
            model_choice, status_display = create_chat_interface(ai_buddy, personality_config, config)
            create_converter_tab(ai_buddy)
            create_online_tab()

        gr.Markdown("---")
        gr.HTML("""
        <div style="text-align: center; color: #666; font-size: 0.9em;">
            <p>本工具基于 llama.cpp 后端，需要先运行 llama-server。</p>
            <p>启动命令示例：<code>llama-server.exe -m model.gguf --host 0.0.0.0 --port 8080</code></p>
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 15px; border-radius: 8px; margin: 15px auto; max-width: 600px;">
                <p style="color: white; font-weight: bold; margin: 5px 0;">🎬 更新请关注B站up主：光影的故事2018</p>
                <p style="color: white; margin: 5px 0;">
                    🔗 <strong>B站主页</strong>: <a href="https://space.bilibili.com/381518712" target="_blank" style="color: #ffdd40; text-decoration: none;">space.bilibili.com/381518712</a>
                </p>
            </div>
            <p>© 原创 WebUI 代码 © 2026 光影紐扣 版权所有</p>
        </div>
        """)

        # 启动时自动检查服务并填充模型列表（使用 gr.update）
        def initial_refresh():
            available, msg = ai_buddy.is_llama_available()
            models = ai_buddy.get_llama_models() if available else []
            if models:
                choices = [(f"{m} (多模态)" if ai_buddy.is_multimodal(m) else m, m) for m in models]
                default = choices[0][1]
            else:
                choices = [("请先启动 llama-server", "none")]
                default = "none"
            status_text = f"✅ 服务正常，模型已加载" if available else f"❌ {msg}"
            return status_text, gr.update(choices=choices, value=default)

        demo.load(fn=initial_refresh, outputs=[status_display, model_choice])

    return demo, ai_buddy

# ========== 健康检查 ==========
def check_llama_health() -> Tuple[bool, str]:
    try:
        response = requests.get("http://127.0.0.1:8080/v1/models", timeout=5)
        if response.status_code != 200:
            return False, "llama.cpp 服务返回非200状态码"
        data = response.json()
        models = data.get("data", [])
        if not models:
            return True, "llama.cpp 服务正在运行，但未检测到任何模型\n请使用 --model 参数指定模型文件"
        return True, f"llama.cpp 服务正常，已加载 {len(models)} 个模型"
    except requests.exceptions.ConnectionError:
        return False, "无法连接到 llama.cpp 服务，请确保已运行 llama-server"
    except requests.exceptions.Timeout:
        return False, "连接 llama.cpp 服务超时"
    except Exception as e:
        return False, f"检查服务时出错：{str(e)}"

# ========== 主程序 ==========
def main():
    print("=" * 60)
    print("正在启动AI助手 - 基于 llama.cpp 后端（聊天 + 转换工具 + 在线入口）...")
    print("=" * 60)

    is_healthy, health_msg = check_llama_health()
    print(health_msg)

    if not is_healthy:
        print("\n请先运行 llama.cpp 服务：")
        print("1. 打开新的命令行窗口")
        print("2. 运行命令示例：llama-server.exe -m your_model.gguf --host 0.0.0.0 --port 8080")
        print("3. 等待服务启动后，重新运行本程序")
        if "未检测到任何模型" in health_msg:
            print("\n需要指定模型文件：")
            print("llama-server.exe -m <模型路径>.gguf --host 0.0.0.0 --port 8080")
        input("\n按Enter键退出...")
        return

    demo, ai_buddy = create_interface()
    print("\nAI助手启动成功！")
    print("浏览器即将打开，如果未自动打开，请访问：http://localhost:7861")
    print("=" * 60)

    try:
        demo.launch(
            server_name="127.0.0.1",
            server_port=7861,
            share=False,
            inbrowser=True
        )
    except Exception as e:
        print(f"启动失败: {e}")
        print("尝试使用其他端口...")
        try:
            demo.launch(
                server_name="127.0.0.1",
                server_port=7866,
                inbrowser=True
            )
        except Exception as e2:
            print(f"再次启动失败: {e2}")
            print("请检查端口7861和7866是否被占用")
            input("按Enter键退出...")
    finally:
        if hasattr(ai_buddy, 'memory_manager'):
            ai_buddy.memory_manager.shutdown()

if __name__ == "__main__":
    # 如果 presets.json 不存在，则创建默认预设
    if not PRESETS_FILE.exists():
        default_presets = [
            {
                "name": "高速纯文本",
                "temperature": 0.7,
                "max_tokens": 2048,
                "gpu_layers": 32,
                "vision_mode": "禁用多模态",
                "thinking_mode": False
            },
            {
                "name": "多模态看图",
                "temperature": 0.8,
                "max_tokens": 4096,
                "gpu_layers": 99,
                "vision_mode": "自动（跟随 GPU 层数）",
                "thinking_mode": True
            }
        ]
        save_presets_to_file(default_presets)
    main()