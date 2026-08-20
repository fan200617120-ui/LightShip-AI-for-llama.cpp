#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DeepSeek-OCR 独立 WebUI (适配 llama.cpp 服务器)
支持多种识别指令，纯文本输出，可下载为 TXT、Word 或 CSV 文件
"""

import gradio as gr
import requests
import base64
import os
import tempfile
import io
import csv
from PIL import Image
from docx import Document

# ========== 配置（适配您的 llama.cpp 环境） ==========
LLAMA_URL = "http://127.0.0.1:60115/v1/chat/completions"   # llamacpp 的 OpenAI 兼容端点
MODEL_NAME = "DeepSeek-OCR-2"                              # 日志中显示的模型名
MAX_TOKENS = 1024
TEMPERATURE = 0.0

PRESET_INSTRUCTIONS = {
    "提取文字": "Extract the text in the image:",
    "自由识别": "Free OCR:",
    "解析图片": "Parse the figure.",
    "转Markdown": "Convert the document to markdown.",
    "图文排版": "<|grounding|>Given the layout of the image."
}

# ========== 函数定义 ==========
def encode_image(image):
    if isinstance(image, str):
        with open(image, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    elif isinstance(image, Image.Image):
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    else:
        raise ValueError("不支持的图片类型")

def recognize(image, instruction):
    if image is None:
        return "请上传图片", ""
    if not instruction.strip():
        instruction = "Extract the text in the image:"

    image_b64 = encode_image(image)
    # 构建 OpenAI 兼容的多模态请求
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
                ]
            }
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "stream": False
    }

    try:
        resp = requests.post(LLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        result = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        result = result.strip() if result else "未识别到内容"
        return result, result
    except requests.exceptions.ConnectionError:
        err = "连接失败，请确保 llama.cpp 服务器已启动 (运行 server.bat)"
        return err, err
    except Exception as e:
        err = f"错误: {str(e)}"
        return err, err

def download_txt(text):
    if not text or text.startswith(("连接失败", "错误:", "请上传图片", "未识别到内容")):
        return None
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        f.write(text)
        return f.name

def download_docx(text):
    if not text or text.startswith(("连接失败", "错误:", "请上传图片", "未识别到内容")):
        return None
    doc = Document()
    doc.add_paragraph(text)
    with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as tmp:
        doc.save(tmp.name)
        return tmp.name

def download_csv(text):
    """将识别结果保存为 CSV 文件（整个文本作为一个单元格）"""
    if not text or text.startswith(("连接失败", "错误:", "请上传图片", "未识别到内容")):
        return None
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8', newline='') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow([text])
        return f.name

# ========== 界面构建 ==========
with gr.Blocks(title="DeepSeek-OCR 识别工具 (llama.cpp)") as demo:
    gr.Markdown("# DeepSeek-OCR 识别工具 (llama.cpp 后端)")
    gr.Markdown("上传图片，选择或输入指令，点击识别即可提取内容。识别后可下载为 TXT、Word 或 CSV 文件。")

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="上传图片", height=300)
            instruction_dropdown = gr.Dropdown(
                choices=list(PRESET_INSTRUCTIONS.keys()),
                value="提取文字",
                label="指令模板"
            )
            instruction_text = gr.Textbox(
                label="自定义指令",
                value=PRESET_INSTRUCTIONS["提取文字"],
                lines=2
            )
            instruction_dropdown.change(
                fn=lambda x: PRESET_INSTRUCTIONS[x],
                inputs=instruction_dropdown,
                outputs=instruction_text
            )
            recognize_btn = gr.Button("开始识别", variant="primary")

        with gr.Column(scale=1):
            output_text = gr.Textbox(label="识别结果", lines=23, interactive=False)
            with gr.Row():
                btn_txt = gr.Button("下载为 TXT", variant="secondary")
                btn_docx = gr.Button("下载为 Word", variant="secondary")
                btn_csv = gr.Button("下载为 CSV", variant="secondary")
            # 用于存放临时文件路径的隐藏组件
            file_txt = gr.File(label="下载 TXT", visible=False)
            file_docx = gr.File(label="下载 Word", visible=False)
            file_csv = gr.File(label="下载 CSV", visible=False)

    # 识别：将结果存入 output_text，同时传给下载文件组件（用于后续生成）
    recognize_btn.click(
        fn=recognize,
        inputs=[image_input, instruction_text],
        outputs=[output_text, output_text]
    )

    # 下载：点击按钮后生成临时文件，并显示对应的下载组件
    btn_txt.click(
        fn=download_txt,
        inputs=output_text,
        outputs=file_txt
    ).then(
        lambda: gr.update(visible=True),
        None,
        file_txt
    )

    btn_docx.click(
        fn=download_docx,
        inputs=output_text,
        outputs=file_docx
    ).then(
        lambda: gr.update(visible=True),
        None,
        file_docx
    )

    btn_csv.click(
        fn=download_csv,
        inputs=output_text,
        outputs=file_csv
    ).then(
        lambda: gr.update(visible=True),
        None,
        file_csv
    )

    # 使用说明
    gr.Markdown("""
    ### 使用说明
    - 确保 llama.cpp 服务器已启动（运行 `server.bat` 或 `llama-server`）
    - 默认监听 `http://127.0.0.1:60115`，模型为 `DeepSeek-OCR-2`
    - 上传图片，选择或输入指令（常用指令已预设）
    - 点击识别，结果将显示在右侧
    - 识别成功后，点击“下载为 TXT”、“下载为 Word”或“下载为 CSV”即可保存文件
    """)

    # ---------- 页脚 ----------
    gr.Markdown("---")
    gr.HTML("""
    <div class="notice">
        注意事项：<br>
        • 本工具仅用于个人学习与视频剪辑使用<br>
        • 禁止用于商业用途及侵权行为<br>            
        • 使用前确保模型与依赖环境正常配置
    </div>
    <div style="text-align: center; color: #666; font-size: 0.9em;">
        <p>本软件包不提供任何模型文件，模型由用户自行从官方渠道获取。用户需自行遵守模型的原许可证。</p>
        <p>本软件包按“原样”提供，不提供任何明示或暗示的担保。使用本软件所产生的一切风险由用户自行承担。</p>
        <p>本软件包开发者不对因使用本软件而导致的任何直接或间接损失负责。</p> 
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 15px; border-radius: 8px; margin: 15px auto; max-width: 600px;">
            <p style="color: white; font-weight: bold; margin: 5px 0; font-size: 1em;">🎬 更新请关注B站up主：光影的故事2018</p>
            <p style="color: white; margin: 5px 0; font-size: 0.9em;">
                🔗 <strong>B站主页</strong>: <a href="https://space.bilibili.com/381518712" target="_blank" style="color: #ffdd40; text-decoration: none; font-weight: bold;">
                    space.bilibili.com/381518712
                </a>
            </p>
        </div>
    </div>
    <div style="text-align: center; color: #666; margin-top: 10px; font-size: 0.9em;">
        © 原创 WebUI 代码 © 2026 光影紐扣 版权所有  |  轻舟渡万境，一智载千寻。 One Ship, All Souls. One AI, All Minds.
    </div>
    """)

# ---------- 启动 ----------
if __name__ == "__main__":
    print("=" * 60)
    print("启动 DeepSeek-OCR 识别工具 (llama.cpp 后端)")
    print(f"API 地址: {LLAMA_URL}")
    print(f"模型名称: {MODEL_NAME}")
    print("⚠️ 请确保 llama.cpp 服务器已启动 (server.bat)")
    print("=" * 60)

    ports_to_try = [7863, 7961, 7861, 7862, 7960]
    launched = False
    for port in ports_to_try:
        try:
            demo.launch(
                server_name="127.0.0.1",
                server_port=port,
                share=False,
                inbrowser=True,
                theme=gr.themes.Default()
            )
            launched = True
            break
        except OSError as e:
            err_str = str(e)
            if ("Address already in use" in err_str or
                "端口" in err_str or
                "Cannot find empty port" in err_str):
                print(f"端口 {port} 被占用，尝试下一个...")
                continue
            else:
                raise e
    if not launched:
        print("所有尝试的端口均被占用，请手动指定空闲端口。")