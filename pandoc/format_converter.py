#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
format_converter.py - 轻舟 AI 工具箱 (v2.7)
修复日志：
- 修复导出 Word 时表格挤压问题（自动清洗 Markdown 表格分隔行）
- 启动时自动探测空闲端口并打开浏览器，避免端口冲突
- _fix_row 语法修复 + 仅处理纯分隔符单元格，避免误删内容
- _resolve_file_path 只信任 path 字段，避免相对路径陷阱
- YAML 清洗限制在文件开头，防止误删正文中的 ---
- convert_files fallback 仅对纯文本启用，避免格式降级
- WebP 转换模式判断修正
- 图片处理增加上下文管理器，避免资源泄漏
- 中文字体检测兼容无 fc-list 的环境
- Gradio 文件对象兼容性增强（支持 file 属性）
- 降级保存错误提示 None 保护
Copyright 2026 光影的故事2018
"""

import os
import subprocess
import gradio as gr
from pathlib import Path
import webbrowser
from PIL import Image, ImageOps
import shutil
import tempfile
import re
import json
import socket
import threading
from datetime import datetime
from typing import Tuple, Optional, Dict, List

# ================= 路径配置 =================
SCRIPT_DIR = Path(__file__).parent.resolve() if "__file__" in dir() else Path.cwd()
PANDOC_PROMPTS_FILE = SCRIPT_DIR / "pandoc_prompts.json"
PANDOC_EXE_FILE = SCRIPT_DIR / "pandoc.exe"       # format_converter.py 本身就在 pandoc 目录下，所以正确
OUTPUT_DIR = SCRIPT_DIR.parent / "output"
IMAGE_OUTPUT_DIR = OUTPUT_DIR / "images"
RENAME_OUTPUT_DIR = OUTPUT_DIR / "renamed"
CHAT_EXPORT_DIR = OUTPUT_DIR / "chat_exports"
TEMPLATE_DIR = SCRIPT_DIR / "templates"

# ================= 文档格式定义 =================
ALL_FORMATS = {
    "Markdown": ".md",
    "Microsoft Word (docx)": ".docx",
    "HTML": ".html",
    "Plain Text": ".txt",
    "reStructuredText": ".rst",
    "EPUB": ".epub",
    "LaTeX": ".tex",
    "OpenDocument": ".odt",
    "Rich Text Format": ".rtf",
    "PDF (需要 LaTeX)": ".pdf",
}

SOURCE_FORMATS = {k: v for k, v in ALL_FORMATS.items() if k != "PDF (需要 LaTeX)"}
TARGET_FORMATS = ALL_FORMATS

FORMAT_ALIASES = {
    ".md": "markdown",
    ".docx": "docx",
    ".html": "html",
    ".txt": "plain",
    ".rst": "rst",
    ".epub": "epub",
    ".tex": "latex",
    ".odt": "odt",
    ".rtf": "rtf",
    ".pdf": "pdf",
}

IMAGE_FORMATS = ["PNG", "JPEG", "WebP", "BMP", "TIFF"]
IMAGE_EXT_MAP = {"PNG": ".png", "JPEG": ".jpg", "WebP": ".webp", "BMP": ".bmp", "TIFF": ".tiff"}

CHINESE_FONTS = ["SimSun", "Microsoft YaHei", "Noto Serif CJK SC", "Noto Sans CJK SC"]

# ================= 在线AI网址 =================
URLS = {
    "DeepL": "https://www.deepl.com/translator",
    "有道翻译": "https://fanyi.youdao.com/",
    "DeepSeek": "https://chat.deepseek.com/",
    "豆包": "https://www.doubao.com/chat/",
    "通义千问": "https://tongyi.aliyun.com/qianwen/",
    "Kimi": "https://kimi.moonshot.cn/",
    "ChatGLM": "https://chatglm.cn/",
    "腾讯元宝": "https://yuanbao.tencent.com/",
}

# ================= 动态提示词配置加载 =================
def load_prompts_from_json() -> Dict:
    if not PANDOC_PROMPTS_FILE.exists():
        return {}
    try:
        with open(PANDOC_PROMPTS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if "分类目录" in data and "角色库" in data:
            return data
        else:
            return {}
    except Exception:
        return {}

PROMPTS_CONFIG = load_prompts_from_json()

def build_category_roles_mapping() -> Dict[str, List[Tuple[str, str]]]:
    mapping = {}
    category_list = PROMPTS_CONFIG.get("分类目录", {})
    role_lib = PROMPTS_CONFIG.get("角色库", {})
    for cat_name in category_list.keys():
        roles = role_lib.get(cat_name, {})
        role_pairs = [(role_id, role_info.get("角色名称", role_id)) for role_id, role_info in roles.items()]
        mapping[cat_name] = role_pairs
    return mapping

CATEGORY_ROLES_MAP = build_category_roles_mapping()
CATEGORY_NAMES = list(CATEGORY_ROLES_MAP.keys())

def get_role_info(category: str, role_id: str) -> dict:
    roles = PROMPTS_CONFIG.get("角色库", {}).get(category, {})
    return roles.get(role_id, {})

# ================= 文件处理辅助 =================
def open_url(url):
    webbrowser.open(url)
    return f"已在浏览器中打开 {url}"

def _resolve_file_path(file_obj):
    """兼容 Gradio 6.x 返回的各种文件对象，只信任 path 字段，避免相对路径陷阱 [FIX]"""
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
        # 某些 Gradio 版本返回的文件对象带有 file 属性（例如 SpooledTemporaryFile）
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

def _open_folder(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    try:
        if os.name == "nt":
            os.startfile(str(path))
        elif os.name == "posix":
            subprocess.run(["xdg-open", str(path)], check=False)
        else:
            webbrowser.open(str(path))
    except Exception:
        webbrowser.open(str(path))
    return f"已打开文件夹：{path}"

def open_output_folder():
    return _open_folder(OUTPUT_DIR)

# ================= Pandoc 相关 =================
def _find_pandoc_executable():
    if PANDOC_EXE_FILE.exists():
        return PANDOC_EXE_FILE
    system_path = shutil.which("pandoc")
    if system_path:
        return Path(system_path)
    return None

def check_pandoc():
    pandoc_exec = _find_pandoc_executable()
    if not pandoc_exec:
        return False, f"未找到 Pandoc：{PANDOC_EXE_FILE}，也未在 PATH 中发现 pandoc。"
    try:
        result = subprocess.run(
            [str(pandoc_exec), "--version"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            return True, f"Pandoc 已就绪：{version_line}"
        else:
            return False, f"Pandoc 执行异常：{result.stderr}"
    except Exception as e:
        return False, f"检查 Pandoc 失败：{str(e)}"

def detect_chinese_font():
    if os.name == "nt":
        return "SimSun"
    # 确保 fc-list 命令存在 [FIX]
    if not shutil.which("fc-list"):
        return None
    for font in CHINESE_FONTS:
        try:
            result = subprocess.run(["fc-list", f":family={font}"], capture_output=True, text=True)
            if result.returncode == 0 and result.stdout.strip():
                return font
        except Exception:
            continue
    return None

def _try_pandoc_convert(pandoc_exe, input_path, output_path, reader, writer, extra_args, timeout=60):
    cmd = [
        str(pandoc_exe), str(input_path),
        "-f", reader, "-t", writer, "-o", str(output_path),
        "--wrap=preserve"
    ] + extra_args
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=True)
        return True, ""
    except subprocess.CalledProcessError as e:
        return False, e.stderr

# ================= 文档转换 =================
def convert_files(files, source_format, target_format, enable_toc=False, reference_doc=None):
    if not files:
        return "请先上传至少一个文件。", ""

    if source_format == "PDF (需要 LaTeX)":
        return "PDF 不能作为源格式，仅支持将其他格式转换为 PDF。请选择正确的源格式。", ""

    pandoc_exec = _find_pandoc_executable()
    if not pandoc_exec:
        return "未找到 Pandoc，可将 pandoc.exe 放置于当前目录或确保 pandoc 已加入系统 PATH。", ""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    src_ext = SOURCE_FORMATS.get(source_format, "")
    tgt_ext = TARGET_FORMATS.get(target_format, "")
    if not src_ext or not tgt_ext:
        return "不支持的格式组合。", ""

    writer = FORMAT_ALIASES.get(tgt_ext, tgt_ext.strip().lstrip('.'))

    pdf_engine = None
    if tgt_ext == ".pdf":
        for eng in ["xelatex", "pdflatex", "lualatex"]:
            try:
                subprocess.run([eng, "--version"], capture_output=True, timeout=2, check=True)
                pdf_engine = eng
                break
            except Exception:
                continue
        if pdf_engine is None:
            return (
                "转换为 PDF 需要安装 LaTeX 引擎（MiKTeX 或 TeX Live）。"
                "若仅需简单的 PDF 转换（无复杂排版/公式），可先将文档转换为 HTML，"
                "再通过浏览器「打印为 PDF」；或使用 Pandoc+wkhtmltopdf（需额外配置），"
                "但兼容性不如 LaTeX 引擎。", ""
            )

    extra_args = []
    if tgt_ext == ".pdf":
        extra_args.extend(["--pdf-engine", pdf_engine])
        font = detect_chinese_font()
        if font and pdf_engine in ["xelatex", "lualatex"]:
            extra_args.extend(["-V", f"mainfont={font}"])
    elif tgt_ext == ".docx":
        if enable_toc:
            extra_args.append("--toc")
        ref_path = _resolve_file_path(reference_doc)
        if ref_path and ref_path.exists() and ref_path.suffix.lower() == '.docx':
            extra_args.extend(["--reference-doc", str(ref_path)])

    success_count = 0
    fail_msgs = []
    output_paths = []

    for file_obj in files:
        input_path = _resolve_file_path(file_obj)
        if input_path is None or not input_path.exists():
            continue
        input_filename = Path(_get_original_filename(file_obj) or input_path.name)
        output_name = input_filename.stem + tgt_ext
        output_path = OUTPUT_DIR / output_name

        counter = 1
        while output_path.exists():
            output_path = OUTPUT_DIR / f"{input_filename.stem}_{counter}{tgt_ext}"
            counter += 1

        default_reader = FORMAT_ALIASES.get(src_ext, src_ext.strip().lstrip('.'))
        success, error_msg = _try_pandoc_convert(pandoc_exec, input_path, output_path, default_reader, writer, extra_args)

        # Markdown 文件尝试去除 YAML 头部后重试
        if not success and src_ext == ".md" and ("YAML" in error_msg or "metadata" in error_msg):
            try:
                content = Path(input_path).read_text(encoding="utf-8")
                # 限定在文件开头剥离 YAML [FIX]
                stripped = re.sub(r'(?s)\A\s*---\n.*?\n---\n', '', content)
                if stripped.strip():
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as tmp:
                        tmp.write(stripped)
                        tmp_path = tmp.name
                    try:
                        success, error_msg = _try_pandoc_convert(pandoc_exec, tmp_path, output_path, default_reader, writer, extra_args)
                    finally:
                        os.unlink(tmp_path)
            except Exception:
                pass

        # 最终兜底：仅当目标为纯文本时降级 reader [FIX]
        if not success and tgt_ext == ".txt":
            success, _ = _try_pandoc_convert(pandoc_exec, input_path, output_path, "plain", writer, extra_args)

        if success:
            success_count += 1
            output_paths.append(str(output_path))
        else:
            fail_msgs.append(f"{input_filename.name}: {error_msg[:200] if error_msg else '未知错误'}")

    msg = f"成功转换 {success_count} 个文件"
    if fail_msgs:
        msg += f"\n失败 {len(fail_msgs)} 个文件：\n" + "\n".join(fail_msgs)
    msg += f"\n输出目录：{OUTPUT_DIR}"
    return msg, "\n".join(output_paths)

# ================= 图片转换 =================
def convert_images(files, target_format, quality=85):
    if not files:
        return "请先上传至少一张图片。", ""
    if target_format not in IMAGE_EXT_MAP:
        return f"不支持的图片格式：{target_format}", ""
    IMAGE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    target_ext = IMAGE_EXT_MAP[target_format]
    success_count = 0
    fail_msgs = []
    output_paths = []

    for file_obj in files:
        input_path = _resolve_file_path(file_obj)
        input_filename = Path(_get_original_filename(file_obj) or (input_path.name if input_path else "unknown"))
        try:
            if not input_path or not input_path.exists():
                raise FileNotFoundError("源文件不存在")
            with Image.open(input_path) as img:                # [FIX] 使用上下文管理器
                img = ImageOps.exif_transpose(img)

                output_name = input_filename.stem + target_ext
                output_path = IMAGE_OUTPUT_DIR / output_name
                counter = 1
                while output_path.exists():
                    output_path = IMAGE_OUTPUT_DIR / f"{input_filename.stem}_{counter}{target_ext}"
                    counter += 1

                if target_format == "JPEG":
                    if img.mode in ("RGBA", "LA", "P"):
                        if img.mode == "P":
                            img = img.convert("RGBA")
                        elif img.mode == "LA":
                            img = img.convert("RGBA")
                        background = Image.new("RGB", img.size, (255, 255, 255))
                        background.paste(img, mask=img.split()[-1])
                        img = background
                    elif img.mode not in ("RGB", "L"):
                        img = img.convert("RGB")
                    img.save(output_path, format="JPEG", quality=quality)
                elif target_format == "WebP":
                    # [FIX] 修正模式判断逻辑
                    if img.mode not in ("RGB", "RGBA"):
                        img = img.convert("RGB")
                    img.save(output_path, format="WEBP", quality=quality)
                else:
                    img.save(output_path, format=target_format)

            success_count += 1
            output_paths.append(str(output_path))
        except Exception as e:
            fail_msgs.append(f"{input_filename.name}: {str(e)}")

    msg = f"成功转换 {success_count} 张图片"
    if fail_msgs:
        msg += f"\n失败 {len(fail_msgs)} 张图片：\n" + "\n".join(fail_msgs)
    msg += f"\n输出目录：{IMAGE_OUTPUT_DIR}"
    return msg, "\n".join(output_paths)

# ================= 批量复制重命名 =================
def batch_copy_rename(files, new_ext):
    if not files:
        return "请先上传至少一个文件。"
    if not new_ext:
        return "请输入新扩展名。"
    new_ext = new_ext.strip()
    if not new_ext.startswith('.'):
        new_ext = '.' + new_ext
    RENAME_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    success_count = 0
    fail_msgs = []

    for file_obj in files:
        input_path = _resolve_file_path(file_obj)
        original_name = _get_original_filename(file_obj) or (input_path.name if input_path else "unknown")
        if not input_path or not input_path.exists():
            fail_msgs.append(f"{original_name}: 文件路径无效或不存在")
            continue
        original_path = Path(original_name)
        new_name = original_path.stem + new_ext
        new_path = RENAME_OUTPUT_DIR / new_name

        counter = 1
        while new_path.exists():
            new_name = f"{original_path.stem}_{counter}{new_ext}"
            new_path = RENAME_OUTPUT_DIR / new_name
            counter += 1

        try:
            shutil.copy2(input_path, new_path)
            success_count += 1
        except Exception as e:
            fail_msgs.append(f"{original_name}: {str(e)}")

    msg = f"成功复制并重命名 {success_count} 个文件"
    if fail_msgs:
        msg += f"\n失败 {len(fail_msgs)} 个：\n" + "\n".join(fail_msgs)
    msg += f"\n输出目录：{RENAME_OUTPUT_DIR}"
    return msg

def open_rename_folder():
    return _open_folder(RENAME_OUTPUT_DIR)

#================= 清洗数据 =================
def fix_markdown_table_separator(text: str) -> str:
    """
    修复 Markdown 表格分隔行：
    1. 将全角破折号 '—' 替换为 '-'
    2. 确保分隔行中每个单元格至少有 '---'
    3. 仅处理纯符号单元格，避免误删内容
    """
    def _fix_row(row: str) -> str:
        row = row.replace('—', '-')
        if re.match(r'^\|?[\s\-:]*\|', row):
            cells = row.split('|')
            fixed_cells = []
            for cell in cells:
                cell = cell.strip()
                if not cell:
                    fixed_cells.append('')
                    continue
                # 只处理纯分隔符单元格 [FIX]
                if not re.match(r'^[\-:\s]+$', cell):
                    fixed_cells.append(cell)
                    continue
                left_colon = cell.startswith(':')
                right_colon = cell.endswith(':')
                new_cell = '-' * 3
                if left_colon:
                    new_cell = ':' + new_cell
                if right_colon:
                    new_cell = new_cell + ':'
                fixed_cells.append(new_cell)
            row = '|'.join(fixed_cells)
        return row

    lines = text.split('\n')
    new_lines = []
    for line in lines:
        if re.search(r'^\s*\|', line) and re.search(r'[-:\—]', line):
            line = _fix_row(line)
        new_lines.append(line)
    return '\n'.join(new_lines)

# ================= 排版导出 =================
def export_content_to_format(content: str, target_format: str, template_file=None):
    if not content.strip():
        return None, "内容为空，无法导出。"
    CHAT_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    pandoc_exec = _find_pandoc_executable()
    if not pandoc_exec:
        return None, "未找到 Pandoc，可将 pandoc.exe 放置于当前目录或确保 pandoc 已加入系统 PATH。"

    pdf_engine = None
    if target_format == "PDF":
        for eng in ["xelatex", "pdflatex", "lualatex"]:
            try:
                subprocess.run([eng, "--version"], capture_output=True, timeout=2, check=True)
                pdf_engine = eng
                break
            except Exception:
                continue
        if pdf_engine is None:
            return None, "转换为 PDF 需要安装 LaTeX 引擎（MiKTeX 或 TeX Live）。"

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
    output_path = CHAT_EXPORT_DIR / f"export_{timestamp}{tgt_ext}"

    def run_pandoc(md_text: str, reader: str = "markdown+hard_line_breaks-yaml_metadata_block") -> Tuple[Optional[str], Optional[str]]:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
            f.write(md_text)
            temp_in = f.name

        extra_args = []
        if target_format == "PDF":
            extra_args = ["--pdf-engine", pdf_engine]
            font = detect_chinese_font()
            if font and pdf_engine in ["xelatex", "lualatex"]:
                extra_args.extend(["-V", f"mainfont={font}"])
        if target_format == "Microsoft Word (docx)" and template_file is not None:
            ref_path = _resolve_file_path(template_file)
            if ref_path and ref_path.exists() and ref_path.suffix.lower() == '.docx':
                extra_args.extend(["--reference-doc", str(ref_path)])

        cmd = [
            str(pandoc_exec), temp_in,
            "-f", reader, "-t", writer,
            "-o", str(output_path), "--wrap=preserve"
        ] + extra_args

        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=True)
            os.unlink(temp_in)
            return str(output_path), None
        except subprocess.CalledProcessError as e:
            os.unlink(temp_in)
            return None, e.stderr
        except Exception as e:
            os.unlink(temp_in)
            return None, str(e)

    # 导出前清洗表格分隔行
    cleaned_content = fix_markdown_table_separator(content)
    out_path, error = run_pandoc(cleaned_content)

    if out_path:
        return out_path, f"导出成功：{out_path}"

    # YAML 头部重试
    if error and ("YAML" in error or "metadata" in error):
        stripped_content = re.sub(r'(?s)\A\s*---\n.*?\n---\n', '', content)  # [FIX] 限定开头
        if stripped_content.strip():
            cleaned_stripped = fix_markdown_table_separator(stripped_content)
            out_path2, error2 = run_pandoc(cleaned_stripped, reader="markdown+hard_line_breaks")
            if out_path2:
                return out_path2, f"导出成功：{out_path2} (已自动忽略 YAML 头部)"
            error = error2

    try:
        txt_output_path = CHAT_EXPORT_DIR / f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(txt_output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return str(txt_output_path), f"Pandoc 转换失败，已降级保存为纯文本：{txt_output_path}"
    except Exception as e:
        return None, f"所有尝试均失败。最后错误：{error or '未知'}；纯文本保存失败：{e}"

def open_chat_export_dir():
    return _open_folder(CHAT_EXPORT_DIR)

def open_template_dir():
    TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
    return _open_folder(TEMPLATE_DIR)

# ================= 端口自动探测与浏览器启动 =================
def find_free_port(start=7969):
    """使用 bind 方式寻找空闲端口"""
    port = start
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('127.0.0.1', port))
                return port
            except OSError:
                port += 1

def open_browser(port):
    webbrowser.open(f"http://127.0.0.1:{port}")

# ================= Gradio 界面 =================
initial_pandoc_status = check_pandoc()[1]

if CATEGORY_NAMES:
    initial_category = CATEGORY_NAMES[0]
    initial_roles = CATEGORY_ROLES_MAP[initial_category]
    first_role_id = initial_roles[0][0] if initial_roles else ""
    first_role_info = get_role_info(initial_category, first_role_id)
else:
    initial_category = ""
    initial_roles = []
    first_role_id = ""
    first_role_info = {}

with gr.Blocks(title="轻舟 AI 工具箱") as demo:
    gr.Markdown("# 轻舟 AI 工具箱")
    gr.Markdown("### 排版导出 · 格式转换 · 图片转换 · 批量改名 · 在线AI入口（动态模板）")

    with gr.Tabs():
        # ---------- Tab1: 排版导出 ----------
        with gr.Tab("排版导出"):
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### 粘贴 Markdown 内容并导出为多种格式")
                    content_input = gr.Textbox(label="Markdown 内容", lines=20, placeholder="在此粘贴 Markdown 文本...")
                    with gr.Accordion("👁️ 实时预览 (点击展开)", open=False):
                        preview_box = gr.Markdown(value="*输入内容后将在此处显示预览...*")
                    export_format = gr.Dropdown(
                        choices=["Microsoft Word (docx)", "HTML", "Plain Text", "Markdown", "PDF"],
                        label="导出格式", value="Microsoft Word (docx)"
                    )
                    with gr.Row():
                        export_btn = gr.Button("开始导出", variant="primary", scale=3)
                        open_export_dir_btn = gr.Button("打开导出目录", variant="secondary", scale=1)
                with gr.Column(scale=1):
                    gr.Markdown("### 高级选项")
                    template_file = gr.File(label="参考模板 (可选 .docx)", file_types=[".docx"], value=None)
                    open_template_btn = gr.Button("打开模板目录", variant="secondary")
                    export_status = gr.Textbox(label="导出状态", interactive=False, lines=3)
                    export_file_download = gr.File(label="下载导出的文件", visible=True)

        def handle_export(content, fmt, template):
            if not content.strip():
                return gr.update(visible=False), "请先输入 Markdown 内容。"
            path, msg = export_content_to_format(content, fmt, template)
            if path:
                return gr.update(value=path, visible=True), msg
            else:
                return gr.update(visible=False), msg

        export_btn.click(fn=handle_export, inputs=[content_input, export_format, template_file], outputs=[export_file_download, export_status])
        open_export_dir_btn.click(fn=open_chat_export_dir, outputs=[export_status])
        open_template_btn.click(fn=open_template_dir, outputs=[export_status])
        content_input.change(fn=lambda txt: txt.strip() or "*内容为空*", inputs=content_input, outputs=preview_box)

        # ---------- Tab2: 文档转换 ----------
        with gr.Tab("文档格式转换"):
            status = gr.Textbox(label="Pandoc 状态", value=initial_pandoc_status, interactive=False)
            with gr.Row():
                with gr.Column(scale=2):
                    file_input = gr.File(label="上传文件（可多选）", file_count="multiple")
                    src_format = gr.Dropdown(choices=list(SOURCE_FORMATS.keys()), label="源格式", value="Markdown")
                    tgt_format = gr.Dropdown(choices=list(TARGET_FORMATS.keys()), label="目标格式", value="Microsoft Word (docx)")
                    with gr.Accordion("高级选项", open=False):
                        enable_toc = gr.Checkbox(label="为 Word 添加目录 (--toc)", value=False)
                        reference_doc_acc = gr.File(label="参考样式模板 (仅 Word，可选 .docx)", file_types=[".docx"])
                    with gr.Row():
                        convert_btn = gr.Button("开始批量转换", variant="primary", scale=3)
                        open_folder_btn = gr.Button("打开输出目录", variant="secondary", scale=1)
                with gr.Column(scale=1):
                    output_msg = gr.Textbox(label="转换结果", interactive=False, lines=5)
                    output_files_list = gr.Textbox(label="输出文件列表", interactive=False, lines=5)

        def convert_files_wrapper(files, src_fmt, tgt_fmt, enable_toc, ref_acc):
            return convert_files(files, src_fmt, tgt_fmt, enable_toc, ref_acc)

        convert_btn.click(fn=convert_files_wrapper, inputs=[file_input, src_format, tgt_format, enable_toc, reference_doc_acc], outputs=[output_msg, output_files_list])
        open_folder_btn.click(fn=open_output_folder, outputs=output_msg)

        # ---------- Tab3: 图片转换 ----------
        with gr.Tab("图片格式转换"):
            with gr.Row():
                with gr.Column(scale=2):
                    img_input = gr.File(label="上传图片（可多选）", file_count="multiple")
                    img_format = gr.Dropdown(choices=IMAGE_FORMATS, label="输出格式", value="PNG")
                    img_quality = gr.Slider(minimum=1, maximum=100, value=85, step=1, label="图片质量 (仅对 JPEG/WebP 有效)")
                    with gr.Row():
                        img_convert_btn = gr.Button("开始转换图片", variant="primary", scale=3)
                        img_open_folder_btn = gr.Button("打开图片输出目录", variant="secondary", scale=1)
                with gr.Column(scale=1):
                    img_output_msg = gr.Textbox(label="转换结果", interactive=False, lines=6)
                    img_output_files_list = gr.Textbox(label="输出文件列表", interactive=False, lines=5)

        img_convert_btn.click(fn=convert_images, inputs=[img_input, img_format, img_quality], outputs=[img_output_msg, img_output_files_list])
        img_open_folder_btn.click(fn=lambda: _open_folder(IMAGE_OUTPUT_DIR), outputs=img_output_msg)

        # ---------- Tab4: 批量复制并重命名 ----------
        with gr.Tab("批量复制并重命名"):
            with gr.Row():
                with gr.Column(scale=2):
                    rename_files = gr.File(label="选择文件（可多选）", file_count="multiple")
                    new_extension = gr.Textbox(label="新扩展名（例如 .txt 或 txt）", placeholder=".md")
                    with gr.Row():
                        rename_btn = gr.Button("开始复制并重命名", variant="primary", scale=3)
                        open_rename_btn = gr.Button("打开输出目录", variant="secondary", scale=1)
                    gr.Markdown("注意：此操作会将文件**复制**到 `output/renamed/` 目录并修改扩展名，**原始文件保持不变**。")
                with gr.Column(scale=1):
                    rename_result = gr.Textbox(label="操作结果", interactive=False, lines=8)

        rename_btn.click(fn=batch_copy_rename, inputs=[rename_files, new_extension], outputs=rename_result)
        open_rename_btn.click(fn=open_rename_folder, outputs=rename_result)

        # ---------- Tab5: 在线AI入口（动态模板） ----------
        with gr.Tab("在线AI入口"):
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
                category_dd = gr.Dropdown(
                    label="选择分类",
                    choices=CATEGORY_NAMES,
                    value=initial_category,
                    interactive=True
                )
                role_dd = gr.Dropdown(
                    label="选择角色",
                    choices=[(name, role_id) for (role_id, name) in initial_roles] if initial_roles else [],
                    value=first_role_id,
                    interactive=True
                )

            current_prompt = gr.Textbox(
                label="系统提示词",
                value=first_role_info.get("系统提示词", ""),
                lines=8,
                interactive=True
            )
            placeholder_box = gr.Textbox(
                label="输入占位符（可复制到需要输入的地方）",
                value=first_role_info.get("输入占位符", ""),
                interactive=False
            )

            gr.Markdown("使用方法：选择分类和角色 → 复制提示词/占位符 → 点击上方按钮打开 AI 网站 → 粘贴使用。")

            def on_category_change(cat):
                roles = CATEGORY_ROLES_MAP.get(cat, [])
                if roles:
                    choices = [(name, role_id) for (role_id, name) in roles]
                    default_role = roles[0][0]
                    return gr.update(choices=choices, value=default_role), gr.update(), gr.update()
                else:
                    return gr.update(choices=[], value=None), gr.update(value=""), gr.update(value="")

            def on_role_change(cat, role):
                info = get_role_info(cat, role)
                prompt = info.get("系统提示词", "")
                placeholder = info.get("输入占位符", "")
                return gr.update(value=prompt), gr.update(value=placeholder)

            category_dd.change(
                fn=on_category_change,
                inputs=[category_dd],
                outputs=[role_dd, current_prompt, placeholder_box]
            )
            role_dd.change(
                fn=on_role_change,
                inputs=[category_dd, role_dd],
                outputs=[current_prompt, placeholder_box]
            )

            btn_deepl.click(fn=lambda: open_url(URLS["DeepL"]), outputs=status_trans)
            btn_youdao.click(fn=lambda: open_url(URLS["有道翻译"]), outputs=status_trans)
            btn_deepseek.click(fn=lambda: open_url(URLS["DeepSeek"]), outputs=status_trans)
            btn_doubao.click(fn=lambda: open_url(URLS["豆包"]), outputs=status_trans)
            btn_qianwen.click(fn=lambda: open_url(URLS["通义千问"]), outputs=status_trans)
            btn_kimi.click(fn=lambda: open_url(URLS["Kimi"]), outputs=status_trans)
            btn_chatglm.click(fn=lambda: open_url(URLS["ChatGLM"]), outputs=status_trans)
            btn_yuanbao.click(fn=lambda: open_url(URLS["腾讯元宝"]), outputs=status_trans)

    gr.Markdown("---")
    gr.HTML("""
      <div style="text-align: center; color: #666; font-size: 0.9em;">
          <p>本工具仅用于个人学习与文档处理，禁止商业用途。</p>
          <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 15px; border-radius: 8px; margin: 15px auto; max-width: 600px;">
              <p style="color: white; font-weight: bold; margin: 5px 0;">更新请关注B站up主：光影的故事2018</p>
              <p style="color: white; margin: 5px 0;">
                  <strong>B站主页</strong>: <a href="https://space.bilibili.com/381518712" target="_blank" style="color: #ffdd40; text-decoration: none;">space.bilibili.com/381518712</a>
              </p>
          </div>
          <p>原创 WebUI 代码 2026 光影紐扣 版权所有</p>
      </div>
    """)

if __name__ == "__main__":
    port = find_free_port(7969)
    print(f"使用端口: {port}")
    threading.Timer(1.5, open_browser, args=(port,)).start()
    demo.launch(
        server_name="127.0.0.1",
        server_port=port,
        share=False,
        theme=gr.themes.Soft()
    )