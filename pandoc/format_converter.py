import os
import subprocess
import gradio as gr
from pathlib import Path
import webbrowser
from PIL import Image, ImageOps
import shutil
import tempfile
import re
from datetime import datetime
from typing import Tuple, Optional

# ================= 配置 =================
SCRIPT_DIR = Path(__file__).parent.resolve()
PANDOC_PATH = SCRIPT_DIR / "pandoc.exe"
OUTPUT_DIR = SCRIPT_DIR.parent / "output"
IMAGE_OUTPUT_DIR = OUTPUT_DIR / "images"
RENAME_OUTPUT_DIR = OUTPUT_DIR / "renamed"
CHAT_EXPORT_DIR = OUTPUT_DIR / "chat_exports"
TEMPLATE_DIR = SCRIPT_DIR / "templates"          # 存放参考模板的目录

# 支持的文档格式
SUPPORTED_FORMATS = {
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

# 中文检测（Windows 下直接返回常用字体）
CHINESE_FONTS = ["SimSun", "Microsoft YaHei", "Noto Serif CJK SC", "Noto Sans CJK SC"]

# ================= 在线AI网址配置 =================
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

# ================= 影视专业提示词 =================
PROMPTS = {
    "专业分镜脚本撰写(影视标准格式)": """你是资深影视导演&专业分镜师。请根据以下需求，撰写符合行业标准的分镜脚本。
要求：
1. 严格遵循镜号、景别、运镜、画面内容、台词/音效、时长的标准分镜格式。
2. 镜头设计贴合叙事逻辑，画面感强，符合影视拍摄规范。
3. 精准标注核心摄影参数、机位调度与光影要求。
4. 直接输出完整分镜脚本，不包含额外解释。

需求内容：
""",
    "镜头语言与运镜方案设计": """你是资深摄影指导&影视导演。请根据以下拍摄场景，设计专业的镜头语言与完整运镜方案。
要求：
1. 明确标注每段镜头的机位、焦段、运镜轨迹、拍摄节奏。
2. 镜头设计服务于叙事与情绪表达，符合电影级拍摄规范。
3. 同步给出适配的器材选型与拍摄参数参考。
4. 输出条理清晰的可执行拍摄方案，无冗余内容。

拍摄场景：
""",
    "短视频爆款拍摄脚本(15-60s竖屏)": """你是短视频爆款导演&商业摄影师。请根据以下需求，撰写15-60s竖屏短视频拍摄脚本。
要求：
1. 强钩子开篇，单镜时长适配短视频快节奏，叙事紧凑抓眼。
2. 明确标注镜号、时长、画面内容、运镜、台词/音效、字幕要点。
3. 镜头设计适配手机竖屏观看逻辑，拍摄落地性强，难度可控。
4. 直接输出完整可执行的拍摄脚本，无额外解释。

需求内容：
""",
    "广告/宣传片创意分镜方案": """你是资深商业广告导演&专业摄影师。请根据以下品牌需求，撰写完整的广告/宣传片创意分镜方案。
要求：
1. 严格遵循行业标准格式，标注镜号、时长、景别、运镜、画面内容、音画设计、品牌植入点位。
2. 画面设计兼具高级感与传播性，深度贴合品牌调性与受众定位。
3. 同步明确核心光影风格、色彩体系与摄影执行标准。
4. 直接输出完整分镜方案，不包含额外解释。

品牌需求：
""",
    "影视台词/旁白专业润色": """你是资深影视导演&台词指导。请对以下影视台词/旁白进行专业润色与校对。
要求：
1. 修正语病与逻辑瑕疵，优化语言节奏，适配口语表达与口型时长。
2. 深度贴合人物设定、剧情语境与核心情绪，完整保留原意。
3. 优化文本的画面适配性，让台词与镜头叙事高度契合。
4. 直接输出润色后的完整文本，无额外解释。

待处理文本：
""",
    "拍摄方案核心要点提炼": """请阅读以下完整拍摄需求/剧本内容，提炼片场执行核心要点。
要求：
1. 精准提炼核心叙事主线、核心拍摄场景、光影风格、器材配置、关键执行节点。
2. 规避冗余信息，使用条理清晰的列表输出，适配片场快速查阅。
3. 同步标注拍摄核心风险点与执行注意事项。

文本内容：
""",
    "片场拍摄执行指令优化": """你是资深影视导演&摄影指导。请将以下模糊拍摄需求，转化为片场可精准执行的标准化专业指令。
要求：
1. 指令按摄影、灯光、置景、收音岗位拆分，表述精准无歧义。
2. 严格遵循影视行业专业术语规范，可直接落地执行。
3. 完整保留核心拍摄诉求，优化执行效率与画面呈现标准。
4. 直接输出优化后的完整执行指令，无额外解释。

需求内容：
"""
}

def open_url(url):
    webbrowser.open(url)
    return f"已在浏览器中打开 {url}"

def update_prompt(selection):
    return PROMPTS.get(selection, "")

def _resolve_file_path(file_obj):
    if file_obj is None:
        return None
    if isinstance(file_obj, (str, Path)):
        return Path(file_obj)
    if hasattr(file_obj, "name"):
        return Path(file_obj.name)
    if isinstance(file_obj, dict) and "name" in file_obj:
        return Path(file_obj["name"])
    raise TypeError(f"Unsupported file object type: {type(file_obj)}")

def _get_original_filename(file_obj):
    if file_obj is None:
        return None
    if hasattr(file_obj, "orig_name") and getattr(file_obj, "orig_name"):
        return Path(file_obj.orig_name).name
    path = _resolve_file_path(file_obj)
    return path.name

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

# ================= Pandoc 文档转换 =================
def _find_pandoc_executable():
    if PANDOC_PATH.exists():
        return PANDOC_PATH
    system_path = shutil.which("pandoc")
    if system_path:
        return Path(system_path)
    return None

def check_pandoc():
    pandoc_exec = _find_pandoc_executable()
    if not pandoc_exec:
        return False, f"未找到 Pandoc：{PANDOC_PATH}，也未在 PATH 中发现 pandoc。"
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
        # Windows 下直接返回默认中文字体
        return "SimSun"
    for font in CHINESE_FONTS:
        try:
            result = subprocess.run(["fc-list", f":family={font}"], capture_output=True, text=True)
            if result.returncode == 0 and result.stdout.strip():
                return font
        except Exception:
            continue
    return None

def convert_files(files, source_format, target_format, enable_toc=False, reference_doc=None):
    if not files:
        return "请先上传至少一个文件。", ""
    pandoc_exec = _find_pandoc_executable()
    if not pandoc_exec:
        return "未找到 Pandoc，可将 pandoc.exe 放置于当前目录或确保 pandoc 已加入系统 PATH。", " "

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    src_ext = SUPPORTED_FORMATS.get(source_format, "")
    tgt_ext = SUPPORTED_FORMATS.get(target_format, "")
    if not src_ext or not tgt_ext:
        return "不支持的格式组合。", ""

    writer = FORMAT_ALIASES.get(tgt_ext, tgt_ext)

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
            return ("转换为 PDF 需要安装 LaTeX 引擎（MiKTeX 或 TeX Live）。 "
                    "若仅需简单的 PDF 转换（无复杂排版/公式），可先将文档转换为 HTML， "
                    "再通过浏览器「打印为 PDF」；或使用 Pandoc+wkhtmltopdf（需额外配置）， "
                    "但兼容性不如 LaTeX 引擎。", "")

    extra_args = []
    if tgt_ext == ".pdf":
        extra_args.extend(["--pdf-engine", pdf_engine])
        font = detect_chinese_font()
        # 修复 Bug 2：仅当引擎支持时才传入 mainfont
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
        if input_path is None:
            continue
        input_filename = Path(_get_original_filename(file_obj))
        output_name = input_filename.stem + tgt_ext
        output_path = OUTPUT_DIR / output_name

        counter = 1
        while output_path.exists():
            output_path = OUTPUT_DIR / f"{input_filename.stem}_{counter}{tgt_ext}"
            counter += 1

        default_reader = FORMAT_ALIASES.get(src_ext, src_ext)
        success = False
        error_msg = ""

        if src_ext == ".md":
            for reader in ["markdown+hard_line_breaks-yaml_metadata_block", "markdown+hard_line_breaks"]:
                try:
                    cmd = [
                        str(pandoc_exec), str(input_path),
                        "-f", reader, "-t", writer, "-o", str(output_path),
                        "--wrap=preserve"
                    ] + extra_args
                    subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=True)
                    success = True
                    break
                except subprocess.CalledProcessError as e:
                    error_msg = e.stderr
                    if "YAML" in error_msg or "metadata" in error_msg:
                        try:
                            with open(input_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                            stripped = re.sub(r'^\s*---\s*\n.*?\n---\s*\n', '', content, flags=re.DOTALL)
                            if stripped != content and stripped.strip():
                                with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as tmp:
                                    tmp.write(stripped)
                                    tmp_path = tmp.name
                                try:
                                    cmd2 = [
                                        str(pandoc_exec), tmp_path,
                                        "-f", "markdown+hard_line_breaks", "-t", writer, "-o", str(output_path),
                                        "--wrap=preserve"
                                    ] + extra_args
                                    subprocess.run(cmd2, capture_output=True, text=True, timeout=60, check=True)
                                    success = True
                                    # 修复 Bug 4：YAML 剥离成功需跳出 for 循环
                                    break
                                finally:
                                    os.unlink(tmp_path)
                        except Exception:
                            pass
                if success:
                    break
        else:
            try:
                cmd = [
                    str(pandoc_exec), str(input_path),
                    "-f", default_reader, "-t", writer, "-o", str(output_path),
                    "--wrap=preserve"
                ] + extra_args
                subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=True)
                success = True
            except subprocess.CalledProcessError as e:
                error_msg = e.stderr

        # 兜底：纯文本提取
        if not success and tgt_ext != ".pdf":
            try:
                cmd = [
                    str(pandoc_exec), str(input_path),
                    "-f", "plain", "-t", writer, "-o", str(output_path),
                    "--wrap=preserve"
                ] + extra_args
                subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=True)
                success = True
                # 修复 Bug 5：删除无效的 output_name 赋值
            except Exception:
                pass

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

# ================= 图片格式转换 =================
def convert_images(files, target_format, quality=85):
    if not files:
        return "请先上传至少一张图片。", ""
    if target_format not in IMAGE_EXT_MAP:
        return f"不支持的图片格式：{target_format}", " "

    IMAGE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    target_ext = IMAGE_EXT_MAP[target_format]
    success_count = 0
    fail_msgs = []
    output_paths = []

    for file_obj in files:
        input_path = _resolve_file_path(file_obj)
        input_filename = Path(_get_original_filename(file_obj))
        try:
            img = Image.open(input_path)
            img = ImageOps.exif_transpose(img)
            output_name = input_filename.stem + target_ext
            output_path = IMAGE_OUTPUT_DIR / output_name

            counter = 1
            while output_path.exists():
                output_path = IMAGE_OUTPUT_DIR / f"{input_filename.stem}_{counter}{target_ext}"
                counter += 1

            if target_format in ["JPEG", "WebP"]:
                # 修复 Bug 3：正确处理 LA 模式的透明通道
                if img.mode in ("RGBA", "LA", "P") and target_format == "JPEG":
                    if img.mode in ("LA", "P"):
                        img = img.convert("RGBA")
                    rgb_img = Image.new("RGB", img.size, (255, 255, 255))
                    rgb_img.paste(img, mask=img.split()[-1])  # 此时 img 必为 RGBA
                    img = rgb_img
                elif img.mode not in ("RGB", "L") and target_format == "JPEG":
                    img = img.convert("RGB")
                img.save(output_path, format=target_format, quality=quality)
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

# ================= 批量复制并重命名 =================
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
        original_name = _get_original_filename(file_obj) or input_path.name
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
            # 修复 Bug 2：仅当引擎支持时才传入 mainfont
            if font and pdf_engine in ["xelatex", "lualatex"]:
                extra_args.extend(["-V", f"mainfont={font}"])
        if target_format == "Microsoft Word (docx)" and template_file is not None:
            ref_path = _resolve_file_path(template_file)
            if ref_path and ref_path.exists() and ref_path.suffix.lower() == '.docx':
                extra_args.extend(["--reference-doc", str(ref_path)])

        cmd = [
            str(pandoc_exec),
            temp_in,
            "-f", reader,
            "-t", writer,
            "-o", str(output_path),
            "--wrap=preserve"
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

    out_path, error = run_pandoc(content, reader="markdown+hard_line_breaks-yaml_metadata_block")
    if out_path:
        return out_path, f"导出成功：{out_path}"

    if error and ("YAML" in error or "metadata" in error):
        stripped_content = re.sub(r'^\s*---\s*\n.*?\n---\s*\n', '', content, flags=re.DOTALL)
        if stripped_content != content and stripped_content.strip():
            out_path, error2 = run_pandoc(stripped_content, reader="markdown+hard_line_breaks-yaml_metadata_block")
            if out_path:
                return out_path, f"导出成功：{out_path} (已自动忽略 YAML 头部)"
            error = error2

    # 修复 Bug 1：兜底保存为纯文本 .txt，而非带有错误扩展名的文件
    try:
        txt_output_path = CHAT_EXPORT_DIR / f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(txt_output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return str(txt_output_path), f"Pandoc 转换失败，已降级保存为纯文本：{txt_output_path}"
    except Exception as e:
        return None, f"所有尝试均失败。最后错误：{error}；纯文本保存失败：{e}"

def open_chat_export_dir():
    return _open_folder(CHAT_EXPORT_DIR)

def open_template_dir():
    TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
    return _open_folder(TEMPLATE_DIR)

# ================= 获取初始 Pandoc 状态 =================
initial_pandoc_status = check_pandoc()[1]

# ================= Gradio 界面 =================
with gr.Blocks(title="轻舟 AI 工具箱") as demo:
    gr.Markdown("# 轻舟 AI 工具箱")
    gr.Markdown("### 排版导出 · 格式转换 · 图片转换 · 批量改名 · 在线AI入口")
    with gr.Tabs():
        # ---------- Tab1: 排版导出 ----------
        with gr.Tab("排版导出"):
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### 粘贴 Markdown 内容并导出为多种格式 ")
                    content_input = gr.Textbox(
                        label="Markdown 内容 ",
                        lines=20,
                        placeholder="在此粘贴 Markdown 文本..."
                    )
                    
                    with gr.Accordion("👁️ 实时预览 (点击展开)", open=False):
                        preview_box = gr.Markdown(value="*输入内容后将在此处显示预览...*")

                    export_format = gr.Dropdown(
                        choices=["Microsoft Word (docx)", "HTML", "Plain Text", "Markdown", "PDF"],
                        label="导出格式 ",
                        value="Microsoft Word (docx)"
                    )
                    with gr.Row():
                        export_btn = gr.Button("开始导出 ", variant="primary", scale=3)
                        open_export_dir_btn = gr.Button("打开导出目录 ", variant="secondary", scale=1)
                with gr.Column(scale=1):
                    gr.Markdown("### 高级选项 ")
                    template_file = gr.File(
                        label="参考模板 (可选 .docx) ",
                        file_types=[".docx"],
                        value=None
                    )
                    open_template_btn = gr.Button("打开模板目录 ", variant="secondary")
                    export_status = gr.Textbox(label="导出状态 ", interactive=False, lines=3)
                    export_file_download = gr.File(label="下载导出的文件 ", visible=True)

            def handle_export(content, fmt, template):
                if not content.strip():
                    return None, "请先输入 Markdown 内容。"
                path, msg = export_content_to_format(content, fmt, template)
                if path:
                    return gr.update(value=path, visible=True), msg
                else:
                    return gr.update(visible=False), msg

            export_btn.click(
                fn=handle_export,
                inputs=[content_input, export_format, template_file],
                outputs=[export_file_download, export_status]
            )
            open_export_dir_btn.click(fn=open_chat_export_dir, outputs=[export_status])
            open_template_btn.click(fn=open_template_dir, outputs=[export_status])
            
            content_input.change(
                fn=lambda txt: txt.strip() or "*内容为空*",
                inputs=content_input,
                outputs=preview_box
            )

        # ---------- Tab2: 文档转换 ----------
        with gr.Tab("文档格式转换"):
            status = gr.Textbox(
                label="Pandoc 状态 ",
                value=initial_pandoc_status,
                interactive=False
            )

            with gr.Row():
                with gr.Column(scale=2):
                    file_input = gr.File(label="上传文件（可多选） ", file_count="multiple")
                    src_format = gr.Dropdown(
                        choices=list(SUPPORTED_FORMATS.keys()),
                        label="源格式 ",
                        value="Markdown"
                    )
                    tgt_format = gr.Dropdown(
                        choices=list(SUPPORTED_FORMATS.keys()),
                        label="目标格式 ",
                        value="Microsoft Word (docx)"
                    )
                    with gr.Accordion("高级选项 ", open=False):
                        enable_toc = gr.Checkbox(label="为 Word 添加目录 (--toc) ", value=False)
                        reference_doc_acc = gr.File(label="参考样式模板 (仅 Word，可选 .docx) ", file_types=[".docx"])

                    with gr.Row():
                        convert_btn = gr.Button("开始批量转换 ", variant="primary", scale=3)
                        open_folder_btn = gr.Button("打开输出目录 ", variant="secondary", scale=1)

                with gr.Column(scale=1):
                    gr.Markdown("### 模板设置 ")
                    reference_doc = gr.File(
                        label="参考模板 (可选 .docx) ",
                        file_types=[".docx"]
                    )
                    open_template_btn2 = gr.Button("打开模板目录 ", variant="secondary")
                    output_msg = gr.Textbox(label="转换结果 ", interactive=False, lines=5)
                    output_files = gr.Textbox(label="输出文件列表 ", interactive=False, lines=5)

            def convert_files_wrapper(files, src_fmt, tgt_fmt, enable_toc, ref_acc, ref_side):
                ref = ref_side if ref_side is not None else ref_acc
                return convert_files(files, src_fmt, tgt_fmt, enable_toc, ref)

            convert_btn.click(
                fn=convert_files_wrapper,
                inputs=[file_input, src_format, tgt_format, enable_toc, reference_doc_acc, reference_doc],
                outputs=[output_msg, output_files]
            )
            open_folder_btn.click(fn=open_output_folder, outputs=output_msg)
            open_template_btn2.click(fn=open_template_dir, outputs=output_msg)

        # ---------- Tab3: 图片转换 ----------
        with gr.Tab("图片格式转换"):
            with gr.Row():
                with gr.Column(scale=2):
                    img_input = gr.File(label="上传图片（可多选） ", file_count="multiple")
                    img_format = gr.Dropdown(
                        choices=IMAGE_FORMATS,
                        label="输出格式 ",
                        value="PNG"
                    )
                    img_quality = gr.Slider(
                        minimum=1, maximum=100, value=85, step=1,
                        label="图片质量 (仅对 JPEG/WebP 有效)"
                    )
                    with gr.Row():
                        img_convert_btn = gr.Button("开始转换图片 ", variant="primary", scale=3)
                        img_open_folder_btn = gr.Button("打开图片输出目录 ", variant="secondary", scale=1)

                with gr.Column(scale=1):
                    img_output_msg = gr.Textbox(label="转换结果 ", interactive=False, lines=6)
                    img_output_files = gr.Textbox(label="输出文件列表 ", interactive=False, lines=5)

            def open_image_folder():
                return _open_folder(IMAGE_OUTPUT_DIR)

            img_convert_btn.click(
                fn=convert_images,
                inputs=[img_input, img_format, img_quality],
                outputs=[img_output_msg, img_output_files]
            )
            img_open_folder_btn.click(fn=open_image_folder, outputs=img_output_msg)

        # ---------- Tab4: 批量复制并重命名 ----------
        with gr.Tab("批量复制并重命名"):
            with gr.Row():
                with gr.Column(scale=2):
                    rename_files = gr.File(label="选择文件（可多选） ", file_count="multiple")
                    new_extension = gr.Textbox(label="新扩展名（例如 .txt 或 txt） ", placeholder=".md")
                    with gr.Row():
                        rename_btn = gr.Button("开始复制并重命名 ", variant="primary", scale=3)
                        open_rename_btn = gr.Button("打开输出目录 ", variant="secondary", scale=1)
                    gr.Markdown("注意：此操作会将文件**复制**到 `output/renamed/` 目录并修改扩展名，**原始文件保持不变**。 ")
                with gr.Column(scale=1):
                    rename_result = gr.Textbox(label="操作结果 ", interactive=False, lines=8)

            rename_btn.click(fn=batch_copy_rename, inputs=[rename_files, new_extension], outputs=rename_result)
            open_rename_btn.click(fn=open_rename_folder, outputs=rename_result)

        # ---------- Tab5: 在线AI入口 ----------
        with gr.Tab("在线AI入口"):
            with gr.Column():
                gr.Markdown("### 快速入口 ")
                with gr.Row(equal_height=True):
                    btn_deepl = gr.Button("DeepL ", variant="secondary")
                    btn_youdao = gr.Button("有道翻译 ", variant="secondary")
                    btn_deepseek = gr.Button("DeepSeek ", variant="secondary")
                    btn_doubao = gr.Button("豆包 ", variant="secondary")
                with gr.Row(equal_height=True):
                    btn_qianwen = gr.Button("通义千问 ", variant="secondary")
                    btn_kimi = gr.Button("Kimi ", variant="secondary")
                    btn_chatglm = gr.Button("ChatGLM ", variant="secondary")
                    btn_yuanbao = gr.Button("腾讯元宝 ", variant="secondary")
                status_trans = gr.Textbox(label=" ", value="点击按钮打开对应网站 ", interactive=False)

                gr.Markdown("---")
                gr.Markdown("### 影视专业提示词模板 ")
                prompt_selector = gr.Dropdown(
                    label="选择提示词类型 ",
                    choices=list(PROMPTS.keys()),
                    value="专业分镜脚本撰写(影视标准格式)",
                )
                prompt_display = gr.Textbox(
                    label="提示词内容 (可直接编辑) ",
                    value=PROMPTS["专业分镜脚本撰写(影视标准格式)"],
                    lines=8,
                    interactive=True,
                )
                gr.Markdown("使用方法：选择模板 → 复制提示词 → 点击上方按钮打开网站 → 粘贴使用。 ")

                btn_deepl.click(fn=lambda: open_url(URLS["DeepL"]), outputs=status_trans)
                btn_youdao.click(fn=lambda: open_url(URLS["有道翻译"]), outputs=status_trans)
                btn_deepseek.click(fn=lambda: open_url(URLS["DeepSeek"]), outputs=status_trans)
                btn_doubao.click(fn=lambda: open_url(URLS["豆包"]), outputs=status_trans)
                btn_qianwen.click(fn=lambda: open_url(URLS["通义千问"]), outputs=status_trans)
                btn_kimi.click(fn=lambda: open_url(URLS["Kimi"]), outputs=status_trans)
                btn_chatglm.click(fn=lambda: open_url(URLS["ChatGLM"]), outputs=status_trans)
                btn_yuanbao.click(fn=lambda: open_url(URLS["腾讯元宝"]), outputs=status_trans)
                prompt_selector.change(fn=update_prompt, inputs=[prompt_selector], outputs=[prompt_display])

    # 页脚
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
    demo.launch(
        server_name="127.0.0.1",
        server_port=7966,
        share=False,
        theme=gr.themes.Soft()
    )