# -*- coding: utf-8 -*-
"""动态重启 llama-server：改 GPU 层数 / 开关视觉模型 / 改上下文"""
import subprocess
import requests
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
BASE_DIR = SCRIPT_DIR.parent
LLAMA_EXE = BASE_DIR / "llama" / "llama-server.exe"
MODELS_DIR = BASE_DIR / "models"
HOST, PORT = "127.0.0.1", 8080
MODELS_URL = f"http://{HOST}:{PORT}/v1/models"

def _find_model_file(model_name: str):
    """在 MODELS_DIR 中查找包含 model_name（去除扩展名后）的第一个 .gguf"""
    stem = Path(model_name).stem.lower()
    for f in MODELS_DIR.glob("*.gguf"):
        if stem in f.stem.lower():
            return f
    return None

def _find_mmproj(model_name: str):
    """
    宽松匹配：只要 models 目录下有 mmproj 文件，就返回第一个。
    如果启用视觉且存在 mmproj，则使用它；否则降级为纯文本。
    """
    mmprojs = list(MODELS_DIR.glob("*mmproj*.gguf"))
    if mmprojs:
        return str(mmprojs[0])
    return None

def restart_server(
    model_name: str,
    gpu_layers: int,
    enable_vision: bool,
    ctx_size: int = 4096,
    timeout: int = 120
):
    """
    重启 llama-server，应用新参数。
    返回 (成功标志, 消息)
    """
    # 1. 找到模型文件
    model_path = _find_model_file(model_name)
    if model_path is None:
        return False, f"❌ 找不到模型文件：{model_name}"

    # 2. 杀掉旧进程
    subprocess.run(["taskkill", "/f", "/im", "llama-server.exe"],
                   capture_output=True, shell=True)
    time.sleep(1)

    # 3. 组装参数
    args = [
        str(LLAMA_EXE),
        "-m", str(model_path),
        "-ngl", str(gpu_layers),
        "-c", str(ctx_size),
        "--host", HOST,
        "--port", str(PORT),
    ]

    mmproj = _find_mmproj(model_name) if enable_vision else None
    if mmproj:
        args += ["--mmproj", mmproj]

    # 4. 后台启动（不弹窗）
    subprocess.Popen(
        args,
        creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
    )

    # 5. 轮询等待服务就绪
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(MODELS_URL, timeout=2).status_code == 200:
                vision_status = "开" if mmproj else "关"
                return True, f"✅ 重载完成 (GPU层数:{gpu_layers}, 上下文:{ctx_size}, 视觉:{vision_status})"
        except Exception:
            pass
        time.sleep(1)

    return False, "❌ 服务重启超时，请检查显存是否够用或模型是否损坏"