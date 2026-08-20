@echo off
chcp 65001 >nul
title DeepSeek-OCR 专用启动器
color 0B
:: 设置路径（请根据实际情况确认）
set "BASE_DIR=%~dp0"
set "LLAMA_SERVER=%BASE_DIR%llama\llama-server.exe"
set "MODEL=%BASE_DIR%models\DeepSeek-OCR-2\deepseek-ocr-2-Q4_K_M.gguf"
set "MMPROJ=%BASE_DIR%models\DeepSeek-OCR-2\mmproj-deepseek-ocr-2-q8_0.gguf"
set "PYTHON=%BASE_DIR%python_embeded\python.exe"
set "WEBUI=%BASE_DIR%core\dsocr_webui.py"

:: 检查关键文件是否存在
if not exist "%LLAMA_SERVER%" (
    echo [错误] 未找到 %LLAMA_SERVER%
    pause
    exit /b 1
)
if not exist "%MODEL%" (
    echo [错误] 未找到模型文件 %MODEL%
    pause
    exit /b 1
)
if not exist "%MMPROJ%" (
    echo [错误] 未找到 mmproj 文件 %MMPROJ%
    pause
    exit /b 1
)
if not exist "%PYTHON%" (
    echo [错误] 未找到 Python 解释器 %PYTHON%
    pause
    exit /b 1
)
if not exist "%WEBUI%" (
    echo [错误] 未找到 WebUI 脚本 %WEBUI%
    pause
    exit /b 1
)

echo ============================================
echo   DeepSeek-OCR 专用启动器
echo   模型: deepseek-ocr-2-Q4_K_M.gguf
echo   mmproj: mmproj-deepseek-ocr-2-q8_0.gguf
echo   端口: 60115
echo ============================================

:: 终止可能占用的进程
taskkill /f /im llama-server.exe 2>nul >nul
timeout /t 2 /nobreak >nul

:: 后台启动 llama-server（新窗口显示日志）
echo 正在启动 llama-server（后台运行）...
start "llama-server-OCR" /D "%BASE_DIR%" "%LLAMA_SERVER%" ^
    --model "%MODEL%" ^
    --mmproj "%MMPROJ%" ^
    --host 127.0.0.1 ^
    --port 50600 ^
    --n-gpu-layers 0 ^
    --ctx-size 8192 ^
    --temp 0.0 ^
    --n-predict 1024

:: 等待模型加载（可根据实际速度调整）
echo 等待模型加载（约 15 秒，若加载较慢请手动延长）...
timeout /t 15 /nobreak >nul

:: 启动 WebUI
echo 正在启动 DeepSeek-OCR WebUI...
"%PYTHON%" "%WEBUI%"

:: 若 WebUI 退出，则自动关闭服务器（可选）
taskkill /f /im llama-server.exe 2>nul >nul
echo 服务已停止。
pause