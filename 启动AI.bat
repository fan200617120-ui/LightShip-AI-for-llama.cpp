@echo off
chcp 65001 >nul
title 轻舟 AI・LightShip AI 启动器 (llama.cpp)
color 0B

:: 获取当前脚本所在目录（绝对路径）
set "BASE_DIR=%~dp0"
set "PYTHON=%BASE_DIR%python_embeded\python.exe"
set "CORE_DIR=%BASE_DIR%core"
set "MODELS_DIR=%BASE_DIR%models"
set "LLAMA_DIR=%BASE_DIR%llama"
set "LLAMA_SERVER=%LLAMA_DIR%\llama-server.exe"
:: 【新增】格式转换工具路径
set "PANDOC_SCRIPT=%BASE_DIR%pandoc\format_converter.py"

:: 确保 models 目录存在
if not exist "%MODELS_DIR%" mkdir "%MODELS_DIR%"

:: 检查 llama-server.exe 是否存在
if not exist "%LLAMA_SERVER%" (
    echo 错误：未找到 %LLAMA_SERVER%
    echo 请确认 llama.cpp 已放置在 %LLAMA_DIR% 目录下
    pause
    exit /b 1
)

:menu
cls
echo ========================================
echo        轻舟 AI・LightShip AI
echo ========================================
echo.
echo              llama.cpp 
echo.
echo ========================================
echo       轻舟渡万境，一智载千寻!
echo ========================================
echo.
echo 模型存储路径：%MODELS_DIR%
echo.
echo 请选择要运行的选项（输入数字或字母）：
echo.
echo   [1] 原生浏览器窗口
echo.
echo   [2] AI助手 (ai_buddy.py)
echo.
echo   [3] 字幕翻译 (subtitle_translator.py)
echo.
echo   [4] 有记忆版聊天 (chat_Ai.py)
echo.
echo   [5] 无记忆版聊天 (chat_Ai_no.py)
echo.
echo   [6] 打开资源管理器到根目录
echo.
echo   [7] 格式转换 (Pandoc 批量markdown word)
echo.
echo   [8] 退出
echo.
set /p choice="请输入数字 (1-8): "

:: 去除无效字符
set "choice=%choice:[=%"
set "choice=%choice:]=%"
set "choice=%choice: =%"

if "%choice%"=="1" goto start_service_and_browser
if "%choice%"=="2" set SCRIPT=ai_buddy.py & goto run_with_service
if "%choice%"=="3" set SCRIPT=subtitle_translator.py & goto run_with_service
if "%choice%"=="4" set SCRIPT=chat_Ai.py & goto run_with_service
if "%choice%"=="5" set SCRIPT=chat_Ai_no.py & goto run_with_service
if "%choice%"=="6" goto explorer
if "%choice%"=="7" goto format_converter
if "%choice%"=="8" goto exit

echo 无效输入，请输入 1-8。
pause
goto menu

:run_with_service
:: 先启动 llama-server 服务，再运行脚本
echo.
echo ========================================
echo 正在关闭可能运行的 llama-server 进程...
echo ========================================
echo.
taskkill /f /im llama-server.exe 2>nul
timeout /t 2 /nobreak >nul

echo 正在启动 llama-server（模型目录：%MODELS_DIR%）...
echo.
echo 服务地址：http://127.0.0.1:8080
echo.
start "llama-server" cmd /c "%LLAMA_SERVER% --models-dir "%MODELS_DIR%""
:: 等待服务完全启动（根据机器性能可适当调整）
echo.
echo 等待服务启动...
timeout /t 5 /nobreak >nul

goto run_script

:run_script
if not exist "%PYTHON%" (
    echo 错误：未找到 %PYTHON%
    pause
    goto menu
)
"%PYTHON%" "%CORE_DIR%\%SCRIPT%"
pause
goto menu

:start_service_and_browser
echo.
echo ========================================
echo 正在关闭可能运行的 llama-server 进程...
echo ========================================
echo.
taskkill /f /im llama-server.exe 2>nul
timeout /t 2 /nobreak >nul

echo 正在启动 llama-server（模型目录：%MODELS_DIR%）...
echo 服务地址：http://127.0.0.1:8080
start "llama-server" cmd /c "%LLAMA_SERVER% --models-dir "%MODELS_DIR%""
:: 等待服务启动
echo 等待服务启动...
timeout /t 5 /nobreak >nul

:: 打开默认浏览器访问 Web UI
echo 正在打开浏览器...
start http://127.0.0.1:8080

echo llama-server 服务已启动，浏览器已打开。
echo.
echo 提示：按任意键返回菜单...
pause
goto menu

:explorer
start "" "%BASE_DIR%"
pause
goto menu

:: ========== 新增：格式转换功能 ==========
:format_converter
echo.
echo 正在启动 Pandoc 格式转换器...
echo 转换界面将在浏览器中打开，地址：http://127.0.0.1:7966
echo 按 Ctrl+C 可停止服务，或直接关闭命令行窗口。
echo.
if not exist "%PYTHON%" (
    echo 错误：未找到 Python 解释器 %PYTHON%
    pause
    goto menu
)
if not exist "%PANDOC_SCRIPT%" (
    echo 错误：未找到格式转换脚本 %PANDOC_SCRIPT%
    pause
    goto menu
)
:: 启动 Gradio 应用（会在新窗口中运行）
start "Pandoc格式转换器" cmd /c "%PYTHON% %PANDOC_SCRIPT%"
:: 等待几秒让服务启动
timeout /t 3 /nobreak >nul
start http://127.0.0.1:7966
echo 转换器已启动。
echo.
pause
goto menu
:: =====================================

:exit
exit