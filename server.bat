@echo off
chcp 65001 >nul
title llama.cpp 服务（前台运行，Ctrl+C 停止）
color 0B

:: ========== 路径配置 ==========
set "BASE_DIR=%~dp0"
set "LLAMA_DIR=%BASE_DIR%llama"
set "MODELS_DIR=%BASE_DIR%models"
set "SERVER_EXE=%LLAMA_DIR%\llama-server.exe"

if not exist "%SERVER_EXE%" (
    echo [错误] 未找到 %SERVER_EXE%
    pause
    exit /b 1
)
if not exist "%MODELS_DIR%" (
    echo [错误] 未找到 %MODELS_DIR%
    pause
    exit /b 1
)

:: ========== 参数设置 ==========
echo ========================================
echo     llama.cpp 服务启动（前台模式）
echo ========================================
echo 提示：直接回车使用默认值
echo.

set /p "GPU_LAYERS=GPU 层数 (0=纯CPU, 默认0，8GB显存建议20~30): "
if "%GPU_LAYERS%"=="" set GPU_LAYERS=0

set /p "CTX_SIZE=上下文大小 (默认4096): "
if "%CTX_SIZE%"=="" set CTX_SIZE=4096

set /p "TEMP=温度 (默认0.5): "
if "%TEMP%"=="" set TEMP=0.5

set /p "N_PREDICT=最大生成 tokens (默认2048): "
if "%N_PREDICT%"=="" set N_PREDICT=2048

set /p "HOST=监听地址 (默认127.0.0.1，局域网输0.0.0.0): "
if "%HOST%"=="" set HOST=127.0.0.1

set /p "PORT=端口 (默认8080): "
if "%PORT%"=="" set PORT=8080

echo.
echo 启动参数：
echo   GPU层数    = %GPU_LAYERS%
echo   上下文大小 = %CTX_SIZE%
echo   温度       = %TEMP%
echo   最大tokens = %N_PREDICT%
echo   监听地址   = %HOST%
echo   端口       = %PORT%
echo.
set /p "CONFIRM=确认启动？(y/n, 默认y): "
if /i not "%CONFIRM%"=="y" if not "%CONFIRM%"=="" exit /b

:: 停止已有服务（简单粗暴，但有效）
taskkill /f /im llama-server.exe 2>nul >nul
timeout /t 1 /nobreak >nul

:: 切换到 llama 目录，确保 dll 能被正确加载
cd /d "%LLAMA_DIR%"

echo.
echo 正在启动服务，日志将显示在下方（包含扫描到的模型列表）...
echo 按 Ctrl+C 可停止服务。
echo ========================================
echo.

:: 直接前台运行
"%SERVER_EXE%" --models-dir "%MODELS_DIR%" --n-gpu-layers %GPU_LAYERS% --ctx-size %CTX_SIZE% --temp %TEMP% --n-predict %N_PREDICT% --host %HOST% --port %PORT%

echo.
echo 服务已停止（按任意键关闭此窗口）
pause >nul