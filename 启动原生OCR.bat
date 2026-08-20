@echo off
chcp 65001 >nul
title OCR 一键启动（自动打开页面）
color 0B

set "BASE_DIR=%~dp0"
set "LLAMA_SERVER=%BASE_DIR%llama\llama-server.exe"
set "HTML_FILE=%BASE_DIR%OCR.html"

:: 检查必要文件
if not exist "%LLAMA_SERVER%" (
    echo [错误] 未找到 %LLAMA_SERVER%
    pause
    exit /b 1
)
if not exist "%HTML_FILE%" (
    echo [警告] 未找到 %HTML_FILE%，尝试打开 ocr_native.html
    set "HTML_FILE=%BASE_DIR%ocr_native.html"
    if not exist "!HTML_FILE!" (
        echo [错误] 未找到任何 HTML 页面
        pause
        exit /b 1
    )
)

:menu
cls
echo ============================================================
echo            🌊 轻舟 OCR · 一键启动
echo ============================================================
echo.
echo   请选择要启动的模型：
echo.
echo     [1]  DeepSeek-OCR-2   (端口 60115)   通用文字提取
echo.
echo     [2]  MinerU2.5-Pro   (端口 50600)    表格/公式识别
echo.
echo     [3]  同时启动两个模型（最强组合）
echo.
echo     [4]  只打开页面（不启动服务，适合服务已运行时）
echo.
echo     [5]  退出
echo.
echo ============================================================
set /p choice="请输入数字 (1-5): "

if "%choice%"=="1" goto start_ds
if "%choice%"=="2" goto start_mineru
if "%choice%"=="3" goto start_both
if "%choice%"=="4" goto open_only
if "%choice%"=="5" exit
echo.
echo 无效输入，请重新选择
pause
goto menu

:start_ds
call :launch "DeepSeek-OCR" "%BASE_DIR%models\DeepSeek-OCR-2\deepseek-ocr-2-Q4_K_M.gguf" "%BASE_DIR%models\DeepSeek-OCR-2\mmproj-deepseek-ocr-2-q8_0.gguf" 60115
goto open_page

:start_mineru
call :launch "MinerU" "%BASE_DIR%models\MinerU2.5-Pro\MinerU2.5-Pro-2605-1.2B-Q4_K_M.gguf" "%BASE_DIR%models\MinerU2.5-Pro\mmproj-MinerU2.5-Pro-2605-1.2B-f16.gguf" 50600
goto open_page

:start_both
call :launch "DeepSeek-OCR" "%BASE_DIR%models\DeepSeek-OCR-2\deepseek-ocr-2-Q4_K_M.gguf" "%BASE_DIR%models\DeepSeek-OCR-2\mmproj-deepseek-ocr-2-q8_0.gguf" 60115
call :launch "MinerU" "%BASE_DIR%models\MinerU2.5-Pro\MinerU2.5-Pro-2605-1.2B-Q4_K_M.gguf" "%BASE_DIR%models\MinerU2.5-Pro\mmproj-MinerU2.5-Pro-2605-1.2B-f16.gguf" 50600
goto open_page

:open_only
echo.
echo 跳过服务启动，直接打开页面...
goto open_page

:open_page
echo.
echo 正在打开 OCR 识别页面 ...
start "" "%HTML_FILE%"
echo.
pause
goto menu

:launch
set "NAME=%~1"
set "MODEL_PATH=%~2"
set "MMPROJ_PATH=%~3"
set "PORT=%~4"

if not exist "%MODEL_PATH%" (
    echo [错误] 模型文件不存在: %MODEL_PATH%
    pause
    goto :eof
)
if not exist "%MMPROJ_PATH%" (
    echo [错误] mmproj 文件不存在: %MMPROJ_PATH%
    pause
    goto :eof
)

echo.
echo [%NAME%] 正在启动服务 (端口 %PORT%) ...
taskkill /f /im llama-server.exe 2>nul >nul
timeout /t 1 /nobreak >nul

start "%NAME%" /D "%BASE_DIR%" "%LLAMA_SERVER%" ^
    --model "%MODEL_PATH%" ^
    --mmproj "%MMPROJ_PATH%" ^
    --host 127.0.0.1 ^
    --port %PORT% ^
    --n-gpu-layers 0 ^
    --ctx-size 8192 ^
    --temp 0.0 ^
    --n-predict 1024

echo [%NAME%] 等待服务加载 (3秒) ...
timeout /t 3 /nobreak >nul
goto :eof