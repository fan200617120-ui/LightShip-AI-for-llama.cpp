@echo off
chcp 65001 >nul
title OCR 服务启动器
color 0B

set "BASE_DIR=%~dp0"
set "LLAMA_SERVER=%BASE_DIR%llama\llama-server.exe"

:: 检查 llama-server 是否存在
if not exist "%LLAMA_SERVER%" (
    echo [错误] 未找到 %LLAMA_SERVER%
    pause
    exit /b 1
)

:menu
cls
echo ========================================
echo         OCR 服务启动器
echo ========================================
echo.
echo   [1] 启动 DeepSeek-OCR (端口 60115)
echo   [2] 启动 MinerU2.5-Pro (端口 50600)
echo   [3] 同时启动两个服务
echo   [4] 停止所有 llama-server 进程
echo   [5] 退出
echo.
set /p choice="请输入数字 (1-5): "

if "%choice%"=="1" goto start_deepseek
if "%choice%"=="2" goto start_mineru
if "%choice%"=="3" goto start_both
if "%choice%"=="4" goto stop_all
if "%choice%"=="5" exit
echo 无效输入，请重新选择
pause
goto menu

:start_deepseek
call :launch "DeepSeek-OCR" "%BASE_DIR%models\DeepSeek-OCR-2\deepseek-ocr-2-Q4_K_M.gguf" "%BASE_DIR%models\DeepSeek-OCR-2\mmproj-deepseek-ocr-2-q8_0.gguf" 60115
goto menu

:start_mineru
call :launch "MinerU" "%BASE_DIR%models\MinerU2.5-Pro\MinerU2.5-Pro-2605-1.2B-Q4_K_M.gguf" "%BASE_DIR%models\MinerU2.5-Pro\mmproj-MinerU2.5-Pro-2605-1.2B-f16.gguf" 50600
goto menu

:start_both
call :launch "DeepSeek-OCR" "%BASE_DIR%models\DeepSeek-OCR-2\deepseek-ocr-2-Q4_K_M.gguf" "%BASE_DIR%models\DeepSeek-OCR-2\mmproj-deepseek-ocr-2-q8_0.gguf" 60115
call :launch "MinerU" "%BASE_DIR%models\MinerU2.5-Pro\MinerU2.5-Pro-2605-1.2B-Q4_K_M.gguf" "%BASE_DIR%models\MinerU2.5-Pro\mmproj-MinerU2.5-Pro-2605-1.2B-f16.gguf" 50600
goto menu

:stop_all
echo 正在停止所有 llama-server 进程...
taskkill /f /im llama-server.exe 2>nul >nul
echo 已停止。
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

echo 正在启动 %NAME% (端口 %PORT%) ...
taskkill /f /im llama-server.exe 2>nul >nul
timeout /t 2 /nobreak >nul

start "%NAME%" /D "%BASE_DIR%" "%LLAMA_SERVER%" ^
    --model "%MODEL_PATH%" ^
    --mmproj "%MMPROJ_PATH%" ^
    --host 127.0.0.1 ^
    --port %PORT% ^
    --n-gpu-layers 0 ^
    --ctx-size 8192 ^
    --temp 0.0 ^
    --n-predict 1024

echo %NAME% 已启动，等待加载...
timeout /t 3 /nobreak >nul
echo.
goto :eof