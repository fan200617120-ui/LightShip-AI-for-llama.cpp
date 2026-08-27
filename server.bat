@echo off
chcp 65001 >nul
title llama.cpp 服务控制 (唯一入口)
color 0B

:: ==================================================
::  用法说明：
::  1. 直接双击        → 弹出确认菜单，可改参数后启动
::  2. server.bat quiet → 跳过所有询问，用默认值直接后台启动
::                        (供 启动AI助手.bat 或 Python 调用)
::  3. server.bat stop  → 只停止服务
:: ==================================================

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

:: ========== 默认参数（在这里改默认值） ==========
set "GPU_LAYERS=30"
set "CTX_SIZE=4096"
set "TEMPERATURE=0.5"
set "N_PREDICT=2048"
set "HOST=127.0.0.1"
set "PORT=8080"

:: ========== 子命令：stop ==========
if /i "%~1"=="stop" goto :stop

:: ========== 子命令：quiet（跳过询问，直接启动） ==========
if /i "%~1"=="quiet" goto :start_server

:: ========== 交互菜单 ==========
cls
echo ========================================
echo  llama.cpp 服务启动 (前台模式)
echo ========================================
echo  提示：直接回车使用默认值
echo.
set /p "GPU_LAYERS=GPU 层数 (0=纯CPU, 默认%GPU_LAYERS%, 8GB显存建议20~35): "
if "%GPU_LAYERS%"=="" set "GPU_LAYERS=30"
set /p "CTX_SIZE=上下文大小 (默认%CTX_SIZE%): "
if "%CTX_SIZE%"=="" set "CTX_SIZE=4096"
set /p "TEMPERATURE=温度 (默认%TEMPERATURE%): "
if "%TEMPERATURE%"=="" set "TEMPERATURE=0.5"
set /p "N_PREDICT=最大生成 tokens (默认%N_PREDICT%): "
if "%N_PREDICT%"=="" set "N_PREDICT=2048"
set /p "HOST=监听地址 (默认%HOST%, 局域网输 0.0.0.0): "
if "%HOST%"=="" set "HOST=127.0.0.1"
set /p "PORT=端口 (默认%PORT%): "
if "%PORT%"=="" set "PORT=8080"

echo.
echo 启动参数：
echo   GPU层数     = %GPU_LAYERS%
echo   上下文大小  = %CTX_SIZE%
echo   温度        = %TEMPERATURE%
echo   最大tokens  = %N_PREDICT%
echo   监听地址    = %HOST%
echo   端口        = %PORT%
echo.
set /p "CONFIRM=确认启动？(y/n, 默认y): "
if /i "%CONFIRM%"=="n" exit /b

:: ========== 启动 ==========
:start_server
call :kill_server
call :run_server
goto :eof

:: ========== 函数区 ==========
:kill_server
taskkill /f /im llama-server.exe 2>nul >nul
timeout /t 1 /nobreak >nul
goto :eof

:run_server
cd /d "%LLAMA_DIR%"
echo.
echo ========================================
echo 正在启动服务 (Ctrl+C 停止)...
echo 服务地址: http://%HOST%:%PORT%
echo ========================================
"%SERVER_EXE%" --models-dir "%MODELS_DIR%" --n-gpu-layers %GPU_LAYERS% --ctx-size %CTX_SIZE% --temp %TEMPERATURE% --n-predict %N_PREDICT% --host %HOST% --port %PORT%
echo.
echo 服务已停止
if /i "%~1"=="quiet" goto :eof
pause >nul
goto :eof

:stop
call :kill_server
echo 已停止 llama-server。
timeout /t 2 /nobreak >nul
goto :eof
