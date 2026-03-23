@echo off
setlocal EnableExtensions EnableDelayedExpansion
title Diabetes AI System

cd /d "%~dp0"

call :banner
call :detect_python || goto :fatal
call :ensure_venv || goto :fatal
call :activate_venv || goto :fatal
call :install_deps || goto :fatal
call :check_env_file
call :check_ollama || goto :fatal

:menu
cls
echo =============================
echo Diabetes AI Control Panel
echo =============================
echo 1. Ingest PDF incremental
echo 2. Ingest PDF full rebuild
echo 3. Test CLI
echo 4. Run Telegram Bot
echo 5. Check environment
echo 6. Exit
echo =============================
set /p choice=Select option: 

if "%choice%"=="1" goto ingest
if "%choice%"=="2" goto ingestfull
if "%choice%"=="3" goto cli
if "%choice%"=="4" goto bot
if "%choice%"=="5" goto doctor
if "%choice%"=="6" goto :eof

echo.
echo [WARN] Invalid option.
pause
goto menu

:banner
cls
echo =====================================
echo Diabetes AI - Auto Setup + Run
echo =====================================
echo.
exit /b 0

:detect_python
set "PY_CMD="
py -3.13 --version >nul 2>&1
if not errorlevel 1 (
    set "PY_CMD=py -3.13"
    echo [OK] Using Python 3.13 via py launcher.
    exit /b 0
)

py --version >nul 2>&1
if not errorlevel 1 (
    for /f "tokens=2" %%V in ('py --version 2^>nul') do set "PY_VER=%%V"
    echo [WARN] Python 3.13 not found. Using default py launcher: !PY_VER!
    set "PY_CMD=py"
    exit /b 0
)

echo [ERROR] Python launcher not found.
echo Install Python first, ideally Python 3.13.
exit /b 1

:ensure_venv
if exist ".venv\Scripts\python.exe" (
    echo [OK] Virtual environment already exists.
    exit /b 0
)

echo [INFO] Creating virtual environment...
%PY_CMD% -m venv .venv
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment.
    exit /b 1
)

echo [OK] Virtual environment created.
exit /b 0

:activate_venv
echo [INFO] Activating virtual environment...
call .venv\Scripts\activate
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment.
    exit /b 1
)
python --version
exit /b 0

:install_deps
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo [ERROR] Failed to upgrade pip.
    exit /b 1
)

echo [INFO] Installing dependencies...
if exist requirements.txt (
    pip install -r requirements.txt
) else (
    echo [WARN] requirements.txt not found. Installing minimal packages...
    pip install python-dotenv python-telegram-bot langchain langchain-community langchain-core langchain-text-splitters langchain-ollama langchain-chroma chromadb pymupdf pytesseract pillow pypdf
)
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies.
    exit /b 1
)

echo [OK] Dependencies installed.
exit /b 0

:check_env_file
if exist ".env" (
    echo [OK] .env file found.
    exit /b 0
)

echo [WARN] .env file not found.
if exist "env.hybrid.example" (
    echo [INFO] Found env.hybrid.example. Copying to .env ...
    copy /Y "env.hybrid.example" ".env" >nul
    if errorlevel 1 (
        echo [WARN] Failed to auto-copy env.hybrid.example.
    ) else (
        echo [OK] .env created from env.hybrid.example.
        echo [ACTION] Please review .env before running the bot.
    )
) else (
    echo [WARN] Example env file not found. Create .env manually.
)
exit /b 0

:check_ollama
where ollama >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Ollama not installed or not in PATH.
    echo Install Ollama, then run this script again.
    exit /b 1
)

echo [INFO] Checking Ollama service...
ollama list >nul 2>&1
if errorlevel 1 (
    echo [WARN] Ollama seems unavailable. Make sure Ollama app/service is running.
)

call :ensure_model "llama3.1:8b"
if errorlevel 1 exit /b 1
call :ensure_model "nomic-embed-text"
if errorlevel 1 exit /b 1

echo [OK] Ollama ready.
exit /b 0

:ensure_model
set "MODEL_NAME=%~1"
ollama list | findstr /i /c:%MODEL_NAME% >nul
if not errorlevel 1 (
    echo [OK] Model found: %MODEL_NAME%
    exit /b 0
)

echo [INFO] Pulling model: %MODEL_NAME%
ollama pull %MODEL_NAME%
if errorlevel 1 (
    echo [ERROR] Failed to pull model: %MODEL_NAME%
    exit /b 1
)
exit /b 0

:doctor
cls
echo =============================
echo Environment Check
echo =============================
echo.
echo [Python]
where python
python --version
echo.
echo [Pip packages]
pip show python-telegram-bot >nul 2>&1 && echo python-telegram-bot : OK || echo python-telegram-bot : MISSING
pip show langchain-ollama >nul 2>&1 && echo langchain-ollama    : OK || echo langchain-ollama    : MISSING
pip show langchain-chroma >nul 2>&1 && echo langchain-chroma    : OK || echo langchain-chroma    : MISSING
pip show chromadb >nul 2>&1 && echo chromadb            : OK || echo chromadb            : MISSING
pip show pymupdf >nul 2>&1 && echo pymupdf             : OK || echo pymupdf             : MISSING
pip show pytesseract >nul 2>&1 && echo pytesseract         : OK || echo pytesseract         : MISSING
echo.
echo [.env]
if exist ".env" (echo .env found) else echo .env missing
echo.
echo [Ollama]
where ollama >nul 2>&1 && (echo ollama found & ollama list) || echo ollama missing
echo.
pause
goto menu

:ingest
cls
echo Running ingest incremental...
python ingest.py
if errorlevel 1 echo [WARN] ingest.py exited with error.
echo.
pause
goto menu

:ingestfull
cls
echo Running ingest full rebuild...
python ingest.py --full-rebuild
if errorlevel 1 echo [WARN] ingest.py full rebuild exited with error.
echo.
pause
goto menu

:cli
cls
echo Running CLI test...
python ask_cli.py
if errorlevel 1 echo [WARN] ask_cli.py exited with error.
echo.
pause
goto menu

:bot
cls
echo Running bot...
python bot.py
if errorlevel 1 echo [WARN] bot.py exited with error.
echo.
pause
goto menu

:fatal
echo.
echo Setup stopped because of an error.
pause
exit /b 1
