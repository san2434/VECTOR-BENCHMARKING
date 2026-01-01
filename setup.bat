@echo off
REM Setup script for RAG Vector Benchmarking System (Windows)

setlocal enabledelayedexpansion

echo.
echo 🚀 RAG Vector Benchmarking System - Setup Script
echo ==================================================
echo.

REM Check Python version
echo ✓ Checking Python version...
for /f "tokens=*" %%i in ('python --version 2^>^&1') do set python_version=%%i
echo   Found %python_version%

REM Create virtual environment
echo.
echo ✓ Creating virtual environment...
python -m venv venv
echo   Created venv\

REM Activate virtual environment
echo.
echo ✓ Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo.
echo ✓ Upgrading pip...
python -m pip install --upgrade pip setuptools wheel > nul 2>&1

REM Install dependencies
echo.
echo ✓ Installing dependencies...
pip install -r requirements.txt

REM Create directories
echo.
echo ✓ Creating directories...
if not exist data mkdir data
if not exist results mkdir results
if not exist logs mkdir logs

REM Check if .env exists
echo.
if exist .env (
    echo ✓ .env file already exists
) else (
    echo ✓ Creating .env from template...
    copy .env.example .env
    echo   ⚠️  Please edit .env with your API keys:
    echo      - OPENAI_API_KEY=your_key_here
    echo      - (Optional) PINECONE_API_KEY
    echo      - (Optional) PostgreSQL credentials
)

echo.
echo ✓ Setup complete!
echo.
echo Next steps:
echo 1. Edit .env with your API keys
echo 2. Run: python main.py
echo.

pause
