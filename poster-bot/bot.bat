@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Сначала один раз запусти run.bat — он поставит библиотеки.
    pause
    exit /b 1
)
if not exist "bot_token.txt" if "%BOT_TOKEN%"=="" (
    echo Положи токен бота от @BotFather в файл bot_token.txt рядом с этим файлом.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" telegram_bot.py
pause
