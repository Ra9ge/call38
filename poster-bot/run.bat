@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Первый запуск: создаю окружение и ставлю библиотеки, это пара минут...
    py -3 -m venv .venv 2>nul || python -m venv .venv
    if not exist ".venv\Scripts\python.exe" (
        echo Не найден Python. Установи Python 3.10+ с python.org и отметь "Add python.exe to PATH".
        pause
        exit /b 1
    )
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt || (pause & exit /b 1)
)
".venv\Scripts\python.exe" make_poster.py %*
echo.
echo Готовые постеры лежат в папке output
pause
