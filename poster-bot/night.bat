@echo off
chcp 65001 >nul
cd /d "%~dp0"
rem Ночной режим: постеры для всех аниме подряд. Просто запусти двойным кликом и иди спать.
rem Можно закрыть окно — при следующем запуске продолжит с того же места.
rem Настройки можно дописать в командную строку, например:
rem   night.bat --hours 8
rem   night.bat --limit 1000 --badge "НОВАЯ ОЗВУЧКА"
rem   night.bat --kind tv --min-score 7
if not exist ".venv\Scripts\python.exe" (
    echo Первый запуск: создаю окружение, это пара минут...
    py -3 -m venv .venv 2>nul || python -m venv .venv
    if not exist ".venv\Scripts\python.exe" (
        echo Не найден Python. Установи Python 3.10+ с python.org и отметь "Add python.exe to PATH".
        pause
        exit /b 1
    )
)
rem библиотеки ставим заново, если поменялся requirements.txt
fc /b requirements.txt ".venv\requirements.installed" >nul 2>&1 || (
    echo Ставлю библиотеки...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt || (pause & exit /b 1)
    copy /y requirements.txt ".venv\requirements.installed" >nul
)
".venv\Scripts\python.exe" batch.py %*
echo.
echo Постеры: output\batch\posters   Отчёт: output\batch\index.html
pause
