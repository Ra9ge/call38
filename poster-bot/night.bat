@echo off
chcp 65001 >nul
cd /d "%~dp0"
rem Ночной режим: постеры для всех аниме подряд. Просто запусти двойным кликом и иди спать.
rem Можно закрыть окно — при следующем запуске продолжит с того же места.
rem Настройки можно дописать в командную строку, например:
rem   night.bat --hours 8
rem   night.bat --limit 1000 --badge "НОВАЯ ОЗВУЧКА"
rem   night.bat --kind tv --min-score 7
call "%~dp0setup.bat" || (pause & exit /b 1)
".venv\Scripts\python.exe" batch.py %*
echo.
echo Постеры: output\batch\posters   Отчёт: output\batch\index.html
pause
