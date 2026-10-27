@echo off
chcp 65001 >nul
cd /d "%~dp0"
call "%~dp0setup.bat" || (pause & exit /b 1)
".venv\Scripts\python.exe" make_poster.py %*
echo.
echo Готовые постеры лежат в папке output
pause
