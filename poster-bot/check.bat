@echo off
chcp 65001 >nul
cd /d "%~dp0"
rem Проверка связи с Shikimori. С прокси: check.bat --proxy socks5://127.0.0.1:10808
call "%~dp0setup.bat" || (pause & exit /b 1)
".venv\Scripts\python.exe" check_shikimori.py %*
pause
