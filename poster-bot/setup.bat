@echo off
chcp 65001 >nul
rem Готовит окружение .venv: подходящий Python (64-bit, 3.11–3.14) и библиотеки.
rem Вызывается из run.bat / night.bat / bot.bat, можно запустить и отдельно.
setlocal
cd /d "%~dp0"

rem окружение на неподходящем Python (например 3.15 или 32-bit) — пересоздаём
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" tools\check_python.py >nul 2>&1
    if errorlevel 1 (
        echo Окружение сделано на неподходящей версии Python - пересоздаю...
        rmdir /s /q .venv
    )
)

if not exist ".venv\Scripts\python.exe" (
    call :find_python
    if not defined PY (
        echo Подходящий Python не найден. Пробую поставить Python 3.13...
        py install 3.13 <nul
        call :find_python
    )
)
if not exist ".venv\Scripts\python.exe" (
    if not defined PY goto :no_python
    echo Создаю окружение, это пара минут...
    %PY% -m venv .venv
    if errorlevel 1 goto :fail
)

rem библиотеки ставим заново, если поменялся requirements.txt
fc /b requirements.txt ".venv\requirements.installed" >nul 2>&1
if errorlevel 1 (
    echo Ставлю библиотеки...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 goto :fail
    copy /y requirements.txt ".venv\requirements.installed" >nul
)
endlocal
exit /b 0

:find_python
set "PY="
for %%V in (3.13 3.12 3.14 3.11) do (
    if not defined PY (
        py -%%V tools\check_python.py <nul >nul 2>&1
        if not errorlevel 1 set "PY=py -%%V"
    )
)
if not defined PY (
    python tools\check_python.py <nul >nul 2>&1
    if not errorlevel 1 set "PY=python"
)
exit /b 0

:no_python
echo.
echo Нужен Python 3.13 (64-bit). Сейчас установлен другой:
python tools\check_python.py 2>nul
echo Под него нет библиотеки нейросетей onnxruntime (она есть только для Python 3.11-3.14, 64-bit).
echo Скачай Python 3.13 "Windows installer (64-bit)" отсюда:
echo     https://www.python.org/downloads/windows/
echo При установке отметь "Add python.exe to PATH". Старый Python удалять не нужно.
echo Потом снова запусти этот файл.
endlocal
exit /b 1

:fail
echo.
echo Не получилось подготовить окружение - посмотри сообщения выше.
endlocal
exit /b 1
