#!/usr/bin/env bash
# Готовит окружение .venv: подходящий Python (64-bit, 3.11–3.14) и библиотеки.
set -e
cd "$(dirname "$0")"
if [ -x .venv/bin/python ] && ! .venv/bin/python tools/check_python.py >/dev/null 2>&1; then
  echo "Окружение сделано на неподходящей версии Python — пересоздаю..."
  rm -rf .venv
fi
if [ ! -x .venv/bin/python ]; then
  PY=""
  for c in python3.13 python3.12 python3.14 python3.11 python3; do
    if command -v "$c" >/dev/null 2>&1 && "$c" tools/check_python.py >/dev/null 2>&1; then PY="$c"; break; fi
  done
  if [ -z "$PY" ]; then
    echo "Нужен Python 3.11–3.14 (64-bit): под другие версии нет библиотеки нейросетей onnxruntime."
    echo "Поставь python3.13 (например: brew install python@3.13 / sudo apt install python3.13-venv)."
    exit 1
  fi
  echo "Создаю окружение ($PY)..."
  "$PY" -m venv .venv
fi
# библиотеки ставим заново, если поменялся requirements.txt
if ! cmp -s requirements.txt .venv/requirements.installed; then
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements.txt
  cp requirements.txt .venv/requirements.installed
fi
