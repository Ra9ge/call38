#!/usr/bin/env bash
# Ночной режим: ./night.sh   (или ./night.sh --hours 8)
set -e
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo "Первый запуск: создаю окружение..."
  python3 -m venv .venv
fi
# библиотеки ставим заново, если поменялся requirements.txt
if ! cmp -s requirements.txt .venv/requirements.installed; then
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements.txt
  cp requirements.txt .venv/requirements.installed
fi
exec .venv/bin/python batch.py "$@"
