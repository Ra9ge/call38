#!/usr/bin/env bash
# Запуск: ./run.sh 13759   (первый запуск сам ставит всё нужное)
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
exec .venv/bin/python make_poster.py "$@"
