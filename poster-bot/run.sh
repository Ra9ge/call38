#!/usr/bin/env bash
# Запуск: ./run.sh 13759   (первый запуск сам ставит всё нужное)
set -e
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo "Первый запуск: создаю окружение и ставлю библиотеки..."
  python3 -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements.txt
fi
exec .venv/bin/python make_poster.py "$@"
