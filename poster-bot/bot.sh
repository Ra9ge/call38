#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
[ -x .venv/bin/python ] || { echo "Сначала запусти ./run.sh — он поставит библиотеки."; exit 1; }
exec .venv/bin/python telegram_bot.py
