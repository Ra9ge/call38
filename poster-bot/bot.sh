#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
./setup.sh
exec .venv/bin/python telegram_bot.py
