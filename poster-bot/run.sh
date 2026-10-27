#!/usr/bin/env bash
# Запуск: ./run.sh 13759
set -e
cd "$(dirname "$0")"
./setup.sh
exec .venv/bin/python make_poster.py "$@"
