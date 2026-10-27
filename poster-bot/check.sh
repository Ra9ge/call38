#!/usr/bin/env bash
# Проверка связи с Shikimori: ./check.sh   (с прокси: ./check.sh --proxy socks5://127.0.0.1:10808)
set -e
cd "$(dirname "$0")"
./setup.sh
exec .venv/bin/python check_shikimori.py "$@"
