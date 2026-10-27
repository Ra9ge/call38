#!/usr/bin/env bash
# Ночной режим: ./night.sh   (или ./night.sh --hours 8)
set -e
cd "$(dirname "$0")"
./setup.sh
exec .venv/bin/python batch.py "$@"
