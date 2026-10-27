#!/usr/bin/env python3
"""Telegram-бот: присылаешь ID/ссылку с Shikimori — получаешь готовый постер.

Запуск:
    BOT_TOKEN=123:ABC python telegram_bot.py
    (или положи токен в файл bot_token.txt рядом со скриптом)

Сообщения боту:
    13759
    https://shikimori.one/animes/z13759-sakura-sou-no-pet-na-kanojo
    13759 | НОВАЯ ОЗВУЧКА          ← текст после | станет плашкой под названием
    13759 | -                      ← без плашки
    картинка + подпись «13759»     ← взять твой постер вместо постера с Shikimori

Переменные окружения:
    BOT_TOKEN       токен от @BotFather
    BADGE           плашка по умолчанию, например «НОВАЯ ОЗВУЧКА» (по умолчанию без неё)
    ALLOWED_USERS   ID пользователей через запятую, кому можно пользоваться ботом (по умолчанию всем)
"""
from __future__ import annotations

import os
import re
import sys
import tempfile
import time
import traceback
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import make_poster  # noqa: E402

HELP = (
    "Пришли ID или ссылку на аниме с Shikimori — сделаю постер с русским названием.\n\n"
    "Примеры:\n"
    "• 13759\n"
    "• https://shikimori.one/animes/z13759-sakura-sou-no-pet-na-kanojo\n"
    "• 13759 | НОВАЯ ОЗВУЧКА — с плашкой под названием\n"
    "• 13759 | - — без плашки\n"
    "• картинка с подписью 13759 — взять твою картинку вместо постера с Shikimori"
)


def load_token() -> str:
    token = os.environ.get("BOT_TOKEN", "").strip()
    f = ROOT / "bot_token.txt"
    if not token and f.exists():
        token = f.read_text(encoding="utf-8").strip()
    if not token:
        sys.exit("Нет токена: задай BOT_TOKEN или положи токен в bot_token.txt")
    return token


def _safe_console():
    # японские символы в логах не должны ронять скрипт на консолях Windows
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace", line_buffering=True)
        except (AttributeError, ValueError):
            pass


class Bot:
    def __init__(self, token: str):
        self.api = f"https://api.telegram.org/bot{token}/"
        self.file_api = f"https://api.telegram.org/file/bot{token}/"
        self.s = requests.Session()
        allowed = os.environ.get("ALLOWED_USERS", "").replace(" ", "")
        self.allowed = {int(x) for x in allowed.split(",") if x.isdigit()}
        self.default_badge = os.environ.get("BADGE") or None

    def call(self, method: str, **kw):
        files = kw.pop("files", None)
        r = self.s.post(self.api + method, data=kw if files else None, json=None if files else kw,
                        files=files, timeout=120)
        data = r.json()
        if not data.get("ok"):
            raise RuntimeError(f"{method}: {data.get('description')}")
        return data["result"]

    def send(self, chat_id, text, reply_to=None):
        return self.call("sendMessage", chat_id=chat_id, text=text,
                         **({"reply_to_message_id": reply_to} if reply_to else {}))

    def download_photo(self, photos: list) -> Path:
        best = max(photos, key=lambda p: p.get("file_size", 0) or p["width"] * p["height"])
        info = self.call("getFile", file_id=best["file_id"])
        r = self.s.get(self.file_api + info["file_path"], timeout=60)
        r.raise_for_status()
        fd, path = tempfile.mkstemp(suffix=Path(info["file_path"]).suffix or ".jpg")
        with os.fdopen(fd, "wb") as f:
            f.write(r.content)
        return Path(path)

    def handle(self, msg: dict):
        chat_id = msg["chat"]["id"]
        user_id = (msg.get("from") or {}).get("id")
        if self.allowed and user_id not in self.allowed:
            self.send(chat_id, "Этот бот приватный.")
            return
        text = (msg.get("text") or msg.get("caption") or "").strip()
        if not text or text.startswith(("/start", "/help")):
            self.send(chat_id, HELP)
            return
        query, _, badge = text.partition("|")
        badge = badge.strip()
        badge = None if badge == "-" else (badge or self.default_badge)
        m = re.search(r"(?:/animes/)?[a-z]*\d+", query.strip(), re.I)
        if not m:
            self.send(chat_id, "Не нашёл ID аниме. " + HELP, msg["message_id"])
            return
        argv = [query.strip(), "--out", str(ROOT / "output")]
        if badge:
            argv += ["--badge", badge]
        photo_path = None
        if msg.get("photo"):
            photo_path = self.download_photo(msg["photo"])
            argv += ["--image", str(photo_path)]
        self.send(chat_id, "Делаю постер, это займёт 10–30 секунд…", msg["message_id"])
        self.call("sendChatAction", chat_id=chat_id, action="upload_photo")
        try:
            args = make_poster.build_parser().parse_args(argv)
            path = make_poster.make_one(argv[0], args)
            data = path.read_bytes()
            self.call("sendPhoto", chat_id=chat_id, files={"photo": (path.name, data, "image/png")})
            self.call("sendDocument", chat_id=chat_id, caption="Без сжатия",
                      files={"document": (path.name, data, "image/png")})
        except Exception as e:
            traceback.print_exc()
            self.send(chat_id, f"Не получилось: {e}")
        finally:
            if photo_path:
                photo_path.unlink(missing_ok=True)

    def run(self):
        me = self.call("getMe")
        print(f"Бот @{me['username']} запущен. Ctrl+C — остановить.")
        offset = 0
        while True:
            try:
                updates = self.s.get(self.api + "getUpdates",
                                     params={"timeout": 50, "offset": offset}, timeout=70).json()
                for upd in updates.get("result", []):
                    offset = upd["update_id"] + 1
                    msg = upd.get("message")
                    if msg:
                        self.handle(msg)
            except KeyboardInterrupt:
                print("Остановлен.")
                return
            except Exception:
                traceback.print_exc()
                time.sleep(5)


if __name__ == "__main__":
    _safe_console()
    Bot(load_token()).run()
