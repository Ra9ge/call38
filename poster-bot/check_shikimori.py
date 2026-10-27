#!/usr/bin/env python3
"""Проверка связи с Shikimori: что именно работает, а что нет, и что с этим делать.

    python check_shikimori.py
    python check_shikimori.py --proxy socks5://127.0.0.1:10808
"""
from __future__ import annotations

import argparse
import socket
import sys
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from make_poster import _safe_console  # noqa: E402
from postergen import shikimori as sh  # noqa: E402

TEST_ID = 13759


def step(name: str, fn):
    t = time.time()
    try:
        res = fn()
        print(f"   [OK]  {name}: {res}  ({time.time() - t:.1f} с)")
        return True, res
    except Exception as e:  # проверка — показываем любую причину
        print(f"   [--]  {name}: {sh.describe_error(e)}  ({time.time() - t:.1f} с)")
        return False, None


def check_domain(domain: str, session) -> bool:
    host = urlparse(domain).hostname
    print(f"\n{domain}")
    if not sh.PROXY:
        step("DNS", lambda: ", ".join(sorted({a[4][0] for a in socket.getaddrinfo(host, 443)}))[:60])

    def rest():
        r = sh._request("GET", f"{domain}/api/animes/{TEST_ID}", session)
        if r.status_code != 200:
            raise sh._http_fail(r, "REST")
        return f"«{sh._json(r, 'REST').get('russian')}»"

    def gql_one():
        items = sh._gql(domain, sh.GRAPHQL_QUERY, session, {"ids": str(TEST_ID)})
        if not items:
            raise sh._Fail("пустой ответ")
        return f"«{items[0].get('russian')}», постер: {(items[0].get('poster') or {}).get('originalUrl')}"

    def gql_list():
        q = "{ animes(page: 1, limit: 5, order: popularity, censored: true, kind: \"tv,movie,ova,ona\") {" \
            + sh.FIELDS + "} }"
        items = sh._gql(domain, q, session)
        return f"{len(items)} шт.: " + ", ".join(a.get("russian") or a.get("name") or "?" for a in items[:3])

    ok_rest, _ = step("REST API (данные аниме)", rest)
    ok_one, _ = step("GraphQL (данные + постер)", gql_one)
    ok_list, _ = step("GraphQL (список аниме для ночного режима)", gql_list)

    ok_img = False
    if ok_one or ok_rest:
        def poster():
            sh._good_domain = domain
            info = sh.fetch_anime(TEST_ID, session)
            data = sh.download_poster_bytes(info, session)
            return f"{len(data) // 1024} КБ"
        ok_img, _ = step("скачивание постера", poster)
    return (ok_one or ok_rest) and ok_img


def main() -> int:
    _safe_console()
    p = argparse.ArgumentParser(description="Проверка связи с Shikimori")
    p.add_argument("--proxy", help="http://127.0.0.1:10809 или socks5://127.0.0.1:10808")
    args = p.parse_args()
    if args.proxy:
        sh.set_proxy(args.proxy)
    print(f"Python {sys.version.split()[0]}")
    system = {k: v for k, v in urllib.request.getproxies().items() if k in ("http", "https", "all")}
    print(f"Прокси: {sh.PROXY or 'нет'}; системный: {system or 'нет'}")
    session = sh._session(None)
    good = [d for d in sh._domains() if check_domain(d, session)]

    print()
    if good:
        print(f"ИТОГ: Shikimori работает через {good[0]} — можно запускать night.bat"
              + (f" --proxy {sh.PROXY}" if sh.PROXY else "") + ".")
        return 0
    print("ИТОГ: ни один адрес Shikimori не работает из этой программы.")
    print("Что сделать:")
    print("  1. Открой https://shikimori.io в браузере. Если не открывается и там — сайт недоступен")
    print("     из твоей сети (блокировка или сбой у Shikimori). Нужен VPN.")
    print("  2. Если в браузере открывается только с VPN-расширением — программа его не видит.")
    print("     Включи VPN для всей системы (режим TUN / «системный прокси» в приложении VPN),")
    print("     или укажи прокси VPN-приложения:  check.bat --proxy socks5://127.0.0.1:10808")
    print("     (порт смотри в настройках VPN-приложения; потом тот же --proxy для night.bat).")
    print("  3. Пришли скриншот этого окна — разберёмся.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
