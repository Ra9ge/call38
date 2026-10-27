#!/usr/bin/env python3
"""Ночной режим: постеры для всех аниме с Shikimori подряд.

Примеры:
    python batch.py                         # все TV, фильмы, OVA, ONA — сначала популярные (без 18+)
    python batch.py --hours 8               # остановиться через 8 часов
    python batch.py --limit 500             # сделать 500 новых постеров
    python batch.py --kind tv --min-score 7 # только TV с оценкой от 7
    python batch.py --season 2015_2026      # только вышедшие в 2015–2026
    python batch.py --ids-file list.txt     # свой список (ID или ссылки, по одному в строке)
    python batch.py 13759 48316             # конкретные аниме
    python batch.py --offline --redo        # перерисовать всё из скачанных оригиналов (после обновления скрипта)

Можно закрыть окно или нажать Ctrl+C — при следующем запуске продолжит с того же места.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from make_poster import _safe_console, hex_color  # noqa: E402
from postergen.batch import BatchConfig, run_batch  # noqa: E402
from postergen.pipeline import Options  # noqa: E402
from postergen.shikimori import ORDERS, set_proxy  # noqa: E402


def main(argv=None) -> int:
    _safe_console()
    p = argparse.ArgumentParser(description="Постеры для всех аниме подряд (ночной режим)")
    p.add_argument("anime", nargs="*", help="конкретные ID или ссылки (по умолчанию — весь Shikimori)")
    p.add_argument("--ids-file", help="файл со списком ID/ссылок, по одному в строке")
    p.add_argument("-o", "--out", default=str(ROOT / "output" / "batch"), help="папка результата")
    p.add_argument("--kind", default="tv,movie,ova,ona",
                   help="типы через запятую: tv, movie, ova, ona, special, tv_special, music… или all")
    p.add_argument("--status", help="released (вышло), ongoing (выходит), anons (анонс) — через запятую")
    p.add_argument("--season", help="годы: 2024, 2015_2026, summer_2024")
    p.add_argument("--min-score", type=int, help="минимальная оценка на Shikimori")
    p.add_argument("--order", choices=ORDERS, default="popularity", help="в каком порядке делать")
    p.add_argument("--limit", type=int, help="сколько новых постеров сделать за этот запуск")
    p.add_argument("--hours", type=float, help="остановиться через столько часов")
    p.add_argument("--redo", action="store_true", help="переделать и уже готовые")
    p.add_argument("--offline", action="store_true", help="только уже скачанные оригиналы, без интернета")
    p.add_argument("--format", choices=["jpg", "png", "webp"], default="jpg",
                   help="формат (jpg — ~0.5 МБ на постер, png — ~1.6 МБ)")
    p.add_argument("--badge", help='плашка под названием на всех постерах, например "НОВАЯ ОЗВУЧКА"')
    p.add_argument("--badge-color", type=hex_color, default=(226, 52, 58))
    p.add_argument("--logo", default=str(ROOT / "assets" / "logo.png"), help="лого в углу; none — без него")
    p.add_argument("--logo-corner", choices=["tl", "tr", "bl", "br"], default="tl")
    p.add_argument("--inpaint", choices=["auto", "lama", "opencv"], default="auto")
    p.add_argument("--proxy", help="прокси для Shikimori, если VPN только в браузере: "
                   "http://127.0.0.1:10809 или socks5://127.0.0.1:10808")
    p.add_argument("--allow-opencv", action="store_true",
                   help="работать, даже если нейросеть LaMa не скачалась (качество хуже)")
    args = p.parse_args(argv)

    if args.proxy:
        set_proxy(args.proxy)
    ids = list(args.anime)
    if args.ids_file:
        lines = Path(args.ids_file).read_text(encoding="utf-8").splitlines()
        ids += [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]
    opts = Options(
        badge=args.badge, badge_color=args.badge_color,
        logo=None if str(args.logo).lower() == "none" else Path(args.logo), logo_corner=args.logo_corner,
        inpaint=args.inpaint,
    )
    cfg = BatchConfig(
        out_dir=Path(args.out), kind=None if args.kind.lower() == "all" else args.kind, status=args.status,
        season=args.season, min_score=args.min_score, order=args.order, ids=ids or None,
        offline=args.offline, limit=args.limit, max_hours=args.hours, redo=args.redo, fmt=args.format,
        allow_opencv=args.allow_opencv or args.inpaint == "opencv", poster=opts,
    )
    return run_batch(cfg)


if __name__ == "__main__":
    sys.exit(main())
