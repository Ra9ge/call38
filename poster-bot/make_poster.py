#!/usr/bin/env python3
"""Постер аниме с русским названием по ID Shikimori.

Примеры:
    python make_poster.py 13759
    python make_poster.py https://shikimori.one/animes/z13759-sakura-sou-no-pet-na-kanojo --badge "НОВАЯ ОЗВУЧКА"
    python make_poster.py 13759 40028 52991          # несколько сразу
    python make_poster.py                            # спросит ID
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from postergen.pipeline import Options, make_poster  # noqa: E402
from postergen.shikimori import AnimeInfo, ShikimoriError, download_poster, fetch_anime, parse_anime_id  # noqa: E402


def slug(text: str) -> str:
    s = re.sub(r"[^\w\-]+", "-", text.lower(), flags=re.UNICODE).strip("-")
    return s[:60] or "poster"


def hex_color(s: str) -> tuple:
    s = s.lstrip("#")
    if len(s) != 6:
        raise argparse.ArgumentTypeError("цвет в формате #RRGGBB")
    return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Постеры аниме с русским названием (Shikimori)")
    p.add_argument("anime", nargs="*", help="ID или ссылка на Shikimori (можно несколько)")
    p.add_argument("-o", "--out", default=str(ROOT / "output"), help="папка для готовых постеров")
    p.add_argument("--badge", help='плашка под названием, например "НОВАЯ ОЗВУЧКА"')
    p.add_argument("--badge-color", type=hex_color, default=(226, 52, 58), help="цвет плашки, #RRGGBB")
    p.add_argument("--mode", choices=["auto", "replace", "overlay"], default="auto",
                   help="auto: заменить японское название, а если его нет — наложить своё")
    p.add_argument("--title", help="своё русское название вместо названия с Shikimori")
    p.add_argument("--label", help='подпись над названием, например "ВТОРОЙ СЕЗОН" ("" — убрать)')
    p.add_argument("--subtitle", help='подпись под названием ("" — убрать)')
    p.add_argument("--font", help="шрифт для замены: comfortaa / rubik / montserrat / путь к .ttf")
    p.add_argument("--no-fun", action="store_true", help="без «прыгающих» букв")
    p.add_argument("--keep-other-text", action="store_true", help="не стирать мелкий японский текст")
    p.add_argument("--inpaint", choices=["auto", "lama", "opencv"], default="auto",
                   help="чем стирать текст (auto: нейросеть LaMa, если скачалась, иначе OpenCV)")
    p.add_argument("--logo", default=str(ROOT / "assets" / "logo.png"), help="водяной знак (PNG); none — без него")
    p.add_argument("--logo-corner", choices=["tl", "tr", "bl", "br"], default="tl")
    p.add_argument("--distress", type=float, default=1.0, help="сила потёртостей (0 — без них)")
    p.add_argument("--image", help="взять постер из файла, а не с Shikimori")
    p.add_argument("--format", choices=["png", "jpg", "webp"], default="png")
    p.add_argument("--debug", action="store_true", help="сохранить промежуточные картинки")
    return p


def make_one(value: str, args) -> Path:
    t0 = time.time()
    info = None
    try:
        print(f"\n=== {value} ===")
        info = fetch_anime(value)
        print(f"Аниме: {info.russian or info.name}  ({info.name})")
    except (ShikimoriError, ValueError, OSError) as e:
        if not args.image:
            raise
        print(f"Shikimori недоступен ({str(e).splitlines()[0]}), беру данные из аргументов.")
    if info is None:
        anime_id = parse_anime_id(value) if re.search(r"\d", value) else 0
        info = AnimeInfo(id=anime_id, name=args.title or "", russian=args.title)

    if args.image:
        from PIL import Image
        poster = Image.open(args.image).convert("RGB")
    else:
        print("Скачиваю постер...")
        poster = download_poster(info)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{info.id}-{slug(info.name or info.russian or 'poster')}"
    opts = Options(
        mode=args.mode, badge=args.badge, badge_color=args.badge_color,
        logo=None if str(args.logo).lower() == "none" else Path(args.logo), logo_corner=args.logo_corner,
        inpaint=args.inpaint, font=args.font, playful=not args.no_fun,
        keep_other_text=args.keep_other_text, title=args.title, subtitle=args.subtitle,
        label=args.label, distress=args.distress,
        debug_dir=out_dir / f"{name}-debug" if args.debug else None,
    )
    result, rep = make_poster(info, poster, opts)
    path = out_dir / f"{name}.{args.format}"
    if args.format == "jpg":
        result.save(path, quality=95)
    elif args.format == "webp":
        result.save(path, quality=95)
    else:
        result.save(path)
    how = "замена японского названия" if rep.mode == "replace" else "наложение названия"
    print(f"Режим: {how}; стирание: {rep.inpaint}")
    if rep.japanese:
        print(f"Найден японский текст: {', '.join(rep.japanese)}")
    print(f"Готово за {time.time() - t0:.1f} с: {path}")
    return path


def _safe_console():
    # японские символы в логах не должны ронять скрипт на консолях Windows
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace", line_buffering=True)
        except (AttributeError, ValueError):
            pass


def main(argv=None) -> int:
    _safe_console()
    args = build_parser().parse_args(argv)
    items = args.anime
    if not items:
        try:
            raw = input("ID или ссылка на аниме с Shikimori (можно несколько через пробел): ").strip()
        except EOFError:
            raw = ""
        items = raw.split()
        if not items:
            print("Ничего не введено.")
            return 1
    failed = 0
    for value in items:
        try:
            make_one(value, args)
        except Exception as e:  # продолжаем с остальными ID
            failed += 1
            print(f"Ошибка для {value}: {e}", file=sys.stderr)
            if args.debug:
                raise
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
