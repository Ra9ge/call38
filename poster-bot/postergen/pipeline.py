"""Весь процесс: постер -> поиск японского текста -> стирание -> русское название."""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter

from .compose import (add_badge, add_watermark, badge_reserve, detect_faces, render_overlay, render_replacement,
                      watermark_rect)
from .detect import detect_text, find_title
from .inpaint import inpaint
from .logo import analyze_logo, text_removal_mask
from .titles import resolve_titles

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Options:
    mode: str = "auto"                  # auto | replace | overlay
    badge: str | None = None            # например «НОВАЯ ОЗВУЧКА»
    badge_color: tuple = (226, 52, 58)
    logo: Path | None = ROOT / "assets" / "logo.png"
    logo_corner: str = "tl"
    inpaint: str = "auto"               # auto | lama | opencv
    font: str | None = None             # принудительно: comfortaa | rubik | montserrat | путь к .ttf
    playful: bool = True                # «прыгающие» буквы для милых логотипов
    keep_other_text: bool = False       # не стирать мелкий японский текст
    title: str | None = None            # своё русское название
    subtitle: str | None = None         # своя подпись снизу ('' — без неё)
    label: str | None = None            # своя подпись сверху, например «ВТОРОЙ СЕЗОН» ('' — без неё)
    distress: float = 1.0               # сила потёртостей в режиме наложения
    min_width: int = 900                # маленькие постеры увеличиваем, чтобы текст был чётким
    debug_dir: Path | None = None


@dataclass
class Report:
    mode: str = ""
    japanese: list[str] = field(default_factory=list)
    styles: list[str] = field(default_factory=list)
    inpaint: str = "none"
    title: str = ""
    label: str | None = None
    subtitle: str | None = None


VERBOSE = True   # в ночном режиме шаги каждого постера не печатаем


def log(msg: str):
    if VERBOSE:
        print(msg, file=sys.stderr, flush=True)


def _prepare(img: Image.Image, min_width: int) -> Image.Image:
    img = img.convert("RGB")
    if img.width < min_width:
        k = min_width / img.width
        img = img.resize((min_width, int(round(img.height * k))), Image.LANCZOS)
        img = img.filter(ImageFilter.UnsharpMask(radius=1.2, percent=60, threshold=2))
    return img


def find_logo_subtitle(boxes, cluster) -> list:
    """Латинская строка прямо у японского логотипа (английское название, как «The Eminence in shadow»)."""
    x0, y0, x1, y1 = cluster.bbox
    size = cluster.size
    out = []
    for b in boxes:
        if b.cjk or b.score < 0.6 or sum(ch.isalpha() for ch in b.text) < 4:
            continue
        if b.x1 < x0 - 0.3 * size or b.x0 > x1 + 0.3 * size:
            continue
        if b.y0 > y1 + 0.6 * size or b.y1 < y0 - 0.6 * size:
            continue
        out.append(b)
    return out


def subtitle_text(ocr: str, info) -> str:
    """Текст распознанной английской строки; если OCR ошибся — берём название с Shikimori."""
    norm = lambda t: re.sub(r"[^a-z0-9]", "", t.lower())
    for cand in (getattr(info, "english", None), getattr(info, "name", None)):
        if not cand:
            continue
        r = SequenceMatcher(None, norm(ocr), norm(cand)).ratio()
        if r >= 0.9:
            return ocr          # совпало — оставляем как на постере (регистр букв и т.п.)
        if r >= 0.7:
            return cand
    return ocr


def make_poster(info, poster: Image.Image, opts: Options | None = None) -> tuple[Image.Image, Report]:
    opts = opts or Options()
    rep = Report()
    img = _prepare(poster, opts.min_width)
    arr = np.asarray(img).copy()
    seed = int(getattr(info, "id", 0) or 0)

    parsed, sub = resolve_titles(info)
    title = opts.title or parsed.title
    label = parsed.label if opts.label is None else (opts.label or None)
    subtitle = sub if opts.subtitle is None else (opts.subtitle or None)
    rep.title, rep.label, rep.subtitle = title, label, subtitle

    log("Ищу текст на постере...")
    boxes = detect_text(arr)
    cluster, others = find_title(boxes, arr.shape)
    rep.japanese = [b.text for b in boxes if b.cjk]

    mode = opts.mode
    if mode == "auto":
        mode = "replace" if cluster is not None else "overlay"
    if mode == "replace" and cluster is None:
        log("Японское название не найдено — делаю наложение.")
        mode = "overlay"
    rep.mode = mode

    mask = np.zeros(arr.shape[:2], np.uint8)
    la = None
    logo_sub, logo_sub_size = None, None
    if mode == "replace":
        # английская строка у логотипа стирается вместе с ним и набирается заново под русским названием
        sub_boxes = find_logo_subtitle(boxes, cluster)
        if sub_boxes:
            ocr = " ".join(b.text for b in sorted(sub_boxes, key=lambda b: (round(b.cy / 20), b.cx)))
            logo_sub = subtitle_text(ocr, info) if opts.subtitle is None else (opts.subtitle or None)
            logo_sub_size = max(b.size for b in sub_boxes)
        la = analyze_logo(arr, cluster, extra=sub_boxes)
        mask |= la.mask
        for b in cluster.boxes:
            st = la.styles[id(b)]
            rep.styles.append(f"{b.text}: {st.kind}, цвета {len(st.glyph_colors)}")
    if not opts.keep_other_text:
        other_boxes = [b for c in others for b in c.boxes]
        if other_boxes:
            mask |= text_removal_mask(arr.shape, other_boxes)

    if mask.any():
        log("Стираю японский текст...")
        clean, rep.inpaint = inpaint(arr, mask, opts.inpaint)
    else:
        clean = arr

    out = Image.fromarray(clean).convert("RGBA")
    # где встанет лого — название его обходит
    logo_rect = watermark_rect(out.size, Path(opts.logo), corner=opts.logo_corner) if opts.logo else None
    badge_gap, badge_cap = 0.8, None
    if mode == "replace":
        log("Рисую русское название в стиле оригинала...")
        badge_gap = -0.3   # плашка чуть заходит на тень/обводку названия
        reserve = badge_reserve(out.height, badge_gap) if opts.badge else 0.0
        area, _ = render_replacement(out, cluster, la, title, font=opts.font, playful=opts.playful, seed=seed,
                                     bottom_reserve=reserve, label=label, subtitle=logo_sub,
                                     sub_size=logo_sub_size, avoid=logo_rect)
    else:
        log("Добавляю название поверх постера...")
        faces = detect_faces(clean)
        avoid = [(logo_rect[0], logo_rect[1], logo_rect[2] - logo_rect[0], logo_rect[3] - logo_rect[1])] \
            if logo_rect else []
        area, title_cap = render_overlay(out, title, label, subtitle, faces, avoid=avoid, seed=seed,
                                         distress=opts.distress,
                                         bottom_reserve=badge_reserve(out.height, badge_gap) if opts.badge else 0.0)
        badge_cap = title_cap * 0.7

    if opts.badge:
        add_badge(out, opts.badge, area, color=opts.badge_color, gap=badge_gap, cap=badge_cap)
    if opts.logo:
        add_watermark(out, Path(opts.logo), corner=opts.logo_corner, avoid=area)

    if opts.debug_dir:
        d = Path(opts.debug_dir)
        d.mkdir(parents=True, exist_ok=True)
        vis = arr.copy()
        for b in boxes:
            cv2.polylines(vis, [np.round(b.quad).astype(np.int32)], True,
                          (255, 0, 0) if b.cjk else (0, 160, 255), 2)
        Image.fromarray(vis).save(d / "1_detected.png")
        Image.fromarray(mask).save(d / "2_mask.png")
        Image.fromarray(clean).save(d / "3_clean.png")
    return out.convert("RGB"), rep
