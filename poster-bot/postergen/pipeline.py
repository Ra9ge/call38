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
from .block import CLASS_RU, main_style, render_block, use_block
from .compose import row_geoms
from .detect import TextBox, detect_text, find_logo, reread_vertical
from .inpaint import halo_mask, inpaint
from .logo import analyze_logo, text_removal_mask
from .titles import clean_title, display_title, resolve_titles, title_match, title_variants

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Options:
    mode: str = "auto"                  # auto | replace | overlay
    badge: str | None = None            # например «НОВАЯ ОЗВУЧКА»
    badge_color: tuple = (226, 52, 58)
    logo: Path | None = ROOT / "assets" / "logo.png"
    logo_corner: str = "tl"
    inpaint: str = "auto"               # auto | lama | opencv
    font: str | None = None             # принудительно: heavy | grunge | rounded | tech | serif | путь к .ttf
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
    logo: str | None = None              # какой логотип нашёлся: «японский», «латиница» или None
    font: str | None = None              # каким шрифтом нарисовано название
    warnings: list[str] = field(default_factory=list)


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


def find_logo_subtitle(boxes, cluster, skip=()) -> list:
    """Латинская строка прямо у японского логотипа (английское название, как «The Eminence in shadow»)."""
    x0, y0, x1, y1 = cluster.bbox
    size = cluster.size
    skip = {id(b) for b in skip} | {id(b) for b in cluster.boxes}
    out = []
    for b in boxes:
        if id(b) in skip or b.cjk or b.score < 0.6 or sum(ch.isalpha() for ch in b.text) < 4:
            continue
        if b.x1 < x0 - 0.3 * size or b.x0 > x1 + 0.3 * size:
            continue
        if b.y0 > y1 + 0.6 * size or b.y1 < y0 - 0.6 * size:
            continue
        out.append(b)
    return out


def _union_boxes(boxes):
    return (min(b.x0 for b in boxes), min(b.y0 for b in boxes), max(b.x1 for b in boxes), max(b.y1 for b in boxes))


def residual_text(clean: np.ndarray, mask: np.ndarray, logo_mask: np.ndarray | None = None,
                  reach: float = 0.0) -> list:
    """Текст, который остался после стирания: на месте стирания или рядом с логотипом —
    кусок названия, который распознаватель сначала не нашёл (верхний иероглиф столбика и т.п.).
    Область логотипа проверяем отдельным крупным кадром: на всём постере мелочь теряется."""
    if not mask.any():
        return []
    H, W = mask.shape
    near = cv2.dilate(mask, np.ones((15, 15), np.uint8)) > 0
    around = None
    regions = []
    if logo_mask is not None and reach > 0 and logo_mask.any():
        around = cv2.distanceTransform((logo_mask == 0).astype(np.uint8), cv2.DIST_L2, 3) <= reach
        regions.append(around)
    regions.append(mask > 0)
    out, seen = [], []
    for reg in regions:
        ys, xs = np.nonzero(reg)
        pad = int(0.03 * H)
        x0, y0 = max(0, xs.min() - pad), max(0, ys.min() - pad)
        x1, y1 = min(W, xs.max() + pad), min(H, ys.max() + pad)
        for b in detect_text(np.ascontiguousarray(clean[y0:y1, x0:x1])):
            if b.score < 0.5 or b.size < 0.012 * H:
                continue
            g = TextBox(b.quad + np.float32([x0, y0]), b.text, b.score)
            if any(abs(g.cx - o.cx) < 0.3 * o.size and abs(g.cy - o.cy) < 0.3 * o.size for o in seen):
                continue                                   # уже нашли в другом кадре
            bm = g.mask((H, W)) > 0
            if not bm.any():
                continue
            wordy = g.cjk or sum(c.isalnum() for c in g.text) >= 2
            if wordy and near[bm].mean() > 0.5:
                out.append(g)
            elif around is not None and around[bm].mean() > 0.5 and (
                    g.cjk or (g.size >= 0.4 * reach and g.score >= 0.7)):
                # рядом с логотипом: японский текст или уверенно найденная буква размером с буквы
                # логотипа (одиночный иероглиф распознаватель часто читает латиницей: «ワ» → «D»)
                out.append(g)
            seen.append(g)
    return out


def _all_caps(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return len(letters) >= 3 and all(c.isupper() for c in letters)


def _case_like(text: str, ref: str) -> str:
    """Регистр как на постере: «SWORD ART ONLINE» → заглавными."""
    letters = [c for c in ref if c.isalpha()]
    if letters and all(c.isupper() for c in letters):
        return text.upper()
    return text


def subtitle_text(ocr: str, info) -> str:
    """Текст распознанной английской строки; если OCR ошибся или слепил слова — берём название с Shikimori."""
    norm = lambda t: re.sub(r"[^a-z0-9]", "", t.lower())
    for cand in (getattr(info, "english", None), getattr(info, "name", None)):
        if not cand:
            continue
        cand = clean_title(cand)
        r = SequenceMatcher(None, norm(ocr), norm(cand)).ratio()
        if norm(ocr) == norm(cand) and ocr.count(" ") >= cand.count(" "):
            return ocr          # совпало буква в букву — оставляем как на постере (регистр и т.п.)
        if r >= 0.7:
            return _case_like(cand, ocr)      # распознано с ошибками — берём название с Shikimori
    return ocr


def latin_logo_subtitle(found, info) -> str | None:
    """Под русским названием — оригинальное название с логотипа (как «HUNTER × HUNTER»)."""
    ocr = " ".join(b.text for r in found.cluster.rows for b in r.boxes)
    cand = found.matched_title if found.match >= 0.6 else (getattr(info, "english", None) or info.name)
    english = getattr(info, "english", None)
    if english and title_match(ocr, [english])[0] >= 0.6:
        cand = english                       # английское название понятнее ромадзи
    if not cand or cand == getattr(info, "russian", None) or not re.search(r"[A-Za-z]", cand):
        cand = getattr(info, "english", None) or getattr(info, "name", None)
    if not cand:
        return None
    text = _case_like(clean_title(cand), ocr)
    if "×" in (getattr(info, "japanese", None) or "") + ocr:
        text = re.sub(r"\s+[xX]\s+", " × ", text)       # HUNTER X HUNTER → HUNTER × HUNTER
    return text


def make_poster(info, poster: Image.Image, opts: Options | None = None) -> tuple[Image.Image, Report]:
    opts = opts or Options()
    rep = Report()
    img = _prepare(poster, opts.min_width)
    arr = np.asarray(img).copy()
    seed = int(getattr(info, "id", 0) or 0)

    parsed, sub = resolve_titles(info)
    label = parsed.label if opts.label is None else (opts.label or None)
    subtitle = sub if opts.subtitle is None else (opts.subtitle or None)

    log("Ищу текст на постере...")
    boxes = reread_vertical(arr, detect_text(arr), title_variants(info))
    found = find_logo(boxes, arr.shape, title_variants(info))
    cluster, others = found.cluster, found.others
    rep.japanese = [b.text for b in boxes if b.cjk]
    rep.logo = None if cluster is None else ("латиница" if found.latin else "японский")

    mode = opts.mode
    if mode == "auto":
        mode = "replace" if cluster is not None else "overlay"
    if mode == "replace" and cluster is None:
        log("Название на постере не найдено — делаю наложение.")
        mode = "overlay"
    rep.mode = mode
    if mode == "replace":
        # на месте логотипа — название с номером сезона в конце («… 2»), без отдельной подписи сезона
        title = opts.title or display_title(info)
        label = opts.label or None
    else:
        title = opts.title or parsed.title     # наложение: подпись «ВТОРОЙ СЕЗОН» над названием, как на образце
    rep.title, rep.label, rep.subtitle = title, label, subtitle if mode == "overlay" else None

    mask = np.zeros(arr.shape[:2], np.uint8)
    la = None
    logo_sub, logo_sub_size = None, None
    if mode == "replace":
        # подпись другим языком у логотипа стирается вместе с ним и набирается заново под русским названием
        sub_boxes = found.subtitle + find_logo_subtitle(boxes, cluster, skip=found.subtitle + found.decor)
        if found.latin:
            # логотип латиницей: под русским — оригинальное название с логотипа
            logo_sub = latin_logo_subtitle(found, info)
            logo_sub_size = cluster.size * 0.32
        elif sub_boxes:
            ocr = " ".join(b.text for b in sorted(sub_boxes, key=lambda b: (round(b.cy / 20), b.cx)))
            logo_sub = subtitle_text(ocr, info)
            logo_sub_size = max(b.size for b in sub_boxes)
        if opts.subtitle is not None:
            logo_sub = opts.subtitle or None
        rep.subtitle = logo_sub
        la = analyze_logo(arr, cluster, extra=sub_boxes + found.decor)
        mask |= la.mask
        mask |= halo_mask(arr, la.mask, cluster.size)      # тень/свечение букв — тоже стираем
        for b in cluster.boxes:
            st = la.styles[id(b)]
            rep.styles.append(f"{b.text}: {st.kind}, цвета {len(st.glyph_colors)}")
    if not opts.keep_other_text:
        other_boxes = [b for c in others for b in c.boxes]
        if other_boxes:
            mask |= text_removal_mask(arr.shape, other_boxes)

    if mask.any():
        log("Стираю старое название...")
        clean, rep.inpaint = inpaint(arr, mask, opts.inpaint)
        # проверка: не осталось ли на этом месте текста; если остался — стираем ещё раз
        logo_mask = la.mask if la is not None else None
        reach = 1.2 * cluster.size if la is not None else 0.0
        for _ in range(2):
            left = residual_text(clean, mask, logo_mask, reach)
            if not left:
                break
            extra = np.zeros_like(mask)
            for b in left:
                piece = cv2.dilate(b.mask(arr.shape), cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (2 * int(b.size * 0.3) + 7,) * 2))
                extra |= piece | halo_mask(arr, piece, b.size)
            mask |= extra
            # дорисовываем только найденные остатки, остальное уже чистое
            clean, rep.inpaint = inpaint(clean, extra, opts.inpaint)
        else:
            if residual_text(clean, mask, logo_mask, reach):
                rep.warnings.append("старый текст стёрся не полностью")
    else:
        clean = arr

    out = Image.fromarray(clean).convert("RGBA")
    # где встанет лого — название его обходит
    logo_rect = watermark_rect(out.size, Path(opts.logo), corner=opts.logo_corner) if opts.logo else None
    badge_gap, badge_cap = 0.8, None
    if mode == "replace":
        log("Рисую русское название...")
        badge_gap = -0.3   # плашка чуть заходит на тень/обводку названия
        reserve = badge_reserve(out.height, badge_gap) if opts.badge else 0.0
        drop = any(g.drop for g in row_geoms(cluster.rows, None))
        if use_block(la) and not (drop and main_style(la, cluster).thin):
            region = _union_boxes(list(cluster.boxes) + list(sub_boxes) + list(found.decor))
            area, cls = render_block(out, cluster, la, title, logo_sub, genres=getattr(info, "genres", ()),
                                     forced_font=opts.font, region=region, avoid=logo_rect, bottom_reserve=reserve)
            rep.font = CLASS_RU.get(cls, cls)
        else:
            # разноцветные буквы, таблички, буквица — копия оригинального логотипа
            area, fc = render_replacement(out, cluster, la, title, font=opts.font, playful=opts.playful, seed=seed,
                                          bottom_reserve=reserve, label=label, subtitle=logo_sub,
                                          sub_size=logo_sub_size, avoid=logo_rect,
                                          upper=found.latin and _all_caps(" ".join(b.text for b in cluster.boxes)))
            rep.font = "копия оригинала"
    else:
        log("Добавляю название поверх постера...")
        faces = detect_faces(clean)
        avoid = [(logo_rect[0], logo_rect[1], logo_rect[2] - logo_rect[0], logo_rect[3] - logo_rect[1])] \
            if logo_rect else []
        # не ставим название поверх надписей, которые остались на постере
        H = arr.shape[0]
        avoid += [(b.x0, b.y0, b.x1 - b.x0, b.y1 - b.y0) for b in boxes if b.size >= 0.015 * H]
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
