"""Раскладка русского названия на постере: замена японского логотипа или наложение."""
from __future__ import annotations

import itertools
import math
import re
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .detect import Cluster, Row
from .elements import distressed_title, outlined_text, petal_text, plate_text, sticker_text
from .logo import BoxStyle, LogoAnalysis
from .titles import ORDINALS_M
from .typo import Element, font_for_cap, measure

# короткие слова, которые нельзя оставлять в конце строки
SHORT_WORDS = {"в", "во", "и", "а", "но", "на", "с", "со", "к", "ко", "у", "о", "об", "от", "до", "из",
               "за", "по", "для", "без", "под", "над", "при", "про", "не", "ни", "или", "the", "of", "a", "to"}

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


@dataclass
class FontChoice:
    name: str
    weight: int
    faux_bold: float = 0.0


CUTE = FontChoice("comfortaa", 700, 0.035)
BOLD = FontChoice("rubik", 800)
CLEAN = FontChoice("montserrat", 800)


# ============================================================ замена логотипа
@dataclass
class Seg:
    kind: str
    style: BoxStyle
    text: str = ""


def _row_segments(row: Row, styles: dict) -> list[Seg]:
    segs: list[Seg] = []
    for b in row.boxes:
        st = styles[id(b)]
        if segs and segs[-1].kind == st.kind and st.kind in ("sticker", "plain"):
            # соседние куски одной строки — один стиль, цвета подряд (копия, оригинал не трогаем)
            prev = segs[-1].style
            colors = prev.glyph_colors + [c for c in st.glyph_colors if c not in prev.glyph_colors]
            segs[-1] = Seg(st.kind, replace(prev, glyph_colors=colors))
            continue
        segs.append(Seg(st.kind, st))
    return segs


def choose_font(styles: list[BoxStyle], forced: str | None = None) -> FontChoice:
    if forced:
        return {"comfortaa": CUTE, "rubik": BOLD, "montserrat": CLEAN}.get(forced, FontChoice(forced, 800))
    kinds = {s.kind for s in styles}
    multicolor = any(len(s.glyph_colors) >= 2 for s in styles if s.kind in ("sticker", "plate"))
    if kinds & {"plate", "badge"} or ("sticker" in kinds and multicolor):
        return CUTE
    if "sticker" in kinds:
        return BOLD
    return CLEAN


def _seg_width(seg: Seg, text: str, fc: FontChoice, cap: float) -> float:
    f = font_for_cap(fc.name, cap, fc.weight)
    w = measure(text, f) * (1 + fc.faux_bold * 0.5)
    if seg.kind == "plate":
        w = measure(text.upper(), f, 0.02) + cap * 1.3
    elif seg.kind == "badge":
        w = measure(text.upper(), f) + cap * 1.6
    return w


def _assign_segments(segs: list[Seg], words: list[str]) -> list[Seg]:
    """Слова строки -> части строки (табличка, лепесток...)."""
    if len(segs) == 1 or len(words) < len(segs):
        main = max(segs, key=lambda s: s.kind != "badge")
        return [Seg(main.kind if main.kind != "badge" else "plate", main.style, " ".join(words))]
    out = [Seg(s.kind, s.style) for s in segs]
    # хвостовым частям (лепесткам) — по одному слову с конца, первой — остальное
    tail = len(segs) - 1
    out[0].text = " ".join(words[:len(words) - tail])
    for i, w in enumerate(words[len(words) - tail:]):
        out[1 + i].text = w
    return out


# служебные подписи японского логотипа -> русские
def _season_word(n: int) -> str:
    return f"{ORDINALS_M.get(n, f'{n}-Й')} СЕЗОН"


_KANJI_NUM = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


def tagline_text(text: str, label: str | None) -> str | None:
    """«TVアニメ» -> «ТВ-АНИМЕ», «第2期» -> «ВТОРОЙ СЕЗОН», «劇場版» -> «ФИЛЬМ»."""
    t = text.replace(" ", "").translate(str.maketrans("０１２３４５６７８９", "0123456789"))
    m = re.search(r"第([0-9]+|[一二三四五六七八九十]+)期", t)
    if m:
        g = m.group(1)
        n = int(g) if g.isdigit() else _KANJI_NUM.get(g[-1], 2)
        return label or _season_word(n)
    m = re.search(r"([0-9]+)(?:st|nd|rd|th)season|シーズン([0-9]+)", t, re.I)
    if m:
        return label or _season_word(int(m.group(1) or m.group(2)))
    if re.search(r"劇場版", t):
        return "ФИЛЬМ"
    if re.search(r"TV\s*アニ|ＴＶアニ|TVアニ", t, re.I):
        return "ТВ-АНИМЕ"
    if re.fullmatch(r"(OVA|OAD|ONA)", t, re.I):
        return t.upper()
    return None


@dataclass
class RowGeom:
    row: Row
    length: float        # вдоль строки
    thick: float         # высота строки
    angle: float         # наклон надписи; -90 — вертикальный логотип
    cx: float
    cy: float
    main: bool = False
    tagline: str | None = None

    @property
    def cap(self) -> float:
        return self.thick * 0.6


def row_geoms(rows: list[Row], label: str | None) -> list[RowGeom]:
    out = []
    for r in rows:
        mb = max(r.boxes, key=lambda b: b.run * b.thick)
        if mb.vertical:
            g = RowGeom(r, mb.thick, mb.run, -90.0, mb.cx, mb.cy)
        else:
            g = RowGeom(r, r.w if abs(r.angle) < 15 else mb.run, r.h, r.angle if abs(r.angle) > 1.5 else 0.0,
                        r.cx, r.cy)
        out.append(g)
    main = max(out, key=lambda g: g.thick * math.sqrt(g.length))
    main.main = True
    for g in out:
        if not g.main:
            text = "".join(b.text for b in g.row.boxes)
            g.tagline = tagline_text(text, label)
    return out


def _avail(g: RowGeom, W: int, H: int) -> float:
    # крупная строка — в ширину оригинала, мелкие таблички могут вырасти
    w = g.length * 1.0 if g.main else max(g.length * 1.35, g.thick * 3)
    return min(w, (H if g.angle == -90 else W) * 0.92)


def _wrap2(words: list[str]) -> list[list[str]]:
    return _balanced_lines(words, 2) if len(words) >= 2 else [words]


def fit_row(g: RowGeom, segs: list[Seg], fc: FontChoice, W: int, H: int):
    """Во сколько раз уменьшить текст, чтобы влез, и нужен ли перенос на 2 строки."""
    avail = _avail(g, W, H)
    cap = g.cap
    s1 = avail / max(1.0, sum(_seg_width(s, s.text, fc, cap) for s in segs))
    if len(segs) == 1 and segs[0].kind in ("sticker", "plain") and len(segs[0].text.split()) >= 2:
        lines = _wrap2(segs[0].text.split())
        lonely = any(len(ln) == 1 and ln[0].lower().strip(",.:") in SHORT_WORDS for ln in lines)
        w2 = max(_seg_width(segs[0], " ".join(ln), fc, cap) for ln in lines)
        s2 = min(avail / max(1.0, w2), 0.8)
        # переносим, только если это заметно крупнее и не оставляет предлог одного в строке
        if not lonely and s2 > s1 * 1.35:
            return s2, lines
    return s1, None


def split_title_over_rows(title: str, geoms: list[RowGeom], styles: dict, fc: FontChoice,
                          W: int, H: int) -> list[list[str]]:
    words = title.split()
    free = [i for i, g in enumerate(geoms) if not g.tagline]
    n = len(free)
    groups: list[list[str]] = [[] for _ in geoms]
    main_i = next(i for i, g in enumerate(geoms) if g.main)
    if n <= 1 or len(words) == 1:
        groups[main_i] = words
        return groups
    best, best_cost = None, 1e9
    total_chars = sum(len(w) for w in words)
    # перебираем все способы разрезать название между строками логотипа
    for cuts in itertools.combinations_with_replacement(range(len(words) + 1), n - 1):
        bounds = (0, *cuts, len(words))
        parts = [words[bounds[k]:bounds[k + 1]] for k in range(n)]
        cand = [[] for _ in geoms]
        for k, i in enumerate(free):
            cand[i] = parts[k]
        if not cand[main_i]:
            continue
        cost = 0.0
        for g, grp in zip(geoms, cand):
            if g.tagline:
                continue
            if not grp:
                cost += 0.35
                continue
            segs = _assign_segments(_row_segments(g.row, styles), grp)
            s, wrapped = fit_row(g, segs, fc, W, H)
            cost += abs(math.log(min(s, 1.8))) + (0.2 if wrapped else 0.0)
            if any(x.kind == "badge" and len(x.text) <= 3 for x in segs):
                cost -= 0.3
        if sum(len(w) for w in cand[main_i]) < 0.35 * total_chars:
            cost += 0.4
        if cost < best_cost:
            best, best_cost = cand, cost
    return best or groups


def _render_segment(seg: Seg, text: str, fc: FontChoice, cap: float, playful: bool, seed: int) -> Element:
    st = seg.style
    if seg.kind == "plate":
        return plate_text(text.upper(), fc.name, fc.weight, cap, st.glyph_colors or [(255, 255, 255)],
                          st.plate_color or (240, 200, 60), st.plate_border or (255, 255, 255),
                          st.outline_color, faux_bold=fc.faux_bold)
    if seg.kind == "badge":
        return petal_text(text, fc.name, fc.weight, cap * 0.95, st.plate_color or (220, 80, 150),
                          st.glyph_colors[0] if st.glyph_colors else (255, 255, 255),
                          (255, 255, 255), faux_bold=fc.faux_bold)
    outline_px = st.outline_ratio * cap / 0.6 if st.outline_color else 0
    cases = None
    jitter = None
    if playful and len(st.glyph_colors) >= 3:
        # «СаКуРАсо»: чередуем регистр и чуть раскачиваем буквы
        cases, k = [], 0
        for c in text:
            cases.append(k % 2 == 0)
            k += 0 if c.isspace() else 1
        jitter = {"rot": 4.0, "dy": 0.035, "seed": seed}
    elif seg.kind == "sticker":
        text = text.upper()
    return sticker_text(text, fc.name, fc.weight, cap, st.glyph_colors, outline_px,
                        st.outline_color, shadow=True, jitter=jitter, cases=cases,
                        faux_bold=fc.faux_bold)


def _hstack(elements: list[Element], gap: float, overlap_last: float = 0.0) -> Element:
    """Склеивает части строки (табличка + лепесток) по горизонтали."""
    if len(elements) == 1:
        return elements[0]
    xs = []
    x = 0.0
    for i, e in enumerate(elements):
        left = e.cx - e.image.getbbox()[0] if e.image.getbbox() else e.cx
        if i:
            x += gap - (overlap_last * e.h if i == len(elements) - 1 else 0)
        xs.append(x + left)
        bb = e.image.getbbox() or (0, 0, *e.image.size)
        x += (bb[2] - bb[0])
    cy = max(e.cy for e in elements)
    hmax = max(e.image.height - e.cy for e in elements)
    W = int(math.ceil(x)) + 4
    H = int(math.ceil(cy + hmax)) + 4
    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    for e, ax in zip(elements, xs):
        canvas.alpha_composite(e.image, (int(round(ax - e.cx)) + 2, int(round(cy - e.cy)) + 2))
    ws = sum(e.w for e in elements)
    return Element(canvas, W / 2, cy + 2, ws, max(e.h for e in elements))


def _vstack(elements: list[Element], gap: float) -> Element:
    """Строки одна под другой, по центру. Точка привязки — центр всего блока букв."""
    if len(elements) == 1:
        return elements[0]
    half_w = max(max(e.cx, e.image.width - e.cx) for e in elements)
    Wd = int(math.ceil(2 * half_w)) + 2
    ink_tops, y = [], 0.0
    for e in elements:
        ink_tops.append(y)
        y += e.h + gap
    total_h = y - gap
    # верх картинки каждой строки = верх её букв минус поля (обводка, тень)
    off = max(0.0, max((e.cy - e.h / 2) - t for e, t in zip(elements, ink_tops)))
    Hd = int(math.ceil(max(off + t - (e.cy - e.h / 2) + e.image.height for e, t in zip(elements, ink_tops)))) + 2
    canvas = Image.new("RGBA", (Wd, Hd), (0, 0, 0, 0))
    for e, t in zip(elements, ink_tops):
        canvas.alpha_composite(e.image, (int(round(Wd / 2 - e.cx)), int(round(off + t - (e.cy - e.h / 2)))))
    return Element(canvas, Wd / 2, off + total_h / 2, max(e.w for e in elements), total_h)


def _place_label(poster, text: str, near_box, cap: float):
    """Подпись «ВТОРОЙ СЕЗОН» над названием, если на постере её не было."""
    el = outlined_text(text, "montserrat", 800, cap, fill=(255, 255, 255), outline=(30, 25, 40),
                       outline_px=cap * 0.18, tracking=0.07, shadow_opacity=0.45)
    x0, y0, x1, y1 = near_box
    W, H = poster.size
    cx = min(max((x0 + x1) / 2, el.w / 2 + 0.04 * W), W - el.w / 2 - 0.04 * W)
    y = y0 - cap * 0.5 - el.h / 2
    if y - el.h < 0.02 * H:
        y = y1 + cap * 0.5 + el.h / 2
    el.paste_on(poster, cx, y)
    return el.bbox_at(cx, y)


def render_replacement(poster: Image.Image, cluster: Cluster, la: LogoAnalysis, title: str,
                       font: str | None = None, playful: bool = True, seed: int = 0,
                       bottom_reserve: float = 0.0, label: str | None = None):
    """Рисует русское название на месте японского. Возвращает занятую область."""
    W, H = poster.size
    styles = la.styles
    fc = choose_font(list(styles.values()), font)
    geoms = row_geoms(cluster.rows, label)
    groups = split_title_over_rows(title, geoms, styles, fc, W, H)
    cx0, cy0, cx1, cy1 = cluster.bbox
    multi = len(geoms) > 1
    placed = []
    label_used = False
    for g, words in zip(geoms, groups):
        row = g.row
        if g.tagline:
            segs = [Seg(_row_segments(row, styles)[0].kind, _row_segments(row, styles)[0].style, g.tagline)]
            if segs[0].kind == "badge":
                segs[0].kind = "plate"
            label_used = label_used or g.tagline == label
        elif words:
            segs = _assign_segments(_row_segments(row, styles), words)
        else:
            continue
        scale, lines = fit_row(g, segs, fc, W, H)
        cap = g.cap * min(1.0, scale)
        fun = playful and g.main

        def build(cap_, lines_):
            if lines_:
                seg = segs[0]
                els = [_render_segment(Seg(seg.kind, seg.style, " ".join(ln)), " ".join(ln), fc, cap_, fun, seed + i)
                       for i, ln in enumerate(lines_)]
                return _vstack(els, gap=cap_ * 0.3)
            els = [_render_segment(s, s.text, fc, cap_, fun, seed) for s in segs]
            return _hstack(els, gap=cap_ * 0.25, overlap_last=0.25 if segs[-1].kind == "badge" else 0)

        el = build(cap, lines)
        bb = el.image.getbbox()
        width = bb[2] - bb[0]
        avail = _avail(g, W, H)
        if width > avail * 1.02:
            el = build(cap * avail / width, lines)
        el = el.rotated(g.angle)
        # выравнивание как в оригинале
        bb = el.image.getbbox()
        left_gap, right_gap = row.x0 - cx0, cx1 - row.x1
        if g.angle == -90:
            x = g.cx
        elif multi and left_gap < 0.3 * right_gap:
            x = row.x0 + el.cx - bb[0]          # левый край как у оригинала
        elif multi and right_gap < 0.3 * left_gap:
            x = row.x1 + el.cx - bb[2]          # правый край как у оригинала
        else:
            x = g.cx
        placed.append([g, el, x, g.cy])

    m = 0.02 * W

    def clamp(p):
        g, el, x, y = p
        x0, y0, x1, y1 = el.bbox_at(x, y)
        if x0 < m:
            x += m - x0
        if x1 > W - m:
            x -= x1 - (W - m)
        x0, y0, x1, y1 = el.bbox_at(x, y)
        if y1 > H - m:
            y -= y1 - (H - m)
        if y0 < m:
            y += m - y0
        p[2], p[3] = x, y

    for p in placed:
        clamp(p)

    # мелкие строки не должны сильно наезжать на крупную
    big = next((p for p in placed if p[0].main), None)
    if big:
        bx0, by0, bx1, by1 = big[1].bbox_at(big[2], big[3])
        for p in placed:
            if p is big:
                continue
            g, el, x, y = p
            x0, y0, x1, y1 = el.bbox_at(x, y)
            if min(x1, bx1) - max(x0, bx0) <= 0 or min(y1, by1) - max(y0, by0) <= 0:
                continue
            allowed = 0.3 * (y1 - y0)
            if g.cy < big[0].cy and y1 - by0 > allowed:
                p[3] -= (y1 - by0) - allowed
            elif g.cy >= big[0].cy and by1 - y0 > allowed:
                p[3] += (by1 - y0) - allowed

    # место под плашку снизу: поднимаем всё название целиком
    if placed and bottom_reserve > 0:
        bottom = max(p[1].bbox_at(p[2], p[3])[3] for p in placed)
        lift = bottom - (H - m - bottom_reserve)
        if lift > 0:
            for p in placed:
                p[3] -= lift

    # сначала крупная строка, мелкие (таблички) — поверх, как в оригинале
    boxes = []
    for g, el, x, y in sorted(placed, key=lambda p: (not p[0].main, -p[0].thick)):
        el.paste_on(poster, x, y)
        boxes.append(el.bbox_at(x, y))
    area = _union(boxes)
    if label and not label_used and area and big:
        boxes.append(_place_label(poster, label, area, max(big[0].cap * 0.3, 0.014 * H)))
        area = _union(boxes)
    return area, fc


def _union(boxes):
    if not boxes:
        return None
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))


# ============================================================ наложение (японского текста нет)
def _balanced_lines(words: list[str], n: int) -> list[list[str]]:
    if n <= 1 or len(words) < n:
        return [words]
    best, best_cost = None, 1e9
    for cuts in itertools.combinations(range(1, len(words)), n - 1):
        b = (0, *cuts, len(words))
        lines = [words[b[i]:b[i + 1]] for i in range(n)]
        lens = [sum(len(w) for w in ln) + len(ln) - 1 for ln in lines]
        cost = max(lens) - min(lens) * 0.3
        # предлог в конце строки — плохо
        cost += sum(6 for ln in lines[:-1] if ln[-1].lower().strip(",.:") in SHORT_WORDS)
        if cost < best_cost:
            best, best_cost = lines, cost
    return best


def _line_width(words: list[str], fc: FontChoice, cap: float, sep: bool) -> float:
    f = font_for_cap(fc.name, cap, fc.weight)
    w = sum(measure(x, f, 0.01) for x in words)
    gaps = len(words) - 1
    return w + gaps * (cap * (2 * 0.42 + 0.48) if sep else cap * 0.55)


def detect_faces(img: np.ndarray) -> list[tuple[int, int, int, int]]:
    path = MODELS_DIR / "lbpcascade_animeface.xml"
    try:
        casc = cv2.CascadeClassifier(str(path))
        if casc.empty():
            return []
        g = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
        s = max(24, int(min(img.shape[:2]) * 0.05))
        return [tuple(map(int, f)) for f in casc.detectMultiScale(g, 1.1, 4, minSize=(s, s))]
    except (cv2.error, AttributeError):
        return []


def render_overlay(poster: Image.Image, title: str, label: str | None, subtitle: str | None,
                   faces: list, avoid: list = (), seed: int = 0, separators: bool = True,
                   distress: float = 1.0, bottom_reserve: float = 0.0):
    """Стиль «ОТЕЛЬ ◆ ДЛЯ ◆ НЕЛЮДЕЙ»: подпись сезона, потёртый заголовок, английское название."""
    W, H = poster.size
    fc = FontChoice("unbounded", 800)
    words = title.upper().split()
    max_cap = 0.052 * H
    max_w = 0.88 * W
    # как на образце: в одну строку, пока буквы не становятся слишком мелкими
    options = []
    for n in (1, 2, 3):
        if n > len(words):
            break
        lines = _balanced_lines(words, n)
        sep = separators and len(words) <= 6
        widest = max(_line_width(ln, fc, 100, sep) for ln in lines)
        options.append((min(max_cap, 100 * max_w / widest), lines, sep))
    good = [o for o in options if o[0] >= 0.026 * H]
    cap, lines, sep = good[0] if good else max(options, key=lambda o: o[0])
    cap = max(cap, 0.022 * H)

    title_el = distressed_title(lines, fc.name, fc.weight, cap, separator="diamond" if sep else None,
                                distress=distress, halo=0.6, seed=seed)
    parts = [title_el]
    label_el = sub_el = None
    if label:
        label_el = outlined_text(label, "montserrat", 800, cap * 0.42, tracking=0.07,
                                 shadow_opacity=0.5, halo_opacity=0.55)
        parts.insert(0, label_el)
    if subtitle:
        sub_cap = cap * 0.42
        f = font_for_cap("unbounded", sub_cap, 700)
        sw = measure(subtitle.upper(), f, 0.07)
        if sw > max_w:
            sub_cap *= max_w / sw
        sub_el = outlined_text(subtitle.upper(), "unbounded", 700, max(sub_cap, 0.014 * H), tracking=0.07,
                               shadow_opacity=0.5, halo_opacity=0.55, distress=0.45 * distress, seed=seed + 1)
        parts.append(sub_el)
    gap = cap * 0.42
    heights = [p.h for p in parts]
    block_h = sum(heights) + gap * (len(parts) - 1)

    # где ставить: внизу, если там нет лиц; иначе ищем место получше
    def overlap(y_top):
        x0, x1 = W * 0.07, W * 0.93
        y1 = y_top + block_h
        tot = 0.0
        for fx, fy, fw, fh in list(faces) + list(avoid):
            ix = max(0, min(x1, fx + fw) - max(x0, fx))
            iy = max(0, min(y1, fy + fh) - max(y_top, fy))
            tot += ix * iy
        return tot

    margin = max(0.07 * H, 0.03 * H + bottom_reserve)
    candidates = [H - margin - block_h - k * 0.04 * H for k in range(0, 5)] + [0.08 * H, 0.12 * H]
    y_top = min(candidates, key=lambda y: (overlap(y) > 0, overlap(y), candidates.index(y)))

    y = y_top
    for p in parts:
        p.paste_on(poster, W / 2, y + p.h / 2)
        y += p.h + gap
    return (W * 0.07, y_top, W * 0.93, y_top + block_h), cap


# ============================================================ общее
def badge_reserve(H: int, gap: float = 0.45) -> float:
    """Сколько места снизу нужно плашке «НОВАЯ ОЗВУЧКА» (gap — зазор в долях высоты букв)."""
    return 0.03 * H * (1.15 + gap) + 0.02 * H


def add_badge(poster: Image.Image, text: str, near_box, color=(226, 52, 58), font: str = "montserrat",
              gap: float = 0.45, cap: float | None = None):
    """Плашка «НОВАЯ ОЗВУЧКА» под названием (или над ним, если внизу нет места).

    gap < 0 — плашка слегка заходит на тень названия, как на образце.
    """
    W, H = poster.size
    cap = min(cap or 0.03 * H, 0.03 * H)
    f = font_for_cap(font, cap, 800)
    w = measure(text.upper(), f, 0.03)
    if w > 0.8 * W:
        cap *= 0.8 * W / w
    el = outlined_text(text.upper(), font, 800, cap, fill=color, outline=(255, 255, 255),
                       outline_px=cap * 0.2, tracking=0.03, shadow_opacity=0.5)
    if near_box:
        x0, y0, x1, y1 = near_box
        cx = min(max((x0 + x1) / 2, el.w / 2 + 0.04 * W), W - el.w / 2 - 0.04 * W)
        y = y1 + cap * gap + el.h / 2
        if y + el.h / 2 > H - 0.012 * H:
            y = y0 - cap * abs(gap) - el.h / 2
    else:
        cx, y = W / 2, H * 0.93
    el.paste_on(poster, cx, y)


def add_watermark(poster: Image.Image, logo_path: Path, rel_width: float = 0.12, corner: str = "tl"):
    if not logo_path or not Path(logo_path).exists():
        return
    W, H = poster.size
    logo = Image.open(logo_path).convert("RGBA")
    lw = int(W * rel_width)
    logo = logo.resize((lw, max(1, int(logo.height * lw / logo.width))), Image.LANCZOS)
    m = int(W * 0.03)
    pos = {"tl": (m, m), "tr": (W - lw - m, m), "bl": (m, H - logo.height - m),
           "br": (W - lw - m, H - logo.height - m)}[corner]
    poster.alpha_composite(logo, pos)
