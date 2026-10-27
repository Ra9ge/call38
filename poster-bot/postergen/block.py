"""Русское название «блоком» на месте оригинального логотипа.

Для большинства постеров копировать оригинальный логотип буква в букву не получается:
шрифты и эффекты у всех разные. Поэтому рисуем аккуратный блок по одним правилам:

- шрифт — по виду оригинального логотипа (тонкий / потёртый / обычный) и жанрам аниме;
- цвета — из оригинала (буквы, обводка, градиент), но только если они читаются на фоне;
- размер — не больше места, которое занимал логотип; 1–3 строки, предлог не остаётся один;
- под названием мелко — оригинальное название (если оно было на логотипе латиницей).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .compose import FontChoice, _balanced_lines, _fit_vertically
from .detect import Cluster
from .elements import outlined_text, sticker_text
from .logo import BoxStyle, LogoAnalysis
from .titles import title_words
from .typo import Element, font_for_cap, measure

FONT_CLASSES = {
    "heavy": FontChoice("dela", 400),        # жирный «японский плакатный»
    "grunge": FontChoice("dirt", 400),       # потёртый
    "rounded": FontChoice("rounded", 900),   # округлый
    "tech": FontChoice("russo", 400),        # «техно»
    "serif": FontChoice("playfair", 900),    # с засечками
}
CLASS_RU = {"heavy": "жирный", "grunge": "потёртый", "rounded": "округлый", "tech": "техно", "serif": "с засечками"}

# какие жанры Shikimori к какому шрифту тянут (вес)
GENRE_WEIGHTS = {
    "rounded": {"Romance": 2, "Comedy": 1, "Slice of Life": 2, "School": 1, "Shoujo": 2, "Kids": 2, "Iyashikei": 2,
                "CGDCT": 2, "Gourmet": 1, "Music": 1, "Idols (Female)": 2, "Idols (Male)": 2, "Girls Love": 1,
                "Boys Love": 1, "Love Polygon": 2, "Childcare": 2, "Educational": 1},
    "heavy": {"Action": 2, "Shounen": 1, "Martial Arts": 2, "Sports": 2, "Team Sports": 1, "Combat Sports": 2,
              "Super Power": 1, "Adventure": 1, "Delinquents": 2, "Military": 1, "Survival": 1, "Strategy Game": 1},
    "grunge": {"Horror": 3, "Gore": 4, "Psychological": 1, "Thriller": 2, "Suspense": 1, "Seinen": 1},
    "tech": {"Mecha": 3, "Space": 2, "Sci-Fi": 2, "Racing": 2, "Video Game": 1, "Military": 1},
    "serif": {"Drama": 1, "Mystery": 2, "Historical": 2, "Supernatural": 1, "Vampire": 2, "Samurai": 1,
              "Mythology": 2, "Detective": 2, "Josei": 1, "Fantasy": 1},
}
ORDER = ("heavy", "rounded", "tech", "serif", "grunge")      # при равенстве


def main_style(la: LogoAnalysis, cluster: Cluster) -> BoxStyle:
    b = max(cluster.boxes, key=lambda b: b.size * b.run)
    return la.styles[id(b)]


def font_class(st: BoxStyle, genres=(), forced: str | None = None) -> str:
    """Какой шрифт: тонкий логотип — с засечками; иначе по жанрам; если жанров нет — жирный.
    (Потёртость по пикселям надёжно не отличить от градиента и шума JPEG, поэтому потёртый
    шрифт выбирается по жанрам: ужасы, жестокость, триллер.)"""
    if forced in FONT_CLASSES:
        return forced
    if st.thin:
        return "serif"
    score = {k: sum(w.get(g, 0) for g in genres or ()) for k, w in GENRE_WEIGHTS.items()}
    best = max(ORDER, key=lambda k: (score[k], -ORDER.index(k)))
    return best if score[best] > 0 else "heavy"


def use_block(la: LogoAnalysis) -> bool:
    """Копию оригинала (как «Сакурасо») делаем только для разноцветных букв и табличек/лепестков."""
    for st in la.styles.values():
        if st.kind in ("plate", "badge"):
            return False
        if st.kind == "sticker" and len(st.glyph_colors) >= 3:
            return False
    return True


# ---------------------------------------------------------------- цвета
def _lab(rgb) -> np.ndarray:
    return cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2LAB)[0, 0].astype(np.float32)


def _readable(a, b, min_dl: float = 45, min_de: float = 55) -> bool:
    la, lb = _lab(a), _lab(b)
    return abs(float(la[0] - lb[0])) >= min_dl or float(np.linalg.norm(la - lb)) >= min_de


def _is_light(rgb) -> bool:
    return float(_lab(rgb)[0]) > 140


def block_colors(st: BoxStyle):
    """(цвет букв, цвет обводки, градиент или None) — из оригинала, с проверкой, что читается."""
    fill = st.glyph_colors[0] if st.glyph_colors else (255, 255, 255)
    bg = st.bg
    gradient = st.gradient
    outline = st.outline_color
    if outline is not None and not _readable(outline, fill, 50, 60):
        outline = None
    if bg is not None and not _readable(fill, bg):
        if outline is not None and _readable(outline, bg):
            pass        # буквы похожи на фон, но их держит обводка (белые буквы с красной обводкой на светлом)
        else:
            # буквы почти сливаются с фоном — берём белые или тёмные
            fill = (255, 255, 255) if not _is_light(bg) else (28, 26, 34)
            gradient = None
            outline = None
    if gradient and bg is not None and not all(_readable(c, bg, 35, 45) for c in gradient):
        gradient = None
    if outline is None:
        outline = (22, 20, 30) if _is_light(fill) else (255, 255, 255)
    return fill, outline, gradient


# ---------------------------------------------------------------- раскладка
def _line_width(text: str, fc: FontChoice, cap: float) -> float:
    return measure(text, font_for_cap(fc.name, cap, fc.weight), 0.02)


GLUE = "\x00"     # «приклеить к предыдущему слову без пробела» (часть слова после дефиса)


def _tokens(title: str) -> list[str]:
    """Слова для переноса; длинные слова через дефис можно разорвать после дефиса
    («ЧЕЛОВЕК-» / «БЕНЗОПИЛА»)."""
    out = []
    for w in title_words(title):
        parts = w.split("-")
        if len(parts) > 1 and all(len(p) >= 3 for p in parts):
            out.append(parts[0] + "-")
            out += [GLUE + p + ("-" if i < len(parts) - 2 else "") for i, p in enumerate(parts[1:])]
        else:
            out.append(w)
    return out


def _join(tokens: list[str]) -> str:
    return " ".join(tokens).replace(" " + GLUE, "").replace(GLUE, "")


def plan_lines(title: str, fc: FontChoice, max_w: float, max_h: float, cap_max: float, min_cap: float):
    """Сколько строк и какого размера: крупнее — лучше, но строк поменьше."""
    words = _tokens(title)
    best = None
    for n in (1, 2, 3):
        if n > len(words):
            break
        lines = [_join(ln) for ln in _balanced_lines(words, n)]
        widest = max(_line_width(ln, fc, 100.0) for ln in lines)
        cap_w = 100.0 * max_w / max(1.0, widest)
        cap_h = max_h / (1 + 1.32 * (n - 1))
        cap = min(cap_w, cap_h, cap_max)
        score = cap * (1 - 0.08 * (n - 1))
        if best is None or score > best[0] * 1.04:
            best = (score, cap, lines)
    _, cap, lines = best
    return max(cap, min_cap), lines


def _stack(els: list[Element], ys: list[float]) -> Element:
    """Строки по центру, точки привязки (середина заглавных) — на высотах ys."""
    left = min(-e.cx for e in els)
    right = max(e.image.width - e.cx for e in els)
    top = min(y - e.cy for e, y in zip(els, ys))
    bottom = max(y - e.cy + e.image.height for e, y in zip(els, ys))
    W, H = int(np.ceil(right - left)) + 2, int(np.ceil(bottom - top)) + 2
    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    for e, y in zip(els, ys):
        canvas.alpha_composite(e.image, (int(round(-left - e.cx)), int(round(y - top - e.cy))))
    return Element(canvas, -left, (ys[0] + ys[-1]) / 2 - top, max(e.w for e in els), ys[-1] - ys[0] + els[0].h)


def render_block(poster: Image.Image, cluster: Cluster, la: LogoAnalysis, title: str, subtitle: str | None,
                 genres=(), forced_font: str | None = None, region=None, avoid=None,
                 bottom_reserve: float = 0.0) -> tuple[tuple, str]:
    """Рисует блок. region — где был логотип (вместе с подписью). Возвращает (занятая область, шрифт)."""
    W, H = poster.size
    st = main_style(la, cluster)
    cls = font_class(st, genres, forced_font)
    fc = FONT_CLASSES[cls]
    if forced_font and forced_font not in FONT_CLASSES:
        fc = FontChoice(forced_font, 900)              # свой шрифт: имя из fonts/ или путь к .ttf
        cls = Path(forced_font).stem
    fill, outline, gradient = block_colors(st)

    rx0, ry0, rx1, ry1 = region or cluster.bbox
    lw, lh = rx1 - rx0, ry1 - ry0
    # японское название столбиком вдоль края — русское тоже пишем вдоль края (повёрнутым),
    # а не поперёк постера через лица
    vertical = lh >= 1.8 * lw and any(b.vertical for b in cluster.boxes)
    if vertical:
        max_w = min(0.86 * H, max(1.1 * lh, 0.5 * H))                       # длина строки — вдоль столбика
        max_h = min(max(1.5 * lw, 0.08 * W), 0.3 * W) / (1.2 if subtitle else 1.0)
    else:
        # блоку можно быть шире узкого логотипа (русские названия длиннее), но не выше, чем был логотип
        max_w = min(0.9 * W, max(1.15 * lw, 0.8 * W))
        # по высоте — чуть больше места логотипа (длинные русские названия), но не на пол-постера
        max_h = min(max(1.45 * lh, 0.09 * H), 0.26 * H) / (1.2 if subtitle else 1.0)
    cap_max = min(0.62 * max(b.size for b in cluster.boxes), 0.11 * H)
    text = title.upper()
    cap, lines = plan_lines(text, fc, max_w, max_h, cap_max, 0.024 * H)

    out_k = {"heavy": 0.10, "rounded": 0.11, "tech": 0.09, "serif": 0.07, "grunge": 0.08}.get(cls, 0.09)

    def build(cap_):
        els = [sticker_text(ln, fc.name, fc.weight, cap_, [fill], cap_ * out_k, outline, shadow=True,
                            gradient=gradient, tracking=0.02, shadow_opacity=0.45, cap_anchor=True) for ln in lines]
        # от строки до строки — по высоте заглавных; у «Й», «Ё» значок над буквой — чуть больше места
        accent = 0.1 if any(ch in "ЙЁ" for ln in lines[1:] for ch in ln) else 0.0
        pitch = cap_ * (1.32 + out_k + accent)
        ys = [i * pitch for i in range(len(els))]
        if subtitle:
            sub_cap = float(np.clip(cap_ * 0.3, 0.013 * H, 0.03 * H))
            f = font_for_cap("montserrat", sub_cap, 800)
            title_w = max(e.w for e in els)
            sw = measure(subtitle, f, 0.1)
            if sw > title_w * 1.05 and sw > 0:
                sub_cap *= max(0.6, title_w * 1.05 / sw)
            els.append(outlined_text(subtitle, "montserrat", 800, sub_cap, fill=fill if not gradient else gradient[0],
                                     outline=outline, outline_px=sub_cap * 0.16, tracking=0.1, shadow_opacity=0.45,
                                     cap_anchor=True))
            ys.append(ys[-1] + cap_ * (0.5 + out_k) + cap_ * 0.4 + sub_cap * 0.5)
        return _stack(els, ys)

    el = build(cap)
    bb = el.image.getbbox()
    limit = 0.94 * (H if vertical else W)
    if bb[2] - bb[0] > limit:                          # на всякий случай — не шире постера
        cap *= limit / (bb[2] - bb[0])
        el = build(cap)
    if vertical:
        # слева читается снизу вверх, справа — сверху вниз (как корешок книги)
        el = el.rotated(-90 if (rx0 + rx1) / 2 < W / 2 else 90)

    cx, cy = (rx0 + rx1) / 2, (ry0 + ry1) / 2
    vis = el.image.getchannel("A").point(lambda a: 255 if a > 40 else 0).getbbox() or (0, 0, *el.image.size)
    m = 0.03 * W
    half_l, half_r = el.cx - vis[0], vis[2] - el.cx
    cx = min(max(cx, m + half_l), W - m - half_r)
    if vertical:     # прижимаем к своему краю, как было
        cx = m + half_l if (rx0 + rx1) / 2 < W / 2 else W - m - half_r
        cx = (cx + min(max((rx0 + rx1) / 2, m + half_l), W - m - half_r)) / 2
    rect = (cx - half_l, cy - (el.cy - vis[1]), cx + half_r, cy + (vis[3] - el.cy))
    cy += _fit_vertically(rect, W, H, bottom_reserve, avoid)
    el.paste_on(poster, cx, cy)
    return el.bbox_at(cx, cy), cls


