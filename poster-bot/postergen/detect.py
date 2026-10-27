"""Поиск текста на постере (RapidOCR) и выделение блока с японским названием."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import cv2
import numpy as np

CJK_RE = re.compile(r"[぀-ヿ㐀-䶿一-鿿豈-﫿ｦ-ﾟ]")

_ENGINE = None


def _engine():
    global _ENGINE
    if _ENGINE is None:
        from rapidocr import RapidOCR
        _ENGINE = RapidOCR(params={"Global.log_level": "error", "Global.text_score": 0.3})
    return _ENGINE


@dataclass(eq=False)
class TextBox:
    quad: np.ndarray          # (4, 2) float32: tl, tr, br, bl
    text: str
    score: float

    def __post_init__(self):
        q = self.quad = np.asarray(self.quad, np.float32).reshape(4, 2)
        top = q[1] - q[0]
        left = q[3] - q[0]
        self.run = float(np.hypot(*top))       # длина вдоль строки
        self.thick = float(np.hypot(*left))    # высота строки
        self.angle = math.degrees(math.atan2(top[1], top[0]))
        self.vertical = self.thick > 1.4 * self.run and len(self.text) >= 2
        self.size = min(self.run, self.thick) if self.vertical else self.thick
        self.x0, self.y0 = q.min(0)
        self.x1, self.y1 = q.max(0)
        self.cx, self.cy = q.mean(0)
        self.cjk = bool(CJK_RE.search(self.text))

    def mask(self, shape) -> np.ndarray:
        m = np.zeros(shape[:2], np.uint8)
        cv2.fillPoly(m, [np.round(self.quad).astype(np.int32)], 255)
        return m


@dataclass
class Row:
    boxes: list[TextBox]

    @property
    def x0(self): return min(b.x0 for b in self.boxes)
    @property
    def x1(self): return max(b.x1 for b in self.boxes)
    @property
    def y0(self): return min(b.y0 for b in self.boxes)
    @property
    def y1(self): return max(b.y1 for b in self.boxes)
    @property
    def h(self): return max(b.size for b in self.boxes)
    @property
    def w(self): return self.x1 - self.x0
    @property
    def cx(self): return (self.x0 + self.x1) / 2
    @property
    def cy(self): return (self.y0 + self.y1) / 2
    @property
    def angle(self):
        main = max(self.boxes, key=lambda b: b.run * b.thick)
        return 0.0 if main.vertical else main.angle


@dataclass
class Cluster:
    boxes: list[TextBox]
    rows: list[Row] = field(default_factory=list)

    @property
    def bbox(self):
        return (min(b.x0 for b in self.boxes), min(b.y0 for b in self.boxes),
                max(b.x1 for b in self.boxes), max(b.y1 for b in self.boxes))

    @property
    def size(self):
        return max(b.size for b in self.boxes)

    @property
    def text(self):
        return " ".join(b.text for b in self.boxes)


def detect_text(img_rgb: np.ndarray) -> list[TextBox]:
    # режимы указываем явно: движок запоминает параметры прошлого вызова
    # use_cls=False: модуль «текст вверх ногами?» на увеличенных постерах переворачивает
    # нормальный японский текст, и он читается как мусор; на постерах перевёрнутого текста не бывает
    res = _engine()(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), use_det=True, use_cls=False, use_rec=True)
    if res.boxes is None:
        return []
    return [TextBox(b, t, float(s)) for b, t, s in zip(res.boxes, res.txts, res.scores) if t.strip()]


def _group(boxes: list[TextBox]) -> list[list[TextBox]]:
    n = len(boxes)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            a, b = boxes[i], boxes[j]
            g = 0.6 * min(a.size, b.size)
            if a.x0 - g < b.x1 and b.x0 - g < a.x1 and a.y0 - g < b.y1 and b.y0 - g < a.y1:
                parent[find(i)] = find(j)
    groups: dict[int, list[TextBox]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(boxes[i])
    return list(groups.values())


def _rows(boxes: list[TextBox]) -> list[Row]:
    rows: list[Row] = []
    for b in sorted(boxes, key=lambda b: b.cy):
        for r in rows:
            ov = min(r.y1, b.y1) - max(r.y0, b.y0)
            similar = min(b.size, r.h) / max(b.size, r.h) >= 0.45
            if not b.vertical and similar and ov >= 0.5 * min(b.size, r.h):
                r.boxes.append(b)
                break
        else:
            rows.append(Row([b]))
    for r in rows:
        r.boxes.sort(key=lambda b: b.cx)
    rows.sort(key=lambda r: r.cy)
    return rows


@dataclass
class LogoFind:
    cluster: Cluster | None                               # главная часть логотипа (её заменяем)
    subtitle: list[TextBox] = field(default_factory=list)  # подпись другим языком рядом (стираем и набираем заново)
    decor: list[TextBox] = field(default_factory=list)     # мелкие надписи внутри логотипа (просто стираем)
    others: list[Cluster] = field(default_factory=list)    # прочий японский текст на постере (стираем)
    latin: bool = False                                    # логотип латиницей (HUNTER×HUNTER, Steins;Gate…)
    match: float = 0.0                                     # насколько совпал с названием аниме
    matched_title: str | None = None


def _letters(t: str) -> int:
    return sum(ch.isalnum() for ch in t)


def find_logo(boxes: list[TextBox], img_shape, variants: list[str] = (), min_score: float = 0.45) -> LogoFind:
    """Ищет логотип с названием — японский или латиницей.

    Кандидаты — заметные надписи. Логотип: крупный японский текст (как раньше), или текст, похожий
    на одно из названий аниме, или очень крупная латиница. Из похожих выбирается крупнейший.
    """
    from .titles import title_match

    H, W = img_shape[:2]
    # крупный текст берём, даже если он прочитан неуверенно (стилизованные логотипы читаются плохо)
    cands = [b for b in boxes if _letters(b.text) >= 1 and b.size >= 0.015 * H
             and (b.score >= min_score or (b.size >= 0.05 * H and b.score >= 0.2))]
    clusters = [Cluster(g) for g in _group(cands)]
    best, best_score, best_info = None, 0.0, (0.0, None)
    for c in clusters:
        main = max(c.boxes, key=lambda b: b.size)
        main_parts = [b for b in c.boxes if b.size >= 0.45 * main.size and b.cjk == main.cjk]
        text = " ".join(b.text for b in sorted(main_parts, key=lambda b: (round(b.cy / max(1.0, main.size)), b.cx)))
        match, which = title_match(text, list(variants)) if variants else (0.0, None)
        for b in main_parts:                       # строка логотипа может быть частью названия
            m, w = title_match(b.text, list(variants)) if variants else (0.0, None)
            if m > match:
                match, which = m, w
        size = main.size / H
        wide = sum(b.run for b in main_parts) >= 0.25 * W
        if main.cjk:
            ok = (size >= 0.045 and main.score >= min_score) or (match >= 0.6 and size >= 0.025)
        else:
            ok = (match >= 0.6 and size >= 0.03 and _letters(text) >= 3) \
                or (size >= 0.07 and _letters(text) >= 3 and main.score >= 0.6)
        # не прочиталось, но это большая широкая текстовая область — почти наверняка логотип
        unreadable = not ok and size >= 0.08 and wide and main.score >= 0.2
        if not (ok or unreadable):
            continue
        score = main.size * math.sqrt(sum(b.run for b in main_parts)) * (1 + 1.5 * match) * (0.5 if unreadable else 1)
        if score > best_score:
            best, best_score, best_info = c, score, (match, which)

    if best is None:
        others = [c for c in clusters if any(b.cjk for b in c.boxes)]
        cjk_others = [Cluster([b for b in c.boxes if b.cjk]) for c in others]
        return LogoFind(None, others=[c for c in cjk_others if c.boxes])

    main = max(best.boxes, key=lambda b: b.size)
    if main.cjk:
        # японский логотип: все японские части остаются (таблички, подписи «TVアニメ»…), латиница — подпись
        core = [b for b in best.boxes if b.cjk]
        subtitle = [b for b in best.boxes if not b.cjk]
        decor = []
    else:
        # логотип латиницей: крупные строки — название, мелкая латиница — подпись, японское — украшение
        core = [b for b in best.boxes if not b.cjk and b.size >= 0.45 * main.size]
        subtitle = [b for b in best.boxes if not b.cjk and b.size < 0.45 * main.size]
        decor = [b for b in best.boxes if b.cjk]
    logo = Cluster(core)
    logo.rows = _rows(core)
    others = []
    for c in clusters:
        if c is best:
            continue
        cjk = [b for b in c.boxes if b.cjk]
        if cjk:
            others.append(Cluster(cjk))
    return LogoFind(logo, subtitle, decor, others, latin=not main.cjk, match=best_info[0],
                    matched_title=best_info[1])


def find_title(boxes: list[TextBox], img_shape, variants: list[str] = ()) -> tuple[Cluster | None, list[Cluster]]:
    """Старый интерфейс: (кластер логотипа или None, прочий японский текст)."""
    f = find_logo(boxes, img_shape, variants)
    return f.cluster, f.others


# ---------------------------------------------------------------- вертикальный текст
def _read_line(img_rgb: np.ndarray) -> tuple[str, float]:
    """Распознать одну горизонтальную строку (без поиска текста)."""
    res = _engine()(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), use_det=False, use_cls=False, use_rec=True)
    txts = getattr(res, "txts", None) or ()
    scores = getattr(res, "scores", None) or ()
    if not txts:
        return "", 0.0
    return str(txts[0]).strip(), float(scores[0])


def _cells_strip(img: np.ndarray, x0: int, x1: int, ys: list[float]) -> np.ndarray:
    """Столбик иероглифов → строка: каждая клетка столбика ставится в ряд слева направо."""
    w = x1 - x0
    side = 64
    cells = []
    for a, b in zip(ys[:-1], ys[1:]):
        c = img[int(round(a)):int(round(b)), x0:x1]
        if c.size == 0:
            continue
        k = side / max(w, b - a)
        c = cv2.resize(c, (max(1, int(w * k)), max(1, int((b - a) * k))), interpolation=cv2.INTER_AREA)
        pad = np.zeros((side, side, 3), np.uint8) + np.median(c.reshape(-1, 3), 0).astype(np.uint8)
        oy, ox = (side - c.shape[0]) // 2, (side - c.shape[1]) // 2
        pad[oy:oy + c.shape[0], ox:ox + c.shape[1]] = c
        cells.append(pad)
    strip = np.concatenate(cells, 1)
    return cv2.copyMakeBorder(strip, 8, 8, 12, 12, cv2.BORDER_REPLICATE)


def _cjk_count(t: str) -> int:
    return len(CJK_RE.findall(t))


def reread_vertical(img: np.ndarray, boxes: list[TextBox], variants=()) -> list[TextBox]:
    """Вертикальные японские названия (столбиком вдоль края постера).

    Распознаватель читает только строки, поэтому столбик читается мусором, режется на куски
    или находится только один иероглиф. Находим высокие узкие столбики (или отдельный крупный
    иероглиф), режем на клетки (подбираем шаг и сдвиг), ставим клетки в ряд и читаем заново;
    столбик продлеваем вверх и вниз, пока клетки читаются как иероглифы.
    """
    from .titles import title_match
    H, W = img.shape[:2]

    def tall_box(b):
        return b.y1 - b.y0 >= 1.5 * (b.x1 - b.x0) and b.x1 - b.x0 >= 0.03 * W

    def one_char(b):
        h, w = b.y1 - b.y0, b.x1 - b.x0
        return len(b.text.strip()) <= 2 and 0.7 <= h / max(1.0, w) <= 1.5 and h >= 0.035 * H

    tall = [b for b in boxes if tall_box(b)]
    cols: list[list[TextBox]] = []
    for b in sorted(tall, key=lambda b: b.y0):
        for col in cols:
            cx0, cx1 = min(c.x0 for c in col), max(c.x1 for c in col)
            ov = min(cx1, b.x1) - max(cx0, b.x0)
            gap = b.y0 - max(c.y1 for c in col)
            if ov >= 0.6 * min(cx1 - cx0, b.x1 - b.x0) and gap < 0.8 * (cx1 - cx0):
                col.append(b)
                break
        else:
            cols.append([b])
    for col in cols:      # кусочки того же столбика сверху/снизу, прочитанные как латиница («S»)
        cx0, cx1 = min(c.x0 for c in col), max(c.x1 for c in col)
        for b in boxes:
            if b in col or b in tall:
                continue
            ov = min(cx1, b.x1) - max(cx0, b.x0)
            near = min(abs(b.y0 - max(c.y1 for c in col)), abs(min(c.y0 for c in col) - b.y1))
            if ov >= 0.6 * (b.x1 - b.x0) and (b.x1 - b.x0) >= 0.5 * (cx1 - cx0) and near < 0.8 * (cx1 - cx0):
                col.append(b)
    used = {id(b) for col in cols for b in col}
    singles = [[b] for b in boxes if id(b) not in used and one_char(b)]

    def score_of(text, sc):
        if _cjk_count(text) < 2:
            return 0.0
        m = title_match(text, list(variants))[0] if variants else 0.0
        return sc * (1 + 2 * m)

    def read(x0, x1, ys):
        return _read_line(_cells_strip(img, x0, x1, ys))

    def grow(x0, x1, ys):
        """Продлить столбик: новая клетка читается вместе с соседней — по одному иероглифу
        распознаватель ошибается чаще."""
        step = ys[1] - ys[0]
        for _ in range(12):
            a = ys[0] - step
            if a < -0.15 * step:
                break
            t, sc = read(x0, x1, [max(0.0, a), ys[0], ys[1]])
            if _cjk_count(t) < 2 or sc < 0.55:
                break
            ys.insert(0, max(0.0, a))
        for _ in range(12):
            b_ = ys[-1] + step
            if b_ > H + 0.15 * step:
                break
            t, sc = read(x0, x1, [ys[-2], ys[-1], min(float(H), b_)])
            if _cjk_count(t) < 2 or sc < 0.55:
                break
            ys.append(min(float(H), b_))
        return ys

    out = list(boxes)
    for col in cols + singles:
        x0 = int(max(0, min(c.x0 for c in col)))
        x1 = int(min(W, max(c.x1 for c in col)))
        y0, y1 = float(min(c.y0 for c in col)), float(max(c.y1 for c in col))
        w, h = x1 - x0, y1 - y0
        old_best = max(score_of(c.text, c.score) for c in col)
        cands = []
        if len(col) == 1 and one_char(col[0]):
            for k in (0.75, 0.88, 1.0, 1.12):        # шаг — высота одного иероглифа (± промежуток)
                ys = grow(x0, x1, [y0, y0 + h * k])
                if len(ys) >= 4:                      # хотя бы три иероглифа столбиком
                    cands.append(ys)
        else:
            for k in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1):
                p = w * k
                for off in (0.0, -p / 3, p / 3):
                    a, b_ = y0 + off, y1 + off
                    n = int(round((b_ - a) / p))
                    if n < 2 or a < 0 or b_ > H:
                        continue
                    cands.append(list(np.linspace(a, b_, n + 1)))
        best = None
        for ys in cands:
            t, sc = read(x0, x1, ys)
            sc_ = score_of(t, sc)
            if best is None or sc_ > best[0]:
                best = (sc_, ys)
        if best is None or best[0] <= 0:
            continue
        ys = best[1] if len(col) == 1 and one_char(col[0]) else grow(x0, x1, list(best[1]))
        text, sc = read(x0, x1, ys)
        if score_of(text, sc) <= max(0.3, old_best):
            continue
        quad = np.float32([[x0, ys[0]], [x1, ys[0]], [x1, ys[-1]], [x0, ys[-1]]])

        def inside(b):          # кусок, который почти целиком внутри нового столбика
            ix = min(x1, b.x1) - max(x0, b.x0)
            iy = min(ys[-1], b.y1) - max(ys[0], b.y0)
            return ix > 0 and iy > 0 and ix * iy >= 0.6 * (b.x1 - b.x0) * (b.y1 - b.y0)

        out = [b for b in out if b not in col and not inside(b)]
        out.append(TextBox(quad, text, sc))
    return out
