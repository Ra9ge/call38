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
    res = _engine()(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
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


def find_title(boxes: list[TextBox], img_shape, min_rel_size: float = 0.045,
               min_score: float = 0.45) -> tuple[Cluster | None, list[Cluster]]:
    """Возвращает (кластер японского названия или None, остальной японский текст)."""
    H = img_shape[0]
    cjk = [b for b in boxes if b.cjk and b.score >= min_score]
    clusters = [Cluster(g) for g in _group(cjk)]
    for c in clusters:
        c.rows = _rows(c.boxes)
    clusters.sort(key=lambda c: c.size * math.sqrt(sum(b.run for b in c.boxes)), reverse=True)
    if clusters and clusters[0].size >= min_rel_size * H:
        return clusters[0], clusters[1:]
    return None, clusters
