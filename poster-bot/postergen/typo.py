"""Типографика: раскладка букв, обводка, тень, потёртости, плашки."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFont

FONTS_DIR = Path(__file__).resolve().parent.parent / "fonts"
FONT_FILES = {
    "comfortaa": "Comfortaa[wght].ttf",    # круглый «милый» — как на постере Сакурасо
    "montserrat": "Montserrat[wght].ttf",  # подписи, «НОВАЯ ОЗВУЧКА»
    "unbounded": "Unbounded[wght].ttf",    # широкий жирный — как «ОТЕЛЬ ◆ ДЛЯ ◆ НЕЛЮДЕЙ»
    "rubik": "Rubik[wght].ttf",            # универсальный жирный
    "tinos": "Tinos-Regular.ttf",          # с засечками — для тонких «минтё»-логотипов
}
SS = 2  # рисуем в 2 раза крупнее и уменьшаем — так края чище


@lru_cache(maxsize=256)
def get_font(name: str, px: int, weight: int | None = None) -> ImageFont.FreeTypeFont:
    path = Path(name)
    if not path.exists():
        path = FONTS_DIR / FONT_FILES.get(name.lower(), name)
    f = ImageFont.truetype(str(path), max(4, int(px)))
    if weight:
        try:
            axes = f.get_variation_axes()
            vals = []
            for ax in axes:
                nm = ax.get("name", b"")
                nm = nm.decode() if isinstance(nm, bytes) else str(nm)
                if nm.lower().startswith("weight") or nm.lower() == "wght":
                    vals.append(float(np.clip(weight, ax["minimum"], ax["maximum"])))
                else:
                    vals.append(ax.get("default", ax["minimum"]))
            f.set_variation_by_axes(vals)
        except (OSError, AttributeError, KeyError):
            pass  # не вариативный шрифт
    return f


def cap_height(font) -> float:
    return float(-font.getbbox("НH", anchor="ls")[1])


def font_for_cap(name: str, cap_px: float, weight: int | None = None):
    probe = get_font(name, 200, weight)
    return get_font(name, max(4, int(round(200 * cap_px / max(1.0, cap_height(probe))))), weight)


def _ell(r: float) -> np.ndarray:
    r = max(1, int(round(r)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


# ---------------------------------------------------------------- раскладка букв
@dataclass
class Shaped:
    w: int
    h: int
    baseline: float
    cap: float
    patches: list = field(default_factory=list)   # (alpha uint8, x, y, char_index)
    chars: list = field(default_factory=list)
    drop: bool = False                             # первая буква — крупная буквица
    extra_bottom: float = 0.0

    def line_alpha(self) -> np.ndarray:
        """Буквы без буквицы — по ним ставим строку на место оригинальной."""
        a = np.zeros((self.h, self.w), np.float32)
        for p, x, y, i in self.patches:
            if not (self.drop and i == 0):
                _blit_max(a, p.astype(np.float32) / 255, x, y)
        return a

    def alpha(self) -> np.ndarray:
        a = np.zeros((self.h, self.w), np.float32)
        for p, x, y, _ in self.patches:
            _blit_max(a, p.astype(np.float32) / 255, x, y)
        return a


def _blit_max(dst, src, x, y):
    H, W = dst.shape[:2]
    h, w = src.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    s = src[y0 - y:y1 - y, x0 - x:x1 - x]
    np.maximum(dst[y0:y1, x0:x1], s, out=dst[y0:y1, x0:x1])


def _drop_cap(sh: "Shaped", ch: str, font, drop_font, pen: float, margin: int) -> float:
    """Крупная первая буква (как огромный «陰» на постере): верх — вровень с заглавными строки."""
    from PIL import ImageDraw
    asc_b, desc_b = drop_font.getmetrics()
    pad_b = int(drop_font.size * 0.35)
    adv = drop_font.getlength(ch)
    img = Image.new("L", (max(1, int(adv + 2 * pad_b)), max(1, int(asc_b + desc_b + 2 * pad_b))), 0)
    ImageDraw.Draw(img).text((pad_b, pad_b + asc_b), ch, font=drop_font, fill=255, anchor="ls")
    cap_top = sh.baseline - sh.cap
    base_b = cap_top - 0.1 * sh.cap + cap_height(drop_font)
    x = int(round(pen - pad_b))
    y = int(round(base_b - pad_b - asc_b))
    sh.patches.append((np.array(img), x, y, 0))
    sh.drop = True
    sh.extra_bottom = max(0.0, base_b + desc_b * 0.3 - (sh.baseline + font.getmetrics()[1]))
    return adv + font.size * 0.06


def shape_text(text: str, font, tracking: float = 0.0, jitter: dict | None = None,
               cases: list[bool] | None = None, faux_bold: float = 0.0, drop_font=None) -> Shaped:
    """Раскладывает строку по буквам. tracking — доля кегля; jitter — «прыгающие» буквы;
    drop_font — шрифт для крупной первой буквы (буквица)."""
    size = font.size
    asc, desc = font.getmetrics()
    margin = int(size * 0.6)
    pad = int(size * 0.35)
    rng = np.random.default_rng((jitter or {}).get("seed", 0))
    pen = float(margin)
    prev = None
    sh = Shaped(0, 0, margin + asc, cap_height(font))
    bold_k = _ell(faux_bold * size) if faux_bold > 0 else None
    for i, ch in enumerate(text):
        if cases is not None and i < len(cases):
            ch = ch.upper() if cases[i] else ch.lower()
        if i == 0 and drop_font is not None and not ch.isspace():
            pen += _drop_cap(sh, ch, font, drop_font, pen, margin)
            sh.chars.append(ch)
            continue
        if prev is not None:
            pen += font.getlength(prev + ch) - font.getlength(prev) - font.getlength(ch)  # кернинг
        adv = font.getlength(ch)
        if not ch.isspace():
            pw, ph = int(adv + 2 * pad), int(asc + desc + 2 * pad)
            img = Image.new("L", (max(1, pw), max(1, ph)), 0)
            from PIL import ImageDraw
            ImageDraw.Draw(img).text((pad, pad + asc), ch, font=font, fill=255, anchor="ls")
            p = np.array(img)
            if bold_k is not None:
                p = cv2.dilate(p, bold_k)
            dy = 0.0
            if jitter:
                rot = rng.uniform(-1, 1) * jitter.get("rot", 0)
                dy = (1 if i % 2 else -1) * jitter.get("dy", 0) * size * rng.uniform(0.4, 1.0)
                if abs(rot) > 0.1:
                    p = np.array(Image.fromarray(p).rotate(rot, resample=Image.BICUBIC, expand=True))
            x = int(round(pen + adv / 2 - p.shape[1] / 2))
            y = int(round(margin + asc + (desc - asc) / 2 - p.shape[0] / 2 + dy))
            sh.patches.append((p, x, y, i))
        sh.chars.append(ch)
        pen += adv + tracking * size
        prev = ch
    sh.w = int(pen - tracking * size + margin)
    sh.h = int(margin * 2 + asc + desc + sh.extra_bottom)
    return sh


def measure(text: str, font, tracking: float = 0.0) -> float:
    return font.getlength(text) + tracking * font.size * max(0, len(text) - 1)


# ---------------------------------------------------------------- слои
class Layer:
    """RGBA-холст с премультиплицированным альфа-каналом (float)."""

    def __init__(self, w: int, h: int):
        self.rgb = np.zeros((h, w, 3), np.float32)
        self.a = np.zeros((h, w), np.float32)

    @property
    def shape(self):
        return self.a.shape

    def over(self, color, alpha: np.ndarray, opacity: float = 1.0):
        al = np.clip(alpha * opacity, 0, 1)
        c = np.asarray(color, np.float32) / 255.0
        if c.ndim == 1:
            c = c[None, None, :]
        self.rgb = c * al[..., None] + self.rgb * (1 - al[..., None])
        self.a = al + self.a * (1 - al)

    def over_patch(self, color, patch: np.ndarray, x: int, y: int):
        """Как over(), но только в клетке буквы (patch — alpha 0..1 с левым верхним углом в x, y)."""
        H, W = self.a.shape
        h, w = patch.shape
        x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
        if x1 <= x0 or y1 <= y0:
            return
        al = np.clip(patch[y0 - y:y1 - y, x0 - x:x1 - x], 0, 1)
        c = np.asarray(color, np.float32) / 255.0
        rgb = self.rgb[y0:y1, x0:x1]
        self.rgb[y0:y1, x0:x1] = c * al[..., None] + rgb * (1 - al[..., None])
        self.a[y0:y1, x0:x1] = al + self.a[y0:y1, x0:x1] * (1 - al)

    def over_layer(self, other: "Layer", x: int = 0, y: int = 0):
        H, W = self.a.shape
        h, w = other.a.shape
        x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
        if x1 <= x0 or y1 <= y0:
            return
        oa = other.a[y0 - y:y1 - y, x0 - x:x1 - x]
        orgb = other.rgb[y0 - y:y1 - y, x0 - x:x1 - x]
        self.rgb[y0:y1, x0:x1] = orgb + self.rgb[y0:y1, x0:x1] * (1 - oa[..., None])
        self.a[y0:y1, x0:x1] = oa + self.a[y0:y1, x0:x1] * (1 - oa)

    def to_image(self) -> Image.Image:
        a = np.clip(self.a, 0, 1)
        rgb = np.where(a[..., None] > 1e-4, self.rgb / np.maximum(a[..., None], 1e-4), 0)
        out = np.dstack([np.clip(rgb, 0, 1) * 255, a * 255]).round().astype(np.uint8)
        return Image.fromarray(out, "RGBA")


def dilate(alpha: np.ndarray, r: float) -> np.ndarray:
    if r <= 0.3:
        return alpha.copy()
    a8 = (np.clip(alpha, 0, 1) * 255).astype(np.uint8)
    d = cv2.dilate(a8, _ell(r)).astype(np.float32) / 255
    return cv2.GaussianBlur(d, (0, 0), 0.6)


def blur(alpha: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(alpha, (0, 0), max(0.1, sigma)) if sigma > 0.1 else alpha


def shift(alpha: np.ndarray, dx: float, dy: float) -> np.ndarray:
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(alpha, M, (alpha.shape[1], alpha.shape[0]))


def per_glyph_fill(sh: Shaped, colors: list) -> tuple[np.ndarray, np.ndarray]:
    """Заливка: каждая буква своим цветом. Возвращает (rgb премульт., alpha)."""
    lay = Layer(sh.w, sh.h)
    for p, x, y, idx in sh.patches:
        lay.over_patch(colors[idx % len(colors)] if colors else (255, 255, 255), p.astype(np.float32) / 255, x, y)
    return lay.rgb, lay.a


def spread_colors(palette: list, chars: list[str]) -> list:
    """Распределяет цвета оригинала по буквам слева направо (как сделал дизайнер)."""
    idx = [i for i, c in enumerate(chars) if not c.isspace()]
    out = [palette[0] if palette else (255, 255, 255)] * len(chars)
    if not palette:
        return out
    n, k = len(idx), len(palette)
    for j, i in enumerate(idx):
        t = j / max(1, n - 1)
        out[i] = palette[min(k - 1, int(round(t * (k - 1))))]
    return out


def grunge(alpha: np.ndarray, seed: int = 0, amount: float = 1.0) -> np.ndarray:
    """Потёртости: мелкие выщербины + царапины, как на постерах в стиле «ОТЕЛЬ ДЛЯ НЕЛЮДЕЙ»."""
    h, w = alpha.shape
    rng = np.random.default_rng(seed)
    fine = cv2.resize(rng.random((h // 3 + 2, w // 3 + 2)).astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    coarse = cv2.resize(rng.random((h // 28 + 2, w // 28 + 2)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    holes = np.clip((fine * (0.55 + 0.7 * coarse) - 0.86) * 6, 0, 1)
    scratches = np.zeros((h, w), np.float32)
    for _ in range(int(w * h / 9000 * amount) + 3):
        x, y = rng.uniform(0, w), rng.uniform(0, h)
        ang = math.radians(rng.uniform(-30, 30) + (90 if rng.random() < 0.15 else 0))
        ln = rng.uniform(0.02, 0.12) * w
        p1 = (int(x), int(y))
        p2 = (int(x + ln * math.cos(ang)), int(y + ln * math.sin(ang)))
        cv2.line(scratches, p1, p2, float(rng.uniform(0.5, 1.0)), int(rng.integers(1, 3)), cv2.LINE_AA)
    dmg = np.clip(holes * 0.95 * amount + scratches * 0.85 * amount, 0, 1)
    return alpha * cv2.GaussianBlur(1 - dmg, (0, 0), 0.5)


def rounded_rect_alpha(w: int, h: int, r: float, pad: int = 0) -> np.ndarray:
    a = np.zeros((h + 2 * pad, w + 2 * pad), np.uint8)
    r = int(max(1, min(r, h / 2, w / 2)))
    x0, y0, x1, y1 = pad, pad, pad + w - 1, pad + h - 1
    cv2.rectangle(a, (x0 + r, y0), (x1 - r, y1), 255, -1)
    cv2.rectangle(a, (x0, y0 + r), (x1, y1 - r), 255, -1)
    for cx, cy in ((x0 + r, y0 + r), (x1 - r, y0 + r), (x0 + r, y1 - r), (x1 - r, y1 - r)):
        cv2.circle(a, (cx, cy), r, 255, -1, cv2.LINE_AA)
    return cv2.GaussianBlur(a.astype(np.float32) / 255, (0, 0), 0.7)


# ---------------------------------------------------------------- готовые элементы
@dataclass
class Element:
    image: Image.Image          # RGBA в итоговом масштабе
    cx: float                   # точка привязки (центр букв) внутри image
    cy: float
    w: float                    # размер букв (без обводки и тени)
    h: float

    def rotated(self, angle_deg: float) -> "Element":
        if abs(angle_deg) < 0.3:
            return self
        # кладём точку привязки в центр квадратного холста и вращаем вокруг неё
        W, H = self.image.size
        r = math.hypot(max(self.cx, W - self.cx), max(self.cy, H - self.cy))
        C = int(math.ceil(r)) + 2
        canvas = Image.new("RGBA", (2 * C, 2 * C), (0, 0, 0, 0))
        canvas.paste(self.image, (int(round(C - self.cx)), int(round(C - self.cy))))
        rot = canvas.rotate(-angle_deg, resample=Image.BICUBIC, center=(C, C))
        bbox = rot.getbbox() or (0, 0, rot.width, rot.height)
        return Element(rot.crop(bbox), C - bbox[0], C - bbox[1], self.w, self.h)

    def paste_on(self, poster: Image.Image, x: float, y: float):
        """Ставит элемент так, чтобы его точка привязки попала в (x, y)."""
        poster.alpha_composite(self.image, (int(round(x - self.cx)), int(round(y - self.cy))))

    def bbox_at(self, x: float, y: float):
        """Видимая область (с обводкой, без почти прозрачного края тени) при установке в (x, y)."""
        bb = self.image.getchannel("A").point(lambda a: 255 if a > 40 else 0).getbbox() \
            or (0, 0, *self.image.size)
        ox, oy = x - self.cx, y - self.cy
        return (ox + bb[0], oy + bb[1], ox + bb[2], oy + bb[3])


def finish(layer: Layer, fill_alpha: np.ndarray, anchor_alpha: np.ndarray | None = None) -> Element:
    """Обрезает пустые края, уменьшает в SS раз, точка привязки — центр букв
    (по вертикали — центр anchor_alpha, если задан: строка без буквицы)."""
    a = layer.a
    ys, xs = np.nonzero(a > 0.004)
    if len(xs) == 0:
        img = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
        return Element(img, 0, 0, 0, 0)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    fy, fx = np.nonzero(fill_alpha > 0.3)
    if len(fx):
        fcx, fcy = (fx.min() + fx.max()) / 2, (fy.min() + fy.max()) / 2
        fw, fh = fx.max() - fx.min(), fy.max() - fy.min()
        if anchor_alpha is not None:
            ay = np.nonzero(anchor_alpha.max(1) > 0.3)[0]
            if len(ay):
                fcy = (ay.min() + ay.max()) / 2
    else:
        fcx, fcy, fw, fh = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
    # выравниваем по сетке SS
    x0, y0 = x0 // SS * SS, y0 // SS * SS
    x1, y1 = -(-x1 // SS) * SS, -(-y1 // SS) * SS
    sub = Layer(x1 - x0, y1 - y0)
    sub.rgb = layer.rgb[y0:y1, x0:x1]
    sub.a = layer.a[y0:y1, x0:x1]
    img = sub.to_image()
    if SS > 1:
        img = img.resize((img.width // SS, img.height // SS), Image.LANCZOS)
    return Element(img, (fcx - x0) / SS, (fcy - y0) / SS, fw / SS, fh / SS)
