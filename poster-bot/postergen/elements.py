"""Готовые надписи: «стикер», табличка, лепесток, потёртый заголовок, подписи."""
from __future__ import annotations

import cv2
import numpy as np

from .typo import (SS, Element, Layer, Shaped, blur, dilate, finish, font_for_cap, grunge,
                   per_glyph_fill, rounded_rect_alpha, shape_text, shift, spread_colors)

SHADOW = (30, 22, 40)


def _glyph_layer(sh: Shaped, colors) -> tuple[Layer, np.ndarray]:
    rgb, a = per_glyph_fill(sh, spread_colors(colors, sh.chars))
    lay = Layer(sh.w, sh.h)
    lay.rgb, lay.a = rgb, a
    return lay, a


def _pad(layer_or_alpha, p):
    if isinstance(layer_or_alpha, Layer):
        out = Layer(layer_or_alpha.a.shape[1] + 2 * p, layer_or_alpha.a.shape[0] + 2 * p)
        out.over_layer(layer_or_alpha, p, p)
        return out
    return np.pad(layer_or_alpha, p)


def sticker_text(text: str, font: str, weight: int, cap: float, colors, outline_px: float,
                 outline_color=(255, 255, 255), shadow: bool = True, jitter=None, cases=None,
                 faux_bold: float = 0.0, tracking: float = 0.0) -> Element:
    """Цветные буквы в толстой общей обводке с мягкой тенью (как «СаКуРАсо»)."""
    f = font_for_cap(font, cap * SS, weight)
    sh = shape_text(text, f, tracking=tracking, jitter=jitter, cases=cases, faux_bold=faux_bold)
    glyphs, fa = _glyph_layer(sh, colors)
    ow = outline_px * SS
    p = int(ow * 2 + cap * SS * 0.25) + 4
    glyphs, fa = _pad(glyphs, p), _pad(fa, p)
    out = Layer(fa.shape[1], fa.shape[0])
    ring = dilate(fa, ow) if ow > 0 else fa
    if shadow:
        sa = blur(shift(ring, ow * 0.35 + cap * SS * 0.02, ow * 0.55 + cap * SS * 0.04), ow * 0.5 + 1.5)
        out.over(SHADOW, sa, 0.5)
    if ow > 0 and outline_color:
        out.over(outline_color, ring)
    out.over_layer(glyphs)
    return finish(out, fa)


def plate_text(text: str, font: str, weight: int, cap: float, glyph_colors, plate_color,
               border_color=(255, 255, 255), glyph_outline=(255, 255, 255), shadow: bool = True,
               faux_bold: float = 0.0, tracking: float = 0.02) -> Element:
    """Цветная табличка с белой каёмкой и разноцветными буквами (как «КОШЕЧКА»)."""
    f = font_for_cap(font, cap * SS, weight)
    sh = shape_text(text, f, tracking=tracking, faux_bold=faux_bold)
    glyphs, fa = _glyph_layer(sh, glyph_colors)
    c = cap * SS
    p = int(c * 1.6)
    glyphs, fa = _pad(glyphs, p), _pad(fa, p)
    H, W = fa.shape
    ys, xs = np.nonzero(fa > 0.3)
    x0, x1 = xs.min(), xs.max()
    cy = (ys.min() + ys.max()) / 2
    pw, ph = int(x1 - x0 + c * 0.9), int(c * 1.75)
    plate = np.zeros((H, W), np.float32)
    rr = rounded_rect_alpha(pw, ph, ph * 0.16)
    py, px = int(cy - ph / 2), int(x0 - c * 0.45)
    plate[py:py + rr.shape[0], px:px + rr.shape[1]] = rr[:H - py, :W - px]
    border = dilate(plate, c * 0.2) if border_color else plate
    out = Layer(W, H)
    if shadow:
        out.over(SHADOW, blur(shift(border, c * 0.06, c * 0.12), c * 0.1 + 1), 0.45)
    if border_color:
        out.over(border_color, border)
    out.over(plate_color, plate)
    if glyph_outline:
        out.over(glyph_outline, dilate(fa, c * 0.09))
    out.over_layer(glyphs)
    return finish(out, np.maximum(fa, plate))


def petal_text(text: str, font: str, weight: int, cap: float, petal_color, text_color=(255, 255, 255),
               border_color=(255, 255, 255), shadow: bool = True, faux_bold: float = 0.0) -> Element:
    """Белые буквы на цветном лепестке (как «ИЗ» на розовом лепестке)."""
    f = font_for_cap(font, cap * SS, weight)
    sh = shape_text(text.upper(), f, tracking=0.02, faux_bold=faux_bold)
    fa = sh.alpha()
    c = cap * SS
    p = int(c * 2)
    fa = _pad(fa, p)
    H, W = fa.shape
    ys, xs = np.nonzero(fa > 0.3)
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    rx = (xs.max() - xs.min()) / 2 + c * 0.55
    ry = max(c * 0.95, rx * 0.72)
    petal = np.zeros((H, W), np.uint8)
    cv2.ellipse(petal, (int(cx), int(cy)), (int(rx), int(ry)), -18, 0, 360, 255, -1, cv2.LINE_AA)
    # выемка сверху справа — похоже на лепесток сакуры
    notch_c = (int(cx + rx * 0.55), int(cy - ry * 0.95))
    cv2.circle(petal, notch_c, int(ry * 0.22), 0, -1, cv2.LINE_AA)
    petal = cv2.GaussianBlur(petal.astype(np.float32) / 255, (0, 0), 0.7)
    border = dilate(petal, c * 0.16) if border_color else petal
    out = Layer(W, H)
    if shadow:
        out.over(SHADOW, blur(shift(border, c * 0.06, c * 0.12), c * 0.1 + 1), 0.45)
    if border_color:
        out.over(border_color, border)
    out.over(petal_color, petal)
    out.over(text_color, fa)
    return finish(out, petal)


def outlined_text(text: str, font: str, weight: int, cap: float, fill=(255, 255, 255),
                  outline=None, outline_px: float = 0.0, shadow_opacity: float = 0.45,
                  halo_opacity: float = 0.0, tracking: float = 0.0, distress: float = 0.0,
                  seed: int = 0) -> Element:
    """Обычная надпись: заливка + обводка/ореол + тень (подписи, «НОВАЯ ОЗВУЧКА»)."""
    f = font_for_cap(font, cap * SS, weight)
    sh = shape_text(text, f, tracking=tracking)
    return _finish_single(sh.alpha(), cap, fill, outline, outline_px, shadow_opacity, halo_opacity,
                          distress, seed)


def _finish_single(fa, cap, fill, outline, outline_px, shadow_opacity, halo_opacity, distress, seed):
    c = cap * SS
    p = int(c * 1.2 + outline_px * SS * 2) + 4
    fa = np.pad(fa, p)
    if distress > 0:
        fa = grunge(fa, seed=seed, amount=distress)
    out = Layer(fa.shape[1], fa.shape[0])
    ring = dilate(fa, outline_px * SS) if outline and outline_px > 0 else fa
    if halo_opacity > 0:
        out.over((0, 0, 0), blur(dilate(fa, c * 0.12), c * 0.35), halo_opacity)
    if shadow_opacity > 0:
        out.over(SHADOW, blur(shift(ring, c * 0.04, c * 0.08), c * 0.08 + 1), shadow_opacity)
    if outline and outline_px > 0:
        out.over(outline, ring)
    out.over(fill, fa)
    return finish(out, fa)


def word_line_alpha(words: list[str], font: str, weight: int, cap: float, tracking: float = 0.0,
                    separator: str | None = "diamond", gap_em: float = 0.42) -> np.ndarray:
    """Строка из слов, между словами — ромбы ◆ (рисуем сами, в шрифтах их часто нет)."""
    f = font_for_cap(font, cap * SS, weight)
    c = cap * SS
    parts = []
    for w in words:
        a = shape_text(w, f, tracking=tracking).alpha()
        cols = np.nonzero(a.max(0) > 0.02)[0]
        parts.append(a[:, cols.min():cols.max() + 1] if len(cols) else a)
    H = parts[0].shape[0]
    rows = np.nonzero(parts[0].max(1) > 0.02)[0]
    mid = (rows.min() + rows.max()) / 2 if len(rows) else H / 2
    # центр ромба — середина высоты заглавной буквы
    word_rows = np.nonzero(np.max([p.max(1) for p in parts], 0) > 0.3)[0]
    cap_mid = (word_rows.min() + word_rows.max()) / 2 if len(word_rows) else mid
    gap = int(c * gap_em)
    d = c * 0.24
    sep_w = int(2 * d) if separator == "diamond" else 0
    total = sum(p.shape[1] for p in parts) + (len(parts) - 1) * (2 * gap + sep_w)
    if separator != "diamond":
        total = sum(p.shape[1] for p in parts) + (len(parts) - 1) * int(c * 0.55)
    out = np.zeros((H, total), np.float32)
    x = 0
    for i, p in enumerate(parts):
        out[:, x:x + p.shape[1]] = np.maximum(out[:, x:x + p.shape[1]], p)
        x += p.shape[1]
        if i < len(parts) - 1:
            if separator == "diamond":
                cx = x + gap + d
                poly = np.array([[cx, cap_mid - d], [cx + d, cap_mid], [cx, cap_mid + d], [cx - d, cap_mid]])
                m = np.zeros((H, total), np.uint8)
                cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 255, cv2.LINE_AA)
                out = np.maximum(out, m.astype(np.float32) / 255)
                x += 2 * gap + sep_w
            else:
                x += int(c * 0.55)
    return out


def distressed_title(lines: list[list[str]], font: str, weight: int, cap: float, fill=(255, 255, 255),
                     line_gap: float = 0.32, separator: str | None = "diamond", tracking: float = 0.01,
                     distress: float = 1.0, halo: float = 0.55, seed: int = 0) -> Element:
    """Крупный заголовок с потёртостями и ромбами (как «ОТЕЛЬ ◆ ДЛЯ ◆ НЕЛЮДЕЙ»)."""
    alphas = [word_line_alpha(ws, font, weight, cap, tracking, separator) for ws in lines]
    c = cap * SS
    W = max(a.shape[1] for a in alphas)
    trimmed = []
    for a in alphas:
        rows = np.nonzero(a.max(1) > 0.02)[0]
        trimmed.append(a[rows.min():rows.max() + 1] if len(rows) else a)
    gap = int(c * line_gap)
    H = sum(a.shape[0] for a in trimmed) + gap * (len(trimmed) - 1)
    fa = np.zeros((H, W), np.float32)
    y = 0
    for a in trimmed:
        x = (W - a.shape[1]) // 2
        fa[y:y + a.shape[0], x:x + a.shape[1]] = a
        y += a.shape[0] + gap
    return _finish_single(fa, cap, fill, None, 0, 0.5, halo, distress, seed)
