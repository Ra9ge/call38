"""Готовые надписи: «стикер», табличка, лепесток, потёртый заголовок, подписи."""
from __future__ import annotations

import cv2
import numpy as np

from .titles import CONNECTORS
from .typo import (SS, Element, Layer, Shaped, blur, dilate, finish, font_for_cap, grunge,
                   measure, per_glyph_fill, rounded_rect_alpha, shape_text, shift, spread_colors)

SHADOW = (30, 22, 40)


def _glyph_layer(sh: Shaped, colors) -> tuple[Layer, np.ndarray]:
    rgb, a = per_glyph_fill(sh, spread_colors(colors, sh.chars))
    lay = Layer(sh.w, sh.h)
    lay.rgb, lay.a = rgb, a
    return lay, a


def cap_band(sh: Shaped) -> np.ndarray:
    """Полоса «от верха заглавных до строки» — точка привязки по высоте заглавных букв,
    чтобы строки с «Й», «Ё», «Щ» стояли ровно."""
    band = np.zeros((sh.h, sh.w), np.float32)
    band[max(0, int(sh.baseline - sh.cap)):int(sh.baseline), :] = 1.0
    return band


def _pad(layer_or_alpha, p):
    if isinstance(layer_or_alpha, Layer):
        out = Layer(layer_or_alpha.a.shape[1] + 2 * p, layer_or_alpha.a.shape[0] + 2 * p)
        out.over_layer(layer_or_alpha, p, p)
        return out
    return np.pad(layer_or_alpha, p)


def sticker_text(text: str, font: str, weight: int, cap: float, colors, outline_px: float,
                 outline_color=(255, 255, 255), shadow: bool = True, jitter=None, cases=None,
                 faux_bold: float = 0.0, tracking: float = 0.0, char_colors=None,
                 drop_ratio: float = 0.0, gradient=None, shadow_opacity: float = 0.5,
                 cap_anchor: bool = False) -> Element:
    """Цветные буквы в общей обводке с мягкой тенью (как «СаКуРАсо»).

    char_colors — свой цвет для каждой буквы (например, красный предлог);
    drop_ratio — во сколько раз первая буква крупнее остальных (буквица).
    """
    f = font_for_cap(font, cap * SS, weight)
    drop_font = font_for_cap(font, cap * SS * drop_ratio, weight) if drop_ratio > 1 else None
    sh = shape_text(text, f, tracking=tracking, jitter=jitter, cases=cases, faux_bold=faux_bold,
                    drop_font=drop_font)
    if char_colors:
        rgb, fa = per_glyph_fill(sh, list(char_colors))
        glyphs = Layer(sh.w, sh.h)
        glyphs.rgb, glyphs.a = rgb, fa
    elif gradient:
        # заливка градиентом сверху вниз (по высоте букв)
        fa = sh.alpha()
        rows = np.nonzero(fa.max(1) > 0.05)[0]
        y0, y1 = (rows.min(), rows.max()) if len(rows) else (0, sh.h - 1)
        t = np.clip((np.arange(sh.h, dtype=np.float32) - y0) / max(1, y1 - y0), 0, 1)[:, None, None]
        top, bot = (np.asarray(c, np.float32) for c in gradient)
        cmap = np.broadcast_to(top * (1 - t) + bot * t, (sh.h, sh.w, 3))
        glyphs = Layer(sh.w, sh.h)
        glyphs.over(cmap, fa)
    else:
        glyphs, fa = _glyph_layer(sh, colors)
    line_a = sh.line_alpha() if sh.drop else (cap_band(sh) if cap_anchor else None)
    ow = outline_px * SS
    p = int(ow * 2 + cap * SS * 0.25) + 4
    glyphs, fa = _pad(glyphs, p), _pad(fa, p)
    if line_a is not None:
        line_a = _pad(line_a, p)
    out = Layer(fa.shape[1], fa.shape[0])
    ring = dilate(fa, ow) if ow > 0 else fa
    if shadow:
        sa = blur(shift(ring, ow * 0.35 + cap * SS * 0.02, ow * 0.55 + cap * SS * 0.04), ow * 0.5 + 1.5)
        out.over(SHADOW, sa, shadow_opacity)
    if ow > 0 and outline_color:
        out.over(outline_color, ring)
    out.over_layer(glyphs)
    return finish(out, fa, line_a)


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
                  seed: int = 0, cap_anchor: bool = False) -> Element:
    """Обычная надпись: заливка + обводка/ореол + тень (подписи, «НОВАЯ ОЗВУЧКА»)."""
    f = font_for_cap(font, cap * SS, weight)
    sh = shape_text(text, f, tracking=tracking)
    return _finish_single(sh.alpha(), cap, fill, outline, outline_px, shadow_opacity, halo_opacity,
                          distress, seed, cap_band(sh) if cap_anchor else None)


def _finish_single(fa, cap, fill, outline, outline_px, shadow_opacity, halo_opacity, distress, seed,
                   anchor=None):
    c = cap * SS
    p = int(c * 1.2 + outline_px * SS * 2) + 4
    fa = np.pad(fa, p)
    if anchor is not None:
        anchor = np.pad(anchor, p)
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
    return finish(out, fa, anchor)


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
    # между словами — ромб; после «×», «&» и т.п. — просто пробел (знак уже стоит)
    kinds = ["space" if separator != "diamond" or words[i].endswith(CONNECTORS) else "diamond"
             for i in range(len(parts) - 1)]
    widths = [2 * gap + int(2 * d) if k == "diamond" else int(c * 0.55) for k in kinds]
    total = sum(p.shape[1] for p in parts) + sum(widths)
    out = np.zeros((H, total), np.float32)
    x = 0
    for i, p in enumerate(parts):
        out[:, x:x + p.shape[1]] = np.maximum(out[:, x:x + p.shape[1]], p)
        x += p.shape[1]
        if i < len(parts) - 1:
            if kinds[i] == "diamond":
                cx = x + gap + d
                poly = np.array([[cx, cap_mid - d], [cx + d, cap_mid], [cx, cap_mid + d], [cx - d, cap_mid]])
                m = np.zeros((H, total), np.uint8)
                cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 255, cv2.LINE_AA)
                out = np.maximum(out, m.astype(np.float32) / 255)
            x += widths[i]
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


def _ink_x(sh: Shaped) -> tuple[float, float]:
    """Левый и правый край закрашенных букв в координатах строки."""
    left, right = 1e9, -1e9
    for p, x, y, _ in sh.patches:
        cols = np.nonzero(p.max(0) > 20)[0]
        if len(cols):
            left, right = min(left, x + cols[0]), max(right, x + cols[-1] + 1)
    return (left, right) if right > left else (0.0, 0.0)


def _paint(layer: Layer, sh: Shaped, colors: list, dx: float, dy: float):
    """Рисует разложенную строку на холсте со сдвигом; colors — цвет каждой буквы."""
    for p, x, y, idx in sh.patches:
        layer.over_patch(colors[idx], p.astype(np.float32) / 255, int(round(x + dx)), int(round(y + dy)))


def dropcap_title(text: str, subtitle: str | None, font: str, weight: int | None, cap: float,
                  sub_cap: float, fill=(255, 255, 255), accent=None, drop_n: int = 0,
                  accent_flags: list[bool] | None = None, tracking: float = 0.03,
                  shadow_opacity: float = 0.5) -> Element:
    """Заглавные с засечками в духе «ВОСХОЖДЕНИЕ В ТЕНИ»: крупная буквица слева (как огромный «陰»),
    справа строка названия, под ней английское название вразрядку до той же ширины.
    accent_flags — какие буквы красить цветом-акцентом (как красные «の», «に»)."""
    text = text.upper()
    flags = accent_flags or [False] * len(text)
    colors = [accent if (f and accent) else fill for f in flags]
    rest = text[drop_n:]
    start = drop_n + len(rest) - len(rest.lstrip())
    drop, main = text[:drop_n], text[start:]
    c = cap * SS
    gap = 0.45 * c
    sh_m = shape_text(main, font_for_cap(font, c, weight), tracking=tracking)
    ml, mr = _ink_x(sh_m)
    main_w = mr - ml

    sh_s = None
    s = sub_cap * SS
    if subtitle:
        fs = font_for_cap(font, s, weight)
        natural = measure(subtitle, fs, 0.04)
        if natural > main_w:                       # не шире строки названия
            s *= main_w / natural
            fs = font_for_cap(font, s, weight)
        w0 = measure(subtitle, fs)
        n = max(1, len(subtitle) - 1)
        t = float(np.clip((main_w * 0.97 - w0) / (fs.size * n), 0.03, 0.35))
        sh_s = shape_text(subtitle, fs, tracking=t)

    pad = int(1.2 * c + 0.6 * s)
    top = pad
    x_main = pad
    drop_base = None
    sh_d = None
    if drop:
        D = (c + gap + s) if sh_s else 1.75 * c       # буквица — от верха названия до низа подписи
        sh_d = shape_text(drop, font_for_cap(font, D, weight), tracking=-0.01)
        dl, dr = _ink_x(sh_d)
        dx_d, dy_d = pad - dl, top - (sh_d.baseline - sh_d.cap)
        x_main = dr + dx_d + 0.14 * c
        drop_base = top + sh_d.cap
    dx_m = x_main - ml
    if drop and not sh_s:
        dy_m = drop_base - sh_m.baseline           # строка стоит на одной линии с буквицей
    else:
        dy_m = top - (sh_m.baseline - sh_m.cap)
    right = mr + dx_m
    bottom = dy_m + sh_m.baseline + sh_m.cap * 0.3
    if sh_s:
        sl, sr = _ink_x(sh_s)
        dx_s = x_main - sl
        dy_s = (top + sh_m.cap + gap + sh_s.cap) - sh_s.baseline
        right = max(right, sr + dx_s)
        bottom = max(bottom, dy_s + sh_s.baseline + sh_s.cap * 0.3)
    if drop_base is not None:
        bottom = max(bottom, drop_base + 0.3 * sh_d.cap)
    W, H = int(right + pad), int(bottom + pad)

    glyphs = Layer(W, H)
    if sh_d is not None:
        _paint(glyphs, sh_d, colors[:drop_n], dx_d, dy_d)
    _paint(glyphs, sh_m, colors[start:], dx_m, dy_m)
    if sh_s is not None:
        _paint(glyphs, sh_s, [fill] * len(subtitle), dx_s, dy_s)
    out = Layer(W, H)
    out.over(SHADOW, blur(shift(glyphs.a, c * 0.03, c * 0.05), c * 0.06 + 1), shadow_opacity)
    out.over_layer(glyphs)
    return finish(out, glyphs.a)
