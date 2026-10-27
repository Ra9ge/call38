"""Разбор оригинального японского логотипа: маска для удаления и стиль (цвета, обводка, плашки)."""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .detect import Cluster, TextBox, _rows


def _ell(r: int) -> np.ndarray:
    r = max(1, int(round(r)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def _grow(m: np.ndarray, r: float) -> np.ndarray:
    """То же, что cv2.dilate с круглым ядром радиуса r, но быстро и для больших r
    (через карту расстояний, время не зависит от радиуса)."""
    r = max(1, int(round(r)))
    if r <= 15:
        return cv2.dilate(m, _ell(r))
    val = m.max()
    if val == 0:
        return np.zeros_like(m)
    d = cv2.distanceTransform((m == 0).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return np.where(d <= r, val, 0).astype(m.dtype)


@dataclass
class BoxStyle:
    kind: str                                   # sticker | plate | badge | plain
    glyph_colors: list[tuple[int, int, int]]    # цвета букв слева направо
    outline_color: tuple[int, int, int] | None
    outline_ratio: float                        # толщина обводки / высота строки
    plate_color: tuple[int, int, int] | None = None
    plate_border: tuple[int, int, int] | None = None
    shadow: bool = True
    mask: np.ndarray | None = field(default=None, repr=False)
    accent: tuple[int, int, int] | None = None  # второй цвет (красные «の», «に») — для предлогов
    thin: bool = False                          # тонкие штрихи (шрифт с засечками, «минтё»)
    strokes: np.ndarray | None = field(default=None, repr=False)  # сами штрихи без запаса
    bg: tuple[int, int, int] | None = None      # цвет фона вокруг надписи
    gradient: tuple | None = None               # (цвет сверху, цвет снизу), если буквы с градиентом


class Analyzer:
    def __init__(self, img: np.ndarray):
        self.img = img
        self.H, self.W = img.shape[:2]
        f = img.astype(np.int16)
        mn, mx = f.min(2), f.max(2)
        self.white = ((mn > 200) & (mx - mn < 48)).astype(np.uint8) * 255
        self.lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        self.colored = ((hsv[..., 1] > 45) & (hsv[..., 2] > 90)).astype(np.uint8) * 255
        self.others = np.zeros(img.shape[:2], np.uint8)   # другие строки логотипа — это не фон

    # ---------- цвета ----------
    def _clusters(self, sel: np.ndarray, k: int = 6, merge_de: float = 22.0, min_share: float = 0.04):
        """k-means по Lab, возвращает [(rgb, share, median_x, median_y)] отсортированные по доле."""
        ys, xs = np.nonzero(sel)
        if len(xs) < 12:
            return []
        if len(xs) > 30000:
            idx = np.random.default_rng(0).choice(len(xs), 30000, replace=False)
            ys, xs = ys[idx], xs[idx]
        pts = self.lab[ys, xs]
        k = int(min(k, max(1, len(pts) // 30)))
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
        _, labels, centers = cv2.kmeans(pts, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
        labels = labels.ravel()
        groups = [[i] for i in range(k)]
        # слить близкие цвета
        changed = True
        while changed:
            changed = False
            for a in range(len(groups)):
                for b in range(a + 1, len(groups)):
                    ca = pts[np.isin(labels, groups[a])].mean(0) if np.isin(labels, groups[a]).any() else centers[groups[a][0]]
                    cb = pts[np.isin(labels, groups[b])].mean(0) if np.isin(labels, groups[b]).any() else centers[groups[b][0]]
                    if np.linalg.norm((ca - cb) * np.array([0.5, 1, 1])) < merge_de:
                        groups[a] += groups.pop(b)
                        changed = True
                        break
                if changed:
                    break
        out = []
        for g in groups:
            m = np.isin(labels, g)
            share = m.mean()
            if share < min_share:
                continue
            lab = np.median(pts[m], 0).astype(np.uint8).reshape(1, 1, 3)
            rgb = tuple(int(v) for v in cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)[0, 0])
            out.append((rgb, float(share), float(np.median(xs[m])), float(np.median(ys[m]))))
        out.sort(key=lambda c: -c[1])
        return out

    @staticmethod
    def _is_whiteish(rgb) -> bool:
        return min(rgb) > 200 and max(rgb) - min(rgb) < 48

    def _is_tint(self, rgb, others) -> bool:
        """Цвет — это смесь белого с другим цветом (сглаживание на краю обводки)?"""
        c, w = self.lab_of(rgb), self.lab_of((255, 255, 255))
        for o in others:
            if o == rgb:
                continue
            ol = self.lab_of(o)
            d = w - ol
            t = float(np.clip(np.dot(c - ol, d) / max(1e-6, np.dot(d, d)), 0, 1))
            if 0.15 < t < 0.95 and np.linalg.norm(ol + t * d - c) < 15:
                return True
        return False

    def _ordered_colors(self, sel: np.ndarray, box: TextBox, extra_refs=(), **kw):
        cl = [c for c in self._clusters(sel, **kw) if not self._is_whiteish(c[0])]
        refs = [c[0] for c in cl] + list(extra_refs)
        cl = [c for c in cl if not self._is_tint(c[0], refs)]
        if not cl:
            return []
        if box.vertical:
            cl.sort(key=lambda c: c[3])
        else:
            a = np.radians(box.angle)
            cl.sort(key=lambda c: c[2] * np.cos(a) + c[3] * np.sin(a))
        return [c[0] for c in cl]

    # ---------- анализ одной строки ----------
    def analyze_box(self, box: TextBox, exclude: np.ndarray | None = None) -> BoxStyle:
        h = box.size
        Q = box.mask(self.img.shape)
        if exclude is not None:
            Q &= ~exclude
        area = max(1, int((Q > 0).sum()))

        # 1) плашка: одна заливка занимает край рамки текста (жёлтая табличка, лепесток)
        band = Q & ~cv2.erode(Q, _ell(max(2, h * 0.16)))
        non_white_band = band & ~self.white
        plate = None
        if (band > 0).sum() > 20:
            cl = self._clusters(non_white_band > 0, k=4, merge_de=18)
            cl = [c for c in cl if max(c[0]) - min(c[0]) > 40]
            if cl:
                rgb = cl[0][0]
                ref = self.lab_of(rgb)
                near = lambda sel: (np.linalg.norm(self.lab[sel] - ref, axis=1) < 32).mean() if sel.any() else 0.0
                in_share = near(band > 0)
                far = (_grow(Q, h * 1.0) & ~_grow(Q, h * 0.5)) > 0
                # у таблички есть край: сразу за ней её цвета уже нет (иначе это просто фон)
                mid = (_grow(Q, h * 0.5) & ~_grow(Q, h * 0.25)) > 0
                if in_share > 0.35 and near(mid) < 0.15 and near(far) < 0.3:
                    plate = rgb
        if plate is not None:
            return self._plate_style(box, Q, plate)

        # 2) «стикер»: цветные буквы в толстой белой обводке
        letters = self._sticker_letters(box, Q)
        if (letters > 0).sum() > 0.06 * area:
            return self._sticker_style(box, Q, letters)

        # 3) всё остальное
        return self._plain_style(box, Q)

    def _plate_style(self, box, Q, plate_rgb):
        h = box.size
        diff = np.linalg.norm(self.lab - self.lab_of(plate_rgb), axis=2)
        plate_px = ((diff < 22) & (self.white == 0)).astype(np.uint8) * 255
        near = _grow(Q, h * 0.45)
        plate_px &= near
        # область плашки = связные куски нужного цвета, касающиеся рамки текста
        n, lab, st, _ = cv2.connectedComponentsWithStats(cv2.morphologyEx(plate_px, cv2.MORPH_CLOSE, _ell(2)))
        region = np.zeros_like(Q)
        for i in range(1, n):
            comp = lab == i
            if (Q[comp] > 0).mean() > 0.15 or ((Q > 0) & comp).sum() > 0.2 * (Q > 0).sum():
                region[comp] = 255
        region |= Q
        region = _fill_holes(region)
        inside = cv2.erode(region, _ell(1)) & ~_grow(plate_px, 1)
        glyph_white = (self.white[inside > 0] > 0).mean() if (inside > 0).any() else 1.0
        glyph_sel = (inside > 0) & (self.white == 0) & (self.colored > 0)
        glyph_colors = self._ordered_colors(cv2.erode(glyph_sel.astype(np.uint8) * 255, _ell(1)) > 0, box,
                                            extra_refs=[plate_rgb])
        # белая каёмка вокруг плашки
        ring = _grow(region, h * 0.2) & ~region
        border = (255, 255, 255) if (self.white[ring > 0] > 0).mean() > 0.35 else None
        mask = region | (self.white & _grow(region, h * 0.2))
        mask = _fill_holes(mask)
        if not glyph_colors or glyph_white > 0.55:
            # белые буквы на цветной подложке (лепесток с «の»)
            return BoxStyle("badge", [(255, 255, 255)], None, 0.0, plate_rgb, border, True, mask)
        outline = (255, 255, 255) if glyph_white > 0.12 else None
        return BoxStyle("plate", glyph_colors, outline, 0.09 if outline else 0.0, plate_rgb, border, True, mask)

    def split_badge(self, box: TextBox, st: BoxStyle):
        """Ищет внутри таблички цветной «лепесток» с белой буквой. -> (табличка, лепесток, цвет)."""
        h = box.size
        Q = box.mask(self.img.shape)
        region = _grow(Q, h * 0.25)
        plate_lab = self.lab_of(st.plate_color)
        far_from_plate = (np.linalg.norm(self.lab - plate_lab, axis=2) > 35).astype(np.uint8) * 255
        hsv_sat = cv2.cvtColor(self.img, cv2.COLOR_RGB2HSV)[..., 1]
        cand = region & far_from_plate & ~self.white & ((hsv_sat > 80).astype(np.uint8) * 255)
        tl, tr, br, bl = box.quad
        u = (tr - tl) / max(1e-6, np.linalg.norm(tr - tl))
        comps = []
        for rgb, *_ in self._clusters(cand > 0, k=6, merge_de=20, min_share=0.05):
            same = (np.linalg.norm(self.lab - self.lab_of(rgb), axis=2) < 24).astype(np.uint8) * 255
            same = cv2.morphologyEx(same & cand, cv2.MORPH_CLOSE, _ell(max(1, h * 0.05)))
            n, lab, stats, _ = cv2.connectedComponentsWithStats(same)
            comps += [((lab == i).astype(np.uint8) * 255, stats[i]) for i in range(1, n)]
        for comp, (x, y, w, hh, area) in comps:
            if area < 0.2 * h * h or hh < 0.55 * h or w > 1.6 * h:
                continue
            filled = _fill_holes(comp)
            hull = cv2.convexHull(np.argwhere(filled > 0)[:, ::-1].astype(np.int32))
            if (filled > 0).sum() < 0.65 * cv2.contourArea(hull):
                continue  # тонкие штрихи — это буква, а не лепесток
            holes = filled & ~comp
            if (self.white[holes > 0] > 0).sum() < 0.05 * area:
                continue  # внутри нет белой буквы
            ys, xs = np.nonzero(comp)
            t = (np.stack([xs, ys], 1) - tl) @ u
            run = np.linalg.norm(tr - tl)
            if t.min() > 0.45 * run:          # лепесток справа
                cut = t.min() - h * 0.05
                plate_q = np.array([tl, tl + u * cut, bl + u * cut, bl])
            elif t.max() < 0.55 * run:        # лепесток слева
                cut = t.max() + h * 0.05
                plate_q = np.array([tl + u * cut, tr, br, bl + u * cut])
            else:
                continue
            badge_q = np.array([[x, y], [x + w, y], [x + w, y + hh], [x, y + hh]], np.float32)
            color = tuple(int(v) for v in np.median(self.img[comp > 0], 0))
            return (TextBox(plate_q.astype(np.float32), box.text, box.score),
                    TextBox(badge_q, box.text[-1:] or "の", box.score), color)
        return None

    def _sticker_letters(self, box, Q):
        h = box.size
        region = _grow(Q, max(2, h * 0.06))
        cand = self.colored & region & ~self.white
        n, lab, st, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
        Qs = Q > 0
        letters = np.zeros_like(Q)
        min_area = max(8, h * h * 0.002)
        for i in range(1, n):
            if st[i, cv2.CC_STAT_AREA] < min_area:
                continue
            x, y, w, hh = st[i, :4]
            pad = 3
            sub = (lab[max(0, y - pad):y + hh + pad, max(0, x - pad):x + w + pad] == i).astype(np.uint8) * 255
            ring = _grow(sub, 2) & ~sub
            wsub = self.white[max(0, y - pad):y + hh + pad, max(0, x - pad):x + w + pad]
            qsub = Qs[max(0, y - pad):y + hh + pad, max(0, x - pad):x + w + pad]
            inside = qsub[sub > 0].mean()
            if ring.any() and inside > 0.5 and (wsub[ring > 0] > 0).mean() > 0.7:
                letters[max(0, y - pad):y + hh + pad, max(0, x - pad):x + w + pad] |= sub
        return letters

    def _sticker_style(self, box, Q, letters):
        h = box.size
        d = max(3, int(h * 0.1))
        stick = letters | (self.white & _grow(letters, d))
        stick = cv2.morphologyEx(stick, cv2.MORPH_CLOSE, _ell(2))
        stick = _fill_holes(stick, max_area=0.08 * h * h * max(1.0, box.run / h))
        # толщина белой обводки
        dist = cv2.distanceTransform((letters == 0).astype(np.uint8), cv2.DIST_L2, 3)
        edge = stick & ~cv2.erode(stick, _ell(1)) & ~_grow(letters, 1)
        ow = float(np.median(dist[edge > 0])) if (edge > 0).any() else h * 0.06
        colors = self._ordered_colors(cv2.erode(letters, _ell(1)) > 0, box, k=8)
        return BoxStyle("sticker", colors or [(240, 90, 140)], (255, 255, 255),
                        float(np.clip(ow / h, 0.035, 0.14)), mask=stick)

    def _plain_style(self, box, Q):
        """Обычные буквы: цвет букв, обводка, цвет-акцент, тонкие ли штрихи.

        Цвета текста — те, что в середине строки встречаются заметно чаще, чем снаружи рамки
        (фон одинаков везде). Из них обводка — та, что окружает другую; акцент — отдельный
        яркий цвет (красные «の», «に»).
        """
        h = box.size
        mid_q = np.zeros_like(Q)
        cv2.fillPoly(mid_q, [np.round(_band(box.quad, 0.3, 0.7, box.vertical)).astype(np.int32)], 255)
        mid = (mid_q & Q) > 0
        edge = (_grow(Q, h * 0.3) & ~Q) > 0
        Qd = _grow(Q, max(2, h * 0.12))
        cands = []
        for rgb, *_ in self._clusters(Q > 0, k=6, merge_de=20, min_share=0.01):
            m = np.linalg.norm(self.lab - self.lab_of(rgb), axis=2) < 22
            c_mid = float(m[mid].mean()) if mid.any() else 0.0
            c_edge = float(m[edge].mean()) if edge.any() else 0.0
            ratio = c_mid / (c_edge + 0.02)
            cands.append((ratio * np.sqrt(c_mid), ratio, c_mid, rgb, self._strokes(m, Qd, mid, edge), m))
        cands.sort(key=lambda c: -c[0])
        text = [c for c in cands if c[1] > 1.3 and c[2] >= 0.01] or cands[:1]
        if not text:
            return BoxStyle("plain", [(255, 255, 255)], (25, 20, 30), 0.06, mask=_grow(Q, h * 0.14))

        r = max(2, int(h * 0.04))

        def surround(a, b):     # какая доля кольца вокруг a занята b
            ring = _grow(a, r) & ~a
            return (ring & b).sum() / max(1, ring.sum())

        fill_c, outline_c = text[0], None
        if len(text) > 1:
            other = text[1]
            s_fo, s_of = surround(fill_c[4], other[4]), surround(other[4], fill_c[4])
            if max(s_fo, s_of) > 0.45:
                if s_of > s_fo:          # «другой» цвет внутри — значит это буквы, а наш — обводка
                    fill_c, outline_c = other, fill_c
                else:
                    outline_c = other
        fill = fill_c[3]
        fill_lab = self.lab_of(fill)
        accent = None
        for c in text:
            if c is fill_c or c is outline_c:
                continue
            rgb = c[3]
            if max(rgb) - min(rgb) > 80 and np.linalg.norm(self.lab_of(rgb) - fill_lab) > 40 \
                    and surround(c[4], fill_c[4]) < 0.3:
                accent = rgb
                break
        # толщина штрихов относительно высоты строки
        fm = fill_c[4] & Q
        thin = (fm > 0).sum() > 50 and stroke_ratio(fm, h) < THIN_RATIO
        # стираем только сами буквы (с запасом на сглаживание и тень), а не всю рамку —
        # так не задеваем фон и соседние надписи вроде английского названия
        # Что стирать: сами буквы. Строгий цвет — чтобы не прилипал похожий фон (подсвеченная
        # стена), затем куски того же цвета, связанные с буквами (кончики штрихов за рамкой OCR),
        # и мягкий край с запасом по цвету (сглаживание, подсветка).
        strict = np.zeros(Q.shape, bool)
        loose = np.zeros(Q.shape, bool)
        for c in text:
            if c is fill_c or c is outline_c or (accent is not None and c[3] == accent):
                d = np.linalg.norm(self.lab - self.lab_of(c[3]), axis=2)
                strict |= d < 24
                loose |= d < 40
        strokes = self._strokes(strict, Qd, None, edge)
        strokes |= self._thin_strokes(fill, Q, strokes, edge, h)
        if strokes.any():
            near = _grow(Q, h * 0.35) > 0
            n, lab = cv2.connectedComponents((strict & near).astype(np.uint8), connectivity=8)
            ids = np.unique(lab[(strokes > 0) & (lab > 0)])
            if len(ids):
                strokes |= np.isin(lab, ids).astype(np.uint8) * 255
            strokes |= (loose & (_grow(strokes, max(2, h * 0.05)) > 0)).astype(np.uint8) * 255

        # Цвет и толщину берём по «телу» букв (буквы вместе с обводкой, тенью, градиентом):
        # заливка — из середины штрихов, обводка — с их края. Так белые буквы с тёмной обводкой,
        # градиент или 3D-тень не путаются с цветом букв.
        found = self._letter_body(box, Q, Qd, mid, edge)
        gradient, bg, far_all = None, None, None
        if found is not None:
            body, outline_part, bg, far_all = found
            got = self._body_colors(body, h, outline_part, bg)
            if got is not None:
                fill, outline_rgb, thin, gradient = got
                outline_c = (None, None, None, outline_rgb) if outline_rgb else None
                if accent is not None and np.linalg.norm(self.lab_of(accent) - self.lab_of(fill)) < 40:
                    accent = None
            strokes |= body
        outline = outline_c[3] if outline_c else None
        if (strokes > 0).sum() > 0.02 * (Q > 0).sum():
            mask = _grow(strokes, max(3, h * 0.08))
        else:
            strokes = None
            mask = _grow(Q, h * 0.14)
        if far_all is not None:
            # всё «не похожее на фон» внутри рамки текста — тоже стираем (подсветка, тени, тонкие хвосты)
            mask |= _grow(far_all, max(2, h * 0.04))
        if h < 0.05 * self.H:
            mask |= _grow(Q, max(3, h * 0.25))       # мелкие строки стираем целиком — так надёжнее
        return BoxStyle("plain", [fill], outline, 0.06 if outline else 0.0, mask=mask, accent=accent, thin=thin,
                        strokes=strokes, bg=bg, gradient=gradient)

    def _letter_body(self, box: TextBox, Q, Qd, mid, edge) -> np.ndarray | None:
        """Буквы вместе с обводкой: всё в рамке, что не похоже на фон вокруг надписи."""
        h = box.size
        ring = (_grow(Q, h * 0.35) & ~_grow(Q, max(3, h * 0.1)) & ~self.others) > 0
        if ring.sum() < 50:
            ring = (_grow(Q, h * 0.35) & ~_grow(Q, max(3, h * 0.1))) > 0
        bg = self._clusters(ring, k=6, merge_de=10, min_share=0.03)
        if not bg:
            return None
        ys, xs = np.nonzero(Qd)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        lab = self.lab[y0:y1, x0:x1]
        dmin = np.full(lab.shape[:2], 1e9, np.float32)
        for rgb, *_ in bg:
            dmin = np.minimum(dmin, np.linalg.norm(lab - self.lab_of(rgb), axis=2))
        far = np.zeros(Q.shape, bool)
        far[y0:y1, x0:x1] = dmin > 28
        bg_rgb = bg[0][0]
        # для стирания: всё, что заметно отличается от фона, внутри рамки (с небольшим запасом)
        loose_far = np.zeros(Q.shape, bool)
        loose_far[y0:y1, x0:x1] = dmin > 18
        far_all = (loose_far & (_grow(Q, max(2, h * 0.08)) > 0)).astype(np.uint8) * 255
        body = self._strokes(far, Qd, mid, edge)
        if (body > 0).sum() <= 0.04 * (Q > 0).sum():
            return None
        # Буквы-контуры (белая середина того же цвета, что фон, и тёмная обводка): нашлась только
        # обводка, а середина — «дыры» в ней. Если дыры больше самих штрихов — это середина букв.
        filled = _fill_holes(body, max_area=0.6 * h * h)
        if (filled > 0).sum() >= 1.8 * (body > 0).sum():
            return filled, body, bg_rgb, far_all      # (буквы целиком, их обводка, фон, всё не-фон)
        return body, None, bg_rgb, far_all

    def _body_colors(self, body: np.ndarray, h: float, outline_part: np.ndarray | None = None, bg=None):
        """(цвет букв, цвет обводки или None, тонкие ли штрихи) по телу букв."""
        to_rgb = lambda lab: tuple(int(v) for v in cv2.cvtColor(
            np.clip(lab, 0, 255).astype(np.uint8).reshape(1, 1, 3), cv2.COLOR_LAB2RGB)[0, 0])
        inside = body > 0
        if inside.sum() < 40:
            return None
        if outline_part is not None:
            # буквы-контуры: середина — самый частый цвет внутри обводки, обводка — сама обводка
            holes = inside & (outline_part == 0)
            cl = self._clusters(holes, k=4, merge_de=15, min_share=0.05)
            if not cl:
                return None
            outline = to_rgb(np.median(self.lab[outline_part > 0], axis=0))
            return cl[0][0], outline, False, None
        d = cv2.distanceTransform(inside.astype(np.uint8), cv2.DIST_L2, 3)
        dmax = max(1.0, float(d.max()))
        thin = stroke_ratio(body, h) < THIN_RATIO
        lab = self.lab
        bg_lab = self.lab_of(bg) if bg is not None else None
        # цвет букв — тот, что глубже всего внутри штрихов, занимает много места и не похож на фон
        best, fill_lab = -1.0, None
        cands = []
        for rgb, share, *_ in self._clusters(inside, k=5, merge_de=18, min_share=0.05):
            c = self.lab_of(rgb)
            # серые/серебряные с градиентом: считаем вместе все пиксели того же оттенка
            if np.hypot(c[1] - 128, c[2] - 128) < 10:
                m = inside & (np.hypot(lab[..., 1] - 128, lab[..., 2] - 128) < 12) & (np.abs(lab[..., 0] - c[0]) < 70)
                share = float(m.sum()) / max(1, inside.sum())
            else:
                m = inside & (np.linalg.norm(lab - c, axis=2) < 24)
            if m.sum() < 30:
                continue
            depth = float(d[m].mean()) / dmax
            # чем сильнее цвет отличается от фона, тем вероятнее это буквы (тень и обводка обычно ближе к фону)
            contrast = 1.0 if bg_lab is None else min(120.0, float(np.linalg.norm(c - bg_lab))) / 120.0
            score = share * contrast * depth
            cands.append((share, contrast, c))
            if score > best:
                best, fill_lab = score, c
        if thin and cands:
            # у тонких букв «глубины» почти нет, края сглажены — берём самый контрастный крупный оттенок
            big = [x for x in cands if x[0] >= 0.2] or cands
            fill_lab = max(big, key=lambda x: x[1])[2]
        if fill_lab is None:
            deep = inside & (d >= max(1.0, float(np.percentile(d[inside], 65))))
            fill_lab = np.median(lab[deep], axis=0)
        # обводка — заметно другой цвет по краю штрихов
        rim = inside & (d <= 3.0) & (np.linalg.norm(lab - fill_lab, axis=2) > 35)
        outline = to_rgb(np.median(lab[rim], axis=0)) if rim.sum() > 0.15 * (inside & (d <= 3.0)).sum() else None
        # градиент сверху вниз: пиксели того же оттенка, что буквы (яркость может меняться)
        gradient = None
        same_hue = inside & (np.linalg.norm((lab - fill_lab)[..., 1:], axis=2) < 18) & (d >= 1.5)
        ys = np.nonzero(same_hue)[0]
        if len(ys) > 80:
            lo, hi = np.percentile(ys, 25), np.percentile(ys, 75)
            rows = np.arange(self.H)[:, None]
            top = np.median(lab[same_hue & (rows <= lo)], axis=0)
            bot = np.median(lab[same_hue & (rows >= hi)], axis=0)
            if np.linalg.norm(top - bot) > 28:
                gradient = (to_rgb(top), to_rgb(bot))
        if gradient:
            fill_lab = (self.lab_of(gradient[0]) + self.lab_of(gradient[1])) / 2
        return to_rgb(fill_lab), outline, thin, gradient

    def _thin_strokes(self, fill, Q, core, edge, h) -> np.ndarray:
        """Тонкие линии светлее (или темнее) своего окружения — штрихи букв. Морфологический
        «top-hat» не реагирует на большие светлые области (подсвеченная стена, небо).
        Берём только то, что лежит в пределах уже найденных букв (их выпуклой оболочки)."""
        if not core.any():
            return np.zeros_like(Q)
        n_c, lab_c, st_c, _ = cv2.connectedComponentsWithStats((core > 0).astype(np.uint8))
        hull_m = np.zeros_like(Q)
        pts = np.argwhere(core > 0)[:, ::-1].astype(np.int32)
        cv2.fillPoly(hull_m, [cv2.convexHull(pts)], 255)
        hull_m = _grow(hull_m, max(2, h * 0.08)) & _grow(Q, max(2, h * 0.08))
        L = self.lab[..., 0]
        fill_L = float(self.lab_of(fill)[0])
        bg_L = float(np.median(L[edge > 0])) if (edge > 0).any() else 128.0
        k = _ell(max(4, h * 0.08))
        if fill_L > bg_L + 40:
            th = cv2.morphologyEx(L, cv2.MORPH_TOPHAT, k)
        elif fill_L < bg_L - 40:
            th = cv2.morphologyEx(L, cv2.MORPH_BLACKHAT, k)
        else:
            return np.zeros_like(Q)
        near_fill = np.linalg.norm(self.lab - self.lab_of(fill), axis=2) < 50
        cand = ((th > 28) & near_fill & (_grow(Q, h * 0.35) > 0)).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
        out = np.zeros_like(Q)
        inside = hull_m > 0
        for i in range(1, n):
            x, y, w, hh, area = st[i]
            sub = lab[y:y + hh, x:x + w] == i
            if area >= 6 and inside[y:y + hh, x:x + w][sub].mean() >= 0.6:
                out[y:y + hh, x:x + w][sub] = 255
        return out

    @staticmethod
    def _strokes(m: np.ndarray, Qd: np.ndarray, mid: np.ndarray, edge: np.ndarray) -> np.ndarray:
        """Только пятна цвета, похожие на буквы: проходят через середину строки и не уходят за рамку
        (так белое небо за белыми буквами не путается с самими буквами)."""
        mm = (m & (Qd > 0)).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(mm, connectivity=8)
        keep = np.zeros(mm.shape, np.uint8)
        for i in range(1, n):
            x, y, w, h, area = st[i]
            sub = lab[y:y + h, x:x + w] == i
            if mid is not None and not mid[y:y + h, x:x + w][sub].any():
                continue
            if edge[y:y + h, x:x + w][sub].sum() > 0.3 * area:
                continue
            keep[y:y + h, x:x + w][sub] = 255
        return keep

    def lab_of(self, rgb) -> np.ndarray:
        return cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2LAB)[0, 0].astype(np.float32)


def _band(q, a: float, b: float, vertical: bool) -> np.ndarray:
    """Полоса внутри рамки текста: от a до b поперёк строки (0..1)."""
    tl, tr, br, bl = q
    if vertical:
        return np.array([tl + a * (tr - tl), tl + b * (tr - tl), bl + b * (br - bl), bl + a * (br - bl)])
    return np.array([tl + a * (bl - tl), tr + a * (br - tr), tr + b * (br - tr), tl + b * (bl - tl)])


THIN_RATIO = 0.08    # толщина штриха / высота строки: меньше — тонкий логотип (засечки, «минтё»)


def stroke_ratio(mask: np.ndarray, h: float) -> float:
    """Толщина штрихов относительно высоты строки. Рваные края и мелкие дырки (потёртые буквы)
    сначала заглаживаются; толщина ≈ 4 × медиана расстояния до края."""
    m = (mask > 0).astype(np.uint8)
    if m.sum() < 20:
        return 0.0
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    d = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    return 4.0 * float(np.median(d[m > 0])) / max(1.0, h)


def _fill_holes(m: np.ndarray, max_area: float | None = None) -> np.ndarray:
    inv = (m == 0).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    out = m.copy()
    H, W = m.shape
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if x == 0 or y == 0 or x + w >= W or y + h >= H:
            continue
        if max_area is None or a <= max_area:
            out[lab == i] = 255
    return out


@dataclass
class LogoAnalysis:
    styles: dict[int, BoxStyle]      # id(box) -> стиль
    mask: np.ndarray                 # что стереть


def _moved(b: TextBox, dx: float, dy: float) -> TextBox:
    return TextBox(b.quad + np.float32([dx, dy]), b.text, b.score)


def analyze_logo(img: np.ndarray, cluster: Cluster, extra: list[TextBox] = ()) -> LogoAnalysis:
    """extra — надписи рядом с логотипом (английское название), их стираем вместе с ним.

    Разбираем только область вокруг логотипа (так в несколько раз быстрее), потом
    переносим результат обратно в координаты всего постера.
    """
    H, W = img.shape[:2]
    allb = list(cluster.boxes) + list(extra)
    pad = int(cluster.size * 1.3) + 8
    x0 = max(0, int(min(b.x0 for b in allb)) - pad)
    y0 = max(0, int(min(b.y0 for b in allb)) - pad)
    x1 = min(W, int(max(b.x1 for b in allb)) + pad + 1)
    y1 = min(H, int(max(b.y1 for b in allb)) + pad + 1)
    local = Cluster([_moved(b, -x0, -y0) for b in cluster.boxes])
    la = _analyze_logo_area(np.ascontiguousarray(img[y0:y1, x0:x1]), local,
                            [_moved(b, -x0, -y0) for b in extra])

    def full(m):
        if m is None:
            return None
        out = np.zeros((H, W), np.uint8)
        out[y0:y1, x0:x1] = m
        return out

    styles = {}
    boxes = []
    for b in local.boxes:            # после анализа рамки могли разделиться (табличка + лепесток)
        gb = _moved(b, x0, y0)
        st = la.styles[id(b)]
        st.mask, st.strokes = full(st.mask), full(st.strokes)
        styles[id(gb)] = st
        boxes.append(gb)
    cluster.boxes[:] = boxes
    cluster.rows = _rows(cluster.boxes)
    return LogoAnalysis(styles, full(la.mask))


def _analyze_logo_area(img: np.ndarray, cluster: Cluster, extra: list[TextBox] = ()) -> LogoAnalysis:
    an = Analyzer(img)

    def excl_for(b):
        # мелкие подписи поверх крупной строки разбираем отдельно
        excl = np.zeros(img.shape[:2], np.uint8)
        for o in cluster.boxes:
            if o is not b and o.size < 0.6 * b.size:
                excl |= _grow(o.mask(img.shape), o.size * 0.35)
        return excl

    def others_for(b):
        m = np.zeros(img.shape[:2], np.uint8)
        for o in list(cluster.boxes) + list(extra):
            if o is not b:
                m |= _grow(o.mask(img.shape), max(2, o.size * 0.15))
        return m

    styles = {}
    for b in list(cluster.boxes):
        an.others = others_for(b)
        st = an.analyze_box(b, excl_for(b))
        styles[id(b)] = st
        if st.kind == "plate":
            # OCR мог склеить табличку и лепесток («さくら荘» + «の») — разделяем
            split = an.split_badge(b, st)
            if split:
                plate_box, badge_box, _ = split
                i = cluster.boxes.index(b)
                cluster.boxes[i:i + 1] = [plate_box, badge_box]
                del styles[id(b)]
                styles[id(plate_box)] = an.analyze_box(plate_box, excl_for(plate_box))
                styles[id(badge_box)] = an.analyze_box(badge_box, excl_for(badge_box))
                if styles[id(badge_box)].kind not in ("badge", "plate"):
                    styles[id(badge_box)] = BoxStyle("badge", [(255, 255, 255)], None, 0.0, split[2], None,
                                                     True, styles[id(badge_box)].mask)
                styles[id(badge_box)].kind = "badge"
                styles[id(badge_box)].glyph_colors = [(255, 255, 255)]
    cluster.rows = _rows(cluster.boxes)
    mask = np.zeros(img.shape[:2], np.uint8)
    for st in styles.values():
        mask |= st.mask
    for b in extra:
        an.others = others_for(b)
        mask |= an.analyze_box(b).mask
    # добираем куски букв, попавшие между рамками соседних частей логотипа
    plain_colors = [c for st in styles.values() if st.kind == "plain"
                    for c in st.glyph_colors[:1] + ([st.accent] if st.accent else [])]
    if plain_colors:
        area = np.zeros_like(mask)
        for b in list(cluster.boxes) + list(extra):
            area |= _grow(b.mask(img.shape), max(2, b.size * 0.08))
        strict = np.zeros(mask.shape, bool)
        for c in plain_colors:
            strict |= np.linalg.norm(an.lab - an.lab_of(c), axis=2) < 26
        rest = (strict & (area > 0) & (mask == 0)).astype(np.uint8)
        n, lab, st_, _ = cv2.connectedComponentsWithStats(rest, connectivity=8)
        for i in range(1, n):
            if st_[i, cv2.CC_STAT_AREA] >= 4:
                mask |= _grow((lab == i).astype(np.uint8) * 255, max(3, cluster.size * 0.04))
    # сглаживание + место под тень
    size = cluster.size
    mask = _grow(mask, max(2, size * 0.02))
    shift = max(1, int(size * 0.025))
    shadow = np.roll(np.roll(mask, shift, 0), shift, 1)
    mask |= shadow & _grow(mask, shift * 2)
    return LogoAnalysis(styles, mask)


def text_removal_mask(img_shape, boxes: list[TextBox]) -> np.ndarray:
    m = np.zeros(img_shape[:2], np.uint8)
    for b in boxes:
        m |= _grow(b.mask(img_shape), max(2, b.size * 0.18))
    return m
