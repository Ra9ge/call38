"""Стирание текста: LaMa (нейросеть, если доступна) или встроенный OpenCV."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# Где брать нейросеть LaMa (по очереди). GitHub (зоопарк моделей OpenCV) открывается почти везде,
# HuggingFace — запасной вариант. (файл, адрес, sha1 или None)
LAMA_SOURCES = [
    ("inpainting_lama_2025jan.onnx",
     "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/inpainting_lama/"
     "inpainting_lama_2025jan.onnx",
     "7c6cdb9362bf73de2a80cfcaf17e121e3302f24c"),
    ("lama_fp32.onnx", "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx", None),
    ("lama_fp32.onnx", "https://hf-mirror.com/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx", None),
]
if os.environ.get("LAMA_URL"):
    LAMA_SOURCES.insert(0, ("lama_custom.onnx", os.environ["LAMA_URL"], None))
LAMA_FILE: Path | None = None          # можно задать вручную (используется в тестах)
FAIL_MARKER = MODELS_DIR / ".lama_download_failed"

_SESSION = None
_LAMA_FAILED = False


def log(msg: str):
    print(msg, file=sys.stderr, flush=True)


def _existing_model() -> Path | None:
    if LAMA_FILE is not None:
        return LAMA_FILE if Path(LAMA_FILE).exists() else None
    for name, _, _ in LAMA_SOURCES:
        if (MODELS_DIR / name).exists():
            return MODELS_DIR / name
    return None


def _download_lama() -> Path | None:
    import hashlib
    import time

    import requests
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    # если недавно не скачалась — не тормозим каждый запуск (повтор через сутки)
    if FAIL_MARKER.exists() and time.time() - FAIL_MARKER.stat().st_mtime < 24 * 3600:
        return None
    for name, url, sha1 in LAMA_SOURCES:
        target = MODELS_DIR / name
        tmp = target.with_suffix(".part")
        try:
            log(f"Скачиваю нейросеть LaMa для чистого удаления текста (~100–200 МБ, один раз): {url}")
            h = hashlib.sha1()
            with requests.get(url, stream=True, timeout=(10, 60)) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                done = 0
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
                        h.update(chunk)
                        done += len(chunk)
                        if total:
                            print(f"\r  {done * 100 // total}%", end="", file=sys.stderr, flush=True)
            print(file=sys.stderr)
            if sha1 and h.hexdigest() != sha1:
                raise ValueError("файл повреждён (не совпала контрольная сумма)")
            tmp.replace(target)
            FAIL_MARKER.unlink(missing_ok=True)
            return target
        except Exception as e:  # сеть/прокси/блокировка — пробуем следующий источник
            log(f"  не получилось: {e}")
            tmp.unlink(missing_ok=True)
    FAIL_MARKER.touch()
    log("LaMa не скачалась; следующая попытка через сутки (или удали models/.lama_download_failed).")
    return None


def _open_session(path: Path):
    """Загружает модель. Оптимизированную копию сохраняем рядом — следующие запуски быстрее."""
    import onnxruntime as ort
    cached = path.with_name(path.stem + ".optimized.onnx")
    if cached.exists() and cached.stat().st_mtime >= path.stat().st_mtime:
        try:
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            so.log_severity_level = 3
            return ort.InferenceSession(str(cached), so, providers=["CPUExecutionProvider"])
        except Exception:
            cached.unlink(missing_ok=True)
    so = ort.SessionOptions()
    so.log_severity_level = 3
    try:
        so.optimized_model_filepath = str(cached)
    except Exception:
        pass
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def _lama_session(allow_download: bool):
    global _SESSION, _LAMA_FAILED
    if _SESSION is not None or _LAMA_FAILED:
        return _SESSION
    path = _existing_model() or (_download_lama() if allow_download else None)
    if path is None:
        _LAMA_FAILED = True
        return None
    try:
        log("Загружаю нейросеть LaMa...")
        _SESSION = _open_session(path)
    except Exception as e:
        log(f"LaMa не загрузилась ({e}), использую OpenCV")
        _LAMA_FAILED = True
    return _SESSION


def _lama_run(sess, crop: np.ndarray, cmask: np.ndarray) -> np.ndarray | None:
    inputs = sess.get_inputs()
    mask_in = next((i for i in inputs if "mask" in i.name.lower()), inputs[-1])
    img_in = next(i for i in inputs if i is not mask_in)
    size = img_in.shape[-1] if isinstance(img_in.shape[-1], int) else 512
    h, w = crop.shape[:2]
    im = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    # маска при уменьшении: пиксель закрыт, если под ним есть хоть кусочек маски (+1 пиксель запаса) —
    # иначе краешки старых букв остаются видны модели, и она дорисовывает из них «призрак» надписи
    m = cv2.resize((cmask > 0).astype(np.float32), (size, size), interpolation=cv2.INTER_AREA) > 0.003
    m = cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32)
    keep = m < 0.5
    im_in = im * keep[..., None]
    # Модель ожидает картинку 0..1; на всякий случай проверяем результат и пробуем 0..255.
    for scale in (1 / 255.0, 1.0):
        x = (im_in * scale).transpose(2, 0, 1)[None].astype(np.float32)
        out = sess.run(None, {img_in.name: x, mask_in.name: m[None, None]})[0][0]
        out = out.transpose(1, 2, 0)
        if out.max() <= 1.5:
            out = out * 255.0
        out = np.clip(out, 0, 255)
        err = np.abs(out[keep] - im[keep]).mean() if keep.any() else 0
        if err < 25:
            return cv2.resize(out.astype(np.uint8), (w, h), interpolation=cv2.INTER_CUBIC)
    return None


def inpaint_lama(img: np.ndarray, mask: np.ndarray, allow_download: bool = True) -> np.ndarray | None:
    sess = _lama_session(allow_download)
    if sess is None:
        return None
    H, W = img.shape[:2]
    out = img.copy()
    n, lab, st, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8))
    # соседние куски обрабатываем одним окном; совсем мелкие надписи (титры, даты) — быстрым
    # способом OpenCV: на тонкой строчке его размытие не видно, а нейросеть — это ~2 с на окно
    boxes = []
    tiny = np.zeros(mask.shape, np.uint8)
    for i in range(1, n):
        x, y, w, h, _ = st[i]
        if h < 0.022 * H:
            tiny[lab == i] = 255
            continue
        boxes.append([x, y, x + w, y + h])
    if tiny.any():
        out = inpaint_opencv(out, tiny)
    merged = []
    for b in sorted(boxes):
        for m in merged:
            if b[0] < m[2] + 40 and b[2] > m[0] - 40 and b[1] < m[3] + 40 and b[3] > m[1] - 40:
                m[:] = [min(m[0], b[0]), min(m[1], b[1]), max(m[2], b[2]), max(m[3], b[3])]
                break
        else:
            merged.append(list(b))
    remaining = mask > 0
    for x0, y0, x1, y1 in merged:
        for ax0, ay0, ax1, ay1, core in _windows(x0, y0, x1, y1, W, H):
            crop = out[ay0:ay1, ax0:ax1]
            hide = remaining[ay0:ay1, ax0:ax1]            # модель не должна видеть ни одной старой буквы,
            cm = hide & core                              # а записываем только свою часть окна
            if not cm.any():
                continue
            res = _lama_run(sess, crop, hide.astype(np.uint8) * 255)
            if res is None:
                log("LaMa дала странный результат, использую OpenCV")
                return None
            soft = cv2.GaussianBlur(cm.astype(np.float32), (0, 0), 1.5)
            soft = np.maximum(soft, cm.astype(np.float32))[..., None]
            out[ay0:ay1, ax0:ax1] = (res * soft + crop * (1 - soft)).astype(np.uint8)
            remaining[ay0:ay1, ax0:ax1] &= ~cm
    return out


def _windows(x0, y0, x1, y1, W, H):
    """Окна для LaMa. Модель работает в 512×512, поэтому большую надпись (логотип во всю ширину)
    не ужимаем в одно окно, а идём окнами с перекрытием почти в родном масштабе: каждое следующее
    окно видит уже дорисованный кусок как готовый фон. core — какую часть окна дорисовывать сейчас
    (хвост оставляем следующему окну — у него будет больше фона вокруг)."""
    bw, bh = x1 - x0, y1 - y0
    short, long_ = min(bw, bh), max(bw, bh)
    if long_ * 1.25 + 64 <= 960:
        # всё помещается в одно окно (уменьшение до ~1.9 раза почти не заметно)
        side = int(min(max(short * 2.2 + 96, long_ * 1.25 + 64, 256), max(H, W)))
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        ax0 = int(np.clip(cx - side // 2, 0, max(0, W - side)))
        ay0 = int(np.clip(cy - side // 2, 0, max(0, H - side)))
        ax1, ay1 = min(W, ax0 + side), min(H, ay0 + side)
        yield ax0, ay0, ax1, ay1, np.ones((ay1 - ay0, ax1 - ax0), bool)
        return
    side = int(min(max(short * 2.2 + 96, 640), 960, H, W))

    def starts(a0, a1, lim):
        """Начала окон вдоль одной оси (с перекрытием), чтобы покрыть [a0, a1]."""
        if a1 - a0 + 0.4 * side <= side:                  # по этой оси хватает одного окна
            return [int(np.clip((a0 + a1) / 2 - side / 2, 0, max(0, lim - side)))]
        out, s = [], a0 - int(side * 0.2)
        while True:
            out.append(int(np.clip(s, 0, max(0, lim - side))))
            if s + side >= a1 + int(side * 0.2) or out[-1] >= lim - side:
                return out
            s += int(side * 0.55)

    xs, ys = starts(x0, x1, W), starts(y0, y1, H)
    cut = int(side * 0.78)                                # хвост окна дорисует следующее окно
    for j, sy in enumerate(ys):
        for i, sx in enumerate(xs):
            ax0, ay0, ax1, ay1 = sx, sy, min(W, sx + side), min(H, sy + side)
            core = np.ones((ay1 - ay0, ax1 - ax0), bool)
            if i < len(xs) - 1:
                core[:, cut:] = False
            if j < len(ys) - 1:
                core[cut:, :] = False
            yield ax0, ay0, ax1, ay1, core


def inpaint_opencv(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Без нейросети: мягкая заливка с нескольких масштабов (меньше «пятен», чем обычный Telea)."""
    m = (mask > 0).astype(np.uint8) * 255
    H, W = m.shape
    ys, xs = np.nonzero(m)
    extent = max(np.ptp(ys), np.ptp(xs)) + 1
    k = 4 if extent > 120 else 2 if extent > 40 else 1
    if k > 1:
        small = cv2.resize(img, (W // k, H // k), interpolation=cv2.INTER_AREA)
        sm = (cv2.resize(m, (W // k, H // k), interpolation=cv2.INTER_AREA) > 0).astype(np.uint8) * 255
        lo = cv2.inpaint(small, sm, 3, cv2.INPAINT_TELEA)
        lo = cv2.GaussianBlur(lo, (0, 0), 1.0)
        up = cv2.resize(lo, (W, H), interpolation=cv2.INTER_CUBIC)
        res = img.copy()
        res[m > 0] = up[m > 0]
        band = m & ~cv2.erode(m, np.ones((5, 5), np.uint8))
        res = cv2.inpaint(res, band, 3, cv2.INPAINT_TELEA)
    else:
        res = cv2.inpaint(img, m, 5, cv2.INPAINT_TELEA)
    # лёгкое зерно, чтобы залитая область не выглядела «пластиковой»
    noise = np.random.default_rng(1).normal(0, 2.0, img.shape).astype(np.float32)
    soft = cv2.GaussianBlur(m.astype(np.float32) / 255, (0, 0), 1.2)[..., None]
    return np.clip(res.astype(np.float32) + noise * soft, 0, 255).astype(np.uint8)


def lama_ready(allow_download: bool = True) -> bool:
    """Нейросеть LaMa на месте и загружена (при необходимости скачивает её)."""
    return _lama_session(allow_download) is not None


def inpaint(img: np.ndarray, mask: np.ndarray, method: str = "auto") -> tuple[np.ndarray, str]:
    if not (mask > 0).any():
        return img.copy(), "none"
    if method in ("auto", "lama"):
        res = inpaint_lama(img, mask, allow_download=True)
        if res is not None:
            return res, "lama"
        log("LaMa недоступна — стираю встроенным способом (OpenCV).")
    return inpaint_opencv(img, mask), "opencv"


def _shift(m: np.ndarray, dx: int, dy: int) -> np.ndarray:
    out = np.zeros_like(m)
    H, W = m.shape[:2]
    xs, xd = (slice(0, W - dx), slice(dx, W)) if dx >= 0 else (slice(-dx, W), slice(0, W + dx))
    ys, yd = (slice(0, H - dy), slice(dy, H)) if dy >= 0 else (slice(-dy, H), slice(0, H + dy))
    out[yd, xd] = m[ys, xs]
    return out


def halo_mask(img: np.ndarray, mask: np.ndarray, h: float) -> np.ndarray:
    """Тень и свечение букв — то, что не попало в маску букв, но после стирания остаётся
    тёмным (или светлым) «призраком» надписи.

    Тень повторяет форму букв со сдвигом, свечение — обводит буквы со всех сторон. Поэтому
    ищем сдвиг, при котором «буквы со сдвигом» ложатся на заметно более тёмную (или светлую)
    полосу вокруг букв, и стираем только эту полосу. Персонажи и предметы рядом с надписью
    по форме на буквы не похожи — их не трогаем."""
    m = mask > 0
    if not m.any() or h <= 0:
        return np.zeros_like(mask)
    r = max(3, int(round(0.2 * h)))
    ys, xs = np.where(m)
    pad = 3 * r
    Hh, Ww = mask.shape[:2]
    y0, y1 = max(0, ys.min() - pad), min(Hh, ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(Ww, xs.max() + pad + 1)
    mm = m[y0:y1, x0:x1].astype(np.uint8)
    lab = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_RGB2LAB).astype(np.float32)

    dist = cv2.distanceTransform((mm == 0).astype(np.uint8), cv2.DIST_L2, 3)   # до ближайшей буквы
    near = (dist <= r) & (mm == 0)
    outer = ((dist <= 2.6 * r) & (dist > 1.4 * r)).astype(np.float32)
    sig = 1.6 * r
    den = cv2.GaussianBlur(outer, (0, 0), sig)
    est = cv2.GaussianBlur(lab[..., 0] * outer, (0, 0), sig) / np.maximum(den, 1e-6)
    valid = near & (den > 0.03)
    dL = lab[..., 0] - est                         # темнее (<0) или светлее (>0) фона вокруг

    steps = sorted({0, r // 4, r // 2, (3 * r) // 4, r} | {-v for v in (r // 4, r // 2, (3 * r) // 4, r)})
    best = (0.0, None)
    for sign in (-1, 1):                           # тень (темнее) или свечение (светлее)
        hit = valid & (sign * dL > 10)
        if hit.sum() < 0.02 * mm.sum():
            continue
        frac = {}
        for dx in steps:
            for dy in steps:
                if dx == 0 and dy == 0:
                    ring = (dist <= 0.5 * r) & (mm == 0) & valid          # свечение со всех сторон
                else:
                    ring = (_shift(mm, dx, dy) > 0) & (mm == 0) & valid   # тень со сдвигом
                n = int(ring.sum())
                frac[dx, dy] = float((ring & hit).sum()) / n if n >= 0.02 * mm.sum() else 0.0
        for (dx, dy), f in frac.items():
            if dx == 0 and dy == 0:
                score = f if f >= 0.45 else 0.0                           # свечение — равномерно вокруг
            else:
                # тень — с одной стороны: в сторону сдвига темно, в обратную — нет
                score = f - frac.get((-dx, -dy), 0.0)
                score = score if f >= 0.25 and score >= 0.12 else 0.0
            if score > best[0]:
                best = (score, (sign, dx, dy))
    if best[1] is None:
        return np.zeros_like(mask)
    sign, dx, dy = best[1]
    base = (dist <= 0.5 * r) if dx == 0 and dy == 0 else (_shift(mm, dx, dy) > 0)
    # полоса тени с размытым краем: «буквы со сдвигом» + немного вокруг, и только где заметно темнее/светлее
    reach = cv2.distanceTransform((~base).astype(np.uint8), cv2.DIST_L2, 3) <= 0.35 * r
    halo = reach & (mm == 0) & valid & (sign * dL > 6)
    halo = cv2.morphologyEx(halo.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    halo = cv2.dilate(halo, np.ones((3, 3), np.uint8))
    out = np.zeros_like(mask)
    out[y0:y1, x0:x1] = np.where(halo > 0, 255, 0).astype(mask.dtype)
    return out
