"""Стирание текста: LaMa (нейросеть, если доступна) или встроенный OpenCV."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
LAMA_FILE = MODELS_DIR / "lama_fp32.onnx"
LAMA_URLS = [u for u in [
    os.environ.get("LAMA_URL"),
    "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx",
    "https://hf-mirror.com/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx",
] if u]

_SESSION = None
_LAMA_FAILED = False


def log(msg: str):
    print(msg, file=sys.stderr, flush=True)


FAIL_MARKER = MODELS_DIR / ".lama_download_failed"


def _download_lama() -> bool:
    import time

    import requests
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    # если недавно не скачалась — не тормозим каждый запуск (повтор через сутки)
    if FAIL_MARKER.exists() and time.time() - FAIL_MARKER.stat().st_mtime < 24 * 3600:
        return False
    tmp = LAMA_FILE.with_suffix(".part")
    for url in LAMA_URLS:
        try:
            log(f"Скачиваю нейросеть LaMa для чистого удаления текста (~200 МБ, один раз): {url}")
            with requests.get(url, stream=True, timeout=(10, 60)) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                done = 0
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
                        done += len(chunk)
                        if total:
                            print(f"\r  {done * 100 // total}%", end="", file=sys.stderr, flush=True)
            print(file=sys.stderr)
            tmp.replace(LAMA_FILE)
            FAIL_MARKER.unlink(missing_ok=True)
            return True
        except Exception as e:  # сеть/прокси/блокировка — пробуем следующее зеркало
            log(f"  не получилось: {e}")
            tmp.unlink(missing_ok=True)
    FAIL_MARKER.touch()
    log("LaMa не скачалась; следующая попытка через сутки (или удали models/.lama_download_failed).")
    return False


def _lama_session(allow_download: bool):
    global _SESSION, _LAMA_FAILED
    if _SESSION is not None or _LAMA_FAILED:
        return _SESSION
    if not LAMA_FILE.exists() and not (allow_download and _download_lama()):
        _LAMA_FAILED = True
        return None
    try:
        import onnxruntime as ort
        _SESSION = ort.InferenceSession(str(LAMA_FILE), providers=["CPUExecutionProvider"])
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
    m = (cv2.resize(cmask, (size, size), interpolation=cv2.INTER_NEAREST) > 0).astype(np.float32)
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
    # соседние куски обрабатываем одним окном
    boxes = []
    for i in range(1, n):
        x, y, w, h, _ = st[i]
        boxes.append([x, y, x + w, y + h])
    merged = []
    for b in sorted(boxes):
        for m in merged:
            if b[0] < m[2] + 40 and b[2] > m[0] - 40 and b[1] < m[3] + 40 and b[3] > m[1] - 40:
                m[:] = [min(m[0], b[0]), min(m[1], b[1]), max(m[2], b[2]), max(m[3], b[3])]
                break
        else:
            merged.append(list(b))
    for x0, y0, x1, y1 in merged:
        side = int(max(x1 - x0, y1 - y0) * 1.6) + 64
        side = min(max(side, 256), max(H, W))
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        ax0 = int(np.clip(cx - side // 2, 0, max(0, W - side)))
        ay0 = int(np.clip(cy - side // 2, 0, max(0, H - side)))
        ax1, ay1 = min(W, ax0 + side), min(H, ay0 + side)
        crop = out[ay0:ay1, ax0:ax1]
        cm = mask[ay0:ay1, ax0:ax1]
        res = _lama_run(sess, crop, cm)
        if res is None:
            log("LaMa дала странный результат, использую OpenCV")
            return None
        soft = cv2.GaussianBlur((cm > 0).astype(np.float32), (0, 0), 1.5)
        soft = np.maximum(soft, (cm > 0).astype(np.float32))[..., None]
        out[ay0:ay1, ax0:ax1] = (res * soft + crop * (1 - soft)).astype(np.uint8)
    return out


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


def inpaint(img: np.ndarray, mask: np.ndarray, method: str = "auto") -> tuple[np.ndarray, str]:
    if not (mask > 0).any():
        return img.copy(), "none"
    if method in ("auto", "lama"):
        res = inpaint_lama(img, mask, allow_download=True)
        if res is not None:
            return res, "lama"
        log("LaMa недоступна — стираю встроенным способом (OpenCV).")
    return inpaint_opencv(img, mask), "opencv"
