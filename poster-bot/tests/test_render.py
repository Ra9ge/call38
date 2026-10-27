import numpy as np
from PIL import Image

from postergen.pipeline import Options, make_poster
from postergen.shikimori import AnimeInfo


def test_overlay_on_textless_poster():
    # градиент без текста -> режим наложения
    h, w = 640, 450
    y = np.linspace(0, 1, h)[:, None, None]
    arr = (np.array([90, 140, 220]) * (1 - y) + np.array([250, 200, 160]) * y).astype(np.uint8)
    arr = np.repeat(arr, w, axis=1)
    info = AnimeInfo(1, "Hotel Inhumans Season 2", "Отель для нелюдей 2", "Hotel Inhumans Season 2")
    out, rep = make_poster(info, Image.fromarray(arr), Options(logo=None, inpaint="opencv", badge="НОВАЯ ОЗВУЧКА"))
    assert rep.mode == "overlay"
    assert rep.label == "ВТОРОЙ СЕЗОН"
    assert out.width >= 900
    # внизу появился белый текст
    bottom = np.asarray(out)[int(out.height * 0.7):]
    assert (bottom.min(2) > 235).mean() > 0.005
