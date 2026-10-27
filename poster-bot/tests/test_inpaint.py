import numpy as np

from postergen.inpaint import _windows, halo_mask


def covered(box, W, H):
    x0, y0, x1, y1 = box
    done = np.zeros((H, W), bool)
    for ax0, ay0, ax1, ay1, core in _windows(x0, y0, x1, y1, W, H):
        assert ax1 - ax0 <= 960 and ay1 - ay0 <= 960          # окно не ужимается сильнее ~1.9 раза
        done[ay0:ay1, ax0:ax1] |= core
    return done[y0:y1, x0:x1].all()


def test_windows_cover_whole_title():
    W, H = 900, 1280
    assert covered((100, 80, 860, 270), W, H)          # широкий логотип — несколько окон
    assert covered((300, 500, 600, 560), W, H)         # маленький — одно окно
    assert covered((30, 150, 200, 1200), W, H)         # высокий столбик вдоль края
    assert covered((0, 0, 900, 400), W, H)             # во всю ширину
    assert covered((0, 100, 900, 1250), W, H)          # почти весь постер — окна в обе стороны
    assert covered((10, 10, 1900, 1100), 1920, 1200)   # широкий кадр


def test_halo_catches_drop_shadow():
    img = np.full((300, 600, 3), 120, np.uint8)
    mask = np.zeros((300, 600), np.uint8)
    mask[100:160, 100:500] = 255                       # буквы
    img[108:172, 108:510] = 40                         # тень сдвинута вниз-вправо
    img[100:160, 100:500] = 250
    halo = halo_mask(img, mask, 60) > 0
    shadow = np.zeros_like(halo)
    shadow[160:172, 108:510] = True
    assert halo[shadow].mean() > 0.9                   # тень попала в маску
    assert not halo[250:, :].any() and not halo[:, 560:].any()   # а фон вдали — нет
