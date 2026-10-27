import io
import json

import numpy as np
import pytest
import requests
from PIL import Image

from postergen import batch
from postergen.batch import BatchConfig, run_batch
from postergen.pipeline import Options
from postergen.shikimori import AnimeInfo


def poster_bytes(seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    h, w = 640, 450
    top, bottom = rng.integers(40, 220, 3), rng.integers(40, 220, 3)
    y = np.linspace(0, 1, h)[:, None, None]
    arr = np.repeat((top * (1 - y) + bottom * y).astype(np.uint8), w, axis=1)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "JPEG")
    return buf.getvalue()


PAGES = {
    1: [AnimeInfo(1, "Hotel Inhumans", "Отель для нелюдей", poster_url="https://x/1.jpg"),
        AnimeInfo(2, "No Poster", "Без постера", poster_url="https://x/missing_original.jpg")],
    2: [AnimeInfo(3, "Kage no Jitsuryokusha", "Восхождение в тени!", poster_url="https://x/3.jpg"),
        AnimeInfo(1, "Hotel Inhumans", "Отель для нелюдей", poster_url="https://x/1.jpg")],  # повтор
}


@pytest.fixture
def fake_shikimori(monkeypatch):
    calls = {"list": 0, "download": []}

    def fake_list(page, *a, **k):
        calls["list"] += 1
        return PAGES.get(page, [])

    def fake_download(info, session=None):
        calls["download"].append(info.id)
        return poster_bytes(info.id)

    monkeypatch.setattr(batch, "list_animes", fake_list)
    monkeypatch.setattr(batch, "download_poster_bytes", fake_download)
    monkeypatch.setattr(batch, "lama_ready", lambda allow_download=True: False)
    monkeypatch.setattr(batch, "RETRY_DELAYS", (0, 0))
    return calls


def config(tmp_path, **kw):
    return BatchConfig(out_dir=tmp_path, allow_opencv=True, poster=Options(logo=None, inpaint="opencv"), **kw)


def states(tmp_path):
    out = {}
    for line in (tmp_path / "_state.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        out[r["id"]] = r["status"]
    return out


def test_full_run_and_resume(tmp_path, fake_shikimori):
    assert run_batch(config(tmp_path)) == 0
    assert states(tmp_path) == {1: "ok", 2: "no_poster", 3: "ok"}
    assert len(list((tmp_path / "posters").glob("*.jpg"))) == 2
    assert (tmp_path / "_originals" / "1.jpg").exists() and (tmp_path / "_originals" / "1.json").exists()
    report = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert "Восхождение в тени!" in report and "Готово: 2" in report
    assert sorted(fake_shikimori["download"]) == [1, 3]       # повтор в списке не скачан дважды

    # второй запуск: всё уже готово — ничего не скачивает и не переделывает
    fake_shikimori["download"].clear()
    lines_before = (tmp_path / "_state.jsonl").read_text(encoding="utf-8").count("\n")
    assert run_batch(config(tmp_path)) == 0
    assert fake_shikimori["download"] == []
    assert (tmp_path / "_state.jsonl").read_text(encoding="utf-8").count("\n") == lines_before


def test_limit(tmp_path, fake_shikimori):
    assert run_batch(config(tmp_path, limit=1)) == 0
    assert states(tmp_path) == {1: "ok"}          # дальше по списку не идёт


def test_offline_redo_uses_cached_originals(tmp_path, fake_shikimori, monkeypatch):
    run_batch(config(tmp_path))

    def no_network(*a, **k):
        raise AssertionError("в офлайн-режиме сеть не нужна")

    monkeypatch.setattr(batch, "list_animes", no_network)
    monkeypatch.setattr(batch, "download_poster_bytes", no_network)
    assert run_batch(config(tmp_path, offline=True, redo=True)) == 0
    assert states(tmp_path)[3] == "ok"


def test_network_errors_are_retried(tmp_path, fake_shikimori, monkeypatch):
    attempts = {"n": 0}

    def flaky(info, session=None):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise requests.ConnectionError("нет интернета")
        return poster_bytes(info.id)

    monkeypatch.setattr(batch, "download_poster_bytes", flaky)
    assert run_batch(config(tmp_path, limit=1)) == 0
    assert states(tmp_path)[1] == "ok" and attempts["n"] == 2


def test_broken_poster_is_logged_and_skipped(tmp_path, fake_shikimori, monkeypatch):
    monkeypatch.setattr(batch, "download_poster_bytes", lambda info, session=None: b"not an image")
    assert run_batch(config(tmp_path)) == 0
    st = states(tmp_path)
    assert st[1] == "error" and st[3] == "error"          # повторятся при следующем запуске
    assert "UnidentifiedImageError" in (tmp_path / "batch.log").read_text(encoding="utf-8")
    assert not (tmp_path / "_originals" / "1.jpg").exists()   # битый файл удалён, скачается заново


def test_stops_without_lama_unless_allowed(tmp_path, fake_shikimori):
    cfg = BatchConfig(out_dir=tmp_path, poster=Options(logo=None))
    assert run_batch(cfg) == 2
    assert not (tmp_path / "_state.jsonl").exists()


def test_list_waits_for_network(tmp_path, fake_shikimori, monkeypatch):
    from postergen.shikimori import ShikimoriUnavailable
    fails = {"n": 0}
    real = batch.list_animes

    def flaky_list(page, *a, **k):
        if page == 1 and fails["n"] < 9:          # больше, чем попыток в RETRY_DELAYS
            fails["n"] += 1
            raise ShikimoriUnavailable("сайт не отвечает")
        return real(page, *a, **k)

    monkeypatch.setattr(batch, "list_animes", flaky_list)
    assert run_batch(config(tmp_path, limit=1)) == 0
    assert states(tmp_path) == {1: "ok"}
