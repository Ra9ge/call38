"""Настоящие HTTP-запросы к локальному «поддельному Shikimori» с тем же API."""
import io
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest
from PIL import Image

from postergen import shikimori
from postergen.batch import BatchConfig, run_batch
from postergen.pipeline import Options

ANIME = {
    13759: {"id": "13759", "name": "Sakurasou no Pet na Kanojo", "russian": "Кошечка из Сакурасо",
            "english": "The Pet Girl of Sakurasou", "kind": "tv", "url": "/animes/z13759",
            "poster": {"originalUrl": "/uploads/poster/animes/13759/a.jpeg"}},
    48316: {"id": "48316", "name": "Kage no Jitsuryokusha ni Naritakute!", "russian": "Восхождение в тени!",
            "english": "The Eminence in Shadow", "kind": "tv", "url": "/animes/48316",
            "poster": {"originalUrl": "/uploads/poster/animes/48316/b.jpeg"}},
    5: {"id": "5", "name": "No Poster", "russian": "Без постера", "kind": "tv", "poster": None},
}


def jpeg(seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    a, b = rng.integers(30, 220, 3), rng.integers(30, 220, 3)
    y = np.linspace(0, 1, 500)[:, None, None]
    arr = np.repeat((a * (1 - y) + b * y).astype(np.uint8), 350, axis=1)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "JPEG")
    return buf.getvalue()


class Handler(BaseHTTPRequestHandler):
    seen = []

    def log_message(self, *a):
        pass

    def send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        Handler.seen.append(("GET", self.path, self.headers.get("User-Agent")))
        m = re.match(r"/uploads/poster/animes/(\d+)/", self.path)
        if m:
            return self.send(200, jpeg(int(m.group(1))), "image/jpeg")
        m = re.match(r"/api/animes/(\d+)$", self.path)
        if m and int(m.group(1)) in ANIME:
            a = ANIME[int(m.group(1))]
            return self.send(200, {"id": int(a["id"]), "name": a["name"], "russian": a["russian"]})
        self.send(404, {"message": "not found"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Handler.seen.append(("POST", body["query"], self.headers.get("User-Agent")))
        ids = (body.get("variables") or {}).get("ids")
        if ids:
            return self.send(200, {"data": {"animes": [ANIME[int(i)] for i in ids.split(",") if int(i) in ANIME]}})
        page = int(re.search(r"page: (\d+)", body["query"]).group(1))
        items = list(ANIME.values()) if page == 1 else []
        self.send(200, {"data": {"animes": items}})


@pytest.fixture
def server(monkeypatch):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    monkeypatch.setenv("SHIKIMORI_URL", url)
    monkeypatch.setattr(shikimori, "DEFAULT_DOMAINS", ())
    monkeypatch.setattr(shikimori, "_good_domain", None)
    monkeypatch.setattr(shikimori, "PROXY", None)
    Handler.seen.clear()
    yield url
    srv.shutdown()


def test_fetch_and_download_over_http(server):
    info = shikimori.fetch_anime("https://shikimori.io/animes/z13759-sakura-sou-no-pet-na-kanojo")
    assert info.russian == "Кошечка из Сакурасо" and info.poster_url.startswith(server)
    assert len(shikimori.download_poster_bytes(info)) > 1000
    assert all(ua == shikimori.USER_AGENT for _, _, ua in Handler.seen)     # User-Agent в каждом запросе


def test_batch_over_http(server, tmp_path, monkeypatch):
    from postergen import batch
    monkeypatch.setattr(batch, "lama_ready", lambda allow_download=True: False)
    cfg = BatchConfig(out_dir=tmp_path, allow_opencv=True, poster=Options(logo=None, inpaint="opencv"))
    assert run_batch(cfg) == 0
    states = {json.loads(ln)["id"]: json.loads(ln)["status"]
              for ln in (tmp_path / "_state.jsonl").read_text(encoding="utf-8").splitlines()}
    assert states == {13759: "ok", 48316: "ok", 5: "no_poster"}
    query = next(q for m, q, _ in Handler.seen if m == "POST")
    assert "censored: true" in query and 'kind: "tv,movie,ova,ona"' in query


def test_unreachable_reason_is_explained(monkeypatch):
    monkeypatch.setenv("SHIKIMORI_URL", "http://127.0.0.1:9")      # порт, где никто не слушает
    monkeypatch.setattr(shikimori, "DEFAULT_DOMAINS", ())
    monkeypatch.setattr(shikimori, "_good_domain", None)
    with pytest.raises(shikimori.ShikimoriUnavailable) as e:
        shikimori.list_animes(1)
    assert "127.0.0.1:9" in str(e.value) and "соединение отклонено" in str(e.value)
