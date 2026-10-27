import pytest
import requests

from postergen.shikimori import fetch_anime, parse_anime_id


@pytest.mark.parametrize("value", ["13759", "13759-sakura-sou-no-pet-na-kanojo", "z13759",
                                   "https://shikimori.one/animes/z13759-sakura-sou-no-pet-na-kanojo"])
def test_parse_id(value):
    assert parse_anime_id(value) == 13759


class FakeResp:
    def __init__(self, status, data):
        self.status_code = status
        self._data = data

    def json(self):
        return self._data


def test_graphql(monkeypatch):
    calls = []

    def fake_request(self, method, url, timeout=None, **kw):
        calls.append((method, url))
        if url.startswith("https://shikimori.io"):
            raise requests.ConnectionError("blocked")
        return FakeResp(200, {"data": {"animes": [{
            "id": "13759", "name": "Sakurasou no Pet na Kanojo", "russian": "Кошечка из Сакурасо",
            "english": "The Pet Girl of Sakurasou", "japanese": "さくら荘のペットな彼女", "kind": "tv",
            "airedOn": {"year": 2012}, "url": "/animes/z13759",
            "poster": {"originalUrl": "https://shiki.one/uploads/poster/animes/13759/x.jpeg"}}]}})

    monkeypatch.setattr(requests.Session, "request", fake_request)
    info = fetch_anime("13759")
    assert info.russian == "Кошечка из Сакурасо"
    assert info.poster_url.endswith("x.jpeg")
    assert info.url == "https://shikimori.one/animes/z13759"
    assert calls[0][1].startswith("https://shikimori.io")  # сначала основной домен, потом запасные


def test_rest_fallback(monkeypatch):
    def fake_request(self, method, url, timeout=None, **kw):
        if url.endswith("/api/graphql"):
            return FakeResp(500, {})
        return FakeResp(200, {"id": 13759, "name": "Sakurasou no Pet na Kanojo", "russian": "Кошечка из Сакурасо",
                              "english": ["The Pet Girl of Sakurasou"], "japanese": ["さくら荘のペットな彼女"],
                              "kind": "tv", "aired_on": "2012-10-09",
                              "image": {"original": "/system/animes/original/13759.jpg"}})

    monkeypatch.setattr(requests.Session, "request", fake_request)
    info = fetch_anime(13759)
    assert info.english == "The Pet Girl of Sakurasou"
    assert info.year == 2012
    assert info.poster_url == "https://shikimori.io/system/animes/original/13759.jpg"
