import pytest
import requests

from postergen import shikimori
from postergen.shikimori import fetch_anime, list_animes, parse_anime_id


@pytest.fixture(autouse=True)
def fresh_domain():
    shikimori._good_domain = None     # скрипт запоминает домен, который ответил; тесты — независимы
    yield
    shikimori._good_domain = None


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


def test_list_graphql_query(monkeypatch):
    sent = []

    def fake_request(self, method, url, timeout=None, **kw):
        sent.append((method, url, kw))
        return FakeResp(200, {"data": {"animes": [{"id": "5", "name": "A", "russian": "А", "kind": "tv",
                                                   "poster": {"originalUrl": "/p/5.jpg"}}]}})

    monkeypatch.setattr(requests.Session, "request", fake_request)
    items = list_animes(3, kind="tv,movie", status="released", score=7)
    q = sent[0][2]["json"]["query"]
    assert "page: 3" in q and "limit: 50" in q and "order: popularity" in q and "censored: true" in q
    assert 'kind: "tv,movie"' in q and 'status: "released"' in q and "score: 7" in q
    assert items[0].id == 5 and items[0].poster_url == "https://shikimori.io/p/5.jpg"


def test_list_rest_fallback_and_bad_args(monkeypatch):
    def fake_request(self, method, url, timeout=None, **kw):
        if url.endswith("/api/graphql"):
            return FakeResp(200, {"errors": [{"message": "boom"}], "data": None})
        assert kw["params"]["censored"] == "true" and kw["params"]["kind"] == "tv"
        return FakeResp(200, [{"id": 9, "name": "B", "russian": "Б", "image": {"original": "/o/9.jpg"}}])

    monkeypatch.setattr(requests.Session, "request", fake_request)
    assert [a.id for a in list_animes(1, kind="tv")] == [9]
    with pytest.raises(ValueError):
        list_animes(1, kind='tv") { evil }')
