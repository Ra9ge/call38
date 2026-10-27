"""Клиент Shikimori: информация об аниме и оригинальный постер."""
from __future__ import annotations

import io
import os
import re
import time
from dataclasses import dataclass, field

import requests
from PIL import Image

# Shikimori несколько раз переезжал между доменами, поэтому пробуем по очереди.
# Свой домен можно задать переменной окружения SHIKIMORI_URL.
DEFAULT_DOMAINS = ("https://shikimori.io", "https://shikimori.one", "https://shiki.one", "https://shikimori.me")
USER_AGENT = os.environ.get("SHIKIMORI_USER_AGENT", "PosterBot/1.0")

GRAPHQL_QUERY = """
query($ids: String!) {
  animes(ids: $ids, limit: 1) {
    id name russian english japanese synonyms kind status episodes season url
    airedOn { year }
    poster { originalUrl mainUrl }
  }
}
"""


class ShikimoriError(RuntimeError):
    pass


@dataclass
class AnimeInfo:
    id: int
    name: str                      # ромадзи: "Sakurasou no Pet na Kanojo"
    russian: str | None            # "Кошечка из Сакурасо"
    english: str | None = None     # "The Pet Girl of Sakurasou"
    japanese: str | None = None    # "さくら荘のペットな彼女"
    synonyms: list[str] = field(default_factory=list)
    kind: str | None = None        # tv / movie / ova / ona / special ...
    year: int | None = None
    poster_url: str | None = None
    url: str | None = None


def parse_anime_id(value: str | int) -> int:
    """'13759', '13759-sakura-sou...', 'z13759', ссылка на shikimori -> 13759."""
    s = str(value).strip()
    m = re.search(r"/animes/[a-z]*(\d+)", s)
    if not m:
        m = re.match(r"^[a-z]*(\d+)", s, re.I)
    if not m:
        raise ValueError(f"Не понял ID аниме: {value!r}")
    return int(m.group(1))


def _domains() -> list[str]:
    custom = os.environ.get("SHIKIMORI_URL")
    doms = [custom.rstrip("/")] if custom else []
    return doms + [d for d in DEFAULT_DOMAINS if d not in doms]


def _request(method: str, url: str, session: requests.Session, **kw) -> requests.Response:
    # Shikimori ограничивает 5 запросов/сек; на 429 ждём и пробуем снова.
    for attempt in range(4):
        r = session.request(method, url, timeout=20, **kw)
        if r.status_code == 429:
            time.sleep(1.5 * (attempt + 1))
            continue
        return r
    return r


def _abs(url: str | None, domain: str) -> str | None:
    if not url:
        return None
    if url.startswith("//"):
        return "https:" + url
    if url.startswith("/"):
        return domain + url
    return url


def _first(v):
    if isinstance(v, list):
        v = [x for x in v if x]
        return v[0] if v else None
    return v or None


def _from_graphql(domain: str, anime_id: int, session: requests.Session) -> AnimeInfo | None:
    r = _request("POST", f"{domain}/api/graphql", session,
                 json={"query": GRAPHQL_QUERY, "variables": {"ids": str(anime_id)}})
    if r.status_code != 200:
        return None
    data = r.json()
    items = (data.get("data") or {}).get("animes") or []
    if not items:
        return None
    a = items[0]
    poster = a.get("poster") or {}
    return AnimeInfo(
        id=int(a["id"]),
        name=a.get("name") or "",
        russian=a.get("russian") or None,
        english=_first(a.get("english")),
        japanese=_first(a.get("japanese")),
        synonyms=a.get("synonyms") or [],
        kind=a.get("kind"),
        year=(a.get("airedOn") or {}).get("year"),
        poster_url=_abs(poster.get("originalUrl") or poster.get("mainUrl"), domain),
        url=_abs(a.get("url"), domain),
    )


def _from_rest(domain: str, anime_id: int, session: requests.Session) -> AnimeInfo | None:
    r = _request("GET", f"{domain}/api/animes/{anime_id}", session)
    if r.status_code == 404:
        raise ShikimoriError(f"Аниме {anime_id} не найдено на Shikimori")
    if r.status_code != 200:
        return None
    a = r.json()
    image = a.get("image") or {}
    aired = a.get("aired_on") or ""
    return AnimeInfo(
        id=int(a["id"]),
        name=a.get("name") or "",
        russian=a.get("russian") or None,
        english=_first(a.get("english")),
        japanese=_first(a.get("japanese")),
        synonyms=a.get("synonyms") or [],
        kind=a.get("kind"),
        year=int(aired[:4]) if aired[:4].isdigit() else None,
        poster_url=_abs(image.get("original"), domain),
        url=_abs(a.get("url"), domain),
    )


def fetch_anime(value: str | int, session: requests.Session | None = None) -> AnimeInfo:
    anime_id = parse_anime_id(value)
    session = session or requests.Session()
    session.headers.setdefault("User-Agent", USER_AGENT)
    errors = []
    for domain in _domains():
        for getter in (_from_graphql, _from_rest):
            try:
                info = getter(domain, anime_id, session)
            except ShikimoriError:
                raise
            except (requests.RequestException, ValueError, KeyError) as e:
                errors.append(f"{domain} ({getter.__name__}): {e}")
                continue
            if info:
                return info
    raise ShikimoriError("Не удалось получить данные с Shikimori:\n  " + "\n  ".join(errors or ["нет ответа"]))


def download_poster(info: AnimeInfo, session: requests.Session | None = None) -> Image.Image:
    if not info.poster_url or "missing" in info.poster_url:
        raise ShikimoriError(f"У аниме {info.id} нет постера на Shikimori")
    session = session or requests.Session()
    session.headers.setdefault("User-Agent", USER_AGENT)
    r = _request("GET", info.poster_url, session)
    if r.status_code != 200:
        raise ShikimoriError(f"Постер не скачался ({r.status_code}): {info.poster_url}")
    return Image.open(io.BytesIO(r.content)).convert("RGB")
