"""Клиент Shikimori: информация об аниме, списки аниме и оригинальные постеры."""
from __future__ import annotations

import collections
import io
import os
import re
import socket
import threading
import time
from dataclasses import dataclass, field

import requests
from PIL import Image

# Shikimori несколько раз переезжал между доменами, поэтому пробуем по очереди.
# Свой домен можно задать переменной окружения SHIKIMORI_URL.
DEFAULT_DOMAINS = ("https://shikimori.io", "https://shikimori.one", "https://shiki.one", "https://shikimori.me")
USER_AGENT = os.environ.get("SHIKIMORI_USER_AGENT", "PosterBot/1.0")

FIELDS_BASE = """
    id name russian english japanese synonyms kind status episodes url
    airedOn { year }
    poster { originalUrl mainUrl }
"""
FIELDS_EXTRA = "    genres { name russian kind }\n"     # жанры — для выбора шрифта
FIELDS = FIELDS_BASE + FIELDS_EXTRA
_extra_ok = True        # если сервер не знает поле genres — дальше спрашиваем без него


def _one_query(fields: str) -> str:
    return "query($ids: String!) { animes(ids: $ids, limit: 1) {" + fields + "} }"


GRAPHQL_QUERY = _one_query(FIELDS)

ORDERS = ("popularity", "ranked", "aired_on", "id", "name")
_SAFE_ARG = re.compile(r"^[a-z0-9_,!\-]+$")


class ShikimoriError(RuntimeError):
    """Окончательная ошибка (аниме нет, постера нет) — повторять бессмысленно."""


class ShikimoriUnavailable(ShikimoriError):
    """Временная ошибка (сеть, сайт не отвечает) — можно повторить позже."""


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
    genres: list[str] = field(default_factory=list)   # английские названия жанров: Action, Romance…

    @property
    def has_poster(self) -> bool:
        return bool(self.poster_url) and "missing" not in self.poster_url


def parse_anime_id(value: str | int) -> int:
    """'13759', '13759-sakura-sou...', 'z13759', ссылка на shikimori -> 13759."""
    s = str(value).strip()
    m = re.search(r"/animes/[a-z]*(\d+)", s)
    if not m:
        m = re.match(r"^[a-z]*(\d+)", s, re.I)
    if not m:
        raise ValueError(f"Не понял ID аниме: {value!r}")
    return int(m.group(1))


# ------------------------------------------------------------------ запросы
class RateLimiter:
    """Не чаще per_sec запросов в секунду и per_min в минуту (у Shikimori лимит 5/с и 90/мин)."""

    def __init__(self, per_sec: float = 4.0, per_min: int = 80):
        self.interval = 1.0 / per_sec
        self.per_min = per_min
        self.times: collections.deque[float] = collections.deque()
        self.lock = threading.Lock()

    def wait(self):
        with self.lock:
            now = time.monotonic()
            while self.times and now - self.times[0] > 60:
                self.times.popleft()
            delay = 0.0
            if self.times:
                delay = max(delay, self.times[-1] + self.interval - now)
            if len(self.times) >= self.per_min:
                delay = max(delay, self.times[0] + 60 - now)
            if delay > 0:
                time.sleep(delay)
            self.times.append(time.monotonic())


LIMITER = RateLimiter()
_good_domain: str | None = None


def _domains() -> list[str]:
    custom = os.environ.get("SHIKIMORI_URL")
    doms = [custom.rstrip("/")] if custom else []
    if _good_domain and _good_domain not in doms:
        doms.append(_good_domain)          # домен, который уже ответил, — первым
    return doms + [d for d in DEFAULT_DOMAINS if d not in doms]


PROXY: str | None = os.environ.get("SHIKIMORI_PROXY") or None
TIMEOUT = (10, 40)    # (соединение, ответ), секунд


def set_proxy(url: str | None):
    """Прокси для Shikimori, например http://127.0.0.1:10809 или socks5://127.0.0.1:10808
    (если VPN включён только в браузере). Без него — системные настройки / HTTPS_PROXY."""
    global PROXY
    PROXY = url or None


class _Fail(Exception):
    """Сайт ответил, но не тем (код ошибки, ошибка GraphQL) — с объяснением."""


def describe_error(e: Exception) -> str:
    """Человеческое объяснение сетевой ошибки."""
    text = str(e)
    if isinstance(e, _Fail):
        return text
    if isinstance(e, socket.gaierror):
        return "адрес не найден (DNS) — домен не работает или заблокирован"
    if isinstance(e, requests.exceptions.ProxyError):
        m = re.search(r"(Tunnel connection failed: [^')\]]+|Unable to connect to proxy|Cannot connect to proxy)", text)
        return f"ошибка прокси {PROXY or '(системный)'}: {m.group(1) if m else 'не отвечает'}"
    if isinstance(e, requests.exceptions.SSLError):
        return "ошибка защищённого соединения (SSL) — часто так выглядит блокировка провайдером"
    if isinstance(e, (requests.exceptions.ConnectTimeout, requests.exceptions.ReadTimeout)):
        return "не отвечает (таймаут) — сайт недоступен из этой сети или заблокирован"
    if isinstance(e, requests.exceptions.ConnectionError):
        low = text.lower()
        if any(k in low for k in ("getaddrinfo", "name or service", "nameresolution", "failed to resolve",
                                  "nodename nor servname", "11001")):
            return "адрес не найден (DNS) — домен не работает или заблокирован"
        if "refused" in low or "10061" in low:
            return "соединение отклонено"
        if "reset" in low or "10054" in low or "aborted" in low:
            return "соединение сброшено — похоже на блокировку"
        return "нет соединения: " + text[:160]
    return f"{e.__class__.__name__}: {text[:160]}"


def _session(session: requests.Session | None) -> requests.Session:
    session = session or requests.Session()
    # Shikimori требует User-Agent с именем приложения; у requests по умолчанию «python-requests/…»,
    # поэтому ставим явно (setdefault тут не сработал бы)
    session.headers["User-Agent"] = USER_AGENT
    if PROXY:
        session.proxies.update({"http": PROXY, "https": PROXY})
    return session


def _request(method: str, url: str, session: requests.Session, **kw) -> requests.Response:
    """Запрос с ограничением частоты. На 429 и ошибки сервера (5xx) — ждём и пробуем снова."""
    r = None
    for attempt in range(4):
        LIMITER.wait()
        r = session.request(method, url, timeout=TIMEOUT, **kw)
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(2.0 * (attempt + 1))
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


def _info_from_gql(a: dict, domain: str) -> AnimeInfo:
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
        genres=_genres(a.get("genres")),
    )


def _genres(items) -> list[str]:
    out = []
    for g in items or []:
        name = g.get("name") if isinstance(g, dict) else str(g)
        if name and name not in out:
            out.append(name)
    return out


def _info_from_rest(a: dict, domain: str) -> AnimeInfo:
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
        genres=_genres(a.get("genres")),
    )


def _http_fail(r: requests.Response, what: str) -> _Fail:
    hint = {403: " (доступ запрещён — защита сайта или блокировка)", 429: " (слишком много запросов)"}
    return _Fail(f"{what}: ответ {r.status_code}{hint.get(r.status_code, '')}")


def _json(r: requests.Response, what: str):
    try:
        return r.json()
    except ValueError:
        snippet = " ".join(r.text[:120].split())
        raise _Fail(f"{what}: пришёл не JSON (возможно, страница-заглушка): {snippet!r}")


def _gql(domain: str, query: str, session: requests.Session, variables: dict | None = None) -> list:
    global _extra_ok
    if not _extra_ok:
        query = query.replace(FIELDS_EXTRA, "")
    try:
        return _gql_raw(domain, query, session, variables)
    except _Fail as e:
        # старый/новый сервер не знает поле жанров — повторяем без него (жанры не обязательны)
        if _extra_ok and FIELDS_EXTRA in query and "genres" in str(e):
            _extra_ok = False
            return _gql_raw(domain, query.replace(FIELDS_EXTRA, ""), session, variables)
        raise


def _gql_raw(domain: str, query: str, session: requests.Session, variables: dict | None = None) -> list:
    r = _request("POST", f"{domain}/api/graphql", session, json={"query": query, "variables": variables or {}})
    if r.status_code != 200:
        raise _http_fail(r, "GraphQL")
    data = _json(r, "GraphQL")
    animes = (data.get("data") or {}).get("animes")
    if data.get("errors") and not animes:
        msg = "; ".join(str(e.get("message", e)) for e in data["errors"][:2])
        raise _Fail(f"GraphQL: ошибка запроса: {msg[:200]}")
    return animes or []


# ------------------------------------------------------------------ одно аниме
def _from_graphql(domain: str, anime_id: int, session: requests.Session) -> AnimeInfo:
    items = _gql(domain, GRAPHQL_QUERY, session, {"ids": str(anime_id)})
    if not items:
        raise _Fail("GraphQL: аниме не найдено")
    return _info_from_gql(items[0], domain)


def _from_rest(domain: str, anime_id: int, session: requests.Session) -> AnimeInfo:
    r = _request("GET", f"{domain}/api/animes/{anime_id}", session)
    if r.status_code == 404:
        raise ShikimoriError(f"Аниме {anime_id} не найдено на Shikimori")
    if r.status_code != 200:
        raise _http_fail(r, "REST")
    return _info_from_rest(_json(r, "REST"), domain)


def _try_domains(getters, session):
    """Пробует домены по очереди; в ошибке — причина по каждому домену."""
    global _good_domain
    errors = []
    for domain in _domains():
        for getter in getters:
            try:
                res = getter(domain, session)
            except ShikimoriError:
                raise
            except requests.RequestException as e:
                errors.append(f"{domain}: {describe_error(e)}")
                break                 # нет связи с доменом — второй способ на нём тоже не сработает
            except (_Fail, ValueError, KeyError, TypeError) as e:
                errors.append(f"{domain}: {describe_error(e)}")
                continue
            _good_domain = domain
            return res
    raise ShikimoriUnavailable("Shikimori недоступен:\n  " + "\n  ".join(errors or ["нет ответа"]))


def fetch_anime(value: str | int, session: requests.Session | None = None) -> AnimeInfo:
    anime_id = parse_anime_id(value)
    session = _session(session)
    return _try_domains([lambda d, s: _from_graphql(d, anime_id, s),
                         lambda d, s: _from_rest(d, anime_id, s)], session)


# ------------------------------------------------------------------ списки
def list_animes(page: int, limit: int = 50, order: str = "popularity", kind: str | None = None,
                status: str | None = None, season: str | None = None, score: int | None = None,
                session: requests.Session | None = None) -> list[AnimeInfo]:
    """Страница списка аниме (без 18+). Пустой список — страницы закончились."""
    if order not in ORDERS:
        raise ValueError(f"order: одно из {', '.join(ORDERS)}")
    for name, v in (("kind", kind), ("status", status), ("season", season)):
        if v and not _SAFE_ARG.match(v):
            raise ValueError(f"Странное значение {name}: {v!r}")
    limit = max(1, min(50, int(limit)))
    session = _session(session)

    args = [f"page: {int(page)}", f"limit: {limit}", f"order: {order}", "censored: true"]
    params = {"page": int(page), "limit": limit, "order": order, "censored": "true"}
    for name, v in (("kind", kind), ("status", status), ("season", season)):
        if v:
            args.append(f'{name}: "{v}"')
            params[name] = v
    if score:
        args.append(f"score: {int(score)}")
        params["score"] = int(score)
    query = "{ animes(" + ", ".join(args) + ") {" + FIELDS + "} }"

    def via_gql(domain, s):
        return [_info_from_gql(a, domain) for a in _gql(domain, query, s)]

    def via_rest(domain, s):
        r = _request("GET", f"{domain}/api/animes", s, params=params)
        if r.status_code != 200:
            raise _http_fail(r, "REST")
        return [_info_from_rest(a, domain) for a in _json(r, "REST")]

    return _try_domains([via_gql, via_rest], session)


# ------------------------------------------------------------------ постеры
def download_poster_bytes(info: AnimeInfo, session: requests.Session | None = None) -> bytes:
    if not info.has_poster:
        raise ShikimoriError(f"У аниме {info.id} нет постера на Shikimori")
    session = _session(session)
    r = _request("GET", info.poster_url, session)
    if r.status_code in (403, 404, 410):
        raise ShikimoriError(f"Постер не найден ({r.status_code}): {info.poster_url}")
    if r.status_code != 200:
        raise ShikimoriUnavailable(f"Постер не скачался ({r.status_code}): {info.poster_url}")
    return r.content


def download_poster(info: AnimeInfo, session: requests.Session | None = None) -> Image.Image:
    return Image.open(io.BytesIO(download_poster_bytes(info, session))).convert("RGB")
