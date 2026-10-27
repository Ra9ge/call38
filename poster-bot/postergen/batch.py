"""Ночной режим: постеры для всех аниме подряд.

- список берётся с Shikimori страницами (сначала популярные, без 18+);
- готовые постеры запоминаются в _state.jsonl — после остановки продолжает с того же места;
- оригиналы сохраняются в _originals (можно перерисовать всё без интернета: --offline --redo);
- постеры скачиваются заранее в фоне, пока рисуется предыдущий;
- обрыв интернета не останавливает работу: повторы с паузами;
- компьютер не уходит в сон (Windows), клик по окну консоли не «замораживает» скрипт;
- Ctrl+C — аккуратная остановка после текущего постера;
- index.html — галерея «было / стало», чтобы утром быстро всё просмотреть.
"""
from __future__ import annotations

import html
import json
import os
import queue
import shutil
import signal
import threading
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import requests
from PIL import Image

from . import pipeline
from .inpaint import lama_ready
from .pipeline import Options, make_poster
from .shikimori import (AnimeInfo, ShikimoriError, ShikimoriUnavailable, describe_error, download_poster_bytes,
                        fetch_anime, list_animes, parse_anime_id)
from .titles import slug

RETRY_DELAYS = (5, 15, 30, 60, 120, 300, 600)   # паузы между повторами; дальше — по 10 минут
MIN_FREE_BYTES = 300 * 1024 * 1024
REPORT_EVERY = 25


@dataclass
class BatchConfig:
    out_dir: Path
    kind: str | None = "tv,movie,ova,ona"   # None — все типы
    status: str | None = None               # released / ongoing / anons (через запятую)
    season: str | None = None               # «2020_2026», «2015», «summer_2024»
    min_score: int | None = None
    order: str = "popularity"
    ids: list[str] | None = None            # свой список вместо всего Shikimori
    offline: bool = False                   # только уже скачанные оригиналы, без интернета
    limit: int | None = None                # сколько новых постеров сделать за запуск
    max_hours: float | None = None          # остановиться через N часов
    redo: bool = False                      # переделать уже готовые
    fmt: str = "jpg"
    allow_opencv: bool = False              # работать, даже если нейросеть LaMa не скачалась
    poster: Options = field(default_factory=Options)


# ---------------------------------------------------------------- вывод
class Log:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()

    def __call__(self, msg: str = "", console: bool = True):
        line = f"{datetime.now():%H:%M:%S} {msg}" if msg else ""
        with self.lock:
            if console:
                try:
                    print(line, flush=True)
                except UnicodeEncodeError:
                    print(line.encode("ascii", "replace").decode(), flush=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")


def fmt_time(sec: float) -> str:
    sec = int(max(0, sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


# ---------------------------------------------------------------- состояние
class State:
    """Что уже сделано: одна строка JSON на постер. Последняя запись по ID — главная."""

    FINAL = ("ok", "no_poster", "not_found")

    def __init__(self, path: Path):
        self.path = path
        self.records: dict[int, dict] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    rec = json.loads(line)
                    self.records[int(rec["id"])] = rec
                except (ValueError, KeyError, TypeError):
                    continue          # недописанная строка (компьютер выключили) — пропускаем

    def finished(self, anime_id: int) -> bool:
        rec = self.records.get(anime_id)
        return rec is not None and rec.get("status") in self.FINAL

    def add(self, rec: dict):
        rec["time"] = datetime.now().isoformat(timespec="seconds")
        self.records[int(rec["id"])] = rec
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()


# ---------------------------------------------------------------- Windows: не спать, не замирать
def keep_awake(on: bool):
    if os.name != "nt":
        return
    try:
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))
    except (AttributeError, OSError):
        pass


def disable_quick_edit():
    """Иначе случайный клик в окне консоли ставит скрипт на паузу до нажатия клавиши."""
    if os.name != "nt":
        return
    try:
        import ctypes
        k = ctypes.windll.kernel32
        h = k.GetStdHandle(-10)
        mode = ctypes.c_uint32()
        if k.GetConsoleMode(h, ctypes.byref(mode)):
            k.SetConsoleMode(h, (mode.value & ~0x0040) | 0x0080)
    except (AttributeError, OSError):
        pass


# ---------------------------------------------------------------- скачивание
def with_retries(fn, what: str, log: Log, stop: threading.Event, forever: bool = False):
    """Повторяет при временных ошибках. Пропал интернет — ждёт сколько угодно (паузы до 10 минут);
    сервер отвечает ошибкой — несколько попыток (~20 минут), если не forever. Окончательные — сразу."""
    attempt = 0
    while True:
        try:
            return fn()
        except ShikimoriUnavailable as e:
            err, network = str(e), False
        except ShikimoriError:
            raise
        except requests.RequestException as e:
            err, network = describe_error(e), True
        attempt += 1
        if stop.is_set():
            raise ShikimoriUnavailable(f"{what}: остановлено")
        if not (forever or network) and attempt > len(RETRY_DELAYS):
            raise ShikimoriUnavailable(f"{what}: {err}")
        delay = RETRY_DELAYS[min(attempt, len(RETRY_DELAYS)) - 1]
        if attempt == 1:     # в первый раз — причина целиком (по каждому адресу), дальше коротко
            log(f"  нет связи ({what}), повтор через {delay} с. Причина: {err}")
            if "Shikimori недоступен" in err:
                log("  Проверь, открывается ли https://shikimori.io в браузере. Если там работает только "
                    "через VPN — включи VPN для всей системы или запусти с --proxy (см. check.bat).")
        else:
            log(f"  всё ещё нет связи ({what}), попытка {attempt + 1}, повтор через {delay} с")
        if stop.wait(delay):
            raise ShikimoriUnavailable(f"{what}: остановлено")


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".gif")


def _suffix(url: str | None) -> str:
    ext = Path((url or "").split("?")[0]).suffix.lower()
    return ext if ext in IMAGE_EXTS else ".jpg"


class Producer(threading.Thread):
    """Фоном перебирает список аниме и заранее скачивает постеры в очередь."""

    def __init__(self, cfg: BatchConfig, state: State, q: queue.Queue, stop: threading.Event, log: Log):
        super().__init__(daemon=True)
        self.cfg, self.state, self.q, self.stop, self.log = cfg, state, q, stop, log
        self.originals = cfg.out_dir / "_originals"
        self.session = requests.Session()
        self.reason = "список закончился"

    def cached(self, anime_id: int) -> Path | None:
        for ext in IMAGE_EXTS:
            p = self.originals / f"{anime_id}{ext}"
            if p.exists():
                return p
        return None

    def put(self, item) -> bool:
        while not self.stop.is_set():
            try:
                self.q.put(item, timeout=1)
                return True
            except queue.Full:
                continue
        return False

    def infos(self):
        cfg = self.cfg
        if cfg.offline:
            for meta in sorted(self.originals.glob("*.json"), key=lambda p: p.stat().st_mtime):
                try:
                    yield AnimeInfo(**json.loads(meta.read_text(encoding="utf-8")))
                except (ValueError, TypeError) as e:
                    self.log(f"  пропускаю {meta.name}: {e}")
            return
        if cfg.ids:
            for v in cfg.ids:
                try:
                    anime_id = parse_anime_id(v)
                except ValueError as e:
                    self.log(f"  {e}")
                    continue
                if self.state.finished(anime_id) and not cfg.redo:
                    continue
                cached = self.originals / f"{anime_id}.json"
                if cached.exists():
                    yield AnimeInfo(**json.loads(cached.read_text(encoding="utf-8")))
                    continue
                try:
                    yield with_retries(lambda: fetch_anime(anime_id, self.session), f"аниме {anime_id}",
                                       self.log, self.stop)
                except ShikimoriUnavailable as e:
                    self.put(("error", AnimeInfo(anime_id, "", None), str(e)))
                except ShikimoriError as e:
                    self.put(("not_found", AnimeInfo(anime_id, "", None), str(e)))
            return
        page = 1
        while not self.stop.is_set():
            items = with_retries(
                lambda: list_animes(page, 50, cfg.order, cfg.kind, cfg.status, cfg.season, cfg.min_score,
                                    self.session),
                f"список, страница {page}", self.log, self.stop, forever=True)
            if not items:
                return
            yield from items
            page += 1

    def run(self):
        cfg = self.cfg
        self.originals.mkdir(parents=True, exist_ok=True)
        seen, queued = set(), 0
        try:
            for info in self.infos():
                if self.stop.is_set():
                    break
                if info.id in seen or (self.state.finished(info.id) and not cfg.redo):
                    continue
                seen.add(info.id)
                path = self.cached(info.id)
                if path is None and cfg.offline:
                    continue
                if path is None:
                    if not info.has_poster:
                        self.put(("no_poster", info, "на Shikimori нет постера"))
                        continue
                    try:
                        data = with_retries(lambda: download_poster_bytes(info, self.session),
                                            f"постер {info.id}", self.log, self.stop)
                    except ShikimoriUnavailable as e:
                        self.put(("error", info, str(e)))
                        continue
                    except ShikimoriError as e:
                        self.put(("no_poster", info, str(e)))
                        continue
                    path = self.originals / f"{info.id}{_suffix(info.poster_url)}"
                    tmp = path.with_suffix(".part")
                    tmp.write_bytes(data)
                    tmp.replace(path)
                    (self.originals / f"{info.id}.json").write_text(
                        json.dumps(asdict(info), ensure_ascii=False), encoding="utf-8")
                if not self.put(("poster", info, path)):
                    break
                queued += 1
                if cfg.limit and queued >= cfg.limit:
                    self.reason = f"сделано {cfg.limit} (лимит)"
                    break
        except ShikimoriUnavailable as e:
            self.put(("fatal", None, f"Shikimori недоступен: {e}"))
        except Exception as e:  # чтобы ночной запуск не завис молча
            self.put(("fatal", None, f"{e.__class__.__name__}: {e}\n{traceback.format_exc()}"))
        finally:
            self.put(None)


# ---------------------------------------------------------------- сохранение и отчёт
def save_image(img: Image.Image, path: Path, fmt: str):
    tmp = path.with_name(path.name + ".part")
    if fmt == "jpg":
        img.save(tmp, "JPEG", quality=95, subsampling=0, optimize=True)
    elif fmt == "webp":
        img.save(tmp, "WEBP", quality=95, method=4)
    else:
        img.save(tmp, "PNG")
    tmp.replace(path)


def save_thumb(img: Image.Image, path: Path, width: int):
    t = img.convert("RGB")
    t.thumbnail((width, width * 2), Image.LANCZOS)
    t.save(path, "JPEG", quality=82)


MODE_RU = {"replace": "замена", "overlay": "наложение"}


def write_report(out_dir: Path, state: State, started: str):
    recs = list(state.records.values())[::-1]          # новые сверху
    ok = [r for r in recs if r.get("status") == "ok"]
    bad = [r for r in recs if r.get("status") == "error"]
    skipped = [r for r in recs if r.get("status") in ("no_poster", "not_found")]
    secs = [r.get("seconds", 0) for r in ok if r.get("seconds")]
    avg = sum(secs) / len(secs) if secs else 0
    e = html.escape

    cards = []
    for r in ok:
        i = int(r["id"])
        link = f'<a href="{e(r.get("url") or "")}" target="_blank">{i}</a>' if r.get("url") else str(i)
        cards.append(
            f'<div class="card"><a href="{e(r["file"])}" target="_blank">'
            f'<img loading="lazy" src="_thumbs/{i}.jpg" alt=""></a>'
            f'<div class="row"><img class="orig" loading="lazy" src="_thumbs/{i}-orig.jpg" alt=""'
            f' title="оригинал"><div><div class="t">{e(r.get("title") or r.get("name") or "")}</div>'
            f'<div class="m">{link} · {e(MODE_RU.get(r.get("mode"), r.get("mode") or ""))}'
            f'{" (" + e(r["logo"]) + ")" if r.get("logo") else ""}'
            f' · {r.get("seconds", "?")} с</div></div></div></div>')
    rows = "".join(f'<tr><td>{int(r["id"])}</td><td>{e(r.get("title") or r.get("name") or "")}</td>'
                   f'<td>{e((r.get("error") or "")[:300])}</td></tr>' for r in bad)
    page = f"""<!doctype html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Постеры — отчёт</title>
<style>
body{{background:#14161c;color:#e8e8ee;font:14px/1.4 system-ui,sans-serif;margin:0;padding:16px}}
h1{{font-size:20px;margin:0 0 6px}} .s{{color:#aab;margin-bottom:14px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:14px}}
.card{{background:#1e2129;border-radius:8px;padding:8px}}
.card img{{width:100%;border-radius:4px;display:block}}
.row{{display:flex;gap:8px;margin-top:8px;align-items:flex-start}}
.row img.orig{{width:44px;flex:none;opacity:.9}}
.t{{font-weight:600}} .m{{color:#99a;font-size:12px}} a{{color:#8cb4ff}}
table{{border-collapse:collapse;margin-bottom:18px}} td{{border-top:1px solid #333;padding:4px 8px;vertical-align:top}}
</style></head><body>
<h1>Постеры</h1>
<div class="s">Готово: {len(ok)} · ошибок: {len(bad)} · без постера: {len(skipped)} ·
среднее время: {avg:.1f} с · запуск: {e(started)} · обновлено: {datetime.now():%Y-%m-%d %H:%M}</div>
{f'<h2>Ошибки (повторятся при следующем запуске)</h2><table>{rows}</table>' if bad else ''}
<div class="grid">{''.join(cards)}</div>
</body></html>"""
    tmp = out_dir / "index.html.part"
    tmp.write_text(page, encoding="utf-8")
    tmp.replace(out_dir / "index.html")


# ---------------------------------------------------------------- главный цикл
def run_batch(cfg: BatchConfig) -> int:
    out = cfg.out_dir
    posters, thumbs = out / "posters", out / "_thumbs"
    for d in (out, posters, thumbs, out / "_originals"):
        d.mkdir(parents=True, exist_ok=True)
    log = Log(out / "batch.log")
    started = datetime.now().strftime("%Y-%m-%d %H:%M")
    log(f"=== Запуск {started} ===")
    where = "скачанные ранее оригиналы" if cfg.offline else (
        f"свой список ({len(cfg.ids)} шт.)" if cfg.ids else
        f"Shikimori: {cfg.kind or 'все типы'}, порядок {cfg.order}"
        + (f", статус {cfg.status}" if cfg.status else "") + (f", сезон {cfg.season}" if cfg.season else "")
        + (f", оценка от {cfg.min_score}" if cfg.min_score else ""))
    log(f"Источник: {where}. Папка: {out}")
    if cfg.limit:
        log(f"Сделать не больше {cfg.limit} новых постеров.")
    if cfg.max_hours:
        log(f"Остановиться через {cfg.max_hours:g} ч.")

    state = State(out / "_state.jsonl")
    done_before = sum(1 for r in state.records.values() if r.get("status") == "ok")
    if done_before and not cfg.redo:
        log(f"Уже готово раньше: {done_before} — их пропускаю.")

    if cfg.poster.inpaint != "opencv" and not lama_ready(allow_download=True):
        if not cfg.allow_opencv:
            log("Нейросеть LaMa недоступна — без неё под стёртым текстом остаётся размытие.")
            log("Скачай модель вручную (ссылка в README) или запусти с --allow-opencv.")
            return 2
        log("Внимание: LaMa недоступна, стираю встроенным способом (качество хуже).")

    pipeline.VERBOSE = False
    keep_awake(True)
    disable_quick_edit()
    stop = threading.Event()
    q: queue.Queue = queue.Queue(maxsize=6)
    producer = Producer(cfg, state, q, stop, log)

    def on_ctrl_c(signum, frame):
        if stop.is_set():
            raise KeyboardInterrupt
        stop.set()
        log("Ctrl+C: доделываю текущий постер и останавливаюсь (ещё раз Ctrl+C — сразу).")

    old_handler = signal.signal(signal.SIGINT, on_ctrl_c)
    t_start = time.time()
    n_ok = n_err = n_skip = 0
    times: list[float] = []
    reason = "список закончился"
    producer.start()
    try:
        while True:
            if cfg.max_hours and time.time() - t_start > cfg.max_hours * 3600:
                reason = f"прошло {cfg.max_hours:g} ч"
                break
            try:
                item = q.get(timeout=1)
            except queue.Empty:
                if stop.is_set() and not producer.is_alive():
                    reason = "остановлено"
                    break
                continue
            if item is None:
                reason = "остановлено" if stop.is_set() else producer.reason
                break
            kind, info, payload = item
            if kind == "fatal":
                log(f"Ошибка: {payload}")
                reason = "ошибка списка"
                break
            if kind in ("no_poster", "not_found"):
                state.add({"id": info.id, "status": kind, "name": info.name, "title": info.russian,
                           "error": payload})
                n_skip += 1
                continue
            if kind == "error":
                state.add({"id": info.id, "status": "error", "name": info.name, "title": info.russian,
                           "error": payload})
                log(f"[ошибка] {info.id}: {payload}")
                n_err += 1
                continue
            if stop.is_set():
                reason = "остановлено"
                break
            if shutil.disk_usage(out).free < MIN_FREE_BYTES:
                reason = "мало места на диске"
                break

            t0 = time.time()
            try:
                try:
                    with Image.open(payload) as im:
                        src = im.convert("RGB")
                except (OSError, ValueError):
                    Path(payload).unlink(missing_ok=True)   # битый файл — в следующий раз скачаем заново
                    raise
                result, rep = make_poster(info, src, cfg.poster)
                name = f"{info.id}-{slug(info.name or info.russian or 'poster')}.{cfg.fmt}"
                save_image(result, posters / name, cfg.fmt)
                save_thumb(result, thumbs / f"{info.id}.jpg", 300)
                save_thumb(src, thumbs / f"{info.id}-orig.jpg", 140)
                dt = time.time() - t0
                times.append(dt)
                n_ok += 1
                state.add({"id": info.id, "status": "ok", "name": info.name, "title": rep.title,
                           "mode": rep.mode, "logo": rep.logo, "inpaint": rep.inpaint, "seconds": round(dt, 1),
                           "file": f"posters/{name}", "url": info.url})
                recent = times[-50:]
                avg = sum(recent) / len(recent)
                elapsed = time.time() - t_start
                per_hour = n_ok / elapsed * 3600 if elapsed > 0 else 0
                tail = ""
                if cfg.limit:
                    tail = f" · осталось ~{fmt_time((cfg.limit - n_ok - n_err) * avg)}"
                log(f"[{n_ok:>5}] {info.id} {(rep.title or info.name)[:40]} — {dt:.1f} с, "
                    f"{MODE_RU.get(rep.mode, rep.mode)} · среднее {avg:.1f} с · ~{per_hour:.0f}/ч · "
                    f"прошло {fmt_time(elapsed)}{tail}")
            except KeyboardInterrupt:
                raise
            except Exception as e:
                n_err += 1
                state.add({"id": info.id, "status": "error", "name": info.name, "title": info.russian,
                           "error": f"{e.__class__.__name__}: {e}"})
                log(f"[ошибка] {info.id} {info.name}: {e.__class__.__name__}: {e}")
                log(traceback.format_exc(), console=False)
            if (n_ok + n_err) % REPORT_EVERY == 0:
                write_report(out, state, started)
    except KeyboardInterrupt:
        reason = "прервано"
    finally:
        stop.set()
        signal.signal(signal.SIGINT, old_handler)
        keep_awake(False)
        pipeline.VERBOSE = True
        write_report(out, state, started)

    elapsed = time.time() - t_start
    avg = sum(times) / len(times) if times else 0
    log(f"=== Готово ({reason}). Сделано: {n_ok}, ошибок: {n_err}, без постера: {n_skip}. "
        f"Время: {fmt_time(elapsed)}, в среднем {avg:.1f} с на постер. ===")
    log(f"Отчёт: {out / 'index.html'}")
    if n_err:
        log("Постеры с ошибками попробуются снова при следующем запуске.")
    return 0
