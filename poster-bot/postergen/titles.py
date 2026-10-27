"""Разбор названий: номер сезона/части, подпись «ВТОРОЙ СЕЗОН», чистое название."""
from __future__ import annotations

import re
from dataclasses import dataclass

ORDINALS_M = {2: "ВТОРОЙ", 3: "ТРЕТИЙ", 4: "ЧЕТВЁРТЫЙ", 5: "ПЯТЫЙ", 6: "ШЕСТОЙ", 7: "СЕДЬМОЙ",
              8: "ВОСЬМОЙ", 9: "ДЕВЯТЫЙ", 10: "ДЕСЯТЫЙ", 11: "ОДИННАДЦАТЫЙ", 12: "ДВЕНАДЦАТЫЙ"}
RU_WORD_NUM = {"перв": 1, "втор": 2, "трет": 3, "четв": 4, "пят": 5, "шест": 6, "седьм": 7,
               "восьм": 8, "девят": 9, "десят": 10}
EN_WORD_NUM = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
               "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10}
ROMAN = {"II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10}

SUBTITLE_SEPARATORS = re.compile(r"\s*(?::\s|\.\s|\s[—–-]\s)\s*")


@dataclass
class ParsedTitle:
    title: str                 # название без «2 сезон» и т.п.
    season: int | None = None
    part: int | None = None

    @property
    def label(self) -> str | None:
        """Подпись над названием, как на постерах: «ВТОРОЙ СЕЗОН», «ЧАСТЬ 2»."""
        bits = []
        if self.season and self.season >= 2:
            bits.append(f"{ORDINALS_M.get(self.season, f'{self.season}-Й')} СЕЗОН")
        if self.part and self.part >= 2:
            bits.append(f"ЧАСТЬ {self.part}")
        return " • ".join(bits) or None


def _word_num(word: str, table: dict[str, int]) -> int | None:
    word = word.lower()
    for k, v in table.items():
        if word.startswith(k):
            return v
    return None


def _strip(title: str, pattern: str) -> tuple[str, re.Match | None]:
    m = re.search(pattern, title, flags=re.I)
    if not m:
        return title, None
    cleaned = (title[:m.start()] + title[m.end():]).strip()
    cleaned = re.sub(r"\s*[:.,–—-]\s*$", "", cleaned)
    cleaned = re.sub(r"\(\s*\)", "", cleaned).strip()
    return cleaned, m


def parse_title(title: str, *, allow_trailing_number: bool = True) -> ParsedTitle:
    """'Отель для нелюдей 2' -> ('Отель для нелюдей', season=2)."""
    t = " ".join((title or "").split())
    season = part = None

    # часть: «Часть 2», «Part 2», «2-я часть», «Cour 2»
    t, m = _strip(t, r"[:.,]?\s*\(?(?:часть|part|cour)\s*(\d{1,2})\)?")
    if m:
        part = int(m.group(1))
    else:
        t, m = _strip(t, r"[:.,]?\s*\(?(\d{1,2})(?:-?я)?\s*часть\)?")
        if m:
            part = int(m.group(1))
    if part is None:
        t, m = _strip(t, r"[:.,]?\s*\(?(перв|втор|трет|четв|пят|шест|седьм|восьм|девят|десят)\w*\s+част\w*\)?")
        if m:
            part = _word_num(m.group(1), RU_WORD_NUM)

    # сезон словами/цифрами
    patterns = [
        (r"[:.,]?\s*\(?(\d{1,2})(?:-?(?:й|ой|ый|ий|st|nd|rd|th))?\s*(?:сезон|season)\)?", lambda m: int(m.group(1))),
        (r"[:.,]?\s*\(?(?:сезон|season)\s*(\d{1,2})\)?", lambda m: int(m.group(1))),
        (r"[:.,]?\s*\(?(перв|втор|трет|четв|пят|шест|седьм|восьм|девят|десят)\w*\s+сезон\)?",
         lambda m: _word_num(m.group(1), RU_WORD_NUM)),
        (r"[:.,]?\s*\(?(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+season\)?",
         lambda m: _word_num(m.group(1), EN_WORD_NUM)),
    ]
    for pat, conv in patterns:
        t2, m = _strip(t, pat)
        if m:
            t, season = t2, conv(m)
            break

    if season is None and allow_trailing_number:
        m = re.search(r"\s(\d{1,2})$", t)
        if m and 2 <= int(m.group(1)) <= 20:
            season, t = int(m.group(1)), t[:m.start()].strip()
        else:
            m = re.search(r"\s(II|III|IV|V|VI|VII|VIII|IX|X)$", t)
            if m:
                season, t = ROMAN[m.group(1)], t[:m.start()].strip()

    return ParsedTitle(title=t.strip(" :.,-–—") or title, season=season, part=part)


def split_subtitle(title: str) -> list[str]:
    """'Клинок, рассекающий демонов: Квартал красных фонарей' -> [главное, подзаголовок]."""
    parts = SUBTITLE_SEPARATORS.split(title, maxsplit=1)
    parts = [p.strip() for p in parts if p.strip()]
    return parts if len(parts) == 2 else [title]


def resolve_titles(info, kind: str | None = None) -> tuple[ParsedTitle, str | None]:
    """Русское название (без номера сезона) + подпись на английском/ромадзи."""
    kind = kind or getattr(info, "kind", None)
    allow_num = kind not in ("movie", "music")
    ru = parse_title(info.russian or info.name, allow_trailing_number=allow_num)
    if ru.season is None or ru.part is None:
        for alt in (info.english, info.name):
            if not alt:
                continue
            p = parse_title(alt, allow_trailing_number=allow_num)
            ru.season = ru.season or p.season
            ru.part = ru.part or p.part
    sub_src = info.english or info.name
    sub = parse_title(sub_src, allow_trailing_number=allow_num).title if sub_src else None
    if sub and ru.title and sub.strip().lower() == ru.title.strip().lower():
        sub = None
    return ru, sub


def slug(text: str) -> str:
    """Безопасное имя файла: 'Sakurasou no Pet na Kanojo' -> 'sakurasou-no-pet-na-kanojo'."""
    s = re.sub(r"[^\w\-]+", "-", (text or "").lower(), flags=re.UNICODE).strip("-")
    return s[:60] or "poster"
