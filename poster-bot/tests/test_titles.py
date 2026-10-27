from postergen.titles import parse_title, split_subtitle
from postergen.compose import tagline_text


def test_season_from_number():
    p = parse_title("Отель для нелюдей 2")
    assert (p.title, p.season, p.label) == ("Отель для нелюдей", 2, "ВТОРОЙ СЕЗОН")


def test_season_and_part():
    p = parse_title("Атака титанов 3: Часть 2")
    assert (p.title, p.season, p.part) == ("Атака титанов", 3, 2)
    assert p.label == "ТРЕТИЙ СЕЗОН • ЧАСТЬ 2"


def test_english_season_words():
    assert parse_title("Spy x Family Second Season").season == 2
    assert parse_title("Overlord IV").season == 4


def test_plain_title_untouched():
    p = parse_title("Кошечка из Сакурасо")
    assert (p.title, p.season, p.label) == ("Кошечка из Сакурасо", None, None)


def test_movie_number_is_not_season():
    assert parse_title("Евангелион 3", allow_trailing_number=False).season is None


def test_subtitle_split():
    assert split_subtitle("Клинок, рассекающий демонов: Квартал красных фонарей") == [
        "Клинок, рассекающий демонов", "Квартал красных фонарей"]
    assert split_subtitle("Re:Zero") == ["Re:Zero"]


def test_taglines():
    assert tagline_text("TVアニメ", None) == "ТВ-АНИМЕ"
    assert tagline_text("第2期", None) == "ВТОРОЙ СЕЗОН"
    assert tagline_text("第三期", None) == "ТРЕТИЙ СЕЗОН"
    assert tagline_text("劇場版", None) == "ФИЛЬМ"
    assert tagline_text("さくら荘", None) is None
