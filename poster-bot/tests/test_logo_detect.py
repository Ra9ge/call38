from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw

from postergen.detect import TextBox, find_logo
from postergen.pipeline import Options, make_poster
from postergen.shikimori import AnimeInfo
from postergen.titles import clean_title, parse_title, title_match, title_variants, title_words
from postergen.typo import get_font, has_glyph

H, W = 1280, 900


def box(text, x, y, w, h, score=0.99):
    return TextBox(np.float32([[x, y], [x + w, y], [x + w, y + h], [x, y + h]]), text, score)


HXH = SimpleNamespace(name="Hunter x Hunter (2011)", english="Hunter x Hunter (2011)",
                      japanese="HUNTER×HUNTER（ハンター×ハンター）", synonyms=[], russian="Охотник × Охотник (2011)")
EMINENCE = SimpleNamespace(name="Kage no Jitsuryokusha ni Naritakute!", english="The Eminence in Shadow",
                           japanese="陰の実力者になりたくて！", synonyms=[], russian="Восхождение в тени!")


def test_title_match():
    v = title_variants(HXH)
    assert title_match("HUNTER", v)[0] > 0.9                  # часть названия
    assert title_match("HUNTEP", v)[0] > 0.7                  # ошибка распознавания
    assert title_match("THE", v)[0] < 0.5                     # случайные буквы
    assert title_match("TV ANIMATION", v)[0] < 0.5


def test_title_words_and_cleanup():
    assert title_words("Охотник × Охотник") == ["Охотник ×", "Охотник"]
    assert clean_title("Hunter x Hunter (2011)") == "Hunter x Hunter"
    assert parse_title("Охотник × Охотник (2011)").title == "Охотник × Охотник"
    p = parse_title("Атака титанов [ТВ-2]")
    assert (p.title, p.label) == ("Атака титанов", "ВТОРОЙ СЕЗОН")


def test_latin_logo_found_and_credits_ignored():
    boxes = [box("HUNTER", 150, 820, 600, 200), box("HUNTER", 150, 1000, 600, 210),
             box("©POT/SHUEISHA HUNTER×HUNTER PROJECT", 100, 1250, 700, 20)]
    f = find_logo(boxes, (H, W), title_variants(HXH))
    assert f.latin and [b.text for r in f.cluster.rows for b in r.boxes] == ["HUNTER", "HUNTER"]


def test_japanese_logo_keeps_english_as_subtitle():
    boxes = [box("险", 80, 90, 245, 175, 0.46), box("の実力者になりたくて!", 300, 120, 500, 91),
             box("The Eminence in shadow", 330, 200, 430, 33)]
    f = find_logo(boxes, (H, W), title_variants(EMINENCE))
    assert not f.latin
    assert [b.text for b in f.subtitle] == ["The Eminence in shadow"]


def test_big_latin_logo_with_small_japanese_reading():
    sao = SimpleNamespace(name="Sword Art Online", english="Sword Art Online", japanese="ソードアート・オンライン",
                          synonyms=[], russian="Мастера меча онлайн")
    boxes = [box("ソードアート・オンライン", 300, 1000, 300, 40), box("SWORD ART ONLINE", 60, 1045, 780, 120)]
    f = find_logo(boxes, (H, W), title_variants(sao))
    assert f.latin and [b.text for b in f.decor] == ["ソードアート・オンライン"]


def test_no_logo():
    f = find_logo([box("©2026 Studio", 300, 1250, 300, 20)], (H, W), title_variants(HXH))
    assert f.cluster is None


def test_missing_glyph_detection():
    assert has_glyph(get_font("rubik", 40), "Ж") and not has_glyph(get_font("rubik", 40), "△")


def test_latin_logo_is_replaced_end_to_end():
    y = np.linspace(0, 1, 640)[:, None, None]
    arr = np.repeat((np.array([70, 90, 160]) * (1 - y) + np.array([200, 160, 120]) * y).astype(np.uint8), 450, axis=1)
    img = Image.fromarray(arr)
    ImageDraw.Draw(img).text((225, 90), "ONE PIECE", font=get_font("montserrat", 64, 900), fill=(255, 210, 40),
                             stroke_width=4, stroke_fill=(60, 30, 10), anchor="mm")
    info = AnimeInfo(21, "One Piece", "Ван-Пис", "One Piece", "ONE PIECE")
    out, rep = make_poster(info, img, Options(logo=None, inpaint="opencv"))
    assert rep.mode == "replace" and rep.logo == "латиница"
    top = np.asarray(out)[: out.height // 4]
    yellow = (top[..., 0] > 200) & (top[..., 1] > 160) & (top[..., 2] < 90)
    assert yellow.mean() > 0.01                                # русское название тем же жёлтым
