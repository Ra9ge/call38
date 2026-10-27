from types import SimpleNamespace

from postergen.block import _join, _tokens, block_colors, font_class
from postergen.logo import BoxStyle
from postergen.titles import display_title


def info(russian, english=None, name="x", kind="tv"):
    return SimpleNamespace(russian=russian, english=english, name=name, kind=kind)


def test_season_number_goes_to_the_end_of_the_title():
    assert display_title(info("Атака титанов [ТВ-2]")) == "Атака титанов 2"
    assert display_title(info("Охотник × Охотник (2011)")) == "Охотник × Охотник"
    assert display_title(info("Код Гиас: Восставший Лелуш R2")) == "Код Гиас: Восставший Лелуш R2"
    assert display_title(info("Отель для нелюдей", english="Hotel Inhumans Season 2")) == "Отель для нелюдей 2"
    assert display_title(info("Атака титанов [ТВ-1]")) == "Атака титанов"


def test_hyphenated_word_can_break_after_hyphen():
    t = _tokens("ЧЕЛОВЕК-БЕНЗОПИЛА")
    assert t[0] == "ЧЕЛОВЕК-" and _join(t) == "ЧЕЛОВЕК-БЕНЗОПИЛА"
    assert _join(t[:1]) == "ЧЕЛОВЕК-" and _join(t[1:]) == "БЕНЗОПИЛА"
    assert _tokens("Re:Zero") == ["Re:Zero"]


def style(**kw):
    st = BoxStyle(kind="plain", glyph_colors=[(255, 255, 255)], outline_color=(20, 20, 20), outline_ratio=0.05)
    for k, v in kw.items():
        setattr(st, k, v)
    return st


def test_font_follows_logo_look_then_genres():
    assert font_class(style(thin=True), ["Action"]) == "serif"            # тонкий логотип — с засечками
    assert font_class(style(), ["Action", "Shounen", "Gore"]) == "grunge"  # «Человек-бензопила»
    assert font_class(style(), ["Action", "Mecha", "Sci-Fi", "Military"]) == "tech"
    assert font_class(style(), ["Romance", "School", "Comedy"]) == "rounded"
    assert font_class(style(), ["Mecha", "Sci-Fi"]) == "tech"
    assert font_class(style(), ["Horror", "Gore"]) == "grunge"
    assert font_class(style(), []) == "heavy"
    assert font_class(style(), ["Romance"], forced="tech") == "tech"


def test_block_colors_stay_readable():
    # белые буквы на белом фоне без обводки → тёмные
    fill, outline, _ = block_colors(style(bg=(250, 250, 250), outline_color=None))
    assert sum(fill) < 200
    # белые буквы с красной обводкой на светлом фоне — оставляем как в оригинале
    fill, outline, _ = block_colors(style(bg=(240, 236, 230), outline_color=(200, 30, 40)))
    assert fill == (255, 255, 255) and outline == (200, 30, 40)
