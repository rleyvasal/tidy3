from __future__ import annotations

import pandas as pd
import pytest

from tidy3 import (
    col,
    filter,
    fixed,
    group_by,
    mutate,
    regex,
    str_c,
    str_count,
    str_detect,
    str_dup,
    str_ends,
    str_extract,
    str_flatten,
    str_length,
    str_pad,
    str_remove,
    str_remove_all,
    str_replace,
    str_replace_all,
    str_replace_na,
    str_split,
    str_split_i,
    str_squish,
    str_starts,
    str_sub,
    str_to_lower,
    str_to_sentence,
    str_to_title,
    str_to_upper,
    str_trim,
    str_trunc,
    summarise,
    tidy,
    transform_source,
)



def values(frame, column):
    return [
        None if pd.api.types.is_scalar(v) and pd.isna(v) else v
        for v in frame.collect(as_="pandas")[column].tolist()
    ]


@pytest.fixture
def cars():
    return tidy(
        {
            "name": ["chevrolet chevelle", "buick skylark", "Chevy nova", None],
            "brand": ["chevrolet", "buick", "chevy", "ford"],
            "mpg": [18, 15, 16, 20],
        }
    )


def test_filter_str_starts(cars):
    out = cars >> filter(str_starts("name", "chev"))
    assert values(out, "name") == ["chevrolet chevelle"]


def test_detect_family_keeps_missing(cars):
    out = cars >> mutate(
        d=str_detect("name", "ev"),
        neg=str_detect("name", "ev", negate=True),
        s=str_starts("name", regex("chev", ignore_case=True)),
        e=str_ends("name", "a$|lle"),
        f=str_detect("name", fixed(".")),
    )
    assert values(out, "d") == [True, False, True, None]
    assert values(out, "neg") == [False, True, False, None]
    assert values(out, "s") == [True, False, True, None]
    assert values(out, "e") == [True, False, True, None]
    assert values(out, "f") == [False, False, False, None]


def test_count_and_length(cars):
    out = cars >> mutate(n=str_count("name", "e"), len=str_length("name"))
    assert values(out, "n") == [5, 0, 1, None]
    assert values(out, "len") == [18, 13, 10, None]


def test_str_sub_is_one_based_with_negatives(cars):
    out = cars >> mutate(
        a=str_sub("brand", 1, 3),
        b=str_sub("brand", -3),
        c=str_sub("brand", 2, -2),
        d=str_sub("brand", 10, 12),
    )
    # R: str_sub(c("chevrolet","buick","chevy","ford"), 1, 3) etc.
    assert values(out, "a") == ["che", "bui", "che", "for"]
    assert values(out, "b") == ["let", "ick", "evy", "ord"]
    assert values(out, "c") == ["hevrole", "uic", "hev", "or"]
    assert values(out, "d") == ["", "", "", ""]


def test_trunc(cars):
    out = cars >> mutate(
        r=str_trunc("brand", 6),
        l=str_trunc("brand", 6, side="left"),
        c=str_trunc("brand", 6, side="center"),
    )
    assert values(out, "r") == ["che...", "buick", "chevy", "ford"]
    assert values(out, "l") == ["...let", "buick", "chevy", "ford"]
    assert values(out, "c") == ["ch...t", "buick", "chevy", "ford"]


def test_replace_and_remove(cars):
    out = cars >> mutate(
        one=str_replace("brand", "[aeiou]", "-"),
        all=str_replace_all("brand", "[aeiou]", "-"),
        ref=str_replace("brand", "(ch)(e)", "\\2\\1"),
        dollar=str_replace("brand", "c", "$"),
        rm=str_remove("brand", "c"),
        rma=str_remove_all("brand", fixed("c")),
        ci=str_replace_all("brand", regex("C", ignore_case=True), "K"),
    )
    assert values(out, "one") == ["ch-vrolet", "b-ick", "ch-vy", "f-rd"]
    assert values(out, "all") == ["ch-vr-l-t", "b--ck", "ch-vy", "f-rd"]
    assert values(out, "ref") == ["echvrolet", "buick", "echvy", "ford"]
    assert values(out, "dollar") == ["$hevrolet", "bui$k", "$hevy", "ford"]
    assert values(out, "rm") == ["hevrolet", "buik", "hevy", "ford"]
    assert values(out, "rma") == ["hevrolet", "buik", "hevy", "ford"]
    assert values(out, "ci") == ["Khevrolet", "buiKk", "Khevy", "ford"]


def test_missing_stays_missing_until_replaced(cars):
    out = cars >> mutate(up=str_to_upper("name"), na=str_replace_na("name"))
    assert values(out, "up")[-1] is None
    assert values(out, "na")[-1] == "NA"


def test_case(cars):
    out = cars >> mutate(
        u=str_to_upper("brand"),
        lo=str_to_lower(str_to_upper("brand")),
        t=str_to_title("name"),
        s=str_to_sentence(str_to_upper("name")),
    )
    assert values(out, "u") == ["CHEVROLET", "BUICK", "CHEVY", "FORD"]
    assert values(out, "lo") == ["chevrolet", "buick", "chevy", "ford"]
    assert values(out, "t") == ["Chevrolet Chevelle", "Buick Skylark", "Chevy Nova", None]
    assert values(out, "s") == ["Chevrolet chevelle", "Buick skylark", "Chevy nova", None]


def test_whitespace_and_padding():
    frame = tidy({"x": ["  a  b ", "c", None]})
    out = frame >> mutate(
        both=str_trim("x"),
        left=str_trim("x", side="left"),
        right=str_trim("x", side="right"),
        squish=str_squish("x"),
    )
    assert values(out, "both") == ["a  b", "c", None]
    assert values(out, "left") == ["a  b ", "c", None]
    assert values(out, "right") == ["  a  b", "c", None]
    assert values(out, "squish") == ["a b", "c", None]
    out = frame >> mutate(
        l=str_pad("x", 4, pad="*"),
        r=str_pad("x", 4, side="right", pad="*"),
        b=str_pad("x", 4, side="both", pad="*"),
        d=str_dup("x", 3),
        z=str_dup("x", 0),
    )
    assert values(out, "l") == ["  a  b ", "***c", None]
    assert values(out, "r") == ["  a  b ", "c***", None]
    # R: str_pad("c", 4, "both", "*") puts the odd pad on the right.
    assert values(out, "b") == ["  a  b ", "*c**", None]
    assert values(out, "d") == ["  a  b   a  b   a  b ", "ccc", None]
    assert values(out, "z") == ["", "", None]


def test_extract_and_split(cars):
    out = cars >> mutate(
        word=str_extract("name", "[a-z]+$"),
        group=str_extract("name", "(\\w+) (\\w+)", group=2),
        none=str_extract("brand", "z"),
        second=str_split_i("name", " ", 2),
        last=str_split_i("name", "\\s+", -1),
        oob=str_split_i("brand", "e", 5),
    )
    assert values(out, "word") == ["chevelle", "skylark", "nova", None]
    assert values(out, "group") == ["chevelle", "skylark", "nova", None]
    assert values(out, "none") == [None, None, None, None]
    assert values(out, "second") == ["chevelle", "skylark", "nova", None]
    assert values(out, "last") == ["chevelle", "skylark", "nova", None]
    assert values(out, "oob") == [None, None, None, None]
    pieces = values(cars >> mutate(p=str_split("brand", "[eo]")), "p")
    assert [list(p) for p in pieces] == [
        ["ch", "vr", "l", "t"], ["buick"], ["ch", "vy"], ["f", "rd"]
    ]


def test_str_c_text_and_columns(cars):
    out = cars >> mutate(
        label=str_c(col("brand"), ": ", col("mpg")),
        sep=str_c(col("brand"), col("name"), sep="|"),
    )
    assert values(out, "label") == ["chevrolet: 18", "buick: 15", "chevy: 16", "ford: 20"]
    assert values(out, "sep")[:2] == ["chevrolet|chevrolet chevelle", "buick|buick skylark"]
    assert values(out, "sep")[-1] is None


def test_str_flatten_aggregates_by_group(cars):
    out = (
        cars
        >> mutate(g=str_starts("brand", "chev"))
        >> group_by("g")
        >> summarise(brands=str_flatten("brand", ", "), names=str_flatten("name", na_rm=True))
    )
    table = out.collect(as_="pandas").set_index("g")
    assert table.loc[True, "brands"] == "chevrolet, chevy"
    assert table.loc[False, "brands"] == "buick, ford"
    assert table.loc[False, "names"] == "buick skylark"
    whole = cars >> summarise(all=str_flatten("name"))
    assert values(whole, "all") == [None]


def test_notebook_bare_names():
    src = 'cars >> filter(str_starts(name, "chev")) >> mutate(x = str_c(brand, " ", mpg))'
    assert transform_source(src) == (
        "cars >> filter(str_starts(col('name'), 'chev')) "
        ">> mutate(x=str_c(col('brand'), ' ', col('mpg')))"
    )


def test_argument_checks():
    with pytest.raises(TypeError):
        str_detect("x", 3)
    with pytest.raises(ValueError):
        str_pad("x", 3, pad="ab")
    with pytest.raises(ValueError):
        str_trim("x", side="middle")
    with pytest.raises(ValueError):
        str_split_i("x", " ", 0)


# ── the rest of stringr ─────────────────────────────────────────────────

from tidy3 import (  # noqa: E402
    boundary,
    coll,
    invert_match,
    str_conv,
    str_equal,
    str_escape,
    str_extract_all,
    str_flatten_comma,
    str_glue,
    str_glue_data,
    str_ilike,
    str_like,
    str_locate,
    str_locate_all,
    str_match,
    str_match_all,
    str_order,
    str_sort,
    str_split_1,
    str_split_fixed,
    str_sub_all,
    str_subset,
    str_unique,
    str_view,
    str_which,
    str_width,
    str_wrap,
    summarise,
    unnest_wider,
    word,
)


def plain(value):
    """Collected cells as plain Python: lists for arrays, None for missing."""
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if hasattr(value, "tolist") and not pd.api.types.is_scalar(value):
        return [plain(v) for v in value.tolist()]
    if isinstance(value, list):
        return [plain(v) for v in value]
    if pd.api.types.is_scalar(value) and pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def cells(frame, column):
    return [plain(v) for v in frame.collect(as_="pandas")[column].tolist()]


@pytest.fixture
def foods():
    return tidy(
        {"x": ["apple pie", "banana split", "Cherry tart", None]}
    )


def test_like_and_equal(foods):
    out = foods >> mutate(
        an=str_like("x", "%an%"),
        c=str_like("x", "c%"),
        ic=str_ilike("x", "c%"),
        one=str_like("x", "_pple%"),
        eq=str_equal("x", "apple pie"),
        eqi=str_equal("x", "cherry TART", ignore_case=True),
    )
    assert cells(out, "an") == [False, True, False, None]
    assert cells(out, "c") == [False, False, False, None]
    assert cells(out, "ic") == [False, False, True, None]
    assert cells(out, "one") == [True, False, False, None]
    assert cells(out, "eq") == [True, False, False, None]
    assert cells(out, "eqi") == [False, False, True, None]


def test_locate(foods):
    out = foods >> mutate(a=str_locate("x", "a"), an=str_locate("x", "an"), all=str_locate_all("x", "an"))
    assert cells(out, "a")[:3] == [
        {"start": 1, "end": 1}, {"start": 2, "end": 2}, {"start": 9, "end": 9}
    ]
    assert cells(out, "an")[0] == {"start": None, "end": None}
    assert cells(out, "all") == [
        [], [{"start": 2, "end": 3}, {"start": 4, "end": 5}], [], None
    ]
    wide = foods >> mutate(loc=str_locate("x", "an")) >> unnest_wider("loc")
    assert cells(wide, "start") == [None, 2, None, None]
    # Positions count characters, not bytes.
    assert str_locate(["café au"], "au") == [{"start": 6, "end": 7}]
    assert invert_match([{"start": 2, "end": 3}]) == [
        {"start": 0, "end": 1}, {"start": 4, "end": -1}
    ]


def test_words_and_pieces(foods):
    out = foods >> mutate(
        first=word("x", 1),
        last=word("x", -1),
        both=word("x", 1, -1),
        none=word("x", 3),
        width=str_width("x"),
        subs=str_sub_all("x", [1, 3], [2, 4]),
    )
    assert cells(out, "first") == ["apple", "banana", "Cherry", None]
    assert cells(out, "last") == ["pie", "split", "tart", None]
    assert cells(out, "both") == ["apple pie", "banana split", "Cherry tart", None]
    assert cells(out, "none") == [None, None, None, None]
    assert cells(out, "width") == [9, 12, 11, None]
    assert cells(out, "subs")[0] == ["ap", "pl"]
    assert word(["a, b,  c"], 2, sep=",\\s*") == ["b"]
    assert word(["a b c"], 2, 3) == ["b c"]
    assert str_width(["abc", "日本", "é"]) == [3, 4, 1]


def test_extract_all_and_match(foods):
    out = foods >> mutate(
        vowels=str_extract_all("x", "[aeiou]"),
        m=str_match("x", "(\\w+) (\\w+)"),
        mall=str_match_all("x", "([a-z])(t)"),
    )
    assert cells(out, "vowels")[0] == ["a", "e", "i", "e"]
    assert cells(out, "vowels")[3] is None
    assert cells(out, "m")[0] == {"match": "apple pie", "1": "apple", "2": "pie"}
    assert cells(out, "mall")[1] == [{"match": "it", "1": "i", "2": "t"}]
    assert str_match(["k=v", "x"], "(?P<key>\\w)=(?P<val>\\w)") == [
        {"match": "k=v", "key": "k", "val": "v"},
        {"match": None, "key": None, "val": None},
    ]


def test_split_variants(foods):
    out = foods >> mutate(
        one=str_split("x", " ", n=1),
        fixed3=str_split_fixed("x", " ", 3),
    )
    assert cells(out, "one")[:3] == [["apple pie"], ["banana split"], ["Cherry tart"]]
    assert cells(out, "fixed3") == [
        ["apple", "pie", ""], ["banana", "split", ""], ["Cherry", "tart", ""], None
    ]
    assert str_split(["a b c"], " ", n=2) == [["a", "b c"]]
    assert str_split(["a1b2c"], "\\d", n=2) == [["a", "b2c"]]
    assert str_split(["abc"], "") == [["a", "b", "c"]]
    assert str_split(["Hi, you there"], boundary("word")) == [["Hi", "you", "there"]]
    assert str_split_1("a,b", ",") == ["a", "b"]
    assert str_count(["it's a dog's life"], boundary("word")) == [4]


def test_str_equal_normalizes_to_nfc():
    frame = tidy(
        {
            "x": ["\u00e9", "\u00c9", "\u212b", "\ufb01", "e\u0301", None],
            "y": ["e\u0301", "e\u0301", "\u00c5", "fi", "\u00e9", None],
        },
    )
    out = frame >> mutate(
        equal=str_equal("x", col("y")),
        ignore=str_equal("x", col("y"), ignore_case=True),
    )
    assert cells(out, "equal") == [True, False, True, False, True, None]
    assert cells(out, "ignore") == [True, True, True, False, True, None]
    assert str_equal(["\u00e9"], "e\u0301") == [True]
    assert str_equal(["\u00e9", None], ["e\u0301", None]) == [True, None]
    assert str_equal(["\u00e9"], "e") == [False]


def test_str_equal_is_not_stringi_collation():
    # stringr equates these. tidy3 compares NFC after optional lowercase.
    family = "\U0001f468\u200d\U0001f469\u200d\U0001f467"
    apart = "\U0001f468\U0001f469\U0001f467"
    assert str_equal([family], [apart]) == [False]
    assert str_equal(["\ufb01"], ["fi"], ignore_case=True) == [False]
    assert str_equal(["\u00df"], ["ss"], ignore_case=True) == [False]


def test_character_boundary_splits_code_points():
    assert str_split(["e\u0301"], boundary("character")) == [["e", "\u0301"]]
    family = "\U0001f468\u200d\U0001f469\u200d\U0001f467"
    assert len(str_split([family], boundary("character"))[0]) == 5


def test_replace_all_with_a_dict(foods):
    out = foods >> mutate(y=str_replace_all("x", {"a": "4", "e": "3"}))
    assert cells(out, "y") == ["4ppl3 pi3", "b4n4n4 split", "Ch3rry t4rt", None]


def test_wrap_and_escape(foods):
    out = foods >> mutate(w=str_wrap("x", 6), e=str_escape(str_c(col("x"), ".*")))
    assert cells(out, "w") == ["apple\npie", "banana\nsplit", "Cherry\ntart", None]
    assert cells(out, "e")[0] == "apple pie\\.\\*"
    assert str_wrap(["The quick brown fox jumps"], width=10) == ["The quick\nbrown fox\njumps"]


def test_flatten_last_and_comma(cars):
    assert str_flatten(["a", "b", "c"], ", ", last=" and ") == "a, b and c"
    assert str_flatten_comma(["a", "b", "c"], ", and ") == "a, b, and c"
    assert str_flatten_comma(["a", "b"], ", and ") == "a and b"
    assert str_flatten_comma(["a", "b"]) == "a, b"
    out = (
        cars
        >> mutate(g=str_starts("brand", "chev"))
        >> group_by("g")
        >> summarise(brands=str_flatten_comma("brand", ", and "))
    )
    table = out.collect(as_="pandas").set_index("g")
    assert table.loc[True, "brands"] == "chevrolet and chevy"
    whole = cars >> summarise(all=str_flatten("brand", ", ", last=" & "))
    assert cells(whole, "all") == ["chevrolet, buick, chevy & ford"]


def test_glue(cars):
    unit = "mpg"
    out = cars >> mutate(
        label=str_glue("{brand} gets {mpg} {unit}"),
        braces=str_glue("{{", "{brand}", "}}", sep=" "),
        na=str_glue("{name}!"),
    )
    assert cells(out, "label")[0] == "chevrolet gets 18 mpg"
    assert cells(out, "braces")[1] == "{ buick }"
    assert cells(out, "na")[3] == "NA!"
    assert str_glue_data({"a": [1, 2]}, "a={a}") == ["a=1", "a=2"]
    with pytest.raises(ValueError):
        str_glue("{mpg + 1}")


def test_vector_helpers():
    fruit = ["apple", "banana", None, "pear"]
    assert str_detect(fruit, "an") == [False, True, None, False]
    assert str_subset(fruit, "an") == ["banana"]
    assert str_subset(fruit, "an", negate=True) == ["apple", "pear"]
    assert str_which(fruit, "a") == [1, 2, 4]
    assert str_sort(["b", "a10", "a9", None]) == ["a10", "a9", "b", None]
    assert str_sort(["b", "a10", "a9", None], numeric=True) == ["a9", "a10", "b", None]
    assert str_sort(["b", "a", None], decreasing=True, na_last=False) == [None, "b", "a"]
    assert str_order(["b", "a", "c"]) == [2, 1, 3]
    assert str_unique(["a", "A", "a", None, None], ignore_case=True) == ["a", None]
    assert str_c(["a", "b"], "-", ["x", None]) == ["a-x", None]
    assert str_detect(["a"], coll("A", ignore_case=True)) == [True]
    assert str_detect(["a\nb"], regex("^b", multiline=True)) == [True]
    assert str_conv([b"caf\xe9"], "latin-1") == ["café"]
    assert str_conv(["Adelie", None], "not-a-codec") == ["Adelie", None]
    assert str(str_view(["apple", "banana"], "an")) == "[2] │ b<an><an>a"
    assert str(str_view(["a", None])) == "[1] │ a\n[2] │ NA"
    with pytest.raises(TypeError):
        str_subset("x", "a")


def test_vector_helpers_read_nan_in_a_list_as_missing():
    # pandas .tolist() gives NaN for missing text.
    values = ["banana", float("nan"), "apple", None]
    assert str_subset(values, "an") == ["banana"]
    assert str_which(values, "a") == [1, 3]
    assert str_sort(values) == ["apple", "banana", None, None]
    assert str_unique(["b", float("nan"), "b", pd.NA]) == ["b", None]
    assert str_subset([float("nan")], "a") == []


def test_str_conv_text_skips_per_value_decode(monkeypatch):
    def explode(value, encoding):
        raise AssertionError(f"per-value decode of {value!r} as {encoding}")

    monkeypatch.setattr("tidy3.stringr._decode", explode)
    out = (
        tidy({"species": ["Adelie", None]})
        >> mutate(out=str_conv("species", "utf-8"))
    ).collect()
    assert out["out"].to_list() == ["Adelie", None]
    assert str_conv(["Adelie", None], "utf-8") == ["Adelie", None]


def test_case_helpers_split_words_like_stringr():
    from tidy3 import str_to_camel, str_to_kebab, str_to_snake

    words = ["Mazda RX4 Wag", "HTTPServer", "camelCaseXML", "  __a-b.C  ", "x1y2", "", None]
    assert str_to_snake(words) == ["mazda_rx_4_wag", "http_server", "camel_case_xml", "a_b_c", "x_1_y_2", "", None]
    assert str_to_kebab(words) == ["mazda-rx-4-wag", "http-server", "camel-case-xml", "a-b-c", "x-1-y-2", "", None]
    assert str_to_camel(words) == ["mazdaRx4Wag", "httpServer", "camelCaseXml", "aBC", "x1Y2", "", None]
    assert str_to_camel(words, first_upper=True)[:2] == ["MazdaRx4Wag", "HttpServer"]
    # A combining accent is not a letter, so it splits a word, as in stringr.
    assert str_to_snake(["\u00c9cole", "e\u0301t\u00e9"]) == ["\u00e9cole", "e_t\u00e9"]

    out = tidy({"model": ["Hornet 4 Drive", "AMC Javelin"]}) >> mutate(
        snake=str_to_snake("model"), camel=str_to_camel(col("model"), first_upper=True)
    )
    assert values(out, "snake") == ["hornet_4_drive", "amc_javelin"]
    assert values(out, "camel") == ["Hornet4Drive", "AmcJavelin"]
    with pytest.raises(TypeError, match="first_upper"):
        str_to_camel("model", first_upper="yes")


def test_case_helpers_python_fallback_matches_polars():
    from tidy3 import str_to_camel, str_to_kebab, str_to_snake
    from tidy3.stringr import eval_pandas

    words = ["Mazda RX4 Wag", "HTTPServer", "aBCd", "ABCDe", "1abc", "\u65e5\u672cText", "don't", "", None]
    series = pd.Series(words, dtype="string")
    frame = pd.DataFrame(index=series.index)
    for name, helper, kw in (
        ("str_to_snake", str_to_snake, {}),
        ("str_to_kebab", str_to_kebab, {}),
        ("str_to_camel", str_to_camel, {"first_upper": False}),
        ("str_to_camel", str_to_camel, {"first_upper": True}),
    ):
        fallback = eval_pandas(name, (series,), kw, frame, None, "window")
        assert [None if pd.isna(v) else v for v in fallback] == helper(words, **kw)


def test_str_rank_ranks_with_ties_and_groups():
    from tidy3 import str_rank
    from tidy3.stringr import eval_pandas

    assert str_rank(["b", "a", None, "c", "a"]) == [3, 1, None, 4, 1]
    assert str_rank(["x10", "x9", "x09"]) == [2, 3, 1]
    assert str_rank(["x10", "x9", "x09"], numeric=True) == [3, 1, 1]
    frame = tidy({"g": [1, 1, 2, 2, 2], "s": ["b", "a", "x10", "x9", None]})
    out = frame >> group_by("g") >> mutate(r=str_rank("s"), n=str_rank("s", numeric=True))
    assert values(out, "r") == [2, 1, 1, 2, None]
    assert values(out, "n") == [2, 1, 2, 1, None]

    series = pd.Series(["b", "a", "x10", "x9", None], dtype="string")
    df = pd.DataFrame({"g": [1, 1, 2, 2, 2], "s": series})
    grouped = eval_pandas("str_rank", (df["s"],), {"numeric": True}, df, ["g"], "window")
    assert [None if pd.isna(v) else v for v in grouped] == [2, 1, 2, 1, None]
    with pytest.raises(ValueError, match="summarise"):
        eval_pandas("str_rank", (df["s"],), {"numeric": False}, df, None, "agg")
    with pytest.raises(TypeError, match="numeric"):
        str_rank("s", numeric=1)


def test_case_and_rank_helpers_take_bare_columns():
    src = "cars >> mutate(a = str_to_snake(name), b = str_to_camel(name), c = str_to_kebab(name), d = str_rank(name))"
    assert transform_source(src) == (
        "cars >> mutate(a=str_to_snake(col('name')), b=str_to_camel(col('name')), "
        "c=str_to_kebab(col('name')), d=str_rank(col('name')))"
    )
