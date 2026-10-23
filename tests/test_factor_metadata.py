"""Imported factors keep unused levels, order, and ordered status."""

import pandas as pd
import polars as pl
import pytest

import tidy3 as t




def factor_frame(*, ordered=True, values=None, categories=None, rows=True):
    categories = ["b", "a", "unused"] if categories is None else categories
    if values is None:
        values = ["a", "b", "a", None, "b"]
    columns = {
        "f": pd.Categorical(values, categories=categories, ordered=ordered),
    }
    if rows:
        columns["row"] = list(range(1, len(values) + 1)) if values else pd.Series([], dtype="int64")
    return t.tidy(pd.DataFrame(columns))


def assert_factor(frame, categories, ordered, column="f"):
    collected = frame.collect(as_="pandas")[column]
    assert collected.cat.categories.tolist() == categories
    assert bool(collected.cat.ordered) is ordered
    assert frame._category_levels[column] == categories
    assert frame._category_levels[column].ordered is ordered
    assert pl.Series(frame._category_levels[column]).to_list() == categories


@pytest.mark.parametrize("ordered", [True, False])
def test_collect_restores_unused_levels_and_ordered_flag(ordered):
    frame = factor_frame(ordered=ordered)
    assert_factor(frame, ["b", "a", "unused"], ordered)
    restored = frame.collect(as_="pandas")
    again = t.tidy(restored)
    assert_factor(again, ["b", "a", "unused"], ordered)
    arrow = frame.collect(as_="pandas", arrow_backed=True)
    assert arrow["f"].cat.categories.tolist() == ["b", "a", "unused"]
    assert bool(arrow["f"].cat.ordered) is ordered
    assert "pyarrow" in str(arrow["row"].dtype)


def test_empty_and_all_missing_factors_keep_levels():
    empty = t.tidy(
        pd.DataFrame({
            "f": pd.Categorical([], categories=["b", "a", "unused"], ordered=True),
            "row": pd.Series([], dtype="int64"),
        }),
    )
    assert_factor(empty, ["b", "a", "unused"], True)
    assert (empty >> t.arrange("f")).collect().height == 0
    missing = factor_frame(ordered=False, values=[None, None])
    assert_factor(missing, ["b", "a", "unused"], False)
    assert missing.collect(as_="pandas")["f"].isna().all()


def test_integer_factor_levels_survive_polars_collection():
    frame = t.tidy(
        pd.DataFrame({"f": pd.Categorical([1, None], categories=[3, 1, 2], ordered=True)}),
    )
    collected = frame.collect(as_="pandas")["f"]
    assert collected.cat.categories.tolist() == [3, 1, 2]
    assert collected.cat.ordered
    assert collected.iloc[0] == 1
    assert pd.isna(collected.iloc[1])


def test_arrange_sorts_factors_by_level_with_missing_last():
    frame = factor_frame(ordered=True)
    assert (frame >> t.arrange("f")).collect()["row"].to_list() == [2, 5, 1, 3, 4]
    assert (frame >> t.arrange(t.col("f"))).collect()["row"].to_list() == [2, 5, 1, 3, 4]
    assert (frame >> t.arrange(t.desc("f"))).collect()["row"].to_list() == [1, 3, 2, 5, 4]
    unordered = factor_frame(ordered=False)
    assert (unordered >> t.arrange("f")).collect()["row"].to_list() == [2, 5, 1, 3, 4]
    grouped = frame >> t.group_by("f") >> t.arrange("row", by_group=True)
    assert grouped.collect()["row"].to_list() == [2, 5, 1, 3, 4]


def test_factor_mutations_keep_the_ordered_flag():
    frame = factor_frame(ordered=True, values=["a", "a", "b"])
    reversed_levels = frame >> t.mutate(f=t.fct_rev("f"))
    assert_factor(reversed_levels, ["unused", "a", "b"], True)
    assert reversed_levels.collect(as_="pandas")["f"].tolist() == ["a", "a", "b"]
    unordered = frame >> t.mutate(f=t.fct_infreq("f", ordered=False))
    assert_factor(unordered, ["a", "b", "unused"], False)
    trimmed = frame >> t.group_by("f", drop=False) >> t.group_trim()
    assert trimmed._category_levels["f"] == ["b", "a"]
    assert trimmed._category_levels["f"].ordered is True


def test_rename_keeps_factor_metadata():
    renamed = factor_frame() >> t.rename(g="f")
    assert_factor(renamed, ["b", "a", "unused"], True, column="g")


def test_overwriting_a_factor_with_new_text_stays_text():
    source = pd.DataFrame({
        "f": pd.Categorical(["a", "b", None], categories=["b", "a", "unused"], ordered=True),
        "label": ["z", "nope", "zz"],
    })
    collected = (
        t.tidy(source) >> t.mutate(f=t.as_character("label"))
    ).collect(as_="pandas")["f"]
    assert collected.tolist() == ["z", "nope", "zz"]
    assert not isinstance(collected.dtype, pd.CategoricalDtype)


def test_factor_levels_lists_each_factor_column_in_order():
    frame = (
        t.tidy({"g": ["a", "b", "a"], "h": ["x", "y", "x"], "n": [1, 2, 3]})
        >> t.mutate(h=t.fct_relevel(t.col("h"), "y"), g=t.fct_infreq(t.col("g")))
    )
    assert frame.factor_levels() == {"g": ["a", "b"], "h": ["y", "x"]}
    assert list(frame.factor_levels()) == ["g", "h"]  # column order


def test_factor_levels_keeps_pandas_categories_and_unused_levels():
    frame = t.tidy(factor_frame())
    assert frame.factor_levels() == {"f": ["b", "a", "unused"]}


def test_factor_levels_includes_polars_enum_columns():
    frame = t.tidy(pl.DataFrame({"e": pl.Series(["lo", "hi"], dtype=pl.Enum(["lo", "mid", "hi"]))}))
    assert frame.factor_levels() == {"e": ["lo", "mid", "hi"]}


def test_factor_levels_leaves_out_dropped_columns():
    frame = (
        t.tidy({"g": ["a", "b"], "x": [1, 2]})
        >> t.mutate(g=t.fct_relevel(t.col("g"), "b"))
        >> t.select("x")
    )
    assert frame.factor_levels() == {}


def test_factor_levels_returns_copies():
    frame = t.tidy({"g": ["a", "b"]}) >> t.mutate(g=t.fct_rev(t.col("g")))
    frame.factor_levels()["g"].append("z")
    assert frame.factor_levels() == {"g": ["b", "a"]}
    assert type(frame.factor_levels()["g"]) is list
