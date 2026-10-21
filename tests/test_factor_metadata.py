"""Imported factors keep unused levels, order, and ordered status."""

import pandas as pd
import polars as pl
import pytest

import tidy3 as t


BACKENDS = ["polars", "pandas"]


def factor_frame(backend, *, ordered=True, values=None, categories=None, rows=True):
    categories = ["b", "a", "unused"] if categories is None else categories
    if values is None:
        values = ["a", "b", "a", None, "b"]
    columns = {
        "f": pd.Categorical(values, categories=categories, ordered=ordered),
    }
    if rows:
        columns["row"] = list(range(1, len(values) + 1)) if values else pd.Series([], dtype="int64")
    return t.tidy(pd.DataFrame(columns), backend=backend)


def assert_factor(frame, categories, ordered, column="f"):
    collected = frame.collect(as_="pandas")[column]
    assert collected.cat.categories.tolist() == categories
    assert bool(collected.cat.ordered) is ordered
    assert frame._category_levels[column] == categories
    assert frame._category_levels[column].ordered is ordered
    assert pl.Series(frame._category_levels[column]).to_list() == categories


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("ordered", [True, False])
def test_collect_restores_unused_levels_and_ordered_flag(backend, ordered):
    frame = factor_frame(backend, ordered=ordered)
    assert_factor(frame, ["b", "a", "unused"], ordered)
    restored = frame.collect(as_="pandas")
    again = t.tidy(restored, backend="polars")
    assert_factor(again, ["b", "a", "unused"], ordered)
    pandas_frame = t.tidy(restored, backend="pandas")
    assert_factor(pandas_frame, ["b", "a", "unused"], ordered)
    if backend == "polars":
        switched = t.tidy(frame, backend="pandas")
        assert_factor(switched, ["b", "a", "unused"], ordered)
        arrow = frame.collect(as_="pandas", arrow_backed=True)
        assert arrow["f"].cat.categories.tolist() == ["b", "a", "unused"]
        assert bool(arrow["f"].cat.ordered) is ordered
        assert "pyarrow" in str(arrow["row"].dtype)


@pytest.mark.parametrize("backend", BACKENDS)
def test_empty_and_all_missing_factors_keep_levels(backend):
    empty = t.tidy(
        pd.DataFrame({
            "f": pd.Categorical([], categories=["b", "a", "unused"], ordered=True),
            "row": pd.Series([], dtype="int64"),
        }),
        backend=backend,
    )
    assert_factor(empty, ["b", "a", "unused"], True)
    assert (empty >> t.arrange("f")).collect().height == 0
    missing = factor_frame(backend, ordered=False, values=[None, None])
    assert_factor(missing, ["b", "a", "unused"], False)
    assert missing.collect(as_="pandas")["f"].isna().all()


@pytest.mark.parametrize("backend", BACKENDS)
def test_integer_factor_levels_survive_polars_collection(backend):
    frame = t.tidy(
        pd.DataFrame({"f": pd.Categorical([1, None], categories=[3, 1, 2], ordered=True)}),
        backend=backend,
    )
    collected = frame.collect(as_="pandas")["f"]
    assert collected.cat.categories.tolist() == [3, 1, 2]
    assert collected.cat.ordered
    assert collected.iloc[0] == 1
    assert pd.isna(collected.iloc[1])


@pytest.mark.parametrize("backend", BACKENDS)
def test_arrange_sorts_factors_by_level_with_missing_last(backend):
    frame = factor_frame(backend, ordered=True)
    assert (frame >> t.arrange("f")).collect()["row"].to_list() == [2, 5, 1, 3, 4]
    assert (frame >> t.arrange(t.col("f"))).collect()["row"].to_list() == [2, 5, 1, 3, 4]
    assert (frame >> t.arrange(t.desc("f"))).collect()["row"].to_list() == [1, 3, 2, 5, 4]
    unordered = factor_frame(backend, ordered=False)
    assert (unordered >> t.arrange("f")).collect()["row"].to_list() == [2, 5, 1, 3, 4]
    grouped = frame >> t.group_by("f") >> t.arrange("row", by_group=True)
    assert grouped.collect()["row"].to_list() == [2, 5, 1, 3, 4]


@pytest.mark.parametrize("backend", BACKENDS)
def test_factor_mutations_keep_the_ordered_flag(backend):
    frame = factor_frame(backend, ordered=True, values=["a", "a", "b"])
    reversed_levels = frame >> t.mutate(f=t.fct_rev("f"))
    assert_factor(reversed_levels, ["unused", "a", "b"], True)
    assert reversed_levels.collect(as_="pandas")["f"].tolist() == ["a", "a", "b"]
    unordered = frame >> t.mutate(f=t.fct_infreq("f", ordered=False))
    assert_factor(unordered, ["a", "b", "unused"], False)
    trimmed = frame >> t.group_by("f", drop=False) >> t.group_trim()
    assert trimmed._category_levels["f"] == ["b", "a"]
    assert trimmed._category_levels["f"].ordered is True


@pytest.mark.parametrize("backend", BACKENDS)
def test_rename_keeps_factor_metadata(backend):
    renamed = factor_frame(backend) >> t.rename(g="f")
    assert_factor(renamed, ["b", "a", "unused"], True, column="g")


@pytest.mark.parametrize("backend", BACKENDS)
def test_overwriting_a_factor_with_new_text_stays_text(backend):
    source = pd.DataFrame({
        "f": pd.Categorical(["a", "b", None], categories=["b", "a", "unused"], ordered=True),
        "label": ["z", "nope", "zz"],
    })
    collected = (
        t.tidy(source, backend=backend) >> t.mutate(f=t.as_character("label"))
    ).collect(as_="pandas")["f"]
    assert collected.tolist() == ["z", "nope", "zz"]
    assert not isinstance(collected.dtype, pd.CategoricalDtype)
