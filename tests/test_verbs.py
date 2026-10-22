"""Core verb correctness vs Polars."""

from __future__ import annotations

import polars as pl
import pytest

from tidy3 import (
    TidyFrame,
    arrange,
    col,
    collect,
    filter,
    group_by,
    mean,
    mutate,
    n,
    select,
    summarise,
    tidy,
)


@pytest.fixture
def cars() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "mpg": [21.0, 22.8, 21.4, 18.7, 18.1, 14.3],
            "cyl": [6, 4, 6, 8, 6, 8],
            "hp": [110, 93, 110, 175, 105, 245],
        }
    )


def test_tidy_wraps_pandas_and_polars(cars):
    import pandas as pd

    t1 = tidy(cars)
    t2 = tidy(cars.to_pandas())
    assert isinstance(t1, TidyFrame)
    assert isinstance(t2, TidyFrame)
    assert t1.collect().shape == cars.shape


def test_pipe_filter_mutate(cars):
    out = (
        tidy(cars)
        >> filter(col("mpg") > 20)
        >> mutate(km=col("mpg") * 1.609)
        >> collect()
    )
    assert out.height == 3
    assert "km" in out.columns
    assert out["km"][0] == pytest.approx(21.0 * 1.609)


def test_method_chain_parity(cars):
    a = (
        tidy(cars)
        >> filter(col("mpg") > 20)
        >> select("mpg", "cyl")
        >> collect()
    )
    b = tidy(cars).filter(col("mpg") > 20).select("mpg", "cyl").collect()
    assert a.equals(b)


def test_group_by_summarise(cars):
    out = (
        tidy(cars)
        >> group_by("cyl")
        >> summarise(n=n(), avg=mean("mpg"))
        >> arrange("cyl")
        >> collect()
    )
    assert set(out.columns) == {"cyl", "n", "avg"}
    assert out.height == 3
    # cyl=4 has one row
    row4 = out.filter(pl.col("cyl") == 4)
    assert row4["n"][0] == 1


def test_lazy_until_collect(cars):
    tf = tidy(cars) >> filter(col("mpg") > 20)
    assert isinstance(tf, TidyFrame)
    assert isinstance(tf.lazy(), pl.LazyFrame)


def test_current_verb_accepts_frame_from_previous_module_copy(cars):
    # Remote re-seeding gives the same qualified class a new identity.  Model
    # an instance retained in the notebook namespace from before that reload.
    StaleTidyFrame = type(
        "TidyFrame",
        (),
        {
            "__module__": "tidy3.frame",
            "__rshift__": lambda self, other: NotImplemented,
        },
    )
    stale = StaleTidyFrame()
    stale._data = cars.lazy()
    stale._groups = ["cyl"]

    out = (stale >> summarise(n=n())).collect()

    assert isinstance(out, pl.DataFrame)
    assert out.height == cars["cyl"].n_unique()


def test_current_frame_accepts_verb_from_previous_module_copy(cars):
    StaleVerb = type(
        "Verb",
        (),
        {
            "__module__": "tidy3.verbs",
            "__rrshift__": lambda self, other: "applied",
        },
    )

    assert (tidy(cars) >> StaleVerb()) == "applied"


# ── dplyr grouped (window) semantics ────────────────────────────────────────


def test_grouped_mutate_is_windowed(cars):
    out = (
        tidy(cars)
        >> group_by("cyl")
        >> mutate(gmean=mean("mpg"))
        >> collect()
    )
    exp = cars.with_columns(pl.col("mpg").mean().over("cyl").alias("gmean"))
    assert (
        out.sort(["cyl", "mpg"])["gmean"].to_list()
        == exp.sort(["cyl", "mpg"])["gmean"].to_list()
    )


def test_grouped_filter_is_windowed(cars):
    out = (
        tidy(cars)
        >> group_by("cyl")
        >> filter(col("mpg") > mean("mpg"))
        >> collect()
    )
    exp = cars.filter((pl.col("mpg") > pl.col("mpg").mean()).over("cyl"))
    assert out.sort("mpg")["mpg"].to_list() == exp.sort("mpg")["mpg"].to_list()


def test_sample_n_stays_lazy_and_exact():
    from tidy3 import sample_n

    df = pl.DataFrame({"x": list(range(1000))})
    tf = tidy(df) >> sample_n(10, seed=42)
    assert isinstance(tf, TidyFrame)  # still lazy, nothing materialized
    out = tf.collect()
    assert out.height == 10
    assert out["x"].n_unique() == 10


def test_grouped_sample_n_per_group(cars):
    from tidy3 import sample_n

    out = (tidy(cars) >> group_by("cyl") >> sample_n(1, seed=1)).collect()
    assert out.height == cars["cyl"].n_unique()


def test_sample_frac_lazy(cars):
    from tidy3 import sample_frac

    out = (tidy(cars) >> sample_frac(0.5, seed=7)).collect()
    assert out.height == 3  # 6 rows * 0.5


def test_grouped_head_per_group(cars):
    from tidy3 import head

    out = (tidy(cars) >> group_by("cyl") >> head(1)).collect()
    assert out.height == cars["cyl"].n_unique()


def test_rename_remaps_groups(cars):
    from tidy3 import rename

    out = (
        tidy(cars)
        >> group_by("cyl")
        >> rename(cylinders="cyl")
        >> summarise(n=n())
    ).collect()
    assert "cylinders" in out.columns
    assert out.height == 3


def test_drop_group_column_raises(cars):
    from tidy3 import drop

    with pytest.raises(ValueError, match="grouping"):
        tidy(cars) >> group_by("cyl") >> drop("cyl")


def test_select_keeps_groups(cars):
    out = (
        tidy(cars)
        >> group_by("cyl")
        >> select("mpg")
        >> summarise(avg=mean("mpg"))
    ).collect()
    assert "cyl" in out.columns


def test_mutate_list_is_one_value_per_row():
    # dplyr: mutate(c = c(0, 1, 0)) is a column. Polars used to repeat the
    # whole list in every row; NumPy arrays already worked.
    import numpy as np

    from tidy3 import mutate, tidy

    df = tidy({"x": [1.0, 2.0, 3.0]})
    out = (df >> mutate(c=[0, 1, 0], d=(5, 6, 7), e=np.array([1, 1, 2]))).collect(as_="pandas")
    assert out["c"].tolist() == [0, 1, 0]
    assert out["d"].tolist() == [5, 6, 7]
    assert out["e"].tolist() == [1, 1, 2]


def test_count_weight_sort_name_collision_and_group_metadata():
    import pandas as pd

    from tidy3 import count, tally

    data = pd.DataFrame(
        {"g": ["a", "a", "b"], "h": [1, 2, 1], "w": [2, 3, 9], "n": [0, 0, 0], "nn": [0, 0, 0]}
    )
    counted = tidy(data) >> group_by("g") >> count(wt="w", sort=True)
    assert counted._groups == ["g"]
    out = counted.collect(as_="pandas")
    assert list(out.columns) == ["g", "nnn"]
    assert out["g"].tolist() == ["b", "a"]
    assert out["nnn"].tolist() == [9, 5]
    tallied = tidy(data) >> group_by("g", "h") >> tally()
    assert tallied._groups == ["g"]
    with pytest.raises(TypeError):
        tally("g")


@pytest.mark.parametrize("rows, expected", [(1, 0), (3, 1), (5, 2), (7, 3)])
def test_sample_frac_rounds_towards_zero(rows, expected):
    from tidy3 import sample_frac

    out = (tidy({"x": list(range(rows))}) >> sample_frac(0.5, seed=3)).collect()
    assert len(out) == expected


def test_negative_sample_sizes_select_the_complement():
    from tidy3 import sample_frac, sample_n

    data = {"x": list(range(8))}
    assert len((tidy(data) >> sample_n(-2, seed=3)).collect()) == 6
    assert len((tidy(data) >> sample_frac(-0.25, seed=3)).collect()) == 6


def test_grouped_sample_frac_truncates_within_each_group():
    from tidy3 import sample_frac

    data = {"g": ["a", "b", "b", "c", "c", "c", "c"]}
    out = (tidy(data) >> group_by("g") >> sample_frac(0.5, seed=3)).collect()
    assert dict(out["g"].value_counts().iter_rows()) == {"c": 2, "b": 1}


def test_distinct_keeps_first_occurrence_order():
    from tidy3 import distinct

    data = {"x": [3, 1, 2, 1, 3], "y": list("abcde")}
    out = (tidy(data) >> distinct("x", keep_all=True)).collect()
    assert out["x"].to_list() == [3, 1, 2]
    assert out["y"].to_list() == ["a", "b", "c"]
