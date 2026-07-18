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
