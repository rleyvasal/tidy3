"""EDA / inspection helpers: names, dim, columns, dtypes."""

from __future__ import annotations

import pandas as pd
import polars as pl
import pytest

from tidy3 import colnames, dim, dtypes, names, ncol, nrow, tidy


def _cars():
    return tidy(
        {
            "cyl": [4, 6, 8],
            "mpg": [22.0, 18.0, 15.0],
            "hp": [90, 110, 200],
        }
    )


def test_columns_and_names_properties():
    cars = _cars()
    assert cars.columns == ["cyl", "mpg", "hp"]
    assert cars.names == cars.columns
    assert cars.width == 3
    assert cars.height == 3
    assert cars.shape == (3, 3)


def test_free_functions_match_properties():
    cars = _cars()
    assert names(cars) == cars.columns
    assert colnames(cars) == ["cyl", "mpg", "hp"]
    assert nrow(cars) == 3
    assert ncol(cars) == 3
    assert dim(cars) == (3, 3)


def test_dtypes_and_schema():
    cars = _cars()
    dt = cars.dtypes
    assert set(dt) == {"cyl", "mpg", "hp"}
    assert dtypes(cars) == dt
    assert "cyl" in cars.schema


def test_names_on_polars_and_pandas():
    pdf = pl.DataFrame({"a": [1], "b": [2]})
    assert names(pdf) == ["a", "b"]
    assert ncol(pdf) == 2
    assert names(pdf.to_pandas()) == ["a", "b"]


def test_columns_schema_only_does_not_need_full_collect():
    from tidy3 import col, filter

    # Lazy with filter still exposes schema columns
    tf = tidy({"x": [1, 2, 3], "y": [4, 5, 6]}) >> filter(col("x") > 0)
    assert "x" in tf.columns and "y" in tf.columns
