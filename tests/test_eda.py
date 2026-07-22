"""EDA / inspection helpers: names, dim, columns, dtypes, paste-ready colnames."""

from __future__ import annotations

import polars as pl

from tidy3 import colnames, dim, dtypes, names, ncol, nrow, tidy
from tidy3.eda import format_colnames, selector_token


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


def test_names_is_plain_list():
    cars = _cars()
    assert names(cars) == ["cyl", "mpg", "hp"]
    assert isinstance(names(cars), list)


def test_colnames_paste_ready_display():
    cars = _cars()
    cn = colnames(cars)
    # Still a list for programming
    assert list(cn) == ["cyl", "mpg", "hp"]
    assert cn[0] == "cyl"
    # Display is paste-ready for select(...)
    text = repr(cn)
    assert text == "cyl,\nmpg,\nhp,\n"
    assert str(cn) == text


def test_colnames_backticks_odd_identifiers():
    tf = tidy({"hp new": [1], "class": [2], "mpg": [3]})
    text = repr(colnames(tf))
    bt = chr(96)
    assert f"{bt}hp new{bt}," in text
    assert f"{bt}class{bt}," in text  # Python keyword
    assert "mpg," in text


def test_selector_token():
    bt = chr(96)
    assert selector_token("mpg") == "mpg"
    assert selector_token("hp new") == f"{bt}hp new{bt}"
    assert selector_token("class") == f"{bt}class{bt}"


def test_format_colnames():
    assert format_colnames(["a", "b"]) == "a,\nb,\n"


def test_free_functions_dim():
    cars = _cars()
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
    assert list(colnames(pdf)) == ["a", "b"]


def test_columns_schema_only_does_not_need_full_collect():
    from tidy3 import col, filter

    tf = tidy({"x": [1, 2, 3], "y": [4, 5, 6]}) >> filter(col("x") > 0)
    assert "x" in tf.columns and "y" in tf.columns
