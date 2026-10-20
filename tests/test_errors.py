from __future__ import annotations

import pickle
import sys

import polars as pl
import pytest

from tidy3 import (
    ColumnNotFoundError,
    arrange,
    col,
    count,
    filter,
    group_by,
    mean,
    mutate,
    rename,
    select,
    summarise,
    tidy,
)

BACKENDS = ["polars", "pandas"]
DATA = {"species": ["Adelie", "Gentoo"], "bill_length_mm": [39.1, 46.1], "body_mass_g": [3750, 4500]}


@pytest.fixture(params=BACKENDS)
def penguins(request):
    return tidy(DATA, backend=request.param)


@pytest.mark.parametrize(
    "verb, name",
    [
        (lambda: mutate(x=col("bill_length") + 1), "mutate"),
        (lambda: filter(col("bill_length") > 40), "filter"),
        (lambda: summarise(m=mean("bill_length")), "summarise"),
        (lambda: arrange("bill_length"), "arrange"),
        (lambda: select("bill_length"), "select"),
        (lambda: group_by("bill_length"), "group_by"),
        (lambda: count("bill_length"), "count"),
    ],
)
def test_missing_column_names_the_verb_when_the_pipe_is_built(penguins, verb, name):
    with pytest.raises(ColumnNotFoundError) as caught:
        penguins >> verb()
    error = caught.value
    assert (error.verb, error.column) == (name, "bill_length")
    assert str(error) == (
        f"{name}(): column 'bill_length' does not exist.\n"
        "Did you mean 'bill_length_mm'?\n"
        "Columns: 'species', 'bill_length_mm', 'body_mass_g'"
    )


def test_error_is_still_a_key_error_and_a_polars_error(penguins):
    with pytest.raises(KeyError):
        penguins >> mutate(x=col("nope"))
    with pytest.raises(pl.exceptions.ColumnNotFoundError):
        penguins >> mutate(x=col("nope"))


def test_assignments_in_order_are_valid(penguins):
    out = penguins >> mutate(x=1, y=col("x") + 1)
    assert out.collect(as_="pandas")["y"].tolist() == [2, 2]


def test_a_renamed_column_is_reported_at_the_later_verb(penguins):
    renamed = penguins >> rename(mass="body_mass_g")
    with pytest.raises(ColumnNotFoundError) as caught:
        renamed >> filter(col("body_mass_g") > 4000)
    assert caught.value.verb == "filter"
    assert caught.value.columns == ["species", "bill_length_mm", "mass"]


def test_long_column_lists_are_shortened_and_unrelated_names_get_no_guess():
    frame = tidy({f"col_{i}": [1] for i in range(20)})
    with pytest.raises(ColumnNotFoundError) as caught:
        frame >> select("price")
    message = str(caught.value)
    assert "Did you mean" not in message
    assert message.endswith("'col_7', … (12 more)")


def test_case_and_typo_suggestions():
    frame = tidy({"Price": [1], "quantity": [2]})
    with pytest.raises(ColumnNotFoundError, match="Did you mean 'Price'"):
        frame >> mutate(x=col("price"))
    with pytest.raises(ColumnNotFoundError, match="Did you mean 'quantity'"):
        frame >> mutate(x=col("quantiy"))


def test_error_pickles():
    error = ColumnNotFoundError("mutate", "a", ["b"])
    copy = pickle.loads(pickle.dumps(error))
    assert str(copy) == str(error)


def test_other_errors_keep_their_type_and_message():
    with pytest.raises(pl.exceptions.InvalidOperationError) as caught:
        tidy({"s": ["a"]}) >> mutate(z=col("s") + 1)
    assert "arithmetic on dtypes str" in str(caught.value)
    if sys.version_info >= (3, 11):
        assert caught.value.__notes__ == ["tidy3: raised in mutate()"]


@pytest.mark.skipif(sys.version_info < (3, 11), reason="exception notes are 3.11+")
def test_a_verb_built_on_others_names_itself():
    with pytest.raises(pl.exceptions.InvalidOperationError) as caught:
        tidy({"s": ["a"]}) >> count(wt=col("s") + 1)
    assert caught.value.__notes__ == ["tidy3: raised in count()"]


def test_failed_preview_is_short_and_escaped():
    frame = tidy({"x": ["1", "a<b>"]}) >> mutate(y=col("x").cast(pl.Int64))
    text = repr(frame)
    assert text.splitlines()[0] == "TidyFrame (preview failed)"
    assert text.splitlines()[1].startswith("InvalidOperationError: conversion from")
    assert text.splitlines()[-1] == "collect() raises the full error."
    assert "Resolved plan" not in text
    html = frame._repr_html_()
    assert "<details" in html and "Technical details" in html
    assert "a<b>" not in html and "a&lt;b&gt;" in html
    assert "Query plan:" in html
    with pytest.raises(pl.exceptions.InvalidOperationError):
        frame.collect()
