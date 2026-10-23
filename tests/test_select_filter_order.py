"""select + row-verb order: filter/arrange/slice may use dropped columns."""

from __future__ import annotations

import polars as pl
import pytest

from tidy3 import (
    arrange,
    col,
    desc,
    filter,
    filter_out,
    select,
    slice_max,
    slice_min,
    str_detect,
    str_length,
    str_replace_na,
    str_to_lower,
    str_trim,
    tidy,
)


def _cars():
    return pl.DataFrame(
        {
            "wt": [1.0, 2.0, 3.0],
            "mpg": [30.0, 20.0, 15.0],
            "cyl": [4, 6, 8],
            "hp": [80, 120, 300],
        }
    )


def test_select_then_filter_uses_dropped_column():
    """User's notebook case: select first, filter on a non-selected column."""
    out = (
        tidy(_cars())
        >> select("wt", "mpg", "cyl")
        >> filter(col("hp") < 250)
    ).collect()
    assert out.columns == ["wt", "mpg", "cyl"]
    assert out.shape == (2, 3)
    assert out["cyl"].to_list() == [4, 6]


def test_filter_then_select_same_result():
    a = (
        tidy(_cars())
        >> select("wt", "mpg", "cyl")
        >> filter(col("hp") < 250)
    ).collect()
    b = (
        tidy(_cars())
        >> filter(col("hp") < 250)
        >> select("wt", "mpg", "cyl")
    ).collect()
    assert a.equals(b)


def test_stacked_filters_after_select():
    out = (
        tidy(_cars())
        >> select("wt", "mpg")
        >> filter(col("hp") < 250)
        >> filter(col("cyl") == 4)
    ).collect()
    assert out.shape == (1, 2)
    assert out["wt"].to_list() == [1.0]


def test_filter_on_selected_column_still_works():
    out = (
        tidy(_cars())
        >> select("wt", "mpg", "cyl")
        >> filter(col("mpg") > 18)
    ).collect()
    assert out["mpg"].to_list() == [30.0, 20.0]


def test_filter_out_after_select():
    out = (
        tidy(_cars())
        >> select("wt", "mpg")
        >> filter_out(col("hp") >= 250)
    ).collect()
    assert out.shape == (2, 2)


def test_missing_column_still_errors():
    with pytest.raises(Exception):
        (
            tidy(_cars())
            >> select("wt", "mpg")
            >> filter(col("nope") > 0)
        ).collect()


def test_select_then_arrange_uses_dropped_column():
    out = (
        tidy(_cars())
        >> select("wt", "mpg")
        >> arrange(desc("hp"))
    ).collect()
    assert out.columns == ["wt", "mpg"]
    # hp 300, 120, 80 → rows wt 3, 2, 1
    assert out["wt"].to_list() == [3.0, 2.0, 1.0]


def test_arrange_then_select_same_result():
    a = (
        tidy(_cars())
        >> select("wt", "mpg")
        >> arrange("hp")
    ).collect()
    b = (
        tidy(_cars())
        >> arrange("hp")
        >> select("wt", "mpg")
    ).collect()
    assert a.equals(b)


def test_select_then_slice_max_uses_dropped_column():
    out = (
        tidy(_cars())
        >> select("wt", "mpg", "cyl")
        >> slice_max(order_by="hp", n=1, with_ties=False)
    ).collect()
    assert out.columns == ["wt", "mpg", "cyl"]
    assert out.shape == (1, 3)
    assert out["cyl"].to_list() == [8]
    assert out["mpg"].to_list() == [15.0]


def test_select_then_slice_min_uses_dropped_column():
    out = (
        tidy(_cars())
        >> select("wt", "mpg")
        >> slice_min(order_by="hp", n=1, with_ties=False)
    ).collect()
    assert out.columns == ["wt", "mpg"]
    assert out["wt"].to_list() == [1.0]


def test_slice_max_then_select_same_result():
    a = (
        tidy(_cars())
        >> select("wt", "mpg")
        >> slice_max(order_by="hp", n=2, with_ties=False)
    ).collect()
    b = (
        tidy(_cars())
        >> slice_max(order_by="hp", n=2, with_ties=False)
        >> select("wt", "mpg")
    ).collect()
    assert a.equals(b)


def test_filter_arrange_slice_stack_after_select():
    out = (
        tidy(_cars())
        >> select("wt", "mpg")
        >> filter(col("hp") < 250)
        >> arrange(desc("hp"))
        >> slice_max(order_by="mpg", n=1, with_ties=False)
    ).collect()
    assert out.columns == ["wt", "mpg"]
    assert out["wt"].to_list() == [1.0]
    assert out["mpg"].to_list() == [30.0]


def _penguins():
    return pl.DataFrame(
        {
            "species": ["Adelie", "Gentoo", "Chinstrap", "Adelie", "Gentoo"],
            "island": ["Biscoe", None, "", "  ", "Dream"],
            "body_mass_g": [3750, 5000, 3500, 3800, 5400],
        }
    )


@pytest.mark.parametrize(
    "predicate, kept",
    [
        (lambda: str_trim(str_replace_na(col("island"), "")) == "", [5000, 3500, 3800]),
        (lambda: str_trim(str_to_lower(col("island"))) == "biscoe", [3750]),
        (lambda: str_detect(str_replace_na(col("island"), ""), "D"), [5400]),
        (lambda: str_length(str_trim(col("island"))) == 0, [3500, 3800]),
    ],
)
def test_filter_with_nested_functions(predicate, kept):
    """A function nested inside another, in filter() (crashed in 0.8.4)."""
    out = (tidy(_penguins()) >> filter(predicate())).collect()
    assert out["body_mass_g"].to_list() == kept
    # Also after select dropped the column the nested call reads.
    dropped = (
        tidy(_penguins()) >> select("body_mass_g") >> filter(predicate())
    ).collect()
    assert dropped.columns == ["body_mass_g"]
    assert dropped["body_mass_g"].to_list() == kept
