"""tidyselect helpers: ! / - negation, set ops, where() predicates."""

from __future__ import annotations

import ast

import pandas as pd
import polars as pl
import pytest

from tidy3 import (
    all_of,
    cols_between,
    col_range,
    contains,
    ends_with,
    everything,
    is_bool,
    is_boolean,
    is_categorical,
    is_character,
    is_datetime,
    is_float,
    is_integer,
    is_numeric,
    is_string,
    last_col,
    matches,
    select,
    starts_with,
    tidy,
    where,
)
from tidy3.masking import apply_masking, default_known_names, rewrite_bang_not


def _cars():
    return tidy(
        {
            "new_a": [1, 2],
            "new_b": [3, 4],
            "mpg": [21.0, 22.5],
            "cyl": [4, 6],
            "hp": [110, 95],
            "name": ["a", "b"],
            "flag": [True, False],
        }
    )


def columns(frame):
    return list(frame.collect().columns)


def test_negation_tilde_and_minus_starts_with():
    cars = _cars()
    assert columns(cars >> select(~starts_with("new"))) == [
        "mpg",
        "cyl",
        "hp",
        "name",
        "flag",
    ]
    # dplyr -helper
    assert columns(cars >> select(-starts_with("new"))) == [
        "mpg",
        "cyl",
        "hp",
        "name",
        "flag",
    ]


def test_bang_preparser():
    assert rewrite_bang_not('select(!starts_with("new"))') == (
        'select(~starts_with("new"))'
    )
    assert rewrite_bang_not("filter(mpg != 20)") == "filter(mpg != 20)"
    assert rewrite_bang_not('x = "a!b"') == 'x = "a!b"'
    # Nested inside tidy3 verb still rewrites
    assert rewrite_bang_not(
        'select(where(is_numeric) & !starts_with("id"))'
    ) == 'select(where(is_numeric) & ~starts_with("id"))'
    assert rewrite_bang_not("filter(!(mpg > 20))") == "filter(~(mpg > 20))"
    # Attribute form: df.select / verbs.select
    assert rewrite_bang_not('df.select(!starts_with("x"))') == (
        'df.select(~starts_with("x"))'
    )


def test_bang_preparser_leaves_shell_and_non_tidy3():
    """Notebook shell / non-tidy3 ``!`` must stay literal (no global rewrite)."""
    assert rewrite_bang_not("!pip install polars") == "!pip install polars"
    assert rewrite_bang_not("!pip install -U tidy3\n") == "!pip install -U tidy3\n"
    assert rewrite_bang_not("!whoami") == "!whoami"
    assert rewrite_bang_not("!whoami\n") == "!whoami\n"
    assert rewrite_bang_not("!!ls -la") == "!!ls -la"
    assert rewrite_bang_not("x = !ls") == "x = !ls"
    assert rewrite_bang_not("  !cd /tmp && pwd") == "  !cd /tmp && pwd"
    # Non-tidy3 call — not our sugar
    assert rewrite_bang_not("print(!foo)") == "print(!foo)"
    assert rewrite_bang_not("my_func(!starts_with('x'))") == (
        "my_func(!starts_with('x'))"
    )
    # Comment with bang outside tidy3
    assert rewrite_bang_not("# use !pip to install\nx = 1") == (
        "# use !pip to install\nx = 1"
    )
    # Mixed cell: shell line + tidy3 line
    mixed = '!pip install polars\nselect(!starts_with("tmp_"))\n'
    assert rewrite_bang_not(mixed) == (
        '!pip install polars\nselect(~starts_with("tmp_"))\n'
    )


def test_backtick_transform_preserves_shell_pip():
    from tidy3.masking import tidy3_backtick_transform

    lines = ["!pip install polars\n"]
    assert tidy3_backtick_transform(lines) == lines
    assert tidy3_backtick_transform(["!whoami\n"]) == ["!whoami\n"]
    lines2 = ['select(!starts_with("new"))\n']
    out2 = "".join(tidy3_backtick_transform(lines2))
    assert "~starts_with" in out2
    assert "!" not in out2 or "!=" in out2


def test_masking_select_bang_starts_with():
    out = apply_masking(
        'select(!starts_with("new"))',
        known=default_known_names(),
    )
    # ! → ~ then unparse may keep ~
    assert "starts_with" in out
    assert "!" not in out or "!=" in out
    tree = ast.parse(out)
    call = tree.body[0].value
    assert isinstance(call, ast.Call)
    assert isinstance(call.args[0], ast.UnaryOp)
    assert isinstance(call.args[0].op, ast.Invert)


def test_masking_select_minus_bare_column():
    out = apply_masking("select(-mpg)", known=default_known_names())
    # -mpg → ~all_of(["mpg"])
    assert "all_of" in out
    assert "mpg" in out


def test_end_to_end_bang_via_transform():
    from tidy3.export import transform_source

    cars = _cars()
    src = transform_source(
        'cars >> select(!starts_with("new"))',
        known=default_known_names({"cars"}),
    )
    ns = {
        "cars": cars,
        "select": select,
        "starts_with": starts_with,
    }
    result = eval(src, ns)
    assert columns(result) == ["mpg", "cyl", "hp", "name", "flag"]


def test_union_intersection_complement():
    cars = _cars()
    # "hp" ends with p; "mpg" ends with g
    assert columns(cars >> select(starts_with("n") | ends_with("p"))) == [
        "new_a",
        "new_b",
        "name",
        "hp",
    ]
    assert columns(cars >> select(starts_with("n") & contains("a"))) == [
        "new_a",
        "name",
    ]
    assert columns(cars >> select(everything() - starts_with("new"))) == [
        "mpg",
        "cyl",
        "hp",
        "name",
        "flag",
    ]
    assert columns(cars >> select(starts_with("new") & ~contains("b"))) == ["new_a"]


def test_everything_all_of_any_of_matches_last_col():
    cars = _cars()
    assert columns(cars >> select(everything())) == list(cars.columns)
    assert columns(cars >> select(all_of(["mpg", "cyl"]))) == ["mpg", "cyl"]
    assert columns(cars >> select(matches(r"^c"))) == ["cyl"]
    assert columns(cars >> select(last_col())) == ["flag"]


def test_col_range_and_cols_between():
    cars = _cars()
    assert columns(cars >> select(col_range("mpg", "hp"))) == ["mpg", "cyl", "hp"]
    assert columns(cars >> select(cols_between("mpg", "hp"))) == ["mpg", "cyl", "hp"]


def test_where_predicates_numeric_string_bool():
    cars = _cars()
    assert columns(cars >> select(where(is_numeric))) == [
        "new_a",
        "new_b",
        "mpg",
        "cyl",
        "hp",
    ]
    assert columns(cars >> select(where(is_integer))) == ["new_a", "new_b", "cyl", "hp"]
    assert columns(cars >> select(where(is_float))) == ["mpg"]
    assert columns(cars >> select(where(is_string))) == ["name"]
    assert columns(cars >> select(where(is_character))) == ["name"]
    assert columns(cars >> select(where(is_boolean))) == ["flag"]
    assert columns(cars >> select(where(is_bool))) == ["flag"]


def test_where_datetime_and_categorical():
    df = tidy(
        {
            "x": [1, 2],
            "when": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "cat": pd.Series(["a", "b"], dtype="category"),
            "label": ["u", "v"],
        }
    )
    assert columns(df >> select(where(is_datetime))) == ["when"]
    assert columns(df >> select(where(is_categorical))) == ["cat"]
    assert "label" in columns(df >> select(where(is_string)))


def test_where_negation_combo():
    cars = _cars()
    # all non-numeric
    assert columns(cars >> select(~where(is_numeric))) == ["name", "flag"]
    # numeric but not starting with new
    assert columns(cars >> select(where(is_numeric) & ~starts_with("new"))) == [
        "mpg",
        "cyl",
        "hp",
    ]


def test_predicate_helpers_on_raw_dtypes():
    assert is_integer(pl.Int64())
    assert is_float(pl.Float64())
    assert is_string(pl.String())
    assert is_boolean(pl.Boolean())
    assert is_datetime(pl.Datetime())
    assert is_categorical(pl.Categorical())
    assert not is_numeric(pl.Boolean())
