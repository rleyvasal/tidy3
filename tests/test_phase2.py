from __future__ import annotations

import pandas as pd
import polars as pl
import pytest

from tidy3 import (
    across,
    all_of,
    any_of,
    col_range,
    col,
    contains,
    drop,
    ends_with,
    everything,
    group_by,
    group_cols,
    if_all,
    if_any,
    is_boolean,
    is_numeric,
    is_string,
    last_col,
    matches,
    mean,
    mutate,
    num_range,
    relocate,
    rename_with,
    select,
    starts_with,
    summarise,
    tidy,
    transmute,
    where,
)


BACKENDS = ["polars", "pandas"]


def frame(backend: str):
    return tidy(
        {
            "id": [1, 2, 3],
            "x_1": [1.0, 2.0, 3.0],
            "x_2": [10.0, 20.0, 30.0],
            "label": ["a", "b", "c"],
            "flag": [True, False, True],
        },
        backend=backend,
    )


def columns(value):
    return list(value.collect(as_="pandas").columns)


@pytest.mark.parametrize("backend", BACKENDS)
def test_name_selectors_and_composition(backend):
    data = frame(backend)
    assert columns(data >> select("id", starts_with("x_"))) == ["id", "x_1", "x_2"]
    assert columns(data >> select(contains("LAB"), ends_with("2"))) == [
        "label",
        "x_2",
    ]
    assert columns(data >> select(matches(r"^[xi]"))) == ["id", "x_1", "x_2"]
    assert columns(data >> select(num_range("x_", [2, 9, 1]))) == ["x_2", "x_1"]
    assert columns(data >> select(any_of(["missing", "label"]))) == ["label"]
    assert columns(data >> select(everything() - starts_with("x_"))) == [
        "id",
        "label",
        "flag",
    ]
    assert columns(data >> select(starts_with("x") | last_col())) == [
        "x_1",
        "x_2",
        "flag",
    ]
    assert columns(data >> select(starts_with(("id", "x_")))) == [
        "id",
        "x_1",
        "x_2",
    ]
    assert columns(data >> select(col_range("x_2", "id"))) == ["x_2", "x_1", "id"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_type_selectors_are_backend_neutral(backend):
    data = frame(backend)
    assert columns(data >> select(where(is_numeric))) == ["id", "x_1", "x_2"]
    assert columns(data >> select(where(is_string))) == ["label"]
    assert columns(data >> select(where(is_boolean))) == ["flag"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_selector_aware_drop_relocate_and_rename_with(backend):
    data = frame(backend)
    assert columns(data >> drop(starts_with("x_"))) == ["id", "label", "flag"]
    moved = data >> relocate(starts_with("x_"), after=last_col())
    assert columns(moved) == ["id", "label", "flag", "x_1", "x_2"]
    renamed = data >> rename_with(str.upper, starts_with("x_"))
    assert columns(renamed) == ["id", "X_1", "X_2", "label", "flag"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_select_rename_and_group_cols_update_metadata(backend):
    data = frame(backend) >> group_by("id")
    selected = data >> select(starts_with("x_"), identifier="id")
    assert columns(selected) == ["x_1", "x_2", "identifier"]
    assert selected._groups == ["identifier"]
    assert columns(data >> select(group_cols(), "label")) == ["id", "label"]
    with pytest.raises(ValueError, match="grouping"):
        data >> drop(group_cols())


def test_all_of_is_strict_and_any_of_is_lenient():
    data = frame("polars")
    with pytest.raises(KeyError, match="missing"):
        data >> select(all_of(["x_1", "missing"]))
    assert columns(data >> select(any_of(["x_1", "missing"]))) == ["x_1"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_across_mutate_transmute_and_names(backend):
    data = frame(backend)
    mutated = data >> mutate(
        across(starts_with("x_"), lambda value: value * 2, names="{col}_double")
    )
    out = mutated.collect(as_="pandas")
    assert out["x_1_double"].tolist() == [2.0, 4.0, 6.0]
    assert out["x_2_double"].tolist() == [20.0, 40.0, 60.0]

    compact = data >> transmute(across(starts_with("x_"), lambda value: value + 1))
    assert columns(compact) == ["x_1", "x_2"]
    assert compact.collect(as_="pandas")["x_1"].tolist() == [2.0, 3.0, 4.0]


@pytest.mark.parametrize("backend", BACKENDS)
def test_across_summarise_multiple_functions_and_groups(backend):
    data = tidy(
        {"g": ["a", "a", "b"], "x": [1.0, 3.0, 10.0], "y": [2.0, 4.0, 20.0]},
        backend=backend,
    )
    result = data >> group_by("g") >> summarise(
        across(
            where(is_numeric),
            {"mean": mean, "max": lambda value: value.max()},
            names="{col}_{fn}",
        )
    )
    out = result.collect(as_="pandas").sort_values("g").reset_index(drop=True)
    assert list(out.columns) == ["g", "x_mean", "x_max", "y_mean", "y_max"]
    assert out.loc[0, ["x_mean", "x_max", "y_mean", "y_max"]].tolist() == [
        2.0,
        3.0,
        3.0,
        4.0,
    ]


@pytest.mark.parametrize("backend", BACKENDS)
def test_if_any_and_if_all_work_in_filter_and_mutate(backend):
    data = frame(backend)
    any_rows = data >> tidy3_filter(
        if_any(starts_with("x_"), lambda value: value > 15)
    )
    assert any_rows.collect(as_="pandas")["id"].tolist() == [2, 3]

    marked = data >> mutate(
        all_positive=if_all(starts_with("x_"), lambda value: value > 0),
        any_large=if_any(starts_with("x_"), lambda value: value >= 30),
    )
    out = marked.collect(as_="pandas")
    assert out["all_positive"].tolist() == [True, True, True]
    assert out["any_large"].tolist() == [False, False, True]


@pytest.mark.parametrize("backend", BACKENDS)
def test_empty_if_any_and_if_all_use_logical_identities(backend):
    data = frame(backend)
    none = data >> tidy3_filter(
        if_any(starts_with("missing"), lambda value: value > 0)
    )
    every = data >> tidy3_filter(
        if_all(starts_with("missing"), lambda value: value > 0)
    )
    assert len(none.collect(as_="pandas")) == 0
    assert len(every.collect(as_="pandas")) == 3


def tidy3_filter(*predicates):
    # Avoid shadowing pytest's own commonly imported helpers at module scope.
    from tidy3 import filter

    return filter(*predicates)


def test_phase2_polars_operations_remain_lazy():
    data = frame("polars")
    results = [
        data >> select(starts_with("x_")),
        data >> mutate(across(starts_with("x_"), lambda value: value * 2)),
        data >> tidy3_filter(if_any(starts_with("x_"), lambda value: value > 0)),
        data >> summarise(across(starts_with("x_"), mean)),
    ]
    assert all(isinstance(result._lf, pl.LazyFrame) for result in results)


def test_select_preserves_computed_expression_position_on_polars():
    result = frame("polars") >> select(
        (col("x_1") * 2).alias("double"), starts_with("x_2")
    )
    assert columns(result) == ["double", "x_2"]


def test_selector_methods_match_pipe_api():
    data = frame("pandas")
    expected = data >> select(starts_with("x_"))
    actual = data.select(starts_with("x_"))
    pd.testing.assert_frame_equal(
        expected.collect(as_="pandas"), actual.collect(as_="pandas")
    )
