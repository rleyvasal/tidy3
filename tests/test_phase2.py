from __future__ import annotations

from collections import namedtuple

import pandas as pd
import polars as pl
import pytest

from tidy3 import (
    across,
    all_of,
    any_of,
    col_range,
    col,
    case_match,
    contains,
    cur_column,
    cur_group,
    cur_group_id,
    cur_group_rows,
    drop,
    ends_with,
    everything,
    group_by,
    group_map,
    group_modify,
    group_nest,
    group_split,
    hoist,
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
    n_groups,
    num_range,
    relocate,
    pack,
    separate_longer_delim,
    separate_wider_delim,
    recode,
    rename_with,
    select,
    sum,
    starts_with,
    summarise,
    tidy,
    transmute,
    unpack,
    where,
    with_groups,
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
def test_across_unpack_mapping_in_mutate_and_summarise(backend):
    data = tidy(
        {"g": ["a", "a", "b"], "x": [1.0, 3.0, 10.0], "y": [2.0, 4.0, 20.0]},
        backend=backend,
    )
    mutated = data >> mutate(
        across(
            starts_with(("x", "y")),
            lambda value: {"double": value * 2, "plus_one": value + 1},
            unpack=True,
        )
    )
    out = mutated.collect(as_="pandas")
    assert list(out.columns) == ["g", "x", "y", "x_double", "x_plus_one", "y_double", "y_plus_one"]
    assert out["x_double"].tolist() == [2.0, 6.0, 20.0]
    assert out["y_plus_one"].tolist() == [3.0, 5.0, 21.0]

    summary = data >> group_by("g") >> summarise(
        across(
            starts_with(("x", "y")),
            lambda value: {"mean": mean(value), "max": value.max()},
            names="{col}",
            unpack="{outer}__{inner}",
        )
    )
    summary_out = summary.collect(as_="pandas").sort_values("g").reset_index(drop=True)
    assert list(summary_out.columns) == ["g", "x__mean", "x__max", "y__mean", "y__max"]
    assert summary_out.loc[0, ["x__mean", "x__max", "y__mean", "y__max"]].tolist() == [
        2.0,
        3.0,
        3.0,
        4.0,
    ]

    Fields = namedtuple("Fields", ["double", "plus_one"])
    structured = data >> mutate(
        across(
            ["x"],
            lambda value: Fields(value * 2, value + 1),
            unpack=True,
        )
    )
    structured_out = structured.collect(as_="pandas")
    assert structured_out[["x_double", "x_plus_one"]].values.tolist() == [
        [2.0, 2.0],
        [6.0, 4.0],
        [20.0, 11.0],
    ]


@pytest.mark.parametrize("backend", BACKENDS)
def test_across_unpack_requires_mapping(backend):
    data = frame(backend)
    with pytest.raises(TypeError, match="must return a mapping"):
        data >> mutate(across(starts_with("x_"), lambda value: value * 2, unpack=True))
    with pytest.raises(TypeError, match="use unpack=True"):
        data >> mutate(
            across(starts_with("x_"), lambda value: {"double": value * 2})
        )


@pytest.mark.parametrize("backend", BACKENDS)
def test_cur_column_is_available_inside_across(backend):
    data = frame(backend)
    result = data >> mutate(
        across(starts_with("x_"), lambda value: value + len(cur_column()))
    )
    out = result.collect(as_="pandas")
    assert out["x_1"].tolist() == [4.0, 5.0, 6.0]
    assert out["x_2"].tolist() == [13.0, 23.0, 33.0]

    with pytest.raises(RuntimeError, match="inside across"):
        cur_column()


@pytest.mark.parametrize("backend", BACKENDS)
def test_cur_group_keys_are_available_inside_across(backend):
    data = tidy({"g": [1, 1, 2], "x": [10.0, 20.0, 30.0]}, backend=backend)
    result = data >> group_by("g") >> mutate(
        across(["x"], lambda value: value + cur_group()["g"])
    )
    out = result.collect(as_="pandas")
    assert out["x"].tolist() == [11.0, 21.0, 32.0]

    with pytest.raises(RuntimeError, match="inside across"):
        cur_group()


@pytest.mark.parametrize("backend", BACKENDS)
def test_cur_group_id_is_available_inside_across(backend):
    data = tidy({"g": [2, 1, 2, 3], "x": [10.0, 20.0, 30.0, 40.0]}, backend=backend)
    result = data >> group_by("g") >> mutate(
        across(["x"], lambda value: value + cur_group_id())
    )
    out = result.collect(as_="pandas")
    assert out["x"].tolist() == [12.0, 21.0, 32.0, 43.0]

    with pytest.raises(RuntimeError, match="inside across"):
        cur_group_id()


@pytest.mark.parametrize("backend", BACKENDS)
def test_n_groups_is_available_inside_across(backend):
    data = tidy({"g": [2, 1, 2, 3], "x": [10.0, 20.0, 30.0, 40.0]}, backend=backend)
    result = data >> group_by("g") >> mutate(
        across(["x"], lambda value: value + n_groups())
    )
    assert result.collect(as_="pandas")["x"].tolist() == [13.0, 23.0, 33.0, 43.0]
    with pytest.raises(RuntimeError, match="inside across"):
        n_groups()


@pytest.mark.parametrize("backend", BACKENDS)
def test_case_match_and_recode_are_backend_neutral(backend):
    data = tidy({"code": [1, 2, 3, None], "label": ["a", "b", "c", "d"]}, backend=backend)
    result = data >> mutate(
        bucket=case_match(
            "code",
            ((1, 2), "low"),
            (3, "high"),
            default="other",
        ),
        recoded=recode("code", {1: "one", 2: "two"}, default="other", missing="missing"),
    )
    out = result.collect(as_="pandas")
    assert out["bucket"].tolist() == ["low", "low", "high", "other"]
    assert out["recoded"].tolist() == ["one", "two", "other", "missing"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_with_groups_restores_prior_grouping(backend):
    data = tidy({"g": ["a", "a", "b"], "x": [1, 3, 10]}, backend=backend)
    result = data >> with_groups(
        "g", lambda grouped: grouped >> mutate(group_total=sum("x"))
    )
    assert result._groups is None
    assert result.collect(as_="pandas")["group_total"].tolist() == [4, 4, 10]


@pytest.mark.parametrize("backend", BACKENDS)
def test_group_split_map_modify_and_nest(backend):
    data = tidy({"g": ["a", "a", "b"], "x": [1, 3, 10]}, backend=backend)
    parts = data >> group_by("g") >> group_split()
    assert [part.collect(as_="pandas")["x"].tolist() for part in parts] == [[1, 3], [10]]

    mapped = data >> group_map(lambda part, key: (key["g"], int(part.collect(as_="pandas")["x"].sum())), "g")
    assert mapped == [("a", 4), ("b", 10)]
    row_mapped = data >> group_map(lambda part, key: cur_group_rows(), "g")
    assert row_mapped == [(0, 1), (2,)]

    modified = data >> group_modify(
        lambda part, key: tidy({"g": [key["g"]], "total": [int(part.collect(as_="pandas")["x"].sum())]}, backend=backend),
        "g",
    )
    assert modified.collect(as_="pandas").to_dict("records") == [
        {"g": "a", "total": 4},
        {"g": "b", "total": 10},
    ]

    nested = data >> group_nest("g", name="rows")
    nested_out = nested.collect(as_="pandas")
    assert nested_out["g"].tolist() == ["a", "b"]
    if backend == "pandas":
        assert nested_out["rows"].iloc[0]["x"].tolist() == [1, 3]
    else:
        assert list(nested_out["rows"].iloc[0]) == [{"x": 1}, {"x": 3}]


@pytest.mark.parametrize("backend", BACKENDS)
def test_tidyr_struct_and_delimited_helpers(backend):
    data = tidy(
        {"id": [1, 2], "code": ["a|b", "c"], "x": [10, 20], "y": [1, 2],
         "obj": [{"score": 3}, {"score": 4}]},
        backend=backend,
    )
    longer = data >> separate_longer_delim("code", "|")
    assert longer.collect(as_="pandas")["code"].tolist() == ["a", "b", "c"]
    wider = data >> separate_wider_delim("code", ["left", "right"], "|")
    wider_out = wider.collect(as_="pandas")
    assert wider_out["right"].iloc[0] == "b"
    assert pd.isna(wider_out["right"].iloc[1])
    extracted = data >> hoist("obj", score="score")
    assert extracted.collect(as_="pandas")["score"].tolist() == [3, 4]
    packed = data >> pack("xy", "x", "y")
    assert packed.collect(as_="pandas")["xy"].iloc[0]["x"] == 10
    unpacked = packed >> unpack("xy")
    assert unpacked.collect(as_="pandas")["x"].tolist() == [10, 20]


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
