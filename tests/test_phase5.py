from __future__ import annotations

import math

import pandas as pd
import pytest

from tidy3 import (
    across,
    case_when,
    coalesce,
    col,
    cumall,
    cumany,
    cume_dist,
    cummean,
    dense_rank,
    desc,
    filter,
    filter_out,
    if_else,
    inner_join,
    lag,
    left_join,
    lead,
    mean,
    min_rank,
    mutate,
    n_distinct,
    ntile,
    percent_rank,
    row_number,
    reframe,
    slice,
    slice_head,
    slice_max,
    slice_min,
    slice_tail,
    starts_with,
    summarise,
    tidy,
    transmute,
)


BACKENDS = ["polars", "pandas"]


def collect(frame):
    return frame.collect(as_="pandas")


@pytest.mark.parametrize("backend", BACKENDS)
def test_per_operation_by_summarise_preserves_first_seen_order(backend):
    frame = tidy(
        {"group": ["b", "a", "b", "a"], "value": [1, 2, 3, 6]},
        backend=backend,
    )
    result = frame >> summarise(avg=mean("value"), by="group")
    assert result._groups is None
    assert collect(result).to_dict(orient="list") == {
        "group": ["b", "a"],
        "avg": [2.0, 4.0],
    }


@pytest.mark.parametrize("backend", BACKENDS)
def test_arrange_descending_is_backend_portable(backend):
    result = tidy({"group": [2, 1, 1], "value": [1, 2, 3]}, backend=backend)
    out = collect(result.arrange("group", desc("value")))
    assert out["value"].tolist() == [3, 2, 1]


@pytest.mark.parametrize("backend", BACKENDS)
def test_per_operation_by_mutate_filter_and_slice(backend):
    frame = tidy(
        {"group": ["a", "a", "b", "b"], "value": [1, 3, 2, 8]},
        backend=backend,
    )
    mutated = frame >> mutate(avg=mean("value"), by="group")
    assert collect(mutated)["avg"].tolist() == [2.0, 2.0, 5.0, 5.0]
    assert mutated._groups is None
    assert collect(frame >> filter(col("value") > mean("value"), by="group"))[
        "value"
    ].tolist() == [3, 8]
    assert collect(frame >> slice_head(n=1, by="group"))["value"].tolist() == [1, 2]


@pytest.mark.parametrize("backend", BACKENDS)
def test_per_operation_groups_are_excluded_from_across(backend):
    frame = tidy(
        {"group": ["a", "a"], "x_one": [1, 3], "x_two": [2, 4]},
        backend=backend,
    )
    result = frame >> mutate(
        across(starts_with("x_"), mean, names="avg_{col}"), by="group"
    )
    assert collect(result)["avg_x_one"].tolist() == [2.0, 2.0]


@pytest.mark.parametrize("backend", BACKENDS)
def test_per_operation_by_remaining_transform_and_slice_verbs(backend):
    frame = tidy(
        {"group": ["a", "a", "b", "b"], "value": [1, 3, 2, 8]},
        backend=backend,
    )
    transformed = frame >> transmute(
        centered=col("value") - mean("value"), by="group"
    )
    assert list(collect(transformed).columns) == ["group", "centered"]
    assert collect(transformed)["centered"].tolist() == [-1.0, 1.0, -3.0, 3.0]
    assert collect(
        frame >> filter_out(col("value") < mean("value"), by="group")
    )["value"].tolist() == [3, 8]
    reframed = frame >> reframe(value=col("value"), by="group")
    assert collect(reframed)["value"].tolist() == [1, 3, 2, 8]
    assert collect(frame >> slice(2, by="group"))["value"].tolist() == [3, 8]
    assert collect(frame >> slice_tail(n=1, by="group"))["value"].tolist() == [3, 8]
    assert collect(frame >> slice_min("value", n=1, by="group"))[
        "value"
    ].tolist() == [1, 2]
    assert collect(frame >> slice_max("value", n=1, by="group"))[
        "value"
    ].tolist() == [3, 8]


@pytest.mark.parametrize("backend", BACKENDS)
def test_by_rejects_already_grouped_frames(backend):
    frame = tidy({"g": [1], "h": [2], "x": [3]}, backend=backend).group_by("g")
    with pytest.raises(ValueError, match="grouped or rowwise"):
        frame >> summarise(total=mean("x"), by="h")


@pytest.mark.parametrize("backend", BACKENDS)
def test_ranking_helpers_and_distinct_count(backend):
    frame = tidy(
        {
            "g": ["a", "a", "a", "b", "b"],
            "x": [10.0, 10.0, 30.0, 5.0, None],
        },
        backend=backend,
    )
    ranked = frame >> mutate(
        sequence=row_number(),
        ordinal=row_number("x"),
        minimum=min_rank("x"),
        dense=dense_rank("x"),
        percent=percent_rank("x"),
        cumulative=cume_dist("x"),
        bucket=ntile("x", 2),
        by="g",
    )
    out = collect(ranked)
    assert out["sequence"].tolist() == [1, 2, 3, 1, 2]
    assert out["minimum"].iloc[:3].tolist() == [1.0, 1.0, 3.0]
    assert out["dense"].iloc[:3].tolist() == [1.0, 1.0, 2.0]
    assert out["bucket"].iloc[:3].tolist() == [1, 1, 2]
    assert math.isclose(out.loc[0, "cumulative"], 2 / 3)
    assert pd.isna(out.loc[4, "minimum"])

    counted = frame >> summarise(unique=n_distinct("x"), by="g")
    assert collect(counted)["unique"].tolist() == [2, 2]


@pytest.mark.parametrize("backend", BACKENDS)
def test_lead_lag_and_cumulative_helpers(backend):
    frame = tidy(
        {"g": ["a", "a", "b", "b"], "x": [1, 3, 2, 6]}, backend=backend
    )
    result = frame >> mutate(
        previous=lag("x", default=-1),
        following=lead("x", default=-1),
        running_mean=cummean("x"),
        all_small=cumall(col("x") < 5),
        any_large=cumany(col("x") > 5),
        by="g",
    )
    out = collect(result)
    assert out["previous"].tolist() == [-1, 1, -1, 2]
    assert out["following"].tolist() == [3, -1, 6, -1]
    assert out["running_mean"].tolist() == [1.0, 2.0, 2.0, 4.0]
    assert out["all_small"].tolist() == [True, True, True, False]
    assert out["any_large"].tolist() == [False, False, False, True]


def test_pandas_grouped_cummean_handles_non_unique_input_index():
    data = pd.DataFrame(
        {"g": ["a", "b", "a", "b"], "x": [1.0, 2.0, 3.0, 6.0]},
        index=[5, 3, 5, 2],
    )
    out = collect(tidy(data, backend="pandas") >> mutate(avg=cummean("x"), by="g"))
    assert out["avg"].tolist() == [1.0, 2.0, 2.0, 4.0]


@pytest.mark.parametrize("backend", BACKENDS)
def test_conditional_and_missing_value_helpers(backend):
    frame = tidy(
        {"x": [1.0, None, 3.0], "fallback": [9.0, 2.0, 8.0]},
        backend=backend,
    )
    result = frame >> mutate(
        filled=coalesce(col("x"), col("fallback"), 0),
        parity=if_else(col("x").is_null(), "missing", "present"),
        band=case_when(
            (col("x").is_null(), "unknown"),
            (col("x") >= 3, "high"),
            default="low",
        ),
    )
    out = collect(result)
    assert out["filled"].tolist() == [1.0, 2.0, 3.0]
    assert out["parity"].tolist() == ["present", "missing", "present"]
    assert out["band"].tolist() == ["low", "unknown", "high"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_join_na_matching_and_keep_controls(backend):
    left = tidy({"key": [None, 1], "x": ["missing", "one"]}, backend=backend)
    right = {"key": [None, 1], "y": ["null", "one"]}
    never = collect(left >> inner_join(right, on="key", na_matches="never"))
    assert never["key"].tolist() == [1.0]

    kept = collect(left >> inner_join(right, on="key", keep=True))
    assert list(kept.columns) == ["key", "x", "key_right", "y"]
    assert pd.isna(kept.loc[0, "key_right"])


@pytest.mark.parametrize("backend", BACKENDS)
def test_join_multiple_first_last_and_any(backend):
    left = tidy({"id": [1], "x": ["x"]}, backend=backend)
    right = {"id": [1, 1], "value": ["first", "last"]}
    assert collect(left >> left_join(right, on="id", multiple="first"))[
        "value"
    ].tolist() == ["first"]
    assert collect(left >> left_join(right, on="id", multiple="last"))[
        "value"
    ].tolist() == ["last"]
    assert collect(left >> left_join(right, on="id", multiple="any"))[
        "value"
    ].tolist() == ["first"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_join_relationship_and_unmatched_guards(backend):
    one = tidy({"id": [1]}, backend=backend)
    many = {"id": [1, 1]}
    with pytest.raises(ValueError, match="many-to-one"):
        collect(one >> left_join(many, on="id", relationship="many-to-one"))

    duplicated_x = tidy({"id": [1, 1]}, backend=backend)
    with pytest.raises(ValueError, match="one-to-many"):
        collect(
            duplicated_x
            >> left_join({"id": [1]}, on="id", relationship="one-to-many")
        )

    with pytest.raises(ValueError, match="rows in y"):
        collect(
            one
            >> left_join(
                {"id": [1, 2]}, on="id", unmatched="error"
            )
        )


def test_polars_join_guards_remain_lazy_until_collect():
    result = tidy({"id": [1]}) >> left_join(
        {"id": [1, 1]}, on="id", relationship="many-to-one"
    )
    with pytest.raises(ValueError, match="many-to-one"):
        result.collect()


def test_join_control_validation_is_clear():
    with pytest.raises(ValueError, match="na_matches"):
        tidy({"id": [1]}) >> left_join(
            {"id": [1]}, on="id", na_matches="sometimes"
        )
