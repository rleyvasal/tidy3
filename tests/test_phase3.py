from __future__ import annotations

import pandas as pd
import polars as pl
import pytest

from tidy3 import (
    across,
    all_of,
    c_across,
    col,
    everything,
    filter,
    group_by,
    mean,
    mutate,
    pick,
    reframe,
    rename,
    rowwise,
    select,
    starts_with,
    std,
    sum,
    summarise,
    tidy,
    ungroup,
)




def numeric_frame():
    return tidy(
        {
            "id": [1, 1, 2],
            "x": [2.0, 4.0, 8.0],
            "y": [10.0, 20.0, 40.0],
        },
    )


def test_rowwise_ordinary_aggregates_are_per_row():
    result = numeric_frame() >> rowwise("id") >> mutate(self_mean=mean("x"))
    out = result.collect(as_="pandas")
    assert out["self_mean"].tolist() == [2.0, 4.0, 8.0]
    assert result._rowwise
    assert result._groups == ["id"]

    kept = result >> filter(col("x") == mean("x"))
    assert len(kept.collect(as_="pandas")) == 3


def test_c_across_horizontal_reductions_and_group_exclusion():
    result = numeric_frame() >> rowwise("id") >> mutate(
        total=sum(c_across(everything())),
        average=mean(c_across(all_of(["x", "y"]))),
        spread=std(c_across(all_of(["x", "y"]))),
    )
    out = result.collect(as_="pandas")
    assert out["total"].tolist() == [12.0, 24.0, 48.0]
    assert out["average"].tolist() == [6.0, 12.0, 24.0]
    assert out["spread"].tolist() == pytest.approx(
        [5.656854249, 11.313708499, 22.627416998]
    )


def test_pick_horizontal_reduction_does_not_require_rowwise():
    result = numeric_frame() >> mutate(
        total=pick(all_of(["x", "y"])).sum()
    )
    assert result.collect(as_="pandas")["total"].tolist() == [12.0, 24.0, 48.0]


def test_pick_can_create_a_structured_column():
    result = numeric_frame() >> mutate(values=pick("x", "y"))
    values = result.collect(as_="pandas")["values"].tolist()
    assert values[0] == {"x": 2.0, "y": 10.0}


def test_c_across_requires_rowwise():
    with pytest.raises(ValueError, match="rowwise"):
        numeric_frame() >> mutate(total=c_across(starts_with("x")).sum())


def test_rowwise_identifiers_follow_select_and_rename():
    result = (
        numeric_frame()
        >> rowwise(starts_with("id"))
        >> select("x", identifier="id")
        >> rename(value="x")
    )
    assert result._rowwise
    assert result._groups == ["identifier"]
    assert list(result.collect(as_="pandas").columns) == ["value", "identifier"]


def test_group_by_and_ungroup_clear_rowwise_state():
    data = numeric_frame() >> rowwise("id")
    grouped = data >> group_by("id")
    assert not grouped._rowwise
    assert not (data >> ungroup())._rowwise


def test_grouped_reframe_returns_each_value_and_recycles_scalars():
    data = tidy(
        {"g": ["b", "b", "a"], "x": [2.0, 4.0, 10.0]}
    )
    result = data >> group_by("g") >> reframe(value=col("x"), avg=mean("x"))
    out = result.collect(as_="pandas")
    expected = pd.DataFrame(
        {
            "g": ["a", "b", "b"],
            "value": [10.0, 2.0, 4.0],
            "avg": [10.0, 3.0, 3.0],
        }
    )
    pd.testing.assert_frame_equal(out, expected, check_dtype=False)
    assert result._groups is None
    assert not result._rowwise


def test_reframe_supports_across_and_ungrouped_vectors():
    data = numeric_frame()
    result = data >> reframe(
        across(all_of(["x", "y"]), lambda value: value, names="copy_{col}")
    )
    out = result.collect(as_="pandas")
    assert list(out.columns) == ["copy_x", "copy_y"]
    assert out["copy_x"].tolist() == [2.0, 4.0, 8.0]


def test_rowwise_reframe_can_expand_each_input_row():
    data = tidy({"id": [1, 2]})
    result = data >> rowwise("id") >> reframe(draw=[10, 20])
    expected = pd.DataFrame({"id": [1, 1, 2, 2], "draw": [10, 20, 10, 20]})
    pd.testing.assert_frame_equal(
        result.collect(as_="pandas"), expected, check_dtype=False
    )


def test_rowwise_summarise_returns_one_row_each_and_keeps_identifiers():
    result = numeric_frame() >> rowwise("id") >> summarise(total=sum(c_across()))
    out = result.collect(as_="pandas")
    assert out["total"].tolist() == [12.0, 24.0, 48.0]
    assert result._groups == ["id"]
    assert not result._rowwise


def test_phase3_polars_operations_remain_lazy():
    data = numeric_frame()
    results = [
        data >> rowwise("id") >> mutate(total=c_across().sum()),
        data >> mutate(total=pick(all_of(["x", "y"])).sum()),
        data >> group_by("id") >> reframe(value=col("x")),
    ]
    assert all(isinstance(result._lf, pl.LazyFrame) for result in results)
