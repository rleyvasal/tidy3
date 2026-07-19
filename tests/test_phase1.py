from __future__ import annotations

import pandas as pd
import polars as pl
import pytest

from tidy3 import (
    add_count,
    add_tally,
    anti_join,
    bind_cols,
    bind_rows,
    col,
    cross_join,
    filter_out,
    full_join,
    glimpse,
    intersect,
    pull,
    relocate,
    rename_with,
    right_join,
    semi_join,
    setdiff,
    setequal,
    slice,
    slice_head,
    slice_max,
    slice_min,
    slice_sample,
    slice_tail,
    symdiff,
    tidy,
    union,
    union_all,
)


BACKENDS = ["polars", "pandas"]


def values(frame, column: str):
    return frame.collect(as_="pandas")[column].tolist()


@pytest.mark.parametrize("backend", BACKENDS)
def test_filter_out_keeps_null_predicates(backend):
    frame = tidy({"x": [1.0, None, 2.0]}, backend=backend)
    result = frame >> filter_out(col("x") == 1)
    out = result.collect(as_="pandas")
    assert len(out) == 2
    assert pd.isna(out.iloc[0]["x"])
    assert out.iloc[1]["x"] == 2


@pytest.mark.parametrize("backend", BACKENDS)
def test_slice_is_one_based_grouped_and_preserves_duplicates(backend):
    frame = tidy(
        {"g": ["a", "a", "b", "b"], "x": [1, 2, 3, 4]}, backend=backend
    ).group_by("g")
    assert values(frame >> slice(2, 1, 2), "x") == [2, 1, 2, 4, 3, 4]
    assert values(frame >> slice(-2), "x") == [1, 3]
    assert len((frame >> slice()).collect(as_="pandas")) == 0


@pytest.mark.parametrize("backend", BACKENDS)
def test_slice_head_tail_and_extremes_are_grouped(backend):
    frame = tidy(
        {
            "g": ["a", "a", "a", "b", "b", "b"],
            "x": [3, 1, 1, 4, 5, 5],
        },
        backend=backend,
    ).group_by("g")
    assert values(frame >> slice_head(prop=0.5), "x") == [3, 4]
    assert values(frame >> slice_tail(n=1), "x") == [1, 5]
    assert values(frame >> slice_min("x", n=1), "x") == [1, 1, 4]
    assert values(frame >> slice_max("x", n=1, with_ties=False), "x") == [3, 5]


@pytest.mark.parametrize("backend", BACKENDS)
def test_slice_extremes_preserve_first_seen_group_order(backend):
    frame = tidy(
        {"g": ["b", "b", "a", "a"], "x": [2, 1, 4, 3]}, backend=backend
    ).group_by("g")
    out = (frame >> slice_min("x", n=1)).collect(as_="pandas")
    assert list(out.itertuples(index=False, name=None)) == [("b", 1), ("a", 3)]


@pytest.mark.parametrize("backend", BACKENDS)
def test_slice_sample_is_grouped_and_keeps_backend_contract(backend):
    frame = tidy(
        {"g": ["a"] * 4 + ["b"] * 4, "x": list(range(8))}, backend=backend
    ).group_by("g")
    result = frame >> slice_sample(n=2, seed=7)
    assert result._groups == ["g"]
    assert result.collect(as_="pandas").groupby("g").size().tolist() == [2, 2]


@pytest.mark.parametrize("backend", BACKENDS)
def test_slice_sample_supports_replacement(backend):
    frame = tidy({"g": ["a", "b"], "x": [1, 2]}, backend=backend).group_by("g")
    out = (frame >> slice_sample(n=2, replace=True, seed=3)).collect(as_="pandas")
    assert out.groupby("g").size().tolist() == [2, 2]


def test_slice_sample_supports_weights_on_pandas():
    frame = tidy({"x": [1, 2], "w": [0.0, 1.0]}, backend="pandas")
    out = frame >> slice_sample(n=3, weight_by="w", replace=True, seed=2)
    assert values(out, "x") == [2, 2, 2]


@pytest.mark.parametrize("backend", BACKENDS)
def test_column_verbs_and_group_metadata(backend):
    frame = tidy({"g": [1, 2], "x": [3, 4], "y": [5, 6]}, backend=backend).group_by("g")
    moved = frame >> relocate("y", before="x")
    assert list(moved.collect(as_="pandas").columns) == ["g", "y", "x"]
    renamed = moved >> rename_with(str.upper, "g", "x")
    assert renamed._groups == ["G"]
    assert list(renamed.collect(as_="pandas").columns) == ["G", "y", "X"]
    assert (frame >> pull(-1)).to_list() == [5, 6]
    named = frame >> pull("x", name="g")
    assert named.to_dict() == {1: 3, 2: 4}


@pytest.mark.parametrize("backend", BACKENDS)
def test_glimpse_prints_and_passes_frame_through(backend, capsys):
    frame = tidy({"x": [1, 2]}, backend=backend)
    result = frame >> glimpse(1)
    printed = capsys.readouterr().out
    assert result is frame
    assert "Columns: 1" in printed
    assert "$ x" in printed


@pytest.mark.parametrize("backend", BACKENDS)
def test_add_count_and_add_tally_preserve_rows_and_groups(backend):
    frame = tidy(
        {"g": ["a", "a", "b"], "x": [1, 1, 1], "w": [2, 3, 5]},
        backend=backend,
    ).group_by("g")
    counted = frame >> add_count("x")
    assert values(counted, "n") == [2, 2, 1]
    assert counted._groups == ["g"]
    tallied = frame >> add_tally(wt="w", name="total")
    assert values(tallied, "total") == [5, 5, 5]
    assert tallied._groups == ["g"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_join_family_matches_nulls_and_preserves_left_groups(backend):
    left = tidy(
        {"key": [1, 2, None], "left": ["a", "b", "null"]}, backend=backend
    ).group_by("key")
    right = {"key": [2, 3, None], "right": ["B", "C", "NULL"]}

    semi = left >> semi_join(right, on="key")
    assert values(semi, "left") == ["b", "null"]
    assert semi._groups == ["key"]
    assert values(left >> anti_join(right, on="key"), "left") == ["a"]

    right_result = (left >> right_join(right, on="key")).collect(as_="pandas")
    assert set(right_result["right"]) == {"B", "C", "NULL"}
    full_result = (left >> full_join(right, on="key")).collect(as_="pandas")
    assert len(full_result) == 4


@pytest.mark.parametrize("backend", BACKENDS)
def test_cross_join_and_bind_verbs(backend):
    left = tidy({"x": [1, 2]}, backend=backend)
    crossed = left >> cross_join({"y": ["a", "b"]})
    assert list(crossed.collect(as_="pandas").itertuples(index=False, name=None)) == [
        (1, "a"),
        (1, "b"),
        (2, "a"),
        (2, "b"),
    ]

    rows = left >> bind_rows({"y": [3]}, id="source")
    out = rows.collect(as_="pandas")
    assert list(out.columns) == ["source", "x", "y"]
    assert out["source"].tolist() == ["1", "1", "2"]

    cols = left >> bind_cols({"y": [3, 4]})
    assert list(cols.collect(as_="pandas").columns) == ["x", "y"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_set_operations_have_relational_set_semantics(backend):
    left = tidy({"x": [1, 1, 2]}, backend=backend)
    right = {"x": [2, 3, 3]}
    assert values(left >> union(right), "x") == [1, 2, 3]
    assert values(left >> union_all(right), "x") == [1, 1, 2, 2, 3, 3]
    assert values(left >> intersect(right), "x") == [2]
    assert values(left >> setdiff(right), "x") == [1]
    assert values(left >> symdiff(right), "x") == [1, 3]
    assert left >> setequal({"x": [2, 1]})
    assert not (left >> setequal(right))


def test_polars_phase1_frame_verbs_remain_lazy():
    frame = tidy(pl.DataFrame({"x": [1, 2], "g": ["a", "a"]}).lazy())
    results = [
        frame >> slice_head(n=1),
        frame >> add_count("g"),
        frame >> semi_join({"x": [1]}, on="x"),
        frame >> bind_rows({"x": [3], "g": ["b"]}),
        frame >> union({"g": ["a"], "x": [1]}),
    ]
    assert all(isinstance(result._lf, pl.LazyFrame) for result in results)
