from __future__ import annotations

import pytest

from tidy3 import (
    coalesce,
    col,
    collect,
    distinct,
    mean,
    mutate,
    starts_with,
    std,
    tidy,
)
from tidy3.bench_suite import geometric_ratio
from tidy3.verbs import _aggregate_cache_plan


BACKENDS = ["polars", "pandas"]


def as_pandas(frame):
    return frame.collect(as_="pandas")


def test_grouped_aggregate_cache_extracts_nested_and_repeated_statistics():
    assignments = {
        "filled": coalesce(col("x"), mean("x")),
        "z": (coalesce(col("x"), mean("x")) - mean("x")) / std("x"),
    }
    cached, rewritten, columns = _aggregate_cache_plan(assignments, ["g", "x"])
    assert len(cached) == 2
    assert len(columns) == 2
    assert all(name.startswith("__tidy3_agg_") for name in columns)
    assert "__tidy3_agg_" in repr(rewritten["z"])


@pytest.mark.parametrize("backend", BACKENDS)
def test_grouped_aggregate_reuse_preserves_mutate_results_and_schema(backend):
    frame = tidy(
        {"g": ["a", "a", "b", "b"], "x": [1.0, None, 2.0, 6.0]},
        backend=backend,
    )
    result = frame >> mutate(
        filled=coalesce(col("x"), mean("x", na_rm=True)),
        centered=(
            coalesce(col("x"), mean("x", na_rm=True))
            - mean("x", na_rm=True)
        ),
        scaled=(
            coalesce(col("x"), mean("x", na_rm=True))
            - mean("x", na_rm=True)
        )
        / std("x", na_rm=True),
        by="g",
    )
    out = as_pandas(result)
    assert list(out.columns) == ["g", "x", "filled", "centered", "scaled"]
    assert out["filled"].tolist() == [1.0, 1.0, 2.0, 6.0]
    assert out["centered"].tolist() == [0.0, 0.0, -2.0, 2.0]
    assert not any("__tidy3_agg_" in column for column in out.columns)
    assert result._groups is None


def test_unknown_polars_window_methods_remain_group_aware():
    frame = tidy({"g": ["a", "a", "b", "b"], "x": [1.0, 3.0, 2.0, 6.0]})
    out = as_pandas(
        frame >> mutate(rolling=col("x").rolling_mean(window_size=2), by="g")
    )
    assert out["rolling"].isna().tolist() == [True, False, True, False]
    assert out["rolling"].dropna().tolist() == [2.0, 4.0]


@pytest.mark.parametrize("backend", BACKENDS)
def test_projected_collection_supports_tidyselect(backend):
    frame = tidy(
        {"id": [1, 2], "x_one": [3, 4], "x_two": [5, 6]},
        backend=backend,
    )
    projected = frame.collect(as_="pandas", columns=starts_with("x_"))
    assert projected.to_dict(orient="list") == {
        "x_one": [3, 4],
        "x_two": [5, 6],
    }
    piped = frame >> collect(as_="pandas", columns=["id", "x_two"])
    assert list(piped.columns) == ["id", "x_two"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_arrow_backed_pandas_collection(backend):
    frame = tidy({"id": [1, 2], "value": [1.5, 2.5]}, backend=backend)
    out = frame.collect(as_="pandas", arrow_backed=True)
    assert out.to_dict(orient="list") == {
        "id": [1, 2],
        "value": [1.5, 2.5],
    }
    assert all("pyarrow" in str(dtype) for dtype in out.dtypes)
    piped = frame >> collect(
        as_="pandas", columns=["value"], arrow_backed=True
    )
    assert list(piped.columns) == ["value"]
    assert "pyarrow" in str(piped["value"].dtype)


def test_arrow_backed_requires_pandas_output():
    with pytest.raises(ValueError, match="requires as_='pandas'"):
        tidy({"x": [1]}).collect(as_="polars", arrow_backed=True)


@pytest.mark.parametrize("backend", BACKENDS)
def test_unordered_distinct_preserves_rows_without_order_contract(backend):
    frame = tidy(
        {"id": [2, 1, 2, 3], "value": ["a", "b", "c", "d"]},
        backend=backend,
    )
    out = as_pandas(frame >> distinct("id", maintain_order=False))
    assert set(out["id"]) == {1, 2, 3}
    assert len(out) == 3


def test_geometric_performance_ratio_respects_minimum_workload_time():
    results = {
        "small": {
            "pandas": {"median_seconds": 0.001, "vs_pandas": 1.0},
            "tidy3[pandas]": {"median_seconds": 0.002, "vs_pandas": 2.0},
        },
        "large": {
            "pandas": {"median_seconds": 0.1, "vs_pandas": 1.0},
            "tidy3[pandas]": {"median_seconds": 0.11, "vs_pandas": 1.1},
        },
    }
    assert geometric_ratio(
        results, "tidy3[pandas]", minimum_pandas_ms=10
    ) == pytest.approx(1.1)
