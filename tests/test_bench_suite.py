from __future__ import annotations

from tidy3 import bench_suite


def test_comprehensive_benchmark_smoke_and_equivalence():
    results = bench_suite.run(rows=1_000, repeat=1, warmup=0, output="native")
    assert len(results) == 15
    assert set(results["ML feature matrix"]) == {
        "pandas",
        "tidy3[pandas]",
        "tidy3[polars]",
    }
    assert all(
        values["median_seconds"] >= 0
        for workload in results.values()
        for values in workload.values()
    )


def test_comprehensive_benchmark_arrow_pandas_handoff():
    results = bench_suite.run(
        rows=1_000,
        repeat=1,
        warmup=0,
        output="pandas-arrow",
        match="mutate 4 features",
    )
    assert set(results) == {"mutate 4 features"}


def test_comprehensive_benchmark_streaming_engine():
    results = bench_suite.run(
        rows=1_000,
        repeat=1,
        warmup=0,
        output="native",
        match="filter + select",
        polars_engine="streaming",
    )
    assert set(results) == {"filter + select"}


def test_comprehensive_benchmark_materialization_workloads():
    results = bench_suite.run(
        rows=1_000,
        repeat=1,
        warmup=0,
        output="native",
        match="group callback,group nest",
    )
    assert set(results) == {"group callback summarize", "group nest and count"}


def test_comprehensive_benchmark_rejects_unknown_engine():
    try:
        bench_suite.run(rows=10, polars_engine="turbo")
    except ValueError as error:
        assert "polars_engine" in str(error)
    else:
        raise AssertionError("unknown Polars engine should fail")
