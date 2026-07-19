from __future__ import annotations

from tidy3 import bench_suite


def test_comprehensive_benchmark_smoke_and_equivalence():
    results = bench_suite.run(rows=1_000, repeat=1, warmup=0, output="native")
    assert len(results) == 12
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
