from polars.testing import assert_frame_equal
import pytest

from tidy3.bench_polars import closest_comparison, closest_inputs, closest_runners, memory_comparison


@pytest.mark.parametrize("groups", [0, 5])
def test_closest_benchmark_compares_complete_outputs_in_original_order(groups):
    case = {"name": "smoke", "rows": 120, "lookup_rows": 12,
            "width": 3 if groups else 5, "groups": groups}
    source, _, _, _, _ = closest_inputs(case)
    functions = closest_runners(case)
    native = functions["raw_polars"]()
    assert native["x"].to_list() == source["x"].to_list()
    assert_frame_equal(native, functions["tidy3"]())
    if not groups:
        assert native["y"].to_list() == [x // 10 * 10 for x in source["x"]]
    report = closest_comparison(case, repeat=2, warmup=0)
    assert len(report["samples_ms"]["raw_polars"]) == 2
    assert len(report["samples_ms"]["tidy3"]) == 2
    assert report["raw_polars_ms"] > 0
    assert report["tidy3_ms"] > 0


def test_closest_memory_workers_materialize_the_same_number_of_rows():
    import sys

    if sys.platform == "win32":
        pytest.skip("RSS measurements use macOS/Linux getrusage")
    case = {"name": "smoke", "rows": 120, "lookup_rows": 12, "width": 5}
    report = memory_comparison(case)
    assert report["raw_polars"]["output_rows"] == 120
    assert report["tidy3"]["output_rows"] == 120
    assert report["raw_polars"]["peak_rss_mib"] > 0
    assert report["tidy3"]["peak_rss_mib"] > 0
