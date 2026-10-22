"""Every public function is classified, and each case matches raw Polars."""

import pytest

from tidy3.bench_api import classification, run_case, work


def test_every_export_and_method_is_classified():
    info = classification()
    assert info["duplicates"] == []
    assert info["unmapped"] == []
    assert info["unmapped_methods"] == []


@pytest.mark.parametrize(
    "name",
    [case.name for case in classification()["cases"]],
)
def test_case_matches_polars(name):
    case = next(item for item in classification()["cases"] if item.name == name)
    run_case(case, work(24), repeat=1, warmup=0)


def test_same_compares_nested_frames_quickly_and_still_finds_differences():
    import polars as pl
    import pytest

    from tidy3.bench_api import same

    nested = pl.DataFrame({
        "g": [1, 2],
        "data": [[{"s": "a", "m": 1.0}, {"s": "b", "m": float("nan")}], []],
    })
    same(nested, nested.clone())
    changed = nested.with_columns(
        pl.col("data").list.eval(pl.element().struct.with_fields(pl.field("m").fill_nan(2.0)))
    )
    with pytest.raises(AssertionError):
        same(nested, changed)
    with pytest.raises(AssertionError):
        same([nested], [nested.head(1)])
