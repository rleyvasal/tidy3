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
