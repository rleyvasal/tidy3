"""On the Polars backend, verbs work in Polars: no detour through pandas.

Each call below runs with tidy3's collect-to-pandas and Polars' to_pandas()
blocked; Polars-native collects are allowed.
"""

from __future__ import annotations

import pandas as pd
import polars as pl
import pytest

import tidy3 as t3
from tidy3.frame import TidyFrame


@pytest.fixture
def no_pandas(monkeypatch):
    original = TidyFrame.collect

    def collect(self, *args, **kwargs):
        if kwargs.get("as_") == "pandas" or (args and args[0] == "pandas"):
            raise AssertionError("converted a Polars frame to pandas")
        return original(self, *args, **kwargs)

    def refuse(*_args, **_kwargs):
        raise AssertionError("converted a Polars frame to pandas")

    monkeypatch.setattr(TidyFrame, "collect", collect)
    monkeypatch.setattr(pl.DataFrame, "to_pandas", refuse)
    yield
    monkeypatch.undo()


def finish(value):
    """Force a lazy result to run (inside the no-pandas block)."""
    return value._lf.collect() if isinstance(value, TidyFrame) else value


CODES = t3.tidy({"id": [1, 2, 3], "code": ["ab-12", "cd-3", None], "date": ["20240115", "20231201", None]})
INFO = t3.tidy({"id": [1, 2], "info": [{"name": "x", "n": "1", "k": 5}, {"name": "y", "n": "2", "k": 6}]})
SALES = t3.tidy({"g": ["b", "a", "b", "a"], "x": [1.0, 2.0, 3.0, 4.0]})
FACTORS = t3.tidy(
    pd.DataFrame({"k": pd.Categorical(["x", "x", None], categories=["x", "y"]), "g": ["b", "a", "b"], "v": [1.0, 2.0, 3.0]})
)

CALLS = {
    "separate_wider_delim": lambda: CODES >> t3.separate_wider_delim("code", ["l", "n"], "-", too_few="align_start"),
    "separate_wider_regex": lambda: CODES >> t3.separate_wider_regex("code", [("l", "[a-z]+"), "-", ("n", "[0-9]+")]),
    "separate_wider_position": lambda: CODES >> t3.separate_wider_position("date", [("y", 4), ("m", 2), ("d", 2)]),
    "separate_longer_delim": lambda: CODES >> t3.separate_longer_delim("code", "-"),
    "separate_longer_position": lambda: CODES >> t3.separate_longer_position("code", 2),
    "hoist": lambda: INFO >> t3.hoist("info", nm="name", n="n", transform={"n": int}),
    "expand_grid": lambda: t3.expand_grid(SALES, z=[1, 2]),
    "crossing": lambda: t3.crossing(x=[3, 1, None], y=["b", "a"]),
    "complete_full_seq": lambda: t3.tidy({"c": ["A", "B"], "year": [1952, 1962]}) >> t3.complete("c", year=t3.full_seq("year", 5)),
    "group_data": lambda: t3.group_data(SALES >> t3.group_by("g")),
    "group_keys": lambda: t3.group_keys(FACTORS >> t3.group_by("g", "k", drop=False)),
    "group_rows": lambda: t3.group_rows(SALES >> t3.group_by("g")),
    "group_indices": lambda: t3.group_indices(SALES >> t3.group_by("g")),
    "group_trim": lambda: t3.group_trim(FACTORS >> t3.group_by("k", drop=False)),
    "group_split": lambda: [finish(p) for p in SALES >> t3.group_by("g") >> t3.group_split()],
    "group_map": lambda: SALES >> t3.group_by("g") >> t3.group_map(lambda part, key: finish(part).height),
    "group_modify": lambda: SALES >> t3.group_by("g") >> t3.group_modify(lambda part, key: part >> t3.slice_head(n=1)),
    "nest_by": lambda: SALES >> t3.nest_by("g"),
    "summarise_drop_false": lambda: FACTORS >> t3.group_by("g", "k", drop=False) >> t3.summarise(n=t3.n()),
    "count_drop_false": lambda: FACTORS >> t3.count("k", "g", drop=False),
    "setequal": lambda: SALES >> t3.setequal(SALES >> t3.arrange("x")),
}


@pytest.mark.parametrize("name", list(CALLS))
def test_polars_backend_stays_in_polars(no_pandas, name):
    result = CALLS[name]()
    if isinstance(result, list):
        assert result is not None
    else:
        finish(result)
