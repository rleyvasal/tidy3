"""symdiff() stacks the two distinct inputs and drops repeats; it must match the two anti joins."""

import datetime as dt
import random

import polars as pl
import pytest

import tidy3.verbs as verbs
from tidy3 import group_by, symdiff, tidy

POOLS = {
    "int": ([1, 2, 3, None], pl.Int64),
    "str": (["a", "b", None, "A"], pl.String),
    "float": ([0.0, -0.0, float("nan"), 1.5, None], pl.Float64),
    "bool": ([True, False, None], pl.Boolean),
    "date": ([dt.date(2024, 1, 1), dt.date(2024, 1, 2), None], pl.Date),
}


def _table(rng, kinds, rows):
    columns, schema = {}, {}
    for position, kind in enumerate(kinds):
        values, dtype = POOLS[kind]
        name = "abcd"[position]
        columns[name] = [rng.choice(values) for _ in range(rows)]
        schema[name] = dtype
    return pl.DataFrame(columns, schema=schema)


def _run(monkeypatch, one_pass, x, y, grouped):
    monkeypatch.setattr(verbs, "_ONE_PASS_SYMDIFF", one_pass)
    try:
        frame = tidy(x)
        if grouped:
            frame = frame >> group_by("a")
        out = frame >> symdiff(y)
        result = out.collect()
    except Exception as error:
        return type(error).__name__, str(error).splitlines()[0]
    # repr compares NaN and signed zero by their text, which == would not.
    return repr(result.to_dicts()), result.columns, result.schema, tuple(out._groups or ())


@pytest.mark.parametrize(
    "kinds",
    [["int"], ["str"], ["float"], ["bool"], ["date"], ["int", "str"], ["float", "float"], ["str", "date", "bool", "int"]],
    ids="-".join,
)
def test_one_pass_symdiff_matches_the_anti_joins(monkeypatch, kinds):
    rng = random.Random(23)
    for rows_x in (0, 1, 3, 8, 14):
        for rows_y in (0, 1, 3, 8, 14):
            x, y = _table(rng, kinds, rows_x), _table(rng, kinds, rows_y)
            for other in (y, y.select(y.columns[::-1])):  # y's columns may come in another order
                for grouped in (False, True):
                    assert _run(monkeypatch, True, x, other, grouped) == _run(monkeypatch, False, x, other, grouped)


def test_differing_column_types_still_go_through_the_joins(monkeypatch):
    x = pl.DataFrame({"a": [1, 2], "b": ["u", "v"]})
    y = pl.DataFrame({"a": ["1", "3"], "b": ["u", "w"]})
    assert _run(monkeypatch, True, x, y, False) == _run(monkeypatch, False, x, y, False)


def test_rows_in_both_inputs_are_dropped_and_each_side_keeps_its_order():
    x = pl.DataFrame({"a": [3, 1, 2, 2, None]})
    y = pl.DataFrame({"a": [None, 5, 2, 4, 5]})
    out = (tidy(x) >> symdiff(y)).collect()
    assert out["a"].to_list() == [3, 1, 5, 4]


def test_symdiff_reads_each_input_once():
    frame = tidy(pl.DataFrame({"a": range(10)}))
    other = pl.DataFrame({"a": range(5, 15)})
    for _ in range(6):
        frame = frame >> symdiff(other)
    assert frame._lf.explain(optimized=False).count("DF [") <= 7
