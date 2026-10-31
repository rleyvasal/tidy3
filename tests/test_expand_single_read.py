"""expand() reads its input once for all columns; it must match one read per column."""

import random

import polars as pl
import pytest

import tidy3.tidyr as tidyr_module
from tidy3 import complete, expand, group_by, nesting, tidy

VERBS = {
    "expand one": expand("a"),
    "expand two": expand("a", "b"),
    "expand three": expand("a", "b", "c"),
    "expand nesting": expand(nesting("a", "b")),
    "expand nesting and column": expand(nesting("a", "b"), "c"),
    "expand with named values": expand("a", b=[1, 2, 9]),
    "complete two": complete("a", "b"),
    "complete nesting and column": complete(nesting("a", "b"), "c"),
    "complete with fill": complete("a", "b", fill={"v": 0}),
    "complete with fill, implicit only": complete("a", "b", fill={"v": 0}, explicit=False),
}


def _table(rng, rows):
    return pl.DataFrame(
        {
            "a": [rng.choice(["p", "q", None, "r"]) for _ in range(rows)],
            "b": [rng.choice([1, 2, None, 3]) for _ in range(rows)],
            "c": [rng.choice([0.0, -0.0, float("nan"), 1.5, None]) for _ in range(rows)],
            "v": list(range(rows)),
        },
        schema={"a": pl.String, "b": pl.Int64, "c": pl.Float64, "v": pl.Int64},
    )


def _run(monkeypatch, one_pass, frame, verb):
    monkeypatch.setattr(tidyr_module, "_SINGLE_READ_EXPAND", one_pass)
    out = (tidy(frame) >> verb).collect()
    # repr compares NaN and signed zero by their text, which == would not.
    return repr(out.to_dicts()), out.columns, out.schema


@pytest.mark.parametrize("name", VERBS)
def test_one_pass_expand_matches_one_read_per_column(monkeypatch, name):
    rng = random.Random(4)
    for rows in (0, 1, 2, 5, 9, 14):
        for _ in range(3):
            frame = _table(rng, rows)
            assert _run(monkeypatch, True, frame, VERBS[name]) == _run(monkeypatch, False, frame, VERBS[name])


def test_expand_reads_the_input_once():
    frame = tidy(_table(random.Random(1), 20))
    for verb in (expand("a", "b", "c"), expand(nesting("a", "b"), "c")):
        plan = (frame >> verb)._lf.explain(optimized=False)
        assert plan.count("DF [") == 1, plan


def test_grouped_expand_is_unchanged(monkeypatch):
    frame = _table(random.Random(2), 12)
    results = []
    for one_pass in (True, False):
        monkeypatch.setattr(tidyr_module, "_SINGLE_READ_EXPAND", one_pass)
        results.append((tidy(frame) >> group_by("v") >> expand("a", "b")).collect())
    assert results[0].equals(results[1])
