"""group_modify() runs a verb pipeline once over the grouped frame.

Each case checks the traced one-pass result against the per-group path,
and that the path taken is the one expected.
"""

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import tidy3 as t3
import tidy3.group_trace as group_trace
import tidy3.verbs as verbs

BASE = pl.DataFrame({
    "g": [2, 1, 2, None, 1, 2, 1],
    "h": ["b", "a", "a", "a", "a", "b", None],
    "x": [5, 4, 3, 2, 1, 6, 7],
    "y": [1.5, None, 2.5, 3.5, 0.5, 2.0, 1.0],
    "s": list("pqrstuv"),
})


def _both_paths(monkeypatch, frame, fn, *cols):
    """The result with the trace and without it, and whether the trace was used."""
    parts = []
    original = verbs._grouped_parts
    with monkeypatch.context() as patched:
        patched.setattr(verbs, "_grouped_parts", lambda *a, **k: parts.append(1) or original(*a, **k))
        traced = (frame >> t3.group_modify(fn, *cols)).collect()
    used_trace = not parts
    with monkeypatch.context() as patched:
        patched.setattr(group_trace, "one_pass", lambda *a, **k: None)
        per_group = (frame >> t3.group_modify(fn, *cols)).collect()
    return traced, per_group, used_trace


ONE_PASS = {
    "slice_head": lambda p, k: p >> t3.slice_head(n=1),
    "slice_tail": lambda p, k: p >> t3.slice_tail(n=2),
    "head": lambda p, k: p.head(2),
    "filter with a summary": lambda p, k: p >> t3.filter(t3.col("x") > t3.mean("x")),
    "window mutate": lambda p, k: p >> t3.mutate(
        n=t3.n(), r=t3.row_number(), l=t3.lag("x"), z=t3.col("x") - t3.mean("x")
    ),
    "mutate the key": lambda p, k: p >> t3.mutate(g=t3.col("g") * 10),
    "rename the key": lambda p, k: p >> t3.rename(gg="g"),
    "select away the key": lambda p, k: p >> t3.select("x", "s"),
    "transmute": lambda p, k: p >> t3.transmute(d=t3.col("x") * 2),
    "summarise": lambda p, k: p >> t3.summarise(m=t3.mean("x"), n=t3.n()),
    "count": lambda p, k: p >> t3.count(),
    "count a column": lambda p, k: p >> t3.count("h"),
    "add_count": lambda p, k: p >> t3.add_count("h"),
    "distinct": lambda p, k: p >> t3.distinct("h"),
    "drop_na": lambda p, k: p >> t3.drop_na(),
    "arrange": lambda p, k: p >> t3.arrange(t3.desc("x")),
    "arrange then head": lambda p, k: p >> t3.arrange("y") >> t3.head(1),
    "slice_max": lambda p, k: p >> t3.slice_max("x", n=1),
    "relocate": lambda p, k: p >> t3.relocate("s"),
    "method chain": lambda p, k: p.filter(t3.col("x") > 1).mutate(w=t3.col("x") + 1),
    "identity": lambda p, k: p,
    "columns read": lambda p, k: p >> t3.select(p.columns[0], "x"),
}


@pytest.mark.parametrize("name", list(ONE_PASS))
def test_pipeline_callbacks_run_in_one_pass(monkeypatch, name):
    frame = t3.tidy(BASE) >> t3.group_by("g")
    traced, per_group, used_trace = _both_paths(monkeypatch, frame, ONE_PASS[name])
    assert used_trace
    assert_frame_equal(traced, per_group, check_dtypes=False)


PER_GROUP = {
    "key read": lambda p, k: p >> t3.mutate(k=k["g"]),
    "collect": lambda p, k: t3.tidy(p.collect().head(1)),
    "pandas": lambda p, k: p.collect(as_="pandas").head(1),
    "cur_group_rows": lambda p, k: p >> t3.mutate(first=t3.cur_group_rows()[0]),
    "height": lambda p, k: p >> t3.slice_head(n=1) if p.height > 2 else p,
    "bind_rows": lambda p, k: p >> t3.bind_rows(p),
    "pivot_wider": lambda p, k: p >> t3.pivot_wider(names_from="h", values_from="x"),
}


@pytest.mark.parametrize("name", list(PER_GROUP))
def test_other_callbacks_run_per_group(monkeypatch, name):
    frame = t3.tidy(BASE) >> t3.group_by("g")
    traced, per_group, used_trace = _both_paths(monkeypatch, frame, PER_GROUP[name])
    assert not used_trace
    assert_frame_equal(traced, per_group, check_dtypes=False)


def test_two_keys_and_explicit_columns(monkeypatch):
    fn = ONE_PASS["slice_head"]
    traced, per_group, used_trace = _both_paths(monkeypatch, t3.tidy(BASE) >> t3.group_by("g", "h"), fn)
    assert used_trace
    assert_frame_equal(traced, per_group)
    traced, per_group, used_trace = _both_paths(monkeypatch, t3.tidy(BASE), fn, "h")
    assert used_trace
    assert_frame_equal(traced, per_group)


def test_float_and_factor_keys_run_per_group(monkeypatch):
    fn = ONE_PASS["slice_head"]
    traced, per_group, used_trace = _both_paths(monkeypatch, t3.tidy(BASE) >> t3.group_by("y"), fn)
    assert not used_trace
    assert_frame_equal(traced, per_group)
    factor = t3.tidy(BASE.with_columns(pl.col("h").cast(pl.Enum(["b", "a"])))) >> t3.group_by("h")
    traced, per_group, used_trace = _both_paths(monkeypatch, factor, fn)
    assert not used_trace
    assert_frame_equal(traced, per_group)


def test_order_is_by_key_then_the_pipeline():
    out = (t3.tidy(BASE) >> t3.group_by("g") >> t3.group_modify(lambda p, k: p >> t3.arrange("x"))).collect()
    assert out["g"].to_list() == [1, 1, 1, 2, 2, 2, None]
    assert out["x"].to_list() == [1, 4, 7, 3, 5, 6, 2]


def test_empty_frame_keeps_its_columns():
    out = (t3.tidy(BASE.head(0)) >> t3.group_by("g") >> t3.group_modify(ONE_PASS["slice_head"])).collect()
    assert out.height == 0
    assert out.columns == BASE.columns


def test_trace_never_computes_or_prints(capsys):
    seen = []

    def spy(p, k):
        seen.append(p)
        return p >> t3.peek()

    out = (t3.tidy(BASE) >> t3.group_by("g") >> t3.group_modify(spy)).collect()
    # peek() is not a one-pass verb: the trace stopped before applying it,
    # so nothing was printed for the probe, and each group then ran.
    assert out.height == BASE.height
    assert len(seen) == 4
    printed = capsys.readouterr().out
    assert printed.count("shape") == 3
