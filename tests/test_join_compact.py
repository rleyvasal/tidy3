"""Equality join_by joins with options run as one join; they must match the general path.

The general path (matched row ids, then re-attached) stays for inequality, rolling
and semi/anti joins and is what these tests compare against."""

import itertools
import random

import polars as pl
import pytest

import tidy3.verbs as verbs
from tidy3 import (
    eq,
    full_join,
    inner_join,
    join_by,
    left_join,
    right_join,
    tidy,
)

JOINS = {"left": left_join, "inner": inner_join, "right": right_join, "full": full_join}
RELATIONSHIPS = [None, "one-to-one", "one-to-many", "many-to-one", "many-to-many"]


def _run(compact, monkeypatch, x, y, how, spec, **kwargs):
    monkeypatch.setattr(verbs, "_COMPACT_EQUALITY_JOINS", compact)
    try:
        return "ok", (tidy(x) >> JOINS[how](y, on=spec, **kwargs)).collect()
    except ValueError as error:
        return "error", str(error).splitlines()[0]


def _table(rng, size, key, value, keys):
    return pl.DataFrame(
        {
            key: [rng.choice([*keys, None]) for _ in range(size)],
            value: list(range(size)),
        },
        schema={key: pl.String, value: pl.Int64},
    )


def test_compact_join_matches_general_path(monkeypatch):
    rng = random.Random(3)
    for _ in range(25):
        x = _table(rng, rng.choice([0, 1, 4, 7]), "k", "a", ["p", "q", "r"])
        renamed = rng.random() < 0.5
        y = _table(rng, rng.choice([0, 1, 4, 7]), "j" if renamed else "k", "b", ["p", "q", "r", "s"])
        spec = join_by(eq("k", "j")) if renamed else join_by("k")
        for how, relationship, na, keep in itertools.product(
            JOINS, RELATIONSHIPS, ["na", "never"], [False, True]
        ):
            options = {"relationship": relationship, "na_matches": na, "keep": keep}
            general = _run(False, monkeypatch, x, y, how, spec, **options)
            compact = _run(True, monkeypatch, x, y, how, spec, **options)
            assert general[0] == compact[0], (how, options)
            if general[0] == "ok":
                assert compact[1].columns == general[1].columns
                assert compact[1].equals(general[1]), (how, options)
            else:
                # With both directions violated either message may come first.
                assert "violated" in compact[1], (how, options)


def test_compact_join_with_two_keys_and_groups(monkeypatch):
    x = pl.DataFrame({"k1": ["p", "p", "q"], "k2": [1, 2, 1], "a": [1, 2, 3]})
    y = pl.DataFrame({"k1": ["p", "q", "q"], "k2": [1, 1, 2], "b": [4, 5, 6]})
    results = []
    for compact in (False, True):
        monkeypatch.setattr(verbs, "_COMPACT_EQUALITY_JOINS", compact)
        grouped = tidy(x).group_by("k1")
        results.append(
            (grouped >> left_join(y, on=join_by("k1", "k2"), relationship="one-to-one")).collect()
        )
    assert results[0].equals(results[1])
    assert results[0].columns == results[1].columns


def test_relationship_errors_name_the_side():
    x = pl.DataFrame({"k": [1, 1], "a": [1, 2]})
    y = pl.DataFrame({"k": [1, 2], "b": [3, 4]})
    with pytest.raises(ValueError, match="a row in x matches multiple rows in y"):
        (tidy(y) >> left_join(x, on=join_by("k"), relationship="one-to-one")).collect()
    with pytest.raises(ValueError, match="a row in y matches multiple rows in x"):
        (tidy(x) >> left_join(y, on=join_by("k"), relationship="one-to-one")).collect()
    with pytest.raises(ValueError, match="a row in x matches multiple rows in y"):
        (tidy(y) >> inner_join(x, on=join_by("k"), relationship="many-to-one")).collect()
    with pytest.raises(ValueError, match="a row in y matches multiple rows in x"):
        (tidy(x) >> inner_join(y, on=join_by("k"), relationship="one-to-many")).collect()


def test_repeated_keys_without_a_partner_do_not_violate():
    # Polars' own validate= would reject this; dplyr and tidy3 only object to
    # a row that really matches more than one row.
    x = pl.DataFrame({"k": ["p", "p", "q"], "a": [1, 2, 3]})
    y = pl.DataFrame({"k": ["q", "s", "s"], "b": [4, 5, 6]})
    out = (tidy(x) >> inner_join(y, on=join_by("k"), relationship="one-to-one")).collect()
    assert out.to_dict(as_series=False) == {"k": ["q"], "a": [3], "b": [4]}
    left = (tidy(x) >> left_join(y, on=join_by("k"), relationship="one-to-one")).collect()
    assert left.height == 3


def test_missing_keys_match_only_when_na_matches_says_so():
    x = pl.DataFrame({"k": [None, None], "a": [1, 2]}, schema={"k": pl.String, "a": pl.Int64})
    y = pl.DataFrame({"k": [None], "b": [3]}, schema={"k": pl.String, "b": pl.Int64})
    # Two missing keys in x both match y's one missing key.
    both = tidy(x) >> left_join(y, on=join_by("k"), relationship="many-to-one", na_matches="na")
    assert both.collect()["b"].to_list() == [3, 3]
    with pytest.raises(ValueError, match="a row in y matches multiple rows in x"):
        (tidy(x) >> left_join(y, on=join_by("k"), relationship="one-to-many", na_matches="na")).collect()
    # Never matching missing keys leaves nothing to violate.
    never = tidy(x) >> left_join(y, on=join_by("k"), relationship="one-to-one", na_matches="never")
    assert never.collect()["b"].to_list() == [None, None]


def test_chained_checked_joins_keep_the_plan_small():
    base = tidy(pl.DataFrame({"id": range(20), "v": range(20)}))
    frame = base
    for i in range(4):
        other = pl.DataFrame({"id": range(0, 20, 2), f"w{i}": range(10)})
        frame = frame >> left_join(other, on=join_by("id"), relationship="one-to-one")
    # The general path repeats the whole earlier plan in every check, which
    # grows past 800,000 characters for these four joins.
    assert len(frame._lf.explain(optimized=False)) < 50_000
    assert frame.collect().height == 20


def test_compact_join_matches_general_path_for_multiple_and_unmatched(monkeypatch):
    rng = random.Random(8)
    for _ in range(12):
        x = _table(rng, rng.choice([0, 2, 5, 8]), "k", "a", ["p", "q", "r"])
        y = _table(rng, rng.choice([0, 2, 5, 8]), "k", "b", ["p", "q", "r", "s"])
        for how, relationship, multiple, unmatched in itertools.product(
            JOINS, [None, "many-to-one"], ["all", "first", "last", "any"], ["drop", "error"]
        ):
            if rng.random() > 0.4:
                continue
            options = {"relationship": relationship, "multiple": multiple, "unmatched": unmatched}
            general = _run(False, monkeypatch, x, y, how, join_by("k"), **options)
            compact = _run(True, monkeypatch, x, y, how, join_by("k"), **options)
            assert general[0] == compact[0], (how, options)
            if general[0] == "ok":
                assert compact[1].columns == general[1].columns
                assert compact[1].equals(general[1]), (how, options)
            else:
                assert "violated" in compact[1] or "unmatched" in compact[1], (how, options)


def test_unmatched_error_names_the_side():
    x = pl.DataFrame({"k": [1, 2], "a": [1, 2]})
    y = pl.DataFrame({"k": [1, 3], "b": [3, 4]})
    with pytest.raises(ValueError, match="rows in x have no match in y"):
        (tidy(x) >> inner_join(y, on=join_by("k"), unmatched="error")).collect()
    with pytest.raises(ValueError, match="rows in y have no match in x"):
        (tidy(x) >> left_join(y, on=join_by("k"), unmatched="error")).collect()
    with pytest.raises(ValueError, match="rows in x have no match in y"):
        (tidy(x) >> right_join(y, on=join_by("k"), unmatched="error")).collect()
    # A full join keeps every row, so nothing can be unmatched.
    out = (tidy(x) >> full_join(y, on=join_by("k"), unmatched="error")).collect()
    assert out.height == 3


def test_multiple_first_leaves_orphaned_y_rows_unmatched_as_in_dplyr():
    # Checked against dplyr 1.2.1: y's second row for key 1 loses its only
    # partner, so it is appended as an unmatched row of y.
    x = pl.DataFrame({"k": [1, 2], "a": [10, 20]})
    y = pl.DataFrame({"k": [1, 1, 3], "b": [1, 2, 3]})
    full = (tidy(x) >> full_join(y, on=join_by("k"), multiple="first")).collect()
    assert full.to_dicts() == [
        {"k": 1, "a": 10, "b": 1},
        {"k": 2, "a": 20, "b": None},
        {"k": 1, "a": None, "b": 2},
        {"k": 3, "a": None, "b": 3},
    ]
    right = (tidy(x) >> right_join(y, on=join_by("k"), multiple="first")).collect()
    assert right.to_dicts() == [
        {"k": 1, "a": 10, "b": 1},
        {"k": 1, "a": None, "b": 2},
        {"k": 3, "a": None, "b": 3},
    ]
    last = (tidy(x) >> right_join(y, on=join_by("k"), multiple="last")).collect()
    assert last.to_dicts() == [
        {"k": 1, "a": 10, "b": 2},
        {"k": 1, "a": None, "b": 1},
        {"k": 3, "a": None, "b": 3},
    ]


def test_multiple_picks_first_or_last_y_row_with_duplicate_keys(monkeypatch):
    x = pl.DataFrame({"k": [2, 1, 9, 1], "a": [0, 1, 2, 3]})
    y = pl.DataFrame({"k": [1, 2, 1, 1, 2], "b": [10, 20, 11, 12, 21]})
    first = (tidy(x) >> left_join(y, on=join_by("k"), multiple="first")).collect()
    last = (tidy(x) >> left_join(y, on=join_by("k"), multiple="last")).collect()
    assert first["b"].to_list() == [20, 10, None, 10]
    assert last["b"].to_list() == [21, 12, None, 12]
    inner = (tidy(x) >> inner_join(y, on=join_by("k"), multiple="first")).collect()
    assert inner["a"].to_list() == [0, 1, 3]
    # The same rows come out when the general path builds every pair first.
    monkeypatch.setattr(verbs, "_COMPACT_EQUALITY_JOINS", False)
    assert (tidy(x) >> left_join(y, on=join_by("k"), multiple="first")).collect().equals(first)
    assert (tidy(x) >> left_join(y, on=join_by("k"), multiple="last")).collect().equals(last)


def test_multiple_with_float_keys_that_compare_equal_but_differ_in_bits(monkeypatch):
    x = pl.DataFrame({"k": [0.0, -0.0, float("nan"), None, 1.5], "a": [0, 1, 2, 3, 4]})
    y = pl.DataFrame({"k": [-0.0, 0.0, float("nan"), float("nan"), None, None, 1.5], "b": list(range(7))})
    for how in ("left", "inner"):
        for multiple in ("first", "last"):
            for na in ("na", "never"):
                options = {"multiple": multiple, "na_matches": na}
                general = _run(False, monkeypatch, x, y, how, join_by("k"), **options)
                compact = _run(True, monkeypatch, x, y, how, join_by("k"), **options)
                assert repr(compact[1].to_dicts()) == repr(general[1].to_dicts()), (how, options)
                # One row of y at most for each row of x.
                assert compact[1].height <= x.height


def test_checks_still_see_every_row_of_y_when_multiple_is_set():
    x = pl.DataFrame({"k": [1, 2], "a": [1, 2]})
    y = pl.DataFrame({"k": [1, 1, 2], "b": [1, 2, 3]})
    # many-to-one would be violated by x's row 1 matching two rows of y; picking
    # one row of y first must not hide that.
    with pytest.raises(ValueError, match="a row in x matches multiple rows in y"):
        (tidy(x) >> left_join(y, on=join_by("k"), multiple="first", relationship="many-to-one")).collect()
    # A y row that is never the first match is still a matched row of y.
    ok = tidy(x) >> left_join(y, on=join_by("k"), multiple="first", unmatched="error")
    assert ok.collect()["b"].to_list() == [1, 3]


def test_multiple_first_and_last_do_not_depend_on_the_thread_count():
    # The general path used to leave the first or last match of a row of x to the
    # order the joins produced, which changes with the number of threads (Polars 2
    # on two threads returned the other match every time). Run it in a process
    # with two threads, on the general path (inequality and equality joins).
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(
        """
        import polars as pl
        import tidy3.verbs as verbs
        from tidy3 import ge, join_by, left_join, tidy

        x = pl.DataFrame({"k": [2, 1, 9, 1], "a": [0, 1, 2, 3], "t": [5, 15, 25, 15]})
        y = pl.DataFrame({"k": [1, 2, 1, 1, 2], "b": [10, 20, 11, 12, 21], "lo": [0, 0, 10, 20, 30]})
        for compact in (True, False):
            verbs._COMPACT_EQUALITY_JOINS = compact
            for multiple, want in (("first", [20, 10, None, 10]), ("last", [21, 12, None, 12])):
                got = (tidy(x) >> left_join(y, on=join_by("k"), multiple=multiple)).collect()["b"].to_list()
                assert got == want, ("equality", compact, multiple, got)
        # An inequality join: the first match is the lowest row of y, the last the highest.
        # t = 5, 15, 25, 15 match y rows (b = 10, 20), (10, 20, 11), (10, 20, 11, 12), (10, 20, 11).
        on = join_by(ge("t", "lo"))
        first = (tidy(x) >> left_join(y.select("lo", "b"), on=on, multiple="first")).collect()["b"].to_list()
        last = (tidy(x) >> left_join(y.select("lo", "b"), on=on, multiple="last")).collect()["b"].to_list()
        assert first == [10, 10, 10, 10], first
        assert last == [20, 11, 12, 11], last
        print("ok")
        """
    )
    for threads in ("1", "2", "3"):
        done = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env={**__import__("os").environ, "POLARS_MAX_THREADS": threads},
        )
        assert done.returncode == 0 and done.stdout.strip() == "ok", (threads, done.stdout, done.stderr[-400:])
