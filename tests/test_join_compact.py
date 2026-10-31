"""Equality join_by joins with options run as one join; they must match the general path."""

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
