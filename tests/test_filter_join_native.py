"""Equality semi_join/anti_join with join_by use Polars' own; they must match the general path."""

import itertools
import random

import polars as pl
import pytest

import tidy3.verbs as verbs
from tidy3 import anti_join, eq, group_by, join_by, semi_join, tidy

ENUM = pl.Enum(["u", "v", "w"])
KINDS = {
    "int": ([1, 2, 3, None], pl.Int64),
    "str": (["a", "b", None, "A"], pl.String),
    "float": ([0.0, -0.0, float("nan"), 1.5, None], pl.Float64),
    "enum": (["u", "v", None], ENUM),
    "categorical": (["u", "v", None], pl.Categorical),
}
JOINS = {"semi": semi_join, "anti": anti_join}


def _table(rng, kind, rows, value, key="k"):
    values, dtype = KINDS[kind]
    return pl.DataFrame(
        {key: [rng.choice(values) for _ in range(rows)], value: list(range(rows))},
        schema={key: dtype, value: pl.Int64},
    )


def _run(monkeypatch, native, x, y, how, spec, na_matches, grouped):
    monkeypatch.setattr(verbs, "_NATIVE_FILTER_JOINS", native)
    try:
        frame = tidy(x)
        if grouped:
            frame = frame >> group_by("a")
        out = frame >> JOINS[how](y, on=spec, na_matches=na_matches)
        result = out.collect()
    except Exception as error:
        return type(error).__name__, str(error).splitlines()[0]
    # repr compares NaN and signed zero by their text, which == would not.
    return repr(result.to_dicts()), result.columns, result.schema, tuple(out._groups or ())


@pytest.mark.parametrize("kind", KINDS)
def test_native_semi_and_anti_match_the_general_path(monkeypatch, kind):
    rng = random.Random(17)
    for rows_x, rows_y in itertools.product((0, 1, 4, 9), (0, 1, 4, 9)):
        renamed = rng.random() < 0.5
        x = _table(rng, kind, rows_x, "a")
        y = _table(rng, kind, rows_y, "b", key="m" if renamed else "k")
        spec = join_by(eq("k", "m")) if renamed else join_by("k")
        for how, na_matches, grouped in itertools.product(JOINS, ("na", "never"), (False, True)):
            assert _run(monkeypatch, True, x, y, how, spec, na_matches, grouped) == _run(
                monkeypatch, False, x, y, how, spec, na_matches, grouped
            ), (kind, how, na_matches, grouped)


def test_native_semi_and_anti_with_two_keys(monkeypatch):
    rng = random.Random(3)

    def table(rows, value):
        return pl.DataFrame(
            {
                "k": [rng.choice([1, 2, None]) for _ in range(rows)],
                "j": [rng.choice(["u", "v", None]) for _ in range(rows)],
                value: list(range(rows)),
            },
            schema={"k": pl.Int64, "j": pl.String, value: pl.Int64},
        )

    for _ in range(30):
        x, y = table(rng.choice([0, 4, 9]), "a"), table(rng.choice([0, 4, 9]), "b")
        for how, na_matches in itertools.product(JOINS, ("na", "never")):
            for spec in (join_by("k", "j"), join_by("k")):
                assert _run(monkeypatch, True, x, y, how, spec, na_matches, False) == _run(
                    monkeypatch, False, x, y, how, spec, na_matches, False
                )


def test_plain_keys_with_na_matches_never_use_the_same_path(monkeypatch):
    x = pl.DataFrame({"k": [1, None, 3], "a": [1, 2, 3]})
    y = pl.DataFrame({"k": [None, 3]})
    for how, expected in (("semi", [3]), ("anti", [1, 2])):
        out = (tidy(x) >> JOINS[how](y, on="k", na_matches="never")).collect()
        assert out["a"].to_list() == expected
        monkeypatch.setattr(verbs, "_NATIVE_FILTER_JOINS", False)
        slow = (tidy(x) >> JOINS[how](y, on="k", na_matches="never")).collect()
        monkeypatch.setattr(verbs, "_NATIVE_FILTER_JOINS", True)
        assert slow.equals(out)


def test_errors_for_missing_key_columns_are_unchanged(monkeypatch):
    x = pl.DataFrame({"k": [1, 2], "a": [1, 2]})
    for native in (True, False):
        monkeypatch.setattr(verbs, "_NATIVE_FILTER_JOINS", native)
        with pytest.raises(KeyError):
            (tidy(x) >> semi_join(pl.DataFrame({"z": [1]}), on=join_by("k"))).collect()
        with pytest.raises(KeyError):
            (tidy(x) >> anti_join(pl.DataFrame({"k": [1]}), on=join_by(eq("q", "k")))).collect()
