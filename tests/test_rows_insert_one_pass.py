"""rows_insert() reads x once; it must match the version that joins x's keys against y."""

import random

import polars as pl
import pytest

import tidy3.verbs as verbs
from tidy3 import rows_insert, tidy

KEY_VALUES = {
    "int": ([1, 2, 3, None], pl.Int64),
    "str": (["a", "b", None, "A"], pl.String),
    "float": ([0.0, -0.0, float("nan"), 1.5, None], pl.Float64),
    "bool": ([True, False, None], pl.Boolean),
}


def _run(monkeypatch, one_pass, x, y, by, conflict):
    monkeypatch.setattr(verbs, "_ONE_PASS_INSERT", one_pass)
    try:
        out = (tidy(x) >> rows_insert(y, by=by, conflict=conflict)).collect()
    except Exception as error:  # the type and first line of the message must agree
        return type(error).__name__, str(error).splitlines()[0]
    # repr compares NaN and signed zero by their text, which == would not.
    return repr(out.to_dicts()), out.columns, out.schema


@pytest.mark.parametrize("kind", KEY_VALUES)
def test_one_pass_insert_matches_the_join_version(monkeypatch, kind):
    values, dtype = KEY_VALUES[kind]
    rng = random.Random(13)

    def table(rows):
        return pl.DataFrame(
            {
                "k": [rng.choice(values) for _ in range(rows)],
                "a": list(range(rows)),
                "extra": [float(i) for i in range(rows)],
            },
            schema={"k": dtype, "a": pl.Int64, "extra": pl.Float64},
        )

    for rows_x in (0, 1, 3, 6, 10):
        for rows_y in (0, 1, 3, 6, 10):
            x, y = table(rows_x), table(rows_y)
            for frame in (y, y.drop("extra")):  # y may lack a column of x
                for conflict in ("error", "ignore"):
                    assert _run(monkeypatch, True, x, frame, "k", conflict) == _run(
                        monkeypatch, False, x, frame, "k", conflict
                    )


def test_one_pass_insert_with_two_keys_and_missing_values(monkeypatch):
    rng = random.Random(2)

    def table(rows):
        return pl.DataFrame(
            {
                "k": [rng.choice([1, 2, None]) for _ in range(rows)],
                "j": [rng.choice(["u", "v", None]) for _ in range(rows)],
                "a": list(range(rows)),
            },
            schema={"k": pl.Int64, "j": pl.String, "a": pl.Int64},
        )

    for _ in range(40):
        x, y = table(rng.choice([0, 3, 8])), table(rng.choice([0, 3, 8]))
        for conflict in ("error", "ignore"):
            assert _run(monkeypatch, True, x, y, ["k", "j"], conflict) == _run(
                monkeypatch, False, x, y, ["k", "j"], conflict
            )


def test_repeated_new_keys_are_all_kept():
    # insert has no uniqueness rule: x=[1], y=[2, 2] gives [1, 2, 2].
    x = pl.DataFrame({"k": [1], "a": [10]})
    y = pl.DataFrame({"k": [2, 2], "a": [20, 21]})
    out = (tidy(x) >> rows_insert(y, by="k")).collect()
    assert out.to_dicts() == [{"k": 1, "a": 10}, {"k": 2, "a": 20}, {"k": 2, "a": 21}]
    # And x's own repeated keys stay put.
    x = pl.DataFrame({"k": [1, 1], "a": [10, 11]})
    out = (tidy(x) >> rows_insert(pl.DataFrame({"k": [3], "a": [30]}), by="k")).collect()
    assert out["a"].to_list() == [10, 11, 30]


def test_conflicts_raise_or_are_skipped():
    x = pl.DataFrame({"k": [1, 2], "a": [10, 20]})
    y = pl.DataFrame({"k": [2, 3, 3], "a": [21, 30, 31]})
    with pytest.raises(ValueError, match="y contains keys that already exist in x"):
        (tidy(x) >> rows_insert(y, by="k")).collect()
    out = (tidy(x) >> rows_insert(y, by="k", conflict="ignore")).collect()
    assert out.to_dicts() == [
        {"k": 1, "a": 10}, {"k": 2, "a": 20}, {"k": 3, "a": 30}, {"k": 3, "a": 31},
    ]
    # A missing key is an existing key when x has one too.
    x = pl.DataFrame({"k": [None, 1], "a": [10, 11]}, schema={"k": pl.Int64, "a": pl.Int64})
    y = pl.DataFrame({"k": [None], "a": [99]}, schema={"k": pl.Int64, "a": pl.Int64})
    with pytest.raises(ValueError, match="already exist"):
        (tidy(x) >> rows_insert(y, by="k")).collect()


def test_mismatched_key_types_still_go_through_the_join(monkeypatch):
    # Stacking would coerce the keys before comparing them; the join decides.
    x = pl.DataFrame({"k": [1, 2], "a": [1, 2]})
    y = pl.DataFrame({"k": [1.0, 5.0], "a": [3, 4]})
    results = []
    for one_pass in (True, False):
        monkeypatch.setattr(verbs, "_ONE_PASS_INSERT", one_pass)
        with pytest.raises(Exception) as caught:
            (tidy(x) >> rows_insert(y, by="k", conflict="ignore")).collect()
        results.append(type(caught.value).__name__)
    assert results[0] == results[1]


def test_chained_inserts_stay_cheap():
    frame = tidy(pl.DataFrame({"k": list(range(50)), "a": list(range(50))}))
    for i in range(1, 13):
        frame = frame >> rows_insert(pl.DataFrame({"k": [1000 + i], "a": [i]}), by="k")
    # Reading x twice per step multiplies the plan walk, which put twelve of
    # these at several seconds; one read keeps the plan to one copy of x.
    assert frame._lf.explain(optimized=False).count("DF [") <= 13 + 1
    assert frame.collect().height == 62
