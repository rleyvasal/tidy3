"""rows_update/patch/upsert read x once; they must match the left join plus second query."""

import itertools
import random

import polars as pl
import pytest

import tidy3.verbs as verbs
from tidy3 import group_by, rows_patch, rows_update, rows_upsert, tidy

VERBS = {"rows_update": rows_update, "rows_patch": rows_patch, "rows_upsert": rows_upsert}


def _x(rng, rows, two_keys):
    columns = {
        "k": [rng.choice([1, 2, 3, None]) for _ in range(rows)],
        "v": [rng.choice([1.5, None, 2.5]) for _ in range(rows)],
        "w": [rng.choice(["p", None, "q"]) for _ in range(rows)],
        "g": [rng.choice([1, 2]) for _ in range(rows)],
    }
    schema = {"k": pl.Int64, "v": pl.Float64, "w": pl.String, "g": pl.Int64}
    if two_keys:
        columns["j"] = [rng.choice(["u", "v", None]) for _ in range(rows)]
        schema["j"] = pl.String
    return pl.DataFrame(columns, schema=schema)


def _y(rng, rows, two_keys):
    if two_keys:
        pairs = rng.sample([(a, b) for a in (1, 2, 3, 4, None) for b in ("u", "z", None)], rows)
        return pl.DataFrame(
            {"k": [a for a, _ in pairs], "j": [b for _, b in pairs], "v": [rng.choice([9.5, None]) for _ in pairs]},
            schema={"k": pl.Int64, "j": pl.String, "v": pl.Float64},
        )
    keys = rng.sample([1, 2, 3, 4, None], rows)
    return pl.DataFrame(
        {"k": keys, "v": [rng.choice([9.5, None, 8.5]) for _ in keys], "w": [rng.choice(["P", None]) for _ in keys]},
        schema={"k": pl.Int64, "v": pl.Float64, "w": pl.String},
    )


def _run(monkeypatch, one_pass, x, y, verb, by, options, grouped, lazy):
    monkeypatch.setattr(verbs, "_ONE_PASS_ROW_CHANGES", one_pass)
    try:
        frame = tidy(x)
        if grouped:
            frame = frame >> group_by("g")
        out = frame >> VERBS[verb](y.lazy() if lazy else y, by=by, **options)
        result = out.collect()
    except Exception as error:
        return type(error).__name__, str(error).splitlines()[0]
    return repr(result.to_dicts()), result.columns, result.schema, tuple(out._groups or ())


@pytest.mark.parametrize("verb", VERBS)
@pytest.mark.parametrize("lazy", [False, True], ids=["table", "lazy"])
def test_one_pass_matches_the_old_path(monkeypatch, verb, lazy):
    rng = random.Random(31)
    for two_keys in (False, True):
        by = ["k", "j"] if two_keys else "k"
        for rows_x, rows_y in itertools.product((0, 1, 3, 7, 12), (0, 1, 2, 4)):
            x, y = _x(rng, rows_x, two_keys), _y(rng, rows_y, two_keys)
            for options in ({}, {"unmatched": "ignore"}):
                if verb == "rows_upsert" and options:
                    continue
                for grouped in (False, True):
                    args = (x, y, verb, by, options, grouped, lazy)
                    assert _run(monkeypatch, True, *args) == _run(monkeypatch, False, *args)


@pytest.mark.parametrize("verb", VERBS)
def test_repeated_y_keys_and_type_differences_are_unchanged(monkeypatch, verb):
    rng = random.Random(5)
    x = _x(rng, 6, False)
    repeated = pl.concat([_y(rng, 3, False), _y(rng, 3, False).head(1)])
    other_type = _y(rng, 3, False).with_columns(pl.col("v").cast(pl.Float32))
    for y in (repeated, other_type):
        args = (x, y, verb, "k", {}, False, False)
        assert _run(monkeypatch, True, *args) == _run(monkeypatch, False, *args)


def test_what_each_verb_does():
    x = pl.DataFrame({"k": [1, 2, 2, None], "v": [10, 20, 21, 30], "w": ["a", "b", "c", "d"]})
    y = pl.DataFrame({"k": [2, None], "v": [None, 99]})
    update = (tidy(x) >> rows_update(y, by="k")).collect()
    assert update["v"].to_list() == [10, None, None, 99]  # y's missing value overwrites
    patch = (tidy(x) >> rows_patch(y, by="k")).collect()
    assert patch["v"].to_list() == [10, 20, 21, 30]  # only x's missing values are filled
    upsert = (tidy(x) >> rows_upsert(pl.DataFrame({"k": [2, 7], "v": [5, 6]}), by="k")).collect()
    assert upsert.to_dicts()[-1] == {"k": 7, "v": 6, "w": None}
    assert upsert["v"].to_list() == [10, 5, 5, 30, 6]
    with pytest.raises(ValueError, match="y contains keys absent from x"):
        (tidy(x) >> rows_update(pl.DataFrame({"k": [9], "v": [1]}), by="k")).collect()
    ignored = (tidy(x) >> rows_update(pl.DataFrame({"k": [9], "v": [1]}), by="k", unmatched="ignore")).collect()
    assert ignored["v"].to_list() == [10, 20, 21, 30]


def test_chained_changes_read_x_once():
    frame = tidy(pl.DataFrame({"k": list(range(20)), "v": [0.0] * 20}))
    for i in range(1, 9):
        frame = frame >> rows_update(pl.DataFrame({"k": [i], "v": [float(i)]}), by="k")
    assert frame._lf.explain(optimized=False).count("DF [") <= 9 + 1
    assert frame.collect()["v"].to_list()[:9] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
