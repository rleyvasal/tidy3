"""slice_sample(weight_by=..., replace=True): draws follow the weights, and rows of
weight 0 are never drawn. A zero-weight row has the same running total as the row
before it, and the tie used to be left to the order of a join, so up to a quarter
of the draws went to such rows."""

import builtins

import polars as pl
import pytest

from tidy3 import group_by, slice_sample, tidy


def _frame(rows=400, equal=False):
    # Every other row has weight 0; the rest weigh 1, or 1, 2 and 3 in turn.
    return pl.DataFrame(
        {
            "id": list(range(rows)),
            "g": [(i // 2) % 4 for i in range(rows)],
            "w": [
                (1.0 if equal else float(1 + (i // 2) % 3)) if i % 2 == 0 else 0.0
                for i in range(rows)
            ],
        }
    )


def _zero_weight_draws():
    """Zero-weight rows drawn over a grid of sizes; which ones showed the bug
    depended on the draw count, grouping and thread count, so try them all."""
    drawn = 0
    for equal in (False, True):
        data = _frame(equal=equal)
        for grouped in (False, True):
            frame = tidy(data) >> group_by("g") if grouped else tidy(data)
            for n in (50, 200, 5_000):
                for seed in range(3):
                    out = (frame >> slice_sample(n=n, weight_by="w", replace=True, seed=seed)).collect()
                    drawn += out.filter(pl.col("w") == 0).height
    return drawn


def test_zero_weight_rows_are_never_drawn():
    assert _zero_weight_draws() == 0


def test_zero_weight_rows_are_never_drawn_on_two_threads():
    import os
    import subprocess
    import sys

    script = (
        "import sys; sys.path.insert(0, %r)\n"
        "from test_weighted_sample_replace import _zero_weight_draws\n"
        "print(_zero_weight_draws())\n" % os.path.dirname(__file__)
    )
    done = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={**os.environ, "POLARS_MAX_THREADS": "2"},
    )
    assert done.returncode == 0, done.stderr[-400:]
    assert done.stdout.strip() == "0", done.stdout


@pytest.mark.parametrize("grouped", [False, True], ids=["ungrouped", "grouped"])
def test_draws_follow_the_weights(grouped):
    data = _frame()
    frame = tidy(data) >> group_by("g") if grouped else tidy(data)
    out = (frame >> slice_sample(n=50_000, weight_by="w", replace=True, seed=5)).collect()
    drawn = dict(out.group_by("w").len().iter_rows())
    mass = {weight: weight * count for weight, count in data.filter(pl.col("w") > 0).group_by("w").len().iter_rows()}
    total = builtins.sum(mass.values())
    for weight, share in mass.items():
        assert abs(drawn.get(weight, 0) / out.height - share / total) < 0.01, (weight, drawn)


def test_leading_and_trailing_zero_weights():
    data = pl.DataFrame({"id": [0, 1, 2, 3, 4], "w": [0.0, 0.0, 5.0, 0.0, 0.0]})
    out = (tidy(data) >> slice_sample(n=100, weight_by="w", replace=True, seed=1)).collect()
    assert set(out["id"].to_list()) == {2}


def test_seeded_draws_repeat():
    data = _frame(60)
    runs = [
        (tidy(data) >> slice_sample(n=25, weight_by="w", replace=True, seed=9)).collect()["id"].to_list()
        for _ in range(3)
    ]
    assert runs[0] == runs[1] == runs[2]
