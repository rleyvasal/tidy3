"""Speed guards for column functions.

Two kinds of check. The structural ones are exact and never flaky: common
text shapes convert natively, without the per-value Python rules, on both
backends. The timing ones compare each function with
the native operation that does the same job, timed on the same machine in
the same run, so a function that falls back to a Python loop (100 times
slower or more) fails while ordinary machine noise does not.

Set TIDY3_SKIP_SPEED=1 to skip the timing checks.
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable

import numpy as np
import pandas as pd
import polars as pl
import pytest

import tidy3 as t3
from tidy3 import mutate, tidy

BACKENDS = ("polars", "pandas")

# Common shapes per function. Number and logical cases include a failure,
# which the grammar settles natively; date cases hold only valid dates,
# since an impossible date is rechecked by the per-value rules on purpose.
CANONICAL: list[tuple[str, Callable[[], Any], list[Any]]] = [
    ("as_double", lambda: t3.as_double("x", strict=False), ["1.5", "abc", None]),
    ("as_integer", lambda: t3.as_integer("x", strict=False), ["12", "1.5", None]),
    ("parse_double", lambda: t3.parse_double("x"), ["1.5", "1,5", None]),
    ("parse_integer", lambda: t3.parse_integer("x"), ["1234", "1,234", None]),
    ("parse_number", lambda: t3.parse_number("x"), ["$1,234.50", "50%", None]),
    ("as_logical", lambda: t3.as_logical("x", strict=False), ["TRUE", "yes", None]),
    ("parse_logical", lambda: t3.parse_logical("x"), ["true", "1", None]),
    ("as_date", lambda: t3.as_date("x", strict=False), ["2024-01-07", "2024-12-31", None]),
    ("ymd", lambda: t3.ymd("x"), ["2024-01-07", "2024/01/07", None]),
    ("mdy", lambda: t3.mdy("x"), ["01/07/2024", "01-07-2024", None]),
    ("dmy", lambda: t3.dmy("x"), ["07/01/2024", "07.01.2024", None]),
    ("ym", lambda: t3.ym("x"), ["2024-01", "2024/02", None]),
    ("parse_date", lambda: t3.parse_date("x", "%d/%m/%Y"), ["07/01/2024", "29/02/2024", None]),
    ("parse_datetime", lambda: t3.parse_datetime("x", "%Y-%m-%d %H:%M:%S"), ["2024-01-07 13:05:09", None]),
    ("as_datetime", lambda: t3.as_datetime("x", strict=False), ["2024-01-07 13:05:09", "2024-01-07", None]),
    ("ymd_hms", lambda: t3.ymd_hms("x"), ["2024-01-07 13:05:09", "2024/01/07T13:05:09", None]),
    ("ymd_hm", lambda: t3.ymd_hm("x"), ["2024-01-07 13:05", None]),
    ("mdy_hms", lambda: t3.mdy_hms("x"), ["01/07/2024 13:05:09", None]),
    ("dmy_hms", lambda: t3.dmy_hms("x", tz="America/New_York"), ["07/01/2024 13:05:09", None]),
]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name, build, values", CANONICAL, ids=[case[0] for case in CANONICAL])
def test_common_shapes_skip_the_per_value_rules(monkeypatch, backend, name, build, values):
    from tidy3 import convert

    def per_value(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(f"{name}() used the per-value Python rules")

    monkeypatch.setattr(convert, "_per_value", per_value)
    monkeypatch.setattr(convert, "convert_batch", per_value)
    frame = tidy({"x": values}, backend=backend) >> mutate(y=build())
    frame.collect()
    if backend == "polars":
        t3.problems(frame).collect()


@pytest.mark.parametrize("backend", BACKENDS)
def test_text_of_numbers_skips_the_per_value_rules(monkeypatch, backend):
    from tidy3 import convert

    monkeypatch.setattr(convert, "_character", lambda *a, **k: (_ for _ in ()).throw(AssertionError))
    frame = tidy({"x": [1.5, None, 3.0], "b": [True, False, None]}, backend=backend)
    (frame >> mutate(x=t3.as_string("x"), b=t3.as_string("b"))).collect()


# ── timing ──────────────────────────────────────────────────────────────

ROWS = 200_000
# A function may cost this many times its native equivalent, or this many
# seconds, whichever is larger. Per-value Python loops measured 100 to 900
# times native before the native paths; these budgets leave room for slow
# and noisy machines while still catching that kind of slide.
RATIO = 30
FLOOR = 0.15

timing = pytest.mark.skipif(
    os.environ.get("TIDY3_SKIP_SPEED") == "1", reason="TIDY3_SKIP_SPEED=1"
)


def _best(run: Callable[[], Any], repeat: int = 3) -> float:
    best = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        run()
        best = min(best, time.perf_counter() - start)
    return best


@pytest.fixture(scope="module")
def data() -> pl.DataFrame:
    rng = np.random.default_rng(0)
    ints = rng.integers(0, 10_000, ROWS)
    month, day = ints % 12 + 1, ints % 28 + 1
    return pl.DataFrame(
        {
            "i": [str(v) for v in ints],
            "num": [f"${v:,}.50" for v in ints],
            "dbl": [f"{v}.25" for v in ints],
            "lgl": ["true" if v % 2 else "false" for v in ints],
            "date": [f"2024-{m:02d}-{d:02d}" for m, d in zip(month, day)],
            "mdy": [f"{m:02d}/{d:02d}/2024" for m, d in zip(month, day)],
            "dt": [f"2024-{m:02d}-{d:02d} {v % 24:02d}:{v % 60:02d}:{v % 60:02d}" for m, d, v in zip(month, day, ints)],
            "x": ints.astype(float),
            "y": ints.astype(float) / 2,
        }
    )


TIMED: list[tuple[str, Callable[[], Any], Callable[[pl.DataFrame], Any]]] = [
    ("parse_integer", lambda: t3.parse_integer("i"), lambda d: d.select(pl.col("i").cast(pl.Int64))),
    ("parse_number", lambda: t3.parse_number("num"), lambda d: d.select(pl.col("num").str.replace_all(r"[^0-9.]", "").cast(pl.Float64))),
    ("as_double", lambda: t3.as_double("dbl"), lambda d: d.select(pl.col("dbl").cast(pl.Float64))),
    ("parse_logical", lambda: t3.parse_logical("lgl"), lambda d: d.select(pl.col("lgl").str.to_lowercase() == "true")),
    ("as_date", lambda: t3.as_date("date"), lambda d: d.select(pl.col("date").str.to_date("%Y-%m-%d"))),
    ("ymd", lambda: t3.ymd("date"), lambda d: d.select(pl.col("date").str.to_date("%Y-%m-%d"))),
    ("mdy", lambda: t3.mdy("mdy"), lambda d: d.select(pl.col("mdy").str.to_date("%m/%d/%Y"))),
    ("parse_datetime", lambda: t3.parse_datetime("dt", "%Y-%m-%d %H:%M:%S"), lambda d: d.select(pl.col("dt").str.to_datetime("%Y-%m-%d %H:%M:%S"))),
    ("as_datetime", lambda: t3.as_datetime("dt"), lambda d: d.select(pl.col("dt").str.to_datetime("%Y-%m-%d %H:%M:%S"))),
    ("ymd_hms", lambda: t3.ymd_hms("dt"), lambda d: d.select(pl.col("dt").str.to_datetime("%Y-%m-%d %H:%M:%S", time_zone="UTC"))),
    ("as_string", lambda: t3.as_string("x"), lambda d: d.select(pl.col("x").cast(pl.String))),
    ("is_na", lambda: t3.is_na("x"), lambda d: d.select(pl.col("x").is_null() | pl.col("x").is_nan())),
    ("pmin", lambda: t3.pmin("x", "y"), lambda d: d.select(pl.min_horizontal("x", "y"))),
    ("row_sums", lambda: t3.row_sums("x", "y", na_rm=True), lambda d: d.select(pl.sum_horizontal("x", "y"))),
    ("round", lambda: t3.round("x", 1), lambda d: d.select(pl.col("x").round(1))),
]


@timing
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name, build, native", TIMED, ids=[case[0] for case in TIMED])
def test_function_speed_is_close_to_native(data, backend, name, build, native):
    frame = tidy(data if backend == "polars" else data.to_pandas(), backend=backend)
    # Match native(data): materialize only the converted column on either backend.
    elapsed = _best(lambda: (frame >> mutate(out=build())).collect(columns="out"))
    baseline = _best(lambda: native(data))
    budget = max(RATIO * baseline, FLOOR)
    assert elapsed <= budget, (
        f"{name}() on {backend} took {elapsed * 1000:.0f} ms for {ROWS:,} rows; "
        f"native takes {baseline * 1000:.1f} ms, budget {budget * 1000:.0f} ms"
    )


@timing
def test_factor_on_a_lazy_scan_builds_quickly(tmp_path):
    path = tmp_path / "penguins.csv"
    species = np.array(["Adelie", "Gentoo", "Chinstrap"])[np.arange(ROWS) % 3]
    pl.DataFrame({"species": species, "mass": np.arange(ROWS, dtype=float)}).write_csv(path)
    build = _best(lambda: t3.scan_csv(path) >> mutate(species=t3.fct_infreq("species")))
    # A factor mutation snapshots the frame, so the baseline reads it all.
    read = _best(lambda: pl.scan_csv(path).collect())
    budget = max(RATIO * read, FLOOR)
    assert build <= budget, f"fct_infreq() build took {build * 1000:.0f} ms; budget {budget * 1000:.0f} ms"
