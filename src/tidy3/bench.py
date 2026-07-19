"""Engine benchmark: tidy3 (polars/pandas) vs datar vs raw pandas/polars.

Runs one canonical dplyr pipeline on each available engine over the same
synthetic data and reports wall times::

    from tidy3 import bench
    bench.run(rows=1_000_000)

Ships with the package, so under CRAFT it also works on the remote kernel
after seeding (install datar there with `!uv pip install datar datar-pandas`).

The pipeline (dplyr terms)::

    filter(x > 0) >> mutate(z = x*2 + y) >> group_by(g)
    >> summarise(n=n(), avg_z=mean(z), sd_y=sd(y)) >> arrange(g)
"""

from __future__ import annotations

import time
from typing import Any, Callable

__all__ = ["run", "run_ops", "make_data"]


def make_data(rows: int, groups: int = 100, seed: int = 0):
    """Synthetic (pandas_df, polars_df) with identical contents."""
    import numpy as np
    import pandas as pd
    import polars as pl

    rng = np.random.default_rng(seed)
    pdf = pd.DataFrame(
        {
            "g": rng.integers(0, groups, rows),
            "x": rng.normal(0.0, 1.0, rows),
            "y": rng.normal(10.0, 5.0, rows),
        }
    )
    return pdf, pl.from_pandas(pdf)


# ── engine implementations (each: data → pandas DataFrame result) ───────


def _tidy3(backend: str):
    from tidy3 import arrange, col, filter, group_by, mean, mutate, n, std, summarise, tidy

    def go(data):
        return (
            tidy(data, backend=backend)
            >> filter(col("x") > 0)
            >> mutate(z=col("x") * 2 + col("y"))
            >> group_by("g")
            >> summarise(n=n(), avg_z=mean("z"), sd_y=std("y"))
            >> arrange("g")
        ).collect(as_="pandas")

    return go


def _raw_pandas(pdf):
    d = pdf[pdf["x"] > 0]
    d = d.assign(z=d["x"] * 2 + d["y"])
    out = (
        d.groupby("g", sort=False, observed=True, dropna=False)
        .agg(n=("z", "size"), avg_z=("z", "mean"), sd_y=("y", "std"))
        .reset_index()
        .sort_values("g")
        .reset_index(drop=True)
    )
    return out


def _raw_polars(pldf):
    import polars as pl

    return (
        pldf.lazy()
        .filter(pl.col("x") > 0)
        .with_columns(z=pl.col("x") * 2 + pl.col("y"))
        .group_by("g")
        .agg(n=pl.len(), avg_z=pl.col("z").mean(), sd_y=pl.col("y").std())
        .sort("g")
        .collect()
        .to_pandas()
    )


def _datar():
    """Build the datar pipeline runner, or raise ImportError."""
    import datar.all as d  # noqa: F401  (requires datar + datar-pandas)

    f = d.f
    filt = getattr(d, "filter", None) or getattr(d, "filter_")
    sd = getattr(d, "sd", None) or getattr(d, "std")

    def go(pdf):
        out = (
            pdf
            >> filt(f.x > 0)
            >> d.mutate(z=f.x * 2 + f.y)
            >> d.group_by(f.g)
            >> d.summarise(n=d.n(), avg_z=d.mean(f.z), sd_y=sd(f.y))
            >> d.arrange(f.g)
        )
        import pandas as pd

        return pd.DataFrame(out)

    return go


def _time(fn: Callable[[], Any], repeat: int) -> tuple[float, Any]:
    best, result = float("inf"), None
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = fn()
        dt = time.perf_counter() - t0
        best = dt if dt < best else best
    return best, result


def _check(name: str, ref, out) -> str:
    import numpy as np

    try:
        a = ref.sort_values("g").reset_index(drop=True)
        b = out.sort_values("g").reset_index(drop=True)
        if len(a) != len(b) or not (
            np.allclose(a["avg_z"], b["avg_z"]) and np.array_equal(a["n"], b["n"])
        ):
            return "MISMATCH"
        sd_ref = a["sd_y"].to_numpy(float)
        sd_out = b["sd_y"].to_numpy(float)
        if np.allclose(sd_ref, sd_out, equal_nan=True):
            return "ok"
        # accept population std (ddof=0) — R-style sd in datar has no ddof
        n_ = a["n"].to_numpy(float)
        pop = np.nan_to_num(sd_ref) * np.sqrt(np.maximum(n_ - 1, 0) / n_)
        if np.allclose(pop, sd_out, equal_nan=True):
            return "ok (sd ddof=0)"
        return "MISMATCH"
    except Exception as e:  # pragma: no cover
        return f"check failed: {e}"


def run(rows: int = 1_000_000, repeat: int = 3, groups: int = 100, seed: int = 0):
    """Benchmark all available engines; prints a table, returns dict of times."""
    print(f"tidy3 bench: rows={rows:,} groups={groups} repeat={repeat} (best-of)")
    pdf, pldf = make_data(rows, groups, seed)

    engines: list[tuple[str, Callable[[], Any]]] = [
        ("tidy3[polars]", lambda: _tidy3("polars")(pldf)),
        ("tidy3[pandas]", lambda: _tidy3("pandas")(pdf)),
        ("polars (raw lazy)", lambda: _raw_polars(pldf)),
        ("pandas (raw)", lambda: _raw_pandas(pdf)),
    ]
    try:
        datar_go = _datar()
        engines.append(("datar[pandas]", lambda: datar_go(pdf)))
    except ImportError as e:
        print(f"  (datar not available: {e} — pip install datar datar-pandas)")

    times: dict[str, float] = {}
    ref = None
    rows_out: list[tuple[str, float, str]] = []
    for name, fn in engines:
        t, out = _time(fn, repeat)
        times[name] = t
        if ref is None:
            ref, status = out, "ref"
        else:
            status = _check(name, ref, out)
        rows_out.append((name, t, status))

    fastest = min(times.values())
    print()
    print(f"  {'engine':<20} {'time':>10}   {'vs fastest':>10}   result")
    print(f"  {'-'*20} {'-'*10}   {'-'*10}   ------")
    for name, t, status in sorted(rows_out, key=lambda r: r[1]):
        print(f"  {name:<20} {t*1000:>8.1f}ms   {t/fastest:>9.1f}x   {status}")
    return times


def run_ops(rows: int = 10_000_000, repeat: int = 3, groups: int = 100, seed: int = 0):
    """Per-operation comparison against raw pandas (the baseline column).

    Times each verb in isolation — same input, same output — for raw
    pandas, tidy3[pandas], datar and tidy3[polars] (which includes
    engine execution + materialization, since polars is lazy).
    """
    from tidy3 import arrange, col, filter, group_by, mean, mutate, n, std, summarise, tidy

    print(f"tidy3 per-op bench: rows={rows:,} groups={groups} repeat={repeat} (best-of)")
    pdf, pldf = make_data(rows, groups, seed)

    try:
        import datar.all as d

        f = d.f
        filt = getattr(d, "filter", None) or getattr(d, "filter_")
        sd = getattr(d, "sd", None) or getattr(d, "std")
        have_datar = True
    except ImportError as e:
        print(f"  (datar not available: {e})")
        have_datar = False

    def t3p(pipe):  # tidy3[pandas]
        return lambda: pipe(tidy(pdf, backend="pandas")).collect(as_="pandas")

    def t3l(pipe):  # tidy3[polars] — includes collect (engine run)
        return lambda: pipe(tidy(pldf)).collect(as_="pandas")

    ops: list[tuple[str, dict[str, Callable[[], Any]]]] = [
        (
            "filter x>0",
            {
                "pandas": lambda: pdf[pdf["x"] > 0],
                "tidy3[pandas]": t3p(lambda tf: tf >> filter(col("x") > 0)),
                "datar": (lambda: pdf >> filt(f.x > 0)) if have_datar else None,
                "tidy3[polars]": t3l(lambda tf: tf >> filter(col("x") > 0)),
            },
        ),
        (
            "mutate z=x*2+y",
            {
                "pandas": lambda: pdf.assign(z=pdf["x"] * 2 + pdf["y"]),
                "tidy3[pandas]": t3p(lambda tf: tf >> mutate(z=col("x") * 2 + col("y"))),
                "datar": (lambda: pdf >> d.mutate(z=f.x * 2 + f.y)) if have_datar else None,
                "tidy3[polars]": t3l(lambda tf: tf >> mutate(z=col("x") * 2 + col("y"))),
            },
        ),
        (
            "group+summarise",
            {
                "pandas": lambda: pdf.groupby("g", sort=False, observed=True, dropna=False)
                .agg(n=("x", "size"), avg_x=("x", "mean"), sd_y=("y", "std"))
                .reset_index(),
                "tidy3[pandas]": t3p(
                    lambda tf: tf >> group_by("g") >> summarise(n=n(), avg_x=mean("x"), sd_y=std("y"))
                ),
                "datar": (
                    lambda: pdf >> d.group_by(f.g) >> d.summarise(n=d.n(), avg_x=d.mean(f.x), sd_y=sd(f.y))
                )
                if have_datar
                else None,
                "tidy3[polars]": t3l(
                    lambda tf: tf >> group_by("g") >> summarise(n=n(), avg_x=mean("x"), sd_y=std("y"))
                ),
            },
        ),
        (
            "arrange y",
            {
                "pandas": lambda: pdf.sort_values("y").reset_index(drop=True),
                "tidy3[pandas]": t3p(lambda tf: tf >> arrange("y")),
                "datar": (lambda: pdf >> d.arrange(f.y)) if have_datar else None,
                "tidy3[polars]": t3l(lambda tf: tf >> arrange("y")),
            },
        ),
    ]

    engines = ["pandas", "tidy3[pandas]", "datar", "tidy3[polars]"]
    print()
    header = f"  {'op':<16}" + "".join(f" {e:>20}" for e in engines)
    print(header)
    print("  " + "-" * (len(header) - 2))
    results: dict[str, dict[str, float]] = {}
    for op_name, impls in ops:
        row = f"  {op_name:<16}"
        results[op_name] = {}
        base = None
        for e in engines:
            fn = impls.get(e)
            if fn is None:
                row += f" {'—':>20}"
                continue
            t, _ = _time(fn, repeat)
            results[op_name][e] = t
            if e == "pandas":
                base = t
                row += f" {t*1000:>14.1f}ms     "
            else:
                row += f" {t*1000:>12.1f}ms {t/base:>4.1f}x"
        print(row)
    print("\n  (ratios are vs raw pandas; tidy3[polars] includes lazy-plan execution + to_pandas)")
    return results
