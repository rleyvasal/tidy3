"""Polars-relative rolling-join benchmarks.

Run ``python -m tidy3.bench_polars --memory --output report.json``.
Inputs are built before timing, outputs are checked including row order and
dtypes, and engine order alternates between repetitions. Memory is measured
separately, in one fresh process per engine, and includes input construction.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal


CLOSEST_CASES = [
    {"name": "original_sweep", "rows": 200_000, "lookup_rows": 100, "width": 30},
    {"name": "wide_200k", "rows": 200_000, "lookup_rows": 1_000, "width": 21},
    {"name": "wide_1m_sparse", "rows": 1_000_000, "lookup_rows": 1_000, "width": 21},
    {"name": "wide_1m_dense", "rows": 1_000_000, "lookup_rows": 100_000, "width": 21},
    {"name": "grouped_1m", "rows": 1_000_000, "lookup_rows": 100_000, "width": 3, "groups": 1_000},
]


def _original_sweep():
    """Reproduce the original 30-column workload without session files."""
    rows = 200_000
    rng = np.random.default_rng(42)
    i = np.arange(rows)
    species = np.array(["Adelie", "Gentoo", "Chinstrap"])[i % 3]
    island = np.array(["Torgersen", "Biscoe", "Dream"])[(i // 3) % 3]
    sex = np.array(["male", "female", None], dtype=object)[rng.integers(0, 3, rows)]
    mass = rng.normal(4200, 800, rows).round(1)
    mask = rng.random(rows) < .03
    at = [datetime(2024, 1, 1) + timedelta(minutes=int(v)) for v in rng.integers(0, 525_600, rows)]
    day = [a.date() for a in at]
    data = pl.DataFrame({
        "id": i, "g": i % 1000, "species": species, "island": island,
        "sex": sex, "year": 2007 + i % 3,
        "mass": pl.Series(mass).set(pl.Series(mask), None),
        "bill": rng.normal(44, 5, rows).round(1),
        "flipper": rng.integers(170, 230, rows),
        "m1": rng.normal(0, 1, rows), "m2": rng.normal(0, 1, rows), "m3": rng.normal(0, 1, rows),
        "name": [f"{s} from {isl} number {k}" for s, isl, k in zip(species, island, i)],
        "csv": [f"a{k % 7},b{k % 5},c{k % 3}" for k in i],
        "code": [f"AB{k % 97:02d}X{k % 1000:04d}" for k in i],
        "num_txt": [f"${k:,}.50" for k in rng.integers(0, 100_000, rows)],
        "int_txt": [str(k) for k in rng.integers(0, 100_000, rows)],
        "dbl_txt": [f"{k}.25" for k in rng.integers(0, 100_000, rows)],
        "lgl_txt": ["true" if k else "false" for k in i % 2],
        "date_txt": [d.isoformat() for d in day],
        "mdy_txt": [d.strftime("%m/%d/%Y") for d in day],
        "dmy_txt": [d.strftime("%d/%m/%Y") for d in day],
        "ym_txt": [d.strftime("%Y-%m") for d in day],
        "dt_txt": [a.strftime("%Y-%m-%d %H:%M:%S") for a in at],
        "day": day, "at": pl.Series(at, dtype=pl.Datetime("us")),
        "serial": rng.integers(30_000, 46_000, rows).astype(float), "w": 1 + i % 2,
    }).with_columns(
        lst=pl.int_ranges(0, pl.col("id") % 3 + 1),
        rec=pl.struct(a=pl.col("id") % 5, b=pl.col("species")),
    )
    right = pl.DataFrame({
        "lo": np.arange(0, 1000, 10) * (rows // 1000),
        "hi": np.arange(9, 1000, 10) * (rows // 1000) + 5,
        "band": np.arange(100),
    })
    return data, right, "id", "lo", []


def closest_inputs(case):
    if case["name"] == "original_sweep":
        return _original_sweep()
    rows, lookups = case["rows"], case["lookup_rows"]
    if case.get("groups"):
        data = pl.DataFrame({"x": np.arange(rows), "g": np.arange(rows) % case["groups"], "h": np.arange(rows) % 3})
        right = data.select(pl.col("x").alias("y"), "g", "h").filter(pl.col("y") % (rows // lookups) == 0)
        return data, right, "x", "y", ["g", "h"]
    data = pl.DataFrame({"x": np.arange(rows, dtype=np.int64)[::-1]}).with_columns(
        *((pl.col("x") * (i + 1)).alias(f"payload_{i}") for i in range(case["width"] - 1)),
    )
    right = pl.DataFrame({"y": np.arange(lookups, dtype=np.int64) * (rows // lookups), "label": np.arange(lookups)})
    return data, right, "x", "y", []


def closest_runners(case):
    import tidy3 as t

    data, right, left_key, right_key, groups = closest_inputs(case)
    frame = t.tidy(data)
    spec = t.join_by(*groups, t.closest(t.ge(left_key, right_key)))

    def raw_polars():
        # These workloads have unique lookup keys. Preserve original input
        # order, all payload columns, and the same complete Polars output.
        return (
            data.lazy().with_row_index("__bench_row").sort(left_key)
            .join_asof(right.lazy().sort(right_key), left_on=left_key,
                       right_on=right_key, by=groups or None,
                       coalesce=False, check_sortedness=False)
            .sort("__bench_row").drop("__bench_row").collect()
        )

    return {"raw_polars": raw_polars, "tidy3": lambda: (frame >> t.left_join(right, by=spec)).collect()}


def closest_comparison(case, *, repeat=3, warmup=1):
    functions = closest_runners(case)
    assert_frame_equal(functions["raw_polars"](), functions["tidy3"]())
    for _ in range(warmup):
        for function in functions.values():
            function()
    samples = {name: [] for name in functions}
    names = list(functions)
    for iteration in range(repeat):
        for name in names[iteration % 2:] + names[:iteration % 2]:
            start = time.perf_counter()
            result = functions[name]()
            samples[name].append((time.perf_counter() - start) * 1000)
            del result
    best = {name: min(values) for name, values in samples.items()}
    return {
        **case, "raw_polars_ms": best["raw_polars"], "tidy3_ms": best["tidy3"],
        "tidy3_over_polars": best["tidy3"] / best["raw_polars"],
        "median_ms": {name: statistics.median(values) for name, values in samples.items()},
        "samples_ms": samples,
    }


def _memory_worker(case, engine):
    import resource

    functions = closest_runners(case)
    result = functions[engine]()
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = peak if sys.platform == "darwin" else peak * 1024
    return {"peak_rss_mib": peak_bytes / 1024**2, "output_rows": result.height}


def memory_comparison(case):
    results = {}
    for engine in ("raw_polars", "tidy3"):
        completed = subprocess.run(
            [sys.executable, "-m", "tidy3.bench_polars", "--memory-worker",
             json.dumps({"case": case, "engine": engine})],
            check=True, capture_output=True, text=True,
        )
        results[engine] = json.loads(completed.stdout)
    return results


def markdown_report(report):
    lines = [
        "# Polars-relative closest() timings", "",
        f"Python {report['python']}; Polars {report['polars']}; {report['platform']}.", "",
        f"One correctness comparison, {report['warmup']} warm-up(s), best of {report['repeat']} alternating timing rounds.",
        "Input construction and validation are excluded from timing. Both engines materialize the same complete Polars frame, including input row order and dtypes.", "",
        "| Workload | Rows | Lookup rows | Width | Raw Polars | tidy3 | tidy3 / Polars |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["comparisons"]:
        lines.append(f"| {row['name']} | {row['rows']:,} | {row['lookup_rows']:,} | {row['width']} | {row['raw_polars_ms']:.2f} ms | {row['tidy3_ms']:.2f} ms | {row['tidy3_over_polars']:.2f}x |")
    if report["memory"]:
        lines += ["", "Peak resident memory is measured separately in fresh processes, once per engine. It includes interpreter imports, input construction, execution and the complete output; it is not incremental operator allocation. These RSS measurements require macOS or Linux.", "",
                  "| Workload | Raw Polars peak RSS | tidy3 peak RSS |",
                  "|---|---:|---:|"]
        for row in report["comparisons"]:
            memory = row["memory"]
            lines.append(f"| {row['name']} | {memory['raw_polars']['peak_rss_mib']:.1f} MiB | {memory['tidy3']['peak_rss_mib']:.1f} MiB |")
    lines += ["", "These measured ratios are separate from CI regression limits. The benchmark covers unique lookup keys and default `multiple=\"all\"`; duplicate-match correctness is covered by the rolling-join tests.", ""]
    return "\n".join(lines)


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", choices=[c["name"] for c in CLOSEST_CASES])
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--memory", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("rolling-joins-polars.json"))
    parser.add_argument("--memory-worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.memory_worker:
        print(json.dumps(_memory_worker(**json.loads(args.memory_worker))))
        return
    if args.repeat < 1 or args.warmup < 0:
        parser.error("repeat must be positive and warmup non-negative")
    report = {
        "python": platform.python_version(), "polars": pl.__version__,
        "platform": f"{platform.system()} {platform.machine()}",
        "repeat": args.repeat, "warmup": args.warmup, "memory": args.memory,
        "comparisons": [],
    }
    for case in CLOSEST_CASES:
        if args.case and case["name"] not in args.case:
            continue
        row = closest_comparison(case, repeat=args.repeat, warmup=args.warmup)
        print(f"{case['name']}: raw Polars {row['raw_polars_ms']:.2f} ms, tidy3 {row['tidy3_ms']:.2f} ms ({row['tidy3_over_polars']:.2f}x)", flush=True)
        if args.memory:
            row["memory"] = memory_comparison(case)
        report["comparisons"].append(row)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    args.output.with_suffix(".md").write_text(markdown_report(report))


if __name__ == "__main__":
    _main()
