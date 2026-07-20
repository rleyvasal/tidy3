"""Comprehensive tidy3 performance suite against raw pandas.

The suite times both everyday operations and multi-step analytical/ML-style
workflows. Data generation, warm-up, and correctness validation are excluded
from the recorded wall time. By default every engine returns a pandas frame,
so the Polars result includes the cost of the common ML handoff::

    python -m tidy3.bench_suite --rows 1000000 --repeat 5

Use ``--output native`` to keep the Polars result as a Polars DataFrame, or
``--output pandas-arrow`` for a zero-copy-friendly pandas handoff backed by
Arrow extension arrays.
The smaller canonical benchmark in ``tidy3.bench`` retains optional datar
coverage as a semantic/performance backstop; datar is intentionally not in
this primary adoption benchmark.
"""

from __future__ import annotations

import argparse
import gc
import math
import statistics
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
import polars as pl

__all__ = ["geometric_ratio", "make_realistic_data", "run"]


@dataclass(frozen=True)
class Workload:
    category: str
    name: str
    pandas: Callable[[], Any]
    tidy_pandas: Callable[[], Any]
    tidy_polars: Callable[[], Any]


def make_realistic_data(rows: int, seed: int = 0):
    """Create repeatable event/fact data plus a customer dimension table."""
    if rows < 1:
        raise ValueError("rows must be positive")
    rng = np.random.default_rng(seed)
    customers = max(1_000, min(100_000, rows // 40))
    customer_id = rng.integers(0, customers, rows, dtype=np.int64)
    amount = rng.lognormal(mean=3.3, sigma=0.9, size=rows)
    feature_a = rng.normal(size=rows)
    feature_a[rng.random(rows) < 0.08] = np.nan
    pdf = pd.DataFrame(
        {
            "customer_id": customer_id,
            "segment": customer_id % 20,
            "channel": rng.integers(0, 6, rows, dtype=np.int16),
            "event_time": np.arange(rows, dtype=np.int64),
            "amount": amount,
            "cost": amount * rng.uniform(0.35, 0.9, rows),
            "discount": rng.beta(2.0, 12.0, rows),
            "feature_a": feature_a,
            "feature_b": rng.normal(5.0, 2.0, rows),
            "score": rng.normal(size=rows),
            "active": rng.random(rows) < 0.82,
        }
    )
    customer = np.arange(customers, dtype=np.int64)
    dimension = pd.DataFrame(
        {
            "customer_id": customer,
            "region": customer % 12,
            "tier": customer % 4,
            "risk": rng.uniform(size=customers),
        }
    )
    return pdf, pl.from_pandas(pdf), dimension, pl.from_pandas(dimension)


def _as_pandas(value: Any) -> pd.DataFrame:
    if isinstance(value, pl.DataFrame):
        return value.to_pandas()
    return pd.DataFrame(value)


def _assert_equivalent(reference: Any, actual: Any, workload: str) -> None:
    left = _as_pandas(reference).reset_index(drop=True)
    right = _as_pandas(actual).reset_index(drop=True)
    if list(left.columns) != list(right.columns) or left.shape != right.shape:
        raise AssertionError(
            f"{workload}: shape/columns differ: "
            f"{left.shape} {list(left.columns)} != {right.shape} {list(right.columns)}"
        )
    for column in left.columns:
        a, b = left[column], right[column]
        if pd.api.types.is_numeric_dtype(a.dtype) and pd.api.types.is_numeric_dtype(
            b.dtype
        ):
            if not np.allclose(
                a.to_numpy(dtype=float, na_value=np.nan),
                b.to_numpy(dtype=float, na_value=np.nan),
                rtol=1e-8,
                atol=1e-10,
                equal_nan=True,
            ):
                raise AssertionError(f"{workload}: numeric column {column!r} differs")
        else:
            a_values = a.astype("string").fillna("<NA>").tolist()
            b_values = b.astype("string").fillna("<NA>").tolist()
            if a_values != b_values:
                raise AssertionError(f"{workload}: column {column!r} differs")


def _build_workloads(
    pdf: pd.DataFrame,
    pldf: pl.DataFrame,
    dimension: pd.DataFrame,
    pl_dimension: pl.DataFrame,
    *,
    output: str,
    polars_engine: Any,
) -> list[Workload]:
    from tidy3 import (
        arrange,
        coalesce,
        col,
        cummean,
        desc,
        distinct,
        filter,
        group_modify,
        group_nest,
        if_else,
        lag,
        left_join,
        max,
        mean,
        mutate,
        n,
        row_number,
        select,
        std,
        sum,
        summarise,
        tidy,
    )

    def collect_polars(frame, *, columns=None):
        if output == "native":
            return frame.collect(
                as_="polars", columns=columns, engine=polars_engine
            )
        return frame.collect(
            as_="pandas",
            columns=columns,
            arrow_backed=output == "pandas-arrow",
            engine=polars_engine,
        )

    def tp(pipe):
        return lambda: pipe(tidy(pdf, backend="pandas")).collect(as_="pandas")

    def tl(pipe):
        return lambda: collect_polars(pipe(tidy(pldf)))

    selected = ["customer_id", "segment", "amount", "score"]

    def raw_filter():
        mask = (pdf["active"] & (pdf["amount"] > 40) & (pdf["score"] > 0))
        return pdf.loc[mask, selected].reset_index(drop=True)

    filter_pipe = lambda frame: (
        frame
        >> filter(
            col("active") & (col("amount") > 40) & (col("score") > 0)
        )
        >> select(*selected)
    )

    def raw_mutate():
        return pdf.assign(
            net=pdf["amount"] * (1 - pdf["discount"]),
            margin=pdf["amount"] - pdf["cost"],
            log_amount=np.log(pdf["amount"] + 1),
            positive=(pdf["score"] > 0).astype(np.int8),
        )

    mutate_pipe = lambda frame: frame >> mutate(
        net=col("amount") * (1 - col("discount")),
        margin=col("amount") - col("cost"),
        log_amount=(col("amount") + 1).log(),
        positive=if_else(col("score") > 0, 1, 0),
    )

    projected_columns = ["customer_id", "net", "margin", "positive"]

    def raw_projected_features():
        return pd.DataFrame(
            {
                "customer_id": pdf["customer_id"],
                "net": pdf["amount"] * (1 - pdf["discount"]),
                "margin": pdf["amount"] - pdf["cost"],
                "positive": (pdf["score"] > 0).astype(np.int8),
            }
        )

    projection_pipe = lambda frame: frame >> mutate(
        net=col("amount") * (1 - col("discount")),
        margin=col("amount") - col("cost"),
        positive=if_else(col("score") > 0, 1, 0),
    )
    projected_pandas = lambda: projection_pipe(
        tidy(pdf, backend="pandas")
    ).collect(as_="pandas", columns=projected_columns)
    projected_polars = lambda: collect_polars(
        projection_pipe(tidy(pldf)), columns=projected_columns
    )

    def raw_group():
        return (
            pdf.groupby(["segment", "channel"], sort=False, observed=True, dropna=False)
            .agg(
                transactions=("amount", "size"),
                revenue=("amount", "sum"),
                average=("amount", "mean"),
                variability=("score", "std"),
                maximum=("amount", "max"),
            )
            .reset_index()
        )

    group_pipe = lambda frame: frame >> summarise(
        transactions=n(),
        revenue=sum("amount", na_rm=True),
        average=mean("amount", na_rm=True),
        variability=std("score", na_rm=True),
        maximum=max("amount", na_rm=True),
        by=["segment", "channel"],
    )

    def raw_sort():
        return pdf.sort_values(
            ["segment", "amount"], ascending=[True, False], kind="stable"
        ).reset_index(drop=True)

    sort_pipe = lambda frame: frame >> arrange("segment", desc("amount"))

    def raw_distinct():
        return (
            pdf.loc[:, ["customer_id", "channel"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )

    distinct_pipe = lambda frame: frame >> distinct("customer_id", "channel")

    def raw_join():
        return pdf.merge(dimension, on="customer_id", how="left", sort=False)

    pandas_join = lambda: (
        tidy(pdf, backend="pandas") >> left_join(dimension, on="customer_id")
    ).collect(as_="pandas")
    polars_join = lambda: collect_polars(
        tidy(pldf) >> left_join(pl_dimension, on="customer_id")
    )

    def raw_window():
        grouped = pdf.groupby("customer_id", sort=False, observed=True, dropna=False)
        return pdf.assign(
            customer_average=grouped["amount"].transform("mean"),
            previous_amount=grouped["amount"].shift(1),
            event_number=grouped.cumcount() + 1,
        )

    window_pipe = lambda frame: frame >> mutate(
        customer_average=mean("amount", na_rm=True),
        previous_amount=lag("amount"),
        event_number=row_number(),
        by="customer_id",
    )

    def raw_customer_features():
        filtered = pdf.loc[pdf["active"] & (pdf["amount"] > 0)]
        result = (
            filtered.groupby("customer_id", sort=False, observed=True, dropna=False)
            .agg(
                transactions=("amount", "size"),
                revenue=("amount", "sum"),
                average_score=("score", "mean"),
                last_event=("event_time", "max"),
            )
            .reset_index()
            .merge(dimension, on="customer_id", how="left", sort=False)
        )
        return result.loc[result["revenue"] > 1_000].reset_index(drop=True)

    def customer_pipe(frame, dim):
        return (
            frame
            >> filter(col("active") & (col("amount") > 0))
            >> summarise(
                transactions=n(),
                revenue=sum("amount", na_rm=True),
                average_score=mean("score", na_rm=True),
                last_event=max("event_time", na_rm=True),
                by="customer_id",
            )
            >> left_join(dim, on="customer_id")
            >> filter(col("revenue") > 1_000)
        )

    customer_pandas = lambda: customer_pipe(
        tidy(pdf, backend="pandas"), dimension
    ).collect(as_="pandas")
    customer_polars = lambda: collect_polars(
        customer_pipe(tidy(pldf), pl_dimension)
    )

    ml_columns = ["segment", "feature_a", "feature_b", "amount", "score"]

    def raw_ml_features():
        result = pdf[ml_columns].copy()
        grouped = result.groupby("segment", sort=False, observed=True, dropna=False)
        group_mean = grouped["feature_a"].transform("mean")
        group_std = grouped["feature_a"].transform("std")
        filled = result["feature_a"].fillna(group_mean)
        return result.assign(
            feature_a_filled=filled,
            feature_a_z=(filled - group_mean) / group_std,
            feature_ratio=result["feature_b"] / (result["amount"] + 1),
            log_amount=np.log(result["amount"] + 1),
            high_score=(result["score"] > 1).astype(np.int8),
        )

    ml_pipe = lambda frame: (
        frame
        >> select(*ml_columns)
        >> mutate(
            feature_a_filled=coalesce(
                col("feature_a"), mean("feature_a", na_rm=True)
            ),
            feature_a_z=(
                coalesce(col("feature_a"), mean("feature_a", na_rm=True))
                - mean("feature_a", na_rm=True)
            )
            / std("feature_a", na_rm=True),
            feature_ratio=col("feature_b") / (col("amount") + 1),
            log_amount=(col("amount") + 1).log(),
            high_score=if_else(col("score") > 1, 1, 0),
            by="segment",
        )
    )

    ml_output_columns = [
        *ml_columns,
        "feature_a_filled",
        "feature_a_z",
        "feature_ratio",
        "log_amount",
        "high_score",
    ]
    ml_staged_pipe = lambda frame: (
        frame
        >> select(*ml_columns)
        >> mutate(
            __feature_mean=mean("feature_a", na_rm=True),
            __feature_std=std("feature_a", na_rm=True),
            by="segment",
        )
        >> mutate(
            feature_a_filled=coalesce(col("feature_a"), col("__feature_mean")),
            feature_a_z=(
                coalesce(col("feature_a"), col("__feature_mean"))
                - col("__feature_mean")
            )
            / col("__feature_std"),
            feature_ratio=col("feature_b") / (col("amount") + 1),
            log_amount=(col("amount") + 1).log(),
            high_score=if_else(col("score") > 1, 1, 0),
        )
        >> select(*ml_output_columns)
    )

    event_columns = [
        "customer_id",
        "event_time",
        "amount",
        "previous_time",
        "gap",
        "event_number",
        "running_amount",
    ]

    def raw_event_features():
        result = pdf.sort_values(
            ["customer_id", "event_time"], kind="stable"
        ).reset_index(drop=True)
        grouped = result.groupby("customer_id", sort=False, observed=True, dropna=False)
        previous = grouped["event_time"].shift(1, fill_value=-1)
        result = result.assign(
            previous_time=previous,
            gap=result["event_time"] - previous,
            event_number=grouped.cumcount() + 1,
            running_amount=grouped["amount"]
            .expanding()
            .mean()
            .droplevel(0)
            .sort_index(),
        )
        return result[event_columns]

    event_pipe = lambda frame: (
        frame
        >> arrange("customer_id", "event_time")
        >> mutate(
            previous_time=lag("event_time", default=-1),
            event_number=row_number(),
            running_amount=cummean("amount"),
            by="customer_id",
        )
        >> mutate(gap=col("event_time") - col("previous_time"))
        >> select(*event_columns)
    )

    def raw_join_aggregate():
        joined = pdf.merge(dimension, on="customer_id", how="left", sort=False)
        filtered = joined.loc[joined["active"] & (joined["risk"] < 0.8)].copy()
        filtered["adjusted"] = filtered["amount"] * (1 - filtered["discount"])
        return (
            filtered.groupby(["region", "channel"], sort=False, observed=True, dropna=False)
            .agg(
                customers=("customer_id", "nunique"),
                events=("amount", "size"),
                revenue=("adjusted", "sum"),
                average_score=("score", "mean"),
            )
            .reset_index()
            .sort_values(["region", "channel"], kind="stable")
            .reset_index(drop=True)
        )

    def join_aggregate_pipe(frame, dim):
        return (
            frame
            >> left_join(dim, on="customer_id")
            >> filter(col("active") & (col("risk") < 0.8))
            >> mutate(adjusted=col("amount") * (1 - col("discount")))
            >> summarise(
                customers=col("customer_id").n_unique(),
                events=n(),
                revenue=sum("adjusted", na_rm=True),
                average_score=mean("score", na_rm=True),
                by=["region", "channel"],
            )
            >> arrange("region", "channel")
        )

    join_aggregate_pandas = lambda: join_aggregate_pipe(
        tidy(pdf, backend="pandas"), dimension
    ).collect(as_="pandas")
    join_aggregate_polars = lambda: collect_polars(
        join_aggregate_pipe(tidy(pldf), pl_dimension)
    )

    def raw_group_callback():
        return (
            pdf.groupby("segment", sort=False, observed=True, dropna=False)
            .agg(rows=("amount", "size"), average=("amount", "mean"))
            .reset_index()
        )

    def callback(part, key):
        values = part.collect(as_="pandas")
        return pd.DataFrame(
            {
                "segment": [key["segment"]],
                "rows": [len(values)],
                "average": [values["amount"].mean()],
            }
        )

    group_callback_pandas = lambda: (
        tidy(pdf, backend="pandas") >> group_modify(callback, "segment")
    ).collect(as_="pandas")
    group_callback_polars = lambda: (
        tidy(pldf) >> group_modify(callback, "segment")
    ).collect(as_="pandas")

    def raw_group_nest():
        return (
            pdf.groupby("segment", sort=False, observed=True, dropna=False)
            .size()
            .rename("rows")
            .reset_index()
        )

    def nested_counts(frame):
        nested = (frame >> group_nest("segment", name="records")).collect(as_="pandas")
        nested["rows"] = nested["records"].map(len)
        return nested[["segment", "rows"]]

    group_nest_pandas = lambda: nested_counts(tidy(pdf, backend="pandas"))
    group_nest_polars = lambda: nested_counts(tidy(pldf))

    return [
        Workload("Everyday", "filter + select", raw_filter, tp(filter_pipe), tl(filter_pipe)),
        Workload("Everyday", "mutate 4 features", raw_mutate, tp(mutate_pipe), tl(mutate_pipe)),
        Workload(
            "Everyday",
            "projected feature handoff",
            raw_projected_features,
            projected_pandas,
            projected_polars,
        ),
        Workload("Everyday", "group + 5 summaries", raw_group, tp(group_pipe), tl(group_pipe)),
        Workload("Everyday", "sort 2 columns", raw_sort, tp(sort_pipe), tl(sort_pipe)),
        Workload("Everyday", "distinct 2 columns", raw_distinct, tp(distinct_pipe), tl(distinct_pipe)),
        Workload("Everyday", "dimension left join", raw_join, pandas_join, polars_join),
        Workload("Everyday", "3 grouped windows", raw_window, tp(window_pipe), tl(window_pipe)),
        Workload(
            "Real-world",
            "customer aggregation",
            raw_customer_features,
            customer_pandas,
            customer_polars,
        ),
        Workload("Real-world", "ML feature matrix", raw_ml_features, tp(ml_pipe), tl(ml_pipe)),
        Workload(
            "Real-world",
            "ML features staged",
            raw_ml_features,
            tp(ml_staged_pipe),
            tl(ml_staged_pipe),
        ),
        Workload("Real-world", "event history features", raw_event_features, tp(event_pipe), tl(event_pipe)),
        Workload(
            "Real-world",
            "join-filter-aggregate",
            raw_join_aggregate,
            join_aggregate_pandas,
            join_aggregate_polars,
        ),
        Workload(
            "Materialization",
            "group callback summarize",
            raw_group_callback,
            group_callback_pandas,
            group_callback_polars,
        ),
        Workload(
            "Materialization",
            "group nest and count",
            raw_group_nest,
            group_nest_pandas,
            group_nest_polars,
        ),
    ]


def _measure(fn: Callable[[], Any], repeat: int) -> tuple[list[float], Any]:
    samples: list[float] = []
    result = None
    for _ in range(repeat):
        gc.collect()
        start = time.perf_counter()
        result = fn()
        samples.append(time.perf_counter() - start)
    return samples, result


def geometric_ratio(
    results: dict[str, dict[str, dict[str, float]]],
    engine: str,
    *,
    minimum_pandas_ms: float = 0.0,
) -> float:
    """Equal-weight geometric mean ratio for budget-worthy workloads."""
    ratios = [
        workload[engine]["vs_pandas"]
        for workload in results.values()
        if workload["pandas"]["median_seconds"] * 1000 >= minimum_pandas_ms
    ]
    if not ratios:
        raise ValueError("no workloads meet the performance-budget minimum")
    return math.exp(sum(math.log(value) for value in ratios) / len(ratios))


def run(
    rows: int = 1_000_000,
    *,
    repeat: int = 5,
    warmup: int = 1,
    seed: int = 0,
    output: str = "pandas",
    match: str | None = None,
    polars_engine: str = "auto",
) -> dict[str, dict[str, dict[str, float]]]:
    """Run all workloads and print median wall times and pandas-relative ratios."""
    if repeat < 1 or warmup < 0:
        raise ValueError("repeat must be positive and warmup non-negative")
    if output not in {"pandas", "pandas-arrow", "native"}:
        raise ValueError(
            "output must be 'pandas', 'pandas-arrow', or 'native'"
        )
    if polars_engine not in {"auto", "streaming", "gpu"}:
        raise ValueError(
            "polars_engine must be 'auto', 'streaming', or 'gpu'"
        )
    execution_engine: Any = polars_engine
    if polars_engine == "gpu":
        # Polars normally falls back to CPU for unsupported GPU queries.
        # Benchmarks must fail instead of reporting disguised CPU timings.
        execution_engine = pl.GPUEngine(raise_on_fail=True)
    pdf, pldf, dimension, pl_dimension = make_realistic_data(rows, seed)
    workloads = _build_workloads(
        pdf,
        pldf,
        dimension,
        pl_dimension,
        output=output,
        polars_engine=execution_engine,
    )
    if match:
        patterns = [value.strip().casefold() for value in match.split(",") if value.strip()]
        workloads = [
            workload
            for workload in workloads
            if any(pattern in workload.name.casefold() for pattern in patterns)
        ]
        if not workloads:
            raise ValueError(f"no workloads match {match!r}")
    engines = ["pandas", "tidy3[pandas]", "tidy3[polars]"]
    print(
        f"tidy3 comprehensive benchmark: rows={rows:,}, repeat={repeat}, "
        f"warmup={warmup}, output={output}, polars_engine={polars_engine}"
    )
    print("times are medians; ratios are relative to raw pandas")

    results: dict[str, dict[str, dict[str, float]]] = {}
    current_category = None
    header = (
        f"  {'workload':<25} {'pandas':>11} "
        f"{'tidy3[pandas]':>20} {'tidy3[polars]':>20}"
    )
    for workload in workloads:
        if workload.category != current_category:
            current_category = workload.category
            print(f"\n{current_category}")
            print(header)
            print("  " + "-" * (len(header) - 2))
        functions = {
            "pandas": workload.pandas,
            "tidy3[pandas]": workload.tidy_pandas,
            "tidy3[polars]": workload.tidy_polars,
        }
        warm_results = {}
        for _ in range(warmup):
            for engine, function in functions.items():
                warm_results[engine] = function()
        if not warm_results:
            warm_results = {engine: function() for engine, function in functions.items()}
        _assert_equivalent(
            warm_results["pandas"], warm_results["tidy3[pandas]"], workload.name
        )
        _assert_equivalent(
            warm_results["pandas"], warm_results["tidy3[polars]"], workload.name
        )

        samples: dict[str, list[float]] = {engine: [] for engine in engines}
        final_results = {}
        for round_index in range(repeat):
            order = engines[round_index % len(engines) :] + engines[: round_index % len(engines)]
            for engine in order:
                measured, final_results[engine] = _measure(functions[engine], 1)
                samples[engine].extend(measured)
        _assert_equivalent(
            final_results["pandas"], final_results["tidy3[pandas]"], workload.name
        )
        _assert_equivalent(
            final_results["pandas"], final_results["tidy3[polars]"], workload.name
        )

        medians = {engine: statistics.median(values) for engine, values in samples.items()}
        baseline = medians["pandas"]
        results[workload.name] = {
            engine: {
                "median_seconds": medians[engine],
                "minimum_seconds": min(samples[engine]),
                "maximum_seconds": max(samples[engine]),
                "vs_pandas": medians[engine] / baseline,
            }
            for engine in engines
        }
        print(
            f"  {workload.name:<25} {baseline * 1000:>9.1f}ms "
            f"{medians['tidy3[pandas]'] * 1000:>9.1f}ms "
            f"{medians['tidy3[pandas]'] / baseline:>5.2f}x "
            f"{medians['tidy3[polars]'] * 1000:>9.1f}ms "
            f"{medians['tidy3[polars]'] / baseline:>5.2f}x"
        )
    return results


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        choices=["pandas", "pandas-arrow", "native"],
        default="pandas",
    )
    parser.add_argument(
        "--match", help="comma-separated workload-name substrings to run"
    )
    parser.add_argument(
        "--polars-engine",
        choices=["auto", "streaming", "gpu"],
        default="auto",
        help="Polars execution mode; gpu is strict and fails on CPU fallback",
    )
    parser.add_argument("--max-tidy-pandas-geo", type=float)
    parser.add_argument("--max-tidy-polars-geo", type=float)
    parser.add_argument("--budget-min-ms", type=float, default=10.0)
    args = parser.parse_args()
    results = run(
        rows=args.rows,
        repeat=args.repeat,
        warmup=args.warmup,
        seed=args.seed,
        output=args.output,
        match=args.match,
        polars_engine=args.polars_engine,
    )
    budgets = {
        "tidy3[pandas]": args.max_tidy_pandas_geo,
        "tidy3[polars]": args.max_tidy_polars_geo,
    }
    failures = []
    for engine, maximum in budgets.items():
        if maximum is None:
            continue
        ratio = geometric_ratio(
            results, engine, minimum_pandas_ms=args.budget_min_ms
        )
        print(f"budget {engine}: {ratio:.3f}x <= {maximum:.3f}x")
        if ratio > maximum:
            failures.append(f"{engine} {ratio:.3f}x exceeds {maximum:.3f}x")
    if failures:
        raise SystemExit("performance budget failed: " + "; ".join(failures))


if __name__ == "__main__":
    _main()
