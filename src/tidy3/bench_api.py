"""Polars-relative benchmark for the public tidy3 API.

Run ``python -m tidy3.bench_api --rows 100000 --output api-polars.json``.

Every exported name and public ``TidyFrame`` method is either a case or an
explicit reason a raw Polars comparison is not meaningful. Native recipes use
Polars only. Outputs are checked, including names, dtypes and order, before
timing. Engine order alternates after the warm-up. Reports rank cases by
absolute excess time (tidy3 minus raw Polars).
"""

from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import platform
import statistics
import tempfile
import textwrap
import time
import unicodedata
import warnings

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal


class Case:
    """One checked comparison. ``tidy`` may call tidy3; ``native`` must not."""

    def __init__(
        self,
        name: str,
        family: str,
        functions: tuple[str, ...] | list[str],
        tidy,
        native,
        methods: tuple[str, ...] | list[str] = (),
        prepare=None,
    ):
        self.name = name
        self.family = family
        self.functions = tuple(functions)
        self.methods = tuple(methods)
        self.tidy = tidy
        self.native = native
        self.prepare = prepare

    def __repr__(self) -> str:
        return f"Case({self.name!r})"


def work(n: int) -> pl.DataFrame:
    """One eager table of ``n`` rows, built with Polars expressions.

    ``g`` is ``id % 8``. Text dates are zero-padded so the uniform parsers
    apply. ``gap`` and ``note`` are the only columns with missing values.
    ``words`` is constant, which ``remove_constant`` is expected to drop.
    """
    if n < 8:
        raise ValueError("work() needs at least 8 rows")
    month = (pl.col("id") % 12) + 1
    day = (pl.col("id") % 28) + 1
    hour = pl.col("id") % 24
    minute = pl.col("id") % 60
    second = (pl.col("id") * 3) % 60
    species_code = pl.col("id") % 3
    frame = pl.DataFrame({"id": pl.int_range(n, eager=True, dtype=pl.Int64)}).with_columns(
        g=pl.col("id") % 8,
        species=(
            pl.when(species_code == 0).then(pl.lit("Adelie"))
            .when(species_code == 1).then(pl.lit("Chinstrap"))
            .otherwise(pl.lit("Gentoo"))
        ),
        island=(
            pl.when(species_code == 0).then(pl.lit("Biscoe"))
            .when(species_code == 1).then(pl.lit("Dream"))
            .otherwise(pl.lit("Torgersen"))
        ),
        mass=(pl.col("id") % 1000).cast(pl.Float64) + 3000.0,
        bill=(pl.col("id") % 50).cast(pl.Float64) + 30.5,
        flag=pl.col("id") % 2 == 0,
        logic=pl.col("id") % 2 == 0,
        whole=pl.col("id") % 100,
        serial=pl.lit(42370) + (pl.col("id") % 28),
        bill1=(pl.col("id") % 7).cast(pl.Float64),
        bill2=(pl.col("id") % 11).cast(pl.Float64),
    )
    return frame.with_columns(
        txt=pl.col("species") + pl.lit(" ") + pl.col("island"),
        code=pl.lit("AB") + (pl.col("id") % 97).cast(pl.String).str.zfill(2),
        date=pl.format("2024-{}-{}", month.cast(pl.String).str.zfill(2), day.cast(pl.String).str.zfill(2)),
        mdy=pl.format("{}/{}/2024", month.cast(pl.String).str.zfill(2), day.cast(pl.String).str.zfill(2)),
        dmy=pl.format("{}/{}/2024", day.cast(pl.String).str.zfill(2), month.cast(pl.String).str.zfill(2)),
        ym=pl.format("2024-{}", month.cast(pl.String).str.zfill(2)),
        when=pl.format(
            "2024-{}-{} {}:{}:{}",
            month.cast(pl.String).str.zfill(2),
            day.cast(pl.String).str.zfill(2),
            hour.cast(pl.String).str.zfill(2),
            minute.cast(pl.String).str.zfill(2),
            second.cast(pl.String).str.zfill(2),
        ),
        num=pl.lit("$") + pl.col("id").cast(pl.String) + pl.lit(".50"),
        csv=pl.col("species") + pl.lit(",") + pl.col("island"),
        mdy_when=pl.format(
            "{}/{}/2024 {}:{}:{}",
            month.cast(pl.String).str.zfill(2),
            day.cast(pl.String).str.zfill(2),
            hour.cast(pl.String).str.zfill(2),
            minute.cast(pl.String).str.zfill(2),
            second.cast(pl.String).str.zfill(2),
        ),
        dmy_when=pl.format(
            "{}/{}/2024 {}:{}:{}",
            day.cast(pl.String).str.zfill(2),
            month.cast(pl.String).str.zfill(2),
            hour.cast(pl.String).str.zfill(2),
            minute.cast(pl.String).str.zfill(2),
            second.cast(pl.String).str.zfill(2),
        ),
        gap=pl.when(pl.col("id") % 17 == 0).then(None).otherwise(pl.col("mass")),
        note=pl.when(pl.col("id") % 19 == 0).then(None).otherwise(pl.col("species")),
        words=pl.lit("alpha beta gamma"),
        lgl=pl.when(pl.col("id") % 2 == 0).then(pl.lit("true")).otherwise(pl.lit("false")),
        digits=pl.col("whole").cast(pl.String),
        dbl=(pl.col("whole").cast(pl.String) + pl.lit(".25")),
        padded=pl.lit("  ") + pl.col("species") + pl.lit("  "),
    ).with_columns(
        hm=pl.col("when").str.slice(0, 16),
        day_d=pl.col("date").str.to_date("%Y-%m-%d"),
        at=pl.col("when").str.to_datetime("%Y-%m-%d %H:%M:%S", time_unit="us"),
        items=pl.int_ranges(0, (pl.col("id") % 3) + 1),
        rec=pl.struct(a=pl.col("id") % 5, b=pl.col("species")),
    )


def _box(value):
    name = type(value).__name__
    if name == "TidyFrame":
        return value.collect()
    if isinstance(value, pl.LazyFrame):
        return value.collect()
    if name == "Table":
        return pl.from_arrow(value)
    if name == "DataFrame" and not isinstance(value, pl.DataFrame):
        return pl.from_pandas(value)
    if isinstance(value, list) and value and type(value[0]).__name__ == "TidyFrame":
        return [item.collect() for item in value]
    return value


def _norm(value):
    if isinstance(value, np.ndarray):
        if value.dtype == object or np.issubdtype(value.dtype, np.number):
            return value
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, list):
        return [_norm(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_norm(item) for item in value)
    return value


def same(left, right) -> None:
    """Fail unless both results are the same complete value."""
    left, right = _box(left), _box(right)
    if isinstance(left, pl.DataFrame) and isinstance(right, pl.DataFrame):
        assert_frame_equal(left, right, check_row_order=True, check_column_order=True)
        return
    if isinstance(left, pl.Series) and isinstance(right, pl.Series):
        assert_series_equal(left, right, check_names=True)
        return
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        if left.shape != right.shape:
            raise AssertionError(f"array shape {left.shape} != {right.shape}")
        if np.issubdtype(left.dtype, np.floating) or np.issubdtype(right.dtype, np.floating):
            if not np.allclose(left, right, equal_nan=True):
                raise AssertionError("floating arrays differ")
            return
        if not np.array_equal(left, right):
            raise AssertionError("arrays differ")
        return
    if isinstance(left, dict) and isinstance(right, dict):
        left = {key: str(item) for key, item in left.items()}
        right = {key: str(item) for key, item in right.items()}
    left, right = _norm(left), _norm(right)
    if isinstance(left, list) and isinstance(right, list) and left and isinstance(left[0], pl.DataFrame):
        if len(left) != len(right):
            raise AssertionError(f"{len(left)} frames != {len(right)} frames")
        for index, (one, other) in enumerate(zip(left, right)):
            assert_frame_equal(one, other, check_row_order=True, check_column_order=True)
        return
    if left != right:
        raise AssertionError(f"{left!r} != {right!r}")


def _cleanup(extra) -> None:
    if isinstance(extra, Path):
        extra.unlink(missing_ok=True)
    elif isinstance(extra, (list, tuple)):
        for item in extra:
            _cleanup(item)


def run_case(case: Case, df: pl.DataFrame, *, repeat: int = 3, warmup: int = 1) -> dict:
    """Check one case, then time it. Preparation is outside the samples."""
    if repeat < 1 or warmup < 0:
        raise ValueError("repeat must be positive and warmup non-negative")
    extra = case.prepare(df) if case.prepare else None
    try:
        same(case.tidy(df, extra), case.native(df, extra))
        functions = {"tidy3": case.tidy, "raw_polars": case.native}
        for _ in range(warmup):
            for function in functions.values():
                function(df, extra)
        samples = {name: [] for name in functions}
        names = list(functions)
        for iteration in range(repeat):
            for name in names[iteration % 2:] + names[:iteration % 2]:
                start = time.perf_counter()
                result = functions[name](df, extra)
                samples[name].append((time.perf_counter() - start) * 1000)
                del result
        best = {name: min(values) for name, values in samples.items()}
        raw = best["raw_polars"]
        info = classification()
        methods = [name for name, owner in info["methods"].items() if owner == case.name]
        return {
            "name": case.name,
            "family": case.family,
            "functions": list(case.functions),
            "methods": methods,
            "rows": df.height,
            "tidy3_ms": best["tidy3"],
            "raw_polars_ms": raw,
            "tidy3_over_polars": (best["tidy3"] / raw) if raw else None,
            "excess_ms": best["tidy3"] - raw,
            "median_ms": {name: statistics.median(values) for name, values in samples.items()},
            "samples_ms": samples,
        }
    finally:
        _cleanup(extra)


def _column(name, family, functions, build, native, methods=()):
    def tidy(df, extra=None):
        import tidy3 as t

        return (t.tidy(df) >> t.mutate(out=build(t, df)) >> t.select("out")).collect()

    def raw(df, extra=None):
        return df.select(native(df).alias("out"))

    return Case(name, family, functions, tidy, raw, methods)


def _summary(name, functions, build, native, family="expr"):
    def tidy(df, extra=None):
        import tidy3 as t

        return (t.tidy(df) >> t.group_by("g") >> t.summarise(out=build(t, df))).collect()

    def raw(df, extra=None):
        return (
            df.group_by("g", maintain_order=True)
            .agg(native(df).alias("out"))
            .sort("g")
        )

    return Case(name, family, functions, tidy, raw)


def _select_where(name, functions, predicate):
    def tidy(df, extra=None):
        import tidy3 as t

        return (t.tidy(df) >> t.select(t.where(predicate))).collect()

    def raw(df, extra=None):
        names = [column for column, dtype in df.schema.items() if predicate_dtype(predicate, dtype)]
        return df.select(names)

    return Case(name, "select", functions, tidy, raw)


def predicate_dtype(predicate, dtype) -> bool:
    """Polars-side of the ``where(is_*)`` predicates used in the selector cases."""
    label = predicate.__name__
    if label == "is_integer":
        return bool(dtype.is_integer())
    if label == "is_float":
        return bool(dtype.is_float())
    if label == "is_numeric":
        return bool(dtype.is_integer() or dtype.is_float())
    if label in {"is_string", "is_character"}:
        return dtype == pl.String
    if label in {"is_bool", "is_boolean"}:
        return dtype == pl.Boolean
    if label == "is_datetime":
        return dtype == pl.Date or isinstance(dtype, pl.Datetime)
    if label == "is_temporal":
        return bool(dtype.is_temporal())
    if label == "is_categorical":
        return isinstance(dtype, (pl.Enum, pl.Categorical))
    raise ValueError(label)


def _class_name(dtype) -> str:
    if dtype.is_float():
        return "numeric"
    if dtype.is_integer():
        return "integer"
    if dtype == pl.Boolean:
        return "logical"
    if dtype == pl.Date:
        return "Date"
    if isinstance(dtype, pl.Datetime):
        return "POSIXct, POSIXt"
    if dtype == pl.String:
        return "character"
    return str(dtype)


def _factor_levels(df: pl.DataFrame, how: str) -> list[str]:
    """Level order for grouped string factors, stated independently of tidy3."""
    present = df.drop_nulls(subset=["species"])
    categories = sorted(present["species"].unique().to_list())
    codes = {level: index for index, level in enumerate(categories)}
    code = pl.col("species").replace_strict(codes, return_dtype=pl.Int64)
    keys = df.select("g").unique(maintain_order=True).sort("g").with_row_index("gid")
    if how == "rev":
        pairs = present.select("g", "species").unique()
        pairs = pairs.with_columns(code=code).join(keys, on="g", how="left")
        pairs = pairs.sort(["gid", "code"], descending=[False, True])
    elif how == "infreq":
        pairs = present.group_by("g", "species").agg(pl.len().alias("score"))
        pairs = pairs.with_columns(code=code).join(keys, on="g", how="left")
        pairs = pairs.sort(["gid", "score", "code"], descending=[False, True, False])
    elif how == "reorder":
        pairs = present.group_by("g", "species").agg(pl.col("mass").median().alias("score"))
        pairs = pairs.with_columns(code=code).join(keys, on="g", how="left")
        pairs = pairs.sort(["gid", "score", "code"])
    elif how == "relevel":
        requested = ["Gentoo"]
        known = [level for level in requested if level in categories]
        rank = {level: index for index, level in enumerate(requested) if level in categories}
        pairs = present.select("g", "species").unique()
        pairs = pairs.with_columns(code=code).join(keys, on="g", how="left").sort(["gid", "code"])
        other = ~pl.col("species").is_in(known) if known else pl.lit(True)
        pairs = pairs.with_columns(
            other=other,
            other_rank=other.cast(pl.Int64).cum_sum().over("gid", order_by="code") - 1,
            n_other=other.cast(pl.Int64).sum().over("gid"),
            req=pl.col("species").replace_strict(rank, default=None),
        )
        before = pl.col("other") & (pl.col("other_rank") < pl.min_horizontal(pl.lit(0), pl.col("n_other")))
        pairs = pairs.with_columns(
            band=pl.when(before).then(0).when(pl.col("other")).then(2).otherwise(1),
            key=pl.when(pl.col("other")).then(pl.col("other_rank")).otherwise(pl.col("req")),
        ).sort(["gid", "band", "key"], maintain_order=True)
    else:
        raise ValueError(how)
    return pairs["species"].unique(maintain_order=True).to_list()


def _cast_factor(df: pl.DataFrame, how: str) -> pl.DataFrame:
    levels = _factor_levels(df, how)
    return df.select(pl.col("species").cast(pl.String).cast(pl.Enum(levels)).alias("out"))


def _grouped_factor(how: str):
    def tidy(df, extra=None):
        import tidy3 as t

        call = {
            "infreq": lambda: t.fct_infreq("species"),
            "reorder": lambda: t.fct_reorder("species", "mass"),
            "rev": lambda: t.fct_rev("species"),
            "relevel": lambda: t.fct_relevel("species", "Gentoo"),
        }[how]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            frame = t.tidy(df) >> t.group_by("g") >> t.mutate(out=call()) >> t.ungroup() >> t.select("out")
            return frame.collect()

    def raw(df, extra=None):
        return _cast_factor(df, how)

    return tidy, raw


def _sub(column: str, start: int, end: int) -> pl.Expr:
    value = pl.col(column).cast(pl.String)
    length = value.str.len_chars().cast(pl.Int64)
    first = pl.lit(start) if start > 0 else length + start + 1
    last = pl.lit(end) if end >= 0 else length + end + 1
    first = pl.max_horizontal(first, pl.lit(1))
    last = pl.min_horizontal(last, length)
    size = pl.max_horizontal(last - first + 1, pl.lit(0))
    return value.str.slice(first - 1, size)


def _width(value: str) -> int:
    total = 0
    for char in value:
        if unicodedata.combining(char):
            continue
        total += 2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1
    return total


def _wrap(value: str) -> str:
    return "\n".join(textwrap.wrap(value, width=8, break_long_words=False, break_on_hyphens=False))


def _tag(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter(pl.col("id") % 2 == 0).select("id", pl.lit("even").alias("tag"))


def _write(df: pl.DataFrame, suffix: str, method: str, separator: str | None = None) -> Path:
    handle = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    handle.close()
    path = Path(handle.name)
    frame = df.select("id", "species", "whole")
    if method == "csv":
        frame.write_csv(path, separator=separator or ",")
    elif method == "parquet":
        frame.write_parquet(path)
    elif method == "ipc":
        frame.write_ipc(path)
    else:
        raise ValueError(method)
    return path


def _group_number(df: pl.DataFrame) -> list[int]:
    keys = df.select("g").unique(maintain_order=True).sort("g").with_row_index("number", offset=1)
    return df.join(keys, on="g", how="left", maintain_order="left")["number"].to_list()


def _first_per_group(df: pl.DataFrame) -> pl.DataFrame:
    return df.sort("id").unique(subset=["g"], keep="first", maintain_order=True).sort("g")


def _constant_columns(df: pl.DataFrame) -> list[str]:
    kept = []
    for name in df.columns:
        column = df[name]
        if column.n_unique() == 1 and column.null_count() == 0:
            continue
        kept.append(name)
    return kept


# Reasons a raw Polars comparison would not describe the function's job.
REASONS = {
    "TidyFrame": "frame type, not an operation",
    "ColumnNotFoundError": "exception type, not an operation",
    "JoinCondition": "join-spec type, not an operation",
    "JoinSpec": "join-spec type, not an operation",
    "Selector": "selector type, not an operation",
    "ExportResult": "notebook-export type, not an operation",
    "ConversionError": "exception type, not an operation",
    "__version__": "package version string",
    "options": "process-wide options, not a frame query",
    "locale": "parser-locale object, not a frame query",
    "env": "data-mask environment, not a frame query",
    "today": "reads the clock when called",
    "now": "reads the clock when called",
    "glimpse": "prints a preview and returns the frame",
    "peek": "prints a preview and returns the frame",
    "str_view": "prints match highlights",
    "nb_export": "rewrites a notebook; not a frame query",
    "transform_cell": "rewrites notebook source; not a frame query",
    "transform_source": "rewrites notebook source; not a frame query",
    "collect_known_names": "notebook name collection, not a frame query",
    "partial_run": "runs a notebook cell; not a frame query",
    "looks_like_tidy_pipe": "detects notebook pipe syntax",
    "maybe_rewrite_cell": "rewrites notebook source; not a frame query",
    "normalize_pipe_source": "rewrites notebook source; not a frame query",
    "sample_n": "random draw; the engines do not share one RNG stream",
    "sample_frac": "random draw; the engines do not share one RNG stream",
    "slice_sample": "random draw; the engines do not share one RNG stream",
    "read_excel": "Excel workbook reader; not a Polars query",
    "excel_sheets": "Excel workbook metadata; not a Polars query",
    "group_walk": "runs a Python callback per group for side effects and returns the input",
}

METHOD_REASONS = {
    "backend": "storage-engine label, not a query",
    "explain": "prints a query plan",
    "preview": "prints a preview",
    "show": "prints a preview",
    "ggplot": "builds a plot",
    "write_excel": "Excel workbook writer; not a Polars query",
}


def _cases() -> list[Case]:
    cases: list[Case] = []

    def add(case: Case) -> None:
        cases.append(case)

    def col(name, family, functions, build, native):
        add(_column(name, family, functions, build, native))

    # ── expressions ────────────────────────────────────────────────────
    col("abs", "expr", ["abs"], lambda t, df: t.abs("mass"), lambda df: pl.col("mass").abs())
    col("floor", "expr", ["floor"], lambda t, df: t.floor("bill"), lambda df: pl.col("bill").floor())
    col("ceiling", "expr", ["ceiling"], lambda t, df: t.ceiling("bill"), lambda df: pl.col("bill").ceil())
    col("round", "expr", ["round"], lambda t, df: t.round("bill", 1), lambda df: pl.col("bill").round(1))
    col("pmin", "expr", ["pmin"], lambda t, df: t.pmin("mass", "bill"), lambda df: pl.min_horizontal("mass", "bill"))
    col("pmax", "expr", ["pmax"], lambda t, df: t.pmax("mass", "bill"), lambda df: pl.max_horizontal("mass", "bill"))
    col("row_sums", "expr", ["row_sums"], lambda t, df: t.row_sums("mass", "bill", na_rm=True), lambda df: pl.sum_horizontal("mass", "bill"))
    col("row_means", "expr", ["row_means"], lambda t, df: t.row_means("mass", "bill", na_rm=True), lambda df: pl.mean_horizontal("mass", "bill"))
    col("is_na", "expr", ["is_na"], lambda t, df: t.is_na("gap"), lambda df: pl.col("gap").is_null())
    col("is_nan", "expr", ["is_nan"], lambda t, df: t.is_nan("mass"), lambda df: pl.col("mass").is_nan())
    col("is_finite", "expr", ["is_finite"], lambda t, df: t.is_finite("mass"), lambda df: pl.col("mass").is_finite())
    col("is_infinite", "expr", ["is_infinite"], lambda t, df: t.is_infinite("mass"), lambda df: pl.col("mass").is_infinite())
    col("lag", "expr", ["lag"], lambda t, df: t.lag("mass"), lambda df: pl.col("mass").shift(1))
    col("lead", "expr", ["lead"], lambda t, df: t.lead("mass"), lambda df: pl.col("mass").shift(-1))
    col("row_number", "expr", ["row_number"], lambda t, df: t.row_number(), lambda df: pl.int_range(1, pl.len() + 1))
    col("min_rank", "expr", ["min_rank"], lambda t, df: t.min_rank("mass"), lambda df: pl.col("mass").rank(method="min"))
    col("dense_rank", "expr", ["dense_rank"], lambda t, df: t.dense_rank("mass"), lambda df: pl.col("mass").rank(method="dense"))
    col("percent_rank", "expr", ["percent_rank"], lambda t, df: t.percent_rank("mass"), lambda df: (pl.col("mass").rank(method="min") - 1) / (pl.col("mass").count() - 1))
    col("cume_dist", "expr", ["cume_dist"], lambda t, df: t.cume_dist("mass"), lambda df: pl.col("mass").rank(method="max") / pl.col("mass").count())
    col(
        "ntile", "expr", ["ntile"],
        lambda t, df: t.ntile("mass", 4),
        lambda df: (((pl.col("mass").rank(method="ordinal") - 1) * 4 / pl.col("mass").count()).floor() + 1).cast(pl.UInt32),
    )
    col(
        "cummean", "expr", ["cummean"],
        lambda t, df: t.cummean("mass"),
        lambda df: pl.col("mass").cum_sum() / pl.col("mass").is_not_null().cast(pl.UInt32).cum_sum(),
    )
    col("cumall", "expr", ["cumall"], lambda t, df: t.cumall("flag"), lambda df: pl.col("flag").cum_min())
    col("cumany", "expr", ["cumany"], lambda t, df: t.cumany("flag"), lambda df: pl.col("flag").cum_max())
    col(
        "consecutive_id", "expr", ["consecutive_id"],
        lambda t, df: t.consecutive_id("species"),
        lambda df: (
            pl.col("species").ne_missing(pl.col("species").shift()) | (pl.int_range(pl.len()) == 0)
        ).cum_sum().cast(pl.UInt32),
    )
    col("if_else", "expr", ["if_else"], lambda t, df: t.if_else(t.col("flag"), "yes", "no"), lambda df: pl.when(pl.col("flag")).then(pl.lit("yes")).otherwise(pl.lit("no")))
    col(
        "case_when", "expr", ["case_when"],
        lambda t, df: t.case_when((t.col("flag"), "even"), default="odd"),
        lambda df: pl.when(pl.col("flag")).then(pl.lit("even")).otherwise(pl.lit("odd")),
    )
    col(
        "case_match", "expr", ["case_match"],
        lambda t, df: t.case_match("species", ("Adelie", "A"), default="other"),
        lambda df: pl.when(pl.col("species") == "Adelie").then(pl.lit("A")).otherwise(pl.lit("other")),
    )
    col("coalesce", "expr", ["coalesce"], lambda t, df: t.coalesce(t.col("note"), t.col("species")), lambda df: pl.coalesce(pl.col("note"), pl.col("species")))
    col(
        "between_values", "expr", ["between_values"],
        lambda t, df: t.between_values(t.col("mass"), 3000, 3500),
        lambda df: (pl.col("mass") >= 3000) & (pl.col("mass") <= 3500),
    )
    col(
        "na_if", "expr", ["na_if"],
        lambda t, df: t.na_if(t.col("species"), "Adelie"),
        lambda df: pl.when(pl.col("species") == "Adelie").then(None).otherwise(pl.col("species")),
    )
    col(
        "near", "expr", ["near"],
        lambda t, df: t.near(t.col("mass"), t.col("mass")),
        lambda df: (pl.col("mass") - pl.col("mass")).abs() <= 1.490116e-8,
    )
    col(
        "recode", "expr", ["recode"],
        lambda t, df: t.recode("species", {"Adelie": "A"}, default="other"),
        lambda df: pl.when(pl.col("species") == "Adelie").then(pl.lit("A")).otherwise(pl.lit("other")),
    )
    col(
        "recode_values", "expr", ["recode_values"],
        lambda t, df: t.recode_values("species", ("Adelie", "A"), default="other"),
        lambda df: pl.when(pl.col("species") == "Adelie").then(pl.lit("A")).otherwise(pl.lit("other")),
    )
    col(
        "replace_values", "expr", ["replace_values"],
        lambda t, df: t.replace_values("species", ("Adelie", "A")),
        lambda df: pl.when(pl.col("species") == "Adelie").then(pl.lit("A")).otherwise(pl.col("species")),
    )
    col(
        "replace_when", "expr", ["replace_when"],
        lambda t, df: t.replace_when("species", (t.col("flag"), "even")),
        lambda df: pl.when(pl.col("flag")).then(pl.lit("even")).otherwise(pl.col("species")),
    )
    col("when_any", "expr", ["when_any"], lambda t, df: t.when_any(t.col("flag"), t.col("whole") > 50), lambda df: pl.col("flag") | (pl.col("whole") > 50))
    col("when_all", "expr", ["when_all"], lambda t, df: t.when_all(t.col("flag"), t.col("whole") > 50), lambda df: pl.col("flag") & (pl.col("whole") > 50))
    col(
        "order_by", "expr", ["order_by"],
        lambda t, df: t.order_by("mass", t.lag("id")),
        lambda df: pl.col("id").sort_by("mass").shift(1).sort_by(pl.col("mass").arg_sort()),
    )
    col(
        "with_order", "expr", ["with_order"],
        lambda t, df: t.with_order("mass", t.lag, "id"),
        lambda df: pl.col("id").sort_by("mass").shift(1).sort_by(pl.col("mass").arg_sort()),
    )
    col(
        "scale", "expr", ["scale"],
        lambda t, df: t.scale("mass"),
        lambda df: (pl.col("mass") - pl.col("mass").mean()) / pl.col("mass").std(ddof=1),
    )
    col(
        "rescale", "expr", ["rescale"],
        lambda t, df: t.rescale("mass"),
        lambda df: (pl.col("mass") - pl.col("mass").min()) / (pl.col("mass").max() - pl.col("mass").min()),
    )
    col("col", "expr", ["col"], lambda t, df: t.col("mass") + 1, lambda df: pl.col("mass") + 1)

    add(_summary("mean", ["mean"], lambda t, df: t.mean("mass"), lambda df: pl.col("mass").mean()))
    add(_summary("sum", ["sum"], lambda t, df: t.sum("mass"), lambda df: pl.col("mass").sum()))
    add(_summary("min", ["min"], lambda t, df: t.min("mass"), lambda df: pl.col("mass").min()))
    add(_summary("max", ["max"], lambda t, df: t.max("mass"), lambda df: pl.col("mass").max()))
    add(_summary("median", ["median"], lambda t, df: t.median("mass"), lambda df: pl.col("mass").median()))
    add(_summary("sd", ["sd"], lambda t, df: t.sd("mass"), lambda df: pl.col("mass").std(ddof=1)))
    add(_summary("std", ["std"], lambda t, df: t.std("mass"), lambda df: pl.col("mass").std(ddof=1)))
    add(_summary("var", ["var"], lambda t, df: t.var("mass"), lambda df: pl.col("mass").var(ddof=1)))
    add(_summary("n", ["n"], lambda t, df: t.n(), lambda df: pl.len()))
    add(_summary("n_distinct", ["n_distinct"], lambda t, df: t.n_distinct("species"), lambda df: pl.col("species").n_unique()))
    add(_summary("first", ["first"], lambda t, df: t.first("mass"), lambda df: pl.col("mass").sort_by("id").first()))
    add(_summary("last", ["last"], lambda t, df: t.last("mass"), lambda df: pl.col("mass").sort_by("id").last()))
    add(_summary("nth", ["nth"], lambda t, df: t.nth("mass", 2), lambda df: pl.col("mass").sort_by("id").get(1)))
    add(_summary("any", ["any"], lambda t, df: t.any("flag"), lambda df: pl.col("flag").any()))
    add(_summary("all", ["all"], lambda t, df: t.all("flag"), lambda df: pl.col("flag").all()))

    def tidy_desc(df, extra=None):
        import tidy3 as t

        return (t.tidy(df) >> t.arrange(t.desc("mass"), "id") >> t.select("id")).collect()

    def raw_desc(df, extra=None):
        return df.sort(["mass", "id"], descending=[True, False], nulls_last=True, maintain_order=True).select("id")

    add(Case("desc", "expr", ["desc"], tidy_desc, raw_desc))

    # ── convert and dates ──────────────────────────────────────────────
    col("as_double", "convert", ["as_double"], lambda t, df: t.as_double("dbl"), lambda df: pl.col("dbl").cast(pl.Float64))
    col("as_numeric", "convert", ["as_numeric"], lambda t, df: t.as_numeric("dbl"), lambda df: pl.col("dbl").cast(pl.Float64))
    col("as_integer", "convert", ["as_integer"], lambda t, df: t.as_integer("digits"), lambda df: pl.col("digits").cast(pl.Int64))
    col("parse_integer", "convert", ["parse_integer"], lambda t, df: t.parse_integer("digits"), lambda df: pl.col("digits").cast(pl.Int64))
    col("parse_double", "convert", ["parse_double"], lambda t, df: t.parse_double("dbl"), lambda df: pl.col("dbl").cast(pl.Float64))
    col("parse_number", "convert", ["parse_number"], lambda t, df: t.parse_number("num"), lambda df: pl.col("num").str.replace_all(r"[^0-9.]", "").cast(pl.Float64))
    col("as_logical", "convert", ["as_logical"], lambda t, df: t.as_logical("lgl", strict=False), lambda df: pl.col("lgl").str.to_lowercase() == "true")
    col("parse_logical", "convert", ["parse_logical"], lambda t, df: t.parse_logical("lgl"), lambda df: pl.col("lgl").str.to_lowercase() == "true")
    col("as_string", "convert", ["as_string"], lambda t, df: t.as_string("whole"), lambda df: pl.col("whole").cast(pl.String))
    col("as_character", "convert", ["as_character"], lambda t, df: t.as_character("whole"), lambda df: pl.col("whole").cast(pl.String))
    col("as_date", "convert", ["as_date"], lambda t, df: t.as_date("date"), lambda df: pl.col("date").str.to_date("%Y-%m-%d"))
    col("ymd", "convert", ["ymd"], lambda t, df: t.ymd("date"), lambda df: pl.col("date").str.to_date("%Y-%m-%d"))
    col("mdy", "convert", ["mdy"], lambda t, df: t.mdy("mdy"), lambda df: pl.col("mdy").str.to_date("%m/%d/%Y"))
    col("dmy", "convert", ["dmy"], lambda t, df: t.dmy("dmy"), lambda df: pl.col("dmy").str.to_date("%d/%m/%Y"))
    col("ym", "convert", ["ym"], lambda t, df: t.ym("ym"), lambda df: pl.col("ym").str.to_date("%Y-%m"))
    col("parse_date", "convert", ["parse_date"], lambda t, df: t.parse_date("date", "%Y-%m-%d"), lambda df: pl.col("date").str.to_date("%Y-%m-%d"))
    col("as_datetime", "convert", ["as_datetime"], lambda t, df: t.as_datetime("when"), lambda df: pl.col("when").str.to_datetime("%Y-%m-%d %H:%M:%S", time_unit="us"))
    col("parse_datetime", "convert", ["parse_datetime"], lambda t, df: t.parse_datetime("when", "%Y-%m-%d %H:%M:%S"), lambda df: pl.col("when").str.to_datetime("%Y-%m-%d %H:%M:%S", time_unit="us"))

    def _clock(fmt: str):
        return (
            pl.col("when").str.to_datetime(fmt, time_unit="us")
            .dt.replace_time_zone("UTC", ambiguous="latest", non_existent="null")
        )

    col("ymd_hms", "convert", ["ymd_hms"], lambda t, df: t.ymd_hms("when"), lambda df: _clock("%Y-%m-%d %H:%M:%S"))
    col("ymd_hm", "convert", ["ymd_hm"], lambda t, df: t.ymd_hm("hm"), lambda df: pl.col("hm").str.to_datetime("%Y-%m-%d %H:%M", time_unit="us").dt.replace_time_zone("UTC", ambiguous="latest", non_existent="null"))
    col("mdy_hms", "convert", ["mdy_hms"], lambda t, df: t.mdy_hms("mdy_when"), lambda df: pl.col("mdy_when").str.to_datetime("%m/%d/%Y %H:%M:%S", time_unit="us").dt.replace_time_zone("UTC", ambiguous="latest", non_existent="null"))
    col("dmy_hms", "convert", ["dmy_hms"], lambda t, df: t.dmy_hms("dmy_when"), lambda df: pl.col("dmy_when").str.to_datetime("%d/%m/%Y %H:%M:%S", time_unit="us").dt.replace_time_zone("UTC", ambiguous="latest", non_existent="null"))

    def tidy_problems(df, extra=None):
        import tidy3 as t

        frame = t.tidy(df) >> t.mutate(out=t.ymd("date"))
        return t.problems(frame).collect()

    def raw_problems(df, extra=None):
        return pl.DataFrame(schema={"row": pl.Int64, "column": pl.String, "original": pl.String, "expected": pl.String})

    add(Case("problems", "convert", ["problems"], tidy_problems, raw_problems))

    def tidy_stop(df, extra=None):
        import tidy3 as t

        frame = t.tidy(df) >> t.mutate(out=t.ymd("date")) >> t.select("out")
        return t.stop_for_problems(frame).collect()

    add(Case("stop_for_problems", "convert", ["stop_for_problems"], tidy_stop, lambda df, extra=None: df.select(pl.col("date").str.to_date("%Y-%m-%d").alias("out"))))

    col("year", "dates", ["year"], lambda t, df: t.year("day_d"), lambda df: pl.col("day_d").dt.year().cast(pl.Int64))
    col("month", "dates", ["month"], lambda t, df: t.month("day_d"), lambda df: pl.col("day_d").dt.month().cast(pl.Int64))
    col("day", "dates", ["day"], lambda t, df: t.day("day_d"), lambda df: pl.col("day_d").dt.day().cast(pl.Int64))
    col("yday", "dates", ["yday"], lambda t, df: t.yday("day_d"), lambda df: pl.col("day_d").dt.ordinal_day().cast(pl.Int64))
    col("week", "dates", ["week"], lambda t, df: t.week("day_d"), lambda df: (((pl.col("day_d").dt.ordinal_day() - 1) // 7) + 1).cast(pl.Int64))
    col("isoweek", "dates", ["isoweek"], lambda t, df: t.isoweek("day_d"), lambda df: pl.col("day_d").dt.week().cast(pl.Int64))
    col("quarter", "dates", ["quarter"], lambda t, df: t.quarter("day_d"), lambda df: (((pl.col("day_d").dt.month() - 1) // 3) + 1).cast(pl.Int64))
    col("wday", "dates", ["wday"], lambda t, df: t.wday("day_d"), lambda df: (pl.col("day_d").dt.weekday() % 7 + 1).cast(pl.Int64))
    col("hour", "dates", ["hour"], lambda t, df: t.hour("at"), lambda df: pl.col("at").dt.hour().cast(pl.Int64))
    col("minute", "dates", ["minute"], lambda t, df: t.minute("at"), lambda df: pl.col("at").dt.minute().cast(pl.Int64))
    col("second", "dates", ["second"], lambda t, df: t.second("at"), lambda df: pl.col("at").dt.second().cast(pl.Float64))
    col("days_in_month", "dates", ["days_in_month"], lambda t, df: t.days_in_month("day_d"), lambda df: pl.col("day_d").dt.month_end().dt.day().cast(pl.Int64))
    col("floor_date", "dates", ["floor_date"], lambda t, df: t.floor_date("day_d", "month"), lambda df: pl.col("day_d").dt.truncate("1mo"))
    col("ceiling_date", "dates", ["ceiling_date"], lambda t, df: t.ceiling_date("day_d", "month"), lambda df: pl.col("day_d").dt.truncate("1mo").dt.offset_by("1mo"))
    col("months", "dates", ["months", "add_with_rollback"], lambda t, df: t.add_with_rollback("day_d", t.months(1)), lambda df: pl.col("day_d").dt.offset_by("1mo"))
    col("years", "dates", ["years"], lambda t, df: t.add_with_rollback("day_d", t.years(1)), lambda df: pl.col("day_d").dt.offset_by("1y"))
    col(
        "with_tz", "dates", ["with_tz"],
        lambda t, df: t.with_tz("at", "America/New_York"),
        lambda df: pl.col("at").dt.replace_time_zone("UTC").dt.convert_time_zone("America/New_York"),
    )
    col(
        "force_tz", "dates", ["force_tz"],
        lambda t, df: t.force_tz("at", "UTC"),
        lambda df: pl.col("at").dt.replace_time_zone("UTC", ambiguous="latest", non_existent="null"),
    )
    col(
        "excel_numeric_to_date", "dates", ["excel_numeric_to_date"],
        lambda t, df: t.excel_numeric_to_date("serial"),
        lambda df: (pl.lit(date(1899, 12, 30)) + pl.duration(days=pl.col("serial"))).cast(pl.Date),
    )

    # ── strings ────────────────────────────────────────────────────────
    col("str_detect", "string", ["str_detect", "fixed"], lambda t, df: t.str_detect("species", t.fixed("el")), lambda df: pl.col("species").str.contains("el", literal=True))
    col("str_starts", "string", ["str_starts", "coll"], lambda t, df: t.str_starts("species", t.coll("Ad")), lambda df: pl.col("species").str.starts_with("Ad"))
    col("str_ends", "string", ["str_ends"], lambda t, df: t.str_ends("species", "ie"), lambda df: pl.col("species").str.ends_with("ie"))
    col("str_count", "string", ["str_count", "regex"], lambda t, df: t.str_count("species", t.regex("e")), lambda df: pl.col("species").str.count_matches("e", literal=True))
    col("str_like", "string", ["str_like"], lambda t, df: t.str_like("species", "A%"), lambda df: pl.col("species").str.contains(r"(?s)^A.*$"))
    col("str_ilike", "string", ["str_ilike"], lambda t, df: t.str_ilike("species", "a%"), lambda df: pl.col("species").str.contains(r"(?si)^a.*$"))
    col("str_equal", "string", ["str_equal"], lambda t, df: t.str_equal("species", "Adelie"), lambda df: pl.col("species").str.normalize("NFC") == pl.lit("Adelie"))
    col("str_length", "string", ["str_length"], lambda t, df: t.str_length("species"), lambda df: pl.col("species").str.len_chars())
    col("str_width", "string", ["str_width"], lambda t, df: t.str_width("species"), lambda df: pl.col("species").map_elements(_width, return_dtype=pl.Int64))
    col("str_sub", "string", ["str_sub"], lambda t, df: t.str_sub("species", 1, 3), lambda df: _sub("species", 1, 3))
    col("str_sub_all", "string", ["str_sub_all"], lambda t, df: t.str_sub_all("species", [1], [3]), lambda df: pl.concat_list([_sub("species", 1, 3)]))
    col("str_trunc", "string", ["str_trunc"], lambda t, df: t.str_trunc("species", 4, ellipsis=".."), lambda df: pl.when(pl.col("species").str.len_chars() > 4).then(pl.col("species").str.slice(0, 2) + pl.lit("..")).otherwise(pl.col("species")))
    col("word", "string", ["word"], lambda t, df: t.word("txt", 1), lambda df: pl.col("txt").str.split(" ").list.get(0))
    col("str_replace", "string", ["str_replace"], lambda t, df: t.str_replace("species", "e", "E"), lambda df: pl.col("species").str.replace("e", "E", literal=True))
    col("str_replace_all", "string", ["str_replace_all"], lambda t, df: t.str_replace_all("species", "e", "E"), lambda df: pl.col("species").str.replace_all("e", "E", literal=True))
    col("str_remove", "string", ["str_remove"], lambda t, df: t.str_remove("species", "e"), lambda df: pl.col("species").str.replace("e", "", literal=True))
    col("str_remove_all", "string", ["str_remove_all"], lambda t, df: t.str_remove_all("species", "e"), lambda df: pl.col("species").str.replace_all("e", "", literal=True))
    col("str_replace_na", "string", ["str_replace_na"], lambda t, df: t.str_replace_na("note"), lambda df: pl.col("note").fill_null("NA"))
    col("str_to_upper", "string", ["str_to_upper"], lambda t, df: t.str_to_upper("species"), lambda df: pl.col("species").str.to_uppercase())
    col("str_to_lower", "string", ["str_to_lower"], lambda t, df: t.str_to_lower("species"), lambda df: pl.col("species").str.to_lowercase())
    col("str_to_title", "string", ["str_to_title"], lambda t, df: t.str_to_title("txt"), lambda df: pl.col("txt").str.to_titlecase())
    col(
        "str_to_sentence", "string", ["str_to_sentence"],
        lambda t, df: t.str_to_sentence("species"),
        lambda df: pl.concat_str([pl.col("species").str.slice(0, 1).str.to_uppercase(), pl.col("species").str.slice(1).str.to_lowercase()]),
    )
    col("str_trim", "string", ["str_trim"], lambda t, df: t.str_trim("padded"), lambda df: pl.col("padded").str.strip_chars())
    col("str_squish", "string", ["str_squish"], lambda t, df: t.str_squish("padded"), lambda df: pl.col("padded").str.strip_chars().str.replace_all(r"\s+", " "))
    col("str_pad", "string", ["str_pad"], lambda t, df: t.str_pad("species", 12), lambda df: pl.col("species").str.pad_start(12, " "))
    col("str_dup", "string", ["str_dup"], lambda t, df: t.str_dup("species", 2), lambda df: pl.concat_str([pl.col("species"), pl.col("species")]))
    col("str_wrap", "string", ["str_wrap"], lambda t, df: t.str_wrap("words", width=8), lambda df: pl.col("words").map_elements(_wrap, return_dtype=pl.String))
    col("str_extract", "string", ["str_extract"], lambda t, df: t.str_extract("species", "el"), lambda df: pl.col("species").str.extract("el", 0))
    col("str_extract_all", "string", ["str_extract_all"], lambda t, df: t.str_extract_all("species", "e"), lambda df: pl.col("species").str.extract_all("e"))
    col(
        "str_locate", "string", ["str_locate"],
        lambda t, df: t.str_locate("species", "el"),
        lambda df: _locate("species", "el"),
    )
    col("str_c", "string", ["str_c"], lambda t, df: t.str_c(t.col("species"), " ", t.col("island")), lambda df: pl.concat_str([pl.col("species"), pl.lit(" "), pl.col("island")]))
    col("str_glue", "string", ["str_glue"], lambda t, df: t.str_glue("{species}-{island}"), lambda df: pl.concat_str([pl.col("species"), pl.lit("-"), pl.col("island")]))
    col("str_escape", "string", ["str_escape"], lambda t, df: t.str_escape("code"), lambda df: pl.col("code").str.replace_all(r"([.^$|?*+()\[\]{}\\])", r"\${1}"))
    col("str_conv", "string", ["str_conv"], lambda t, df: t.str_conv("species", "utf-8"), lambda df: pl.col("species"))
    col("str_split", "string", ["str_split", "boundary"], lambda t, df: t.str_split("txt", t.boundary("word")), lambda df: pl.col("txt").str.extract_all(r"\w+(?:['’]\w+)*"))
    col("str_split_fixed", "string", ["str_split_fixed"], lambda t, df: t.str_split_fixed("txt", " ", 2), lambda df: _split_fixed("txt"))
    col("str_split_i", "string", ["str_split_i"], lambda t, df: t.str_split_i("txt", " ", 2), lambda df: pl.col("txt").str.split(" ").list.get(1))

    def tidy_flatten(df, extra=None):
        import tidy3 as t

        return (t.tidy(df) >> t.group_by("g") >> t.summarise(out=t.str_flatten("species", collapse="|"))).collect()

    def raw_flatten(df, extra=None):
        return df.group_by("g", maintain_order=True).agg(pl.col("species").sort_by("id").str.join("|").alias("out")).sort("g")

    add(Case("str_flatten", "string", ["str_flatten"], tidy_flatten, raw_flatten))

    def tidy_flatten_comma(df, extra=None):
        import tidy3 as t

        return (t.tidy(df) >> t.group_by("g") >> t.summarise(out=t.str_flatten_comma("species"))).collect()

    def raw_flatten_comma(df, extra=None):
        return df.group_by("g", maintain_order=True).agg(pl.col("species").sort_by("id").str.join(", ").alias("out")).sort("g")

    add(Case("str_flatten_comma", "string", ["str_flatten_comma"], tidy_flatten_comma, raw_flatten_comma))

    def tidy_glue_data(df, extra=None):
        import tidy3 as t

        return t.str_glue_data(df.select("species", "island"), "{species}-{island}")

    def raw_glue_data(df, extra=None):
        return (df["species"] + "-" + df["island"]).to_list()

    add(Case("str_glue_data", "string", ["str_glue_data"], tidy_glue_data, raw_glue_data))

    def tidy_sort(df, extra=None):
        import tidy3 as t

        return t.str_sort(df["code"].to_list())

    add(Case("str_sort", "string", ["str_sort"], tidy_sort, lambda df, extra=None: df["code"].sort().to_list()))

    def tidy_order(df, extra=None):
        import tidy3 as t

        return t.str_order(df["code"].to_list())

    add(Case("str_order", "string", ["str_order"], tidy_order, lambda df, extra=None: (df["code"].arg_sort() + 1).to_list()))

    def tidy_unique(df, extra=None):
        import tidy3 as t

        return t.str_unique(df["species"].to_list())

    add(Case("str_unique", "string", ["str_unique"], tidy_unique, lambda df, extra=None: df["species"].unique(maintain_order=True).to_list()))

    def tidy_subset(df, extra=None):
        import tidy3 as t

        return t.str_subset(df["species"].to_list(), "el")

    add(Case("str_subset", "string", ["str_subset"], tidy_subset, lambda df, extra=None: df.filter(pl.col("species").str.contains("el", literal=True))["species"].to_list()))

    def tidy_which(df, extra=None):
        import tidy3 as t

        return t.str_which(df["species"].to_list(), "el")

    def raw_which(df, extra=None):
        hits = df.with_row_index("row").filter(pl.col("species").str.contains("el", literal=True))
        return (hits["row"] + 1).to_list()

    add(Case("str_which", "string", ["str_which"], tidy_which, raw_which))

    def tidy_split_1(df, extra=None):
        import tidy3 as t

        return t.str_split_1("a,b,c", ",")

    add(Case("str_split_1", "string", ["str_split_1"], tidy_split_1, lambda df, extra=None: ["a", "b", "c"]))

    def tidy_invert(df, extra=None):
        import tidy3 as t

        return t.invert_match([{"start": 2, "end": 3}])

    add(Case(
        "invert_match", "string", ["invert_match"], tidy_invert,
        lambda df, extra=None: [{"start": 0, "end": 1}, {"start": 4, "end": -1}],
    ))

    def tidy_locate_all(df, extra=None):
        import tidy3 as t

        return (t.tidy(df) >> t.mutate(out=t.str_locate_all("species", "e")) >> t.select("out")).collect()

    def raw_locate_all(df, extra=None):
        import re

        pattern = re.compile("e")

        def locate(value: str):
            return [{"start": match.start() + 1, "end": match.end()} for match in pattern.finditer(value)]

        return df.select(pl.col("species").map_elements(locate, return_dtype=pl.List(pl.Struct({"start": pl.Int32, "end": pl.Int32}))).alias("out"))

    add(Case("str_locate_all", "string", ["str_locate_all"], tidy_locate_all, raw_locate_all))

    def tidy_match(df, extra=None):
        import tidy3 as t

        return (t.tidy(df) >> t.mutate(out=t.str_match("species", "(A)(.*)")) >> t.select("out")).collect()

    def raw_match(df, extra=None):
        value = pl.col("species")
        pattern = "(A)(.*)"
        return df.select(pl.struct(
            value.str.extract(pattern, 0).alias("match"),
            value.str.extract(pattern, 1).alias("1"),
            value.str.extract(pattern, 2).alias("2"),
        ).alias("out"))

    add(Case("str_match", "string", ["str_match"], tidy_match, raw_match))

    def tidy_match_all(df, extra=None):
        import tidy3 as t

        return (t.tidy(df) >> t.mutate(out=t.str_match_all("species", "(e)")) >> t.select("out")).collect()

    def raw_match_all(df, extra=None):
        import re

        pattern = re.compile("(e)")

        def rows(value: str):
            return [{"match": match.group(0), "1": match.group(1)} for match in pattern.finditer(value)]

        dtype = pl.List(pl.Struct({"match": pl.String, "1": pl.String}))
        return df.select(pl.col("species").map_elements(rows, return_dtype=dtype).alias("out"))

    add(Case("str_match_all", "string", ["str_match_all"], tidy_match_all, raw_match_all))

    # ── factors ────────────────────────────────────────────────────────
    for how, names in (
        ("infreq", ["fct_infreq"]),
        ("reorder", ["fct_reorder"]),
        ("rev", ["fct_rev"]),
        ("relevel", ["fct_relevel"]),
    ):
        tidy, raw = _grouped_factor(how)
        add(Case(f"fct_{how}" if how != "infreq" else "fct_infreq", "forcats", names, tidy, raw))

    from tidy3.bench_rest import attach

    attach(add, Case)
    return cases


def _locate(column: str, pattern: str) -> pl.Expr:
    value = pl.col(column)
    before = value.str.extract(rf"^((?s:.*?))(?:{pattern})", 1)
    found = value.str.extract(pattern, 0)
    start = (before.str.len_chars() + 1).cast(pl.Int32)
    end = start + found.str.len_chars().cast(pl.Int32) - 1
    return pl.struct(start.alias("start"), end.alias("end"))


def _split_fixed(column: str) -> pl.Expr:
    fields = pl.col(column).str.splitn(" ", 2)
    pieces = [
        fields.struct.field(name).fill_null("") for name in ("field_0", "field_1")
    ]
    return pl.when(pl.col(column).is_null()).then(None).otherwise(pl.concat_list(pieces))


_INFO: dict | None = None


def classification() -> dict:
    """Map every public name to one case or one explicit reason."""
    global _INFO
    if _INFO is None:
        _INFO = _classify()
    return _INFO


def _classify() -> dict:
    import tidy3
    from tidy3.frame import TidyFrame

    cases = _cases()
    functions: dict[str, str] = {}
    duplicates: list[str] = []
    case_names: set[str] = set()
    for case in cases:
        if case.name in case_names:
            duplicates.append(f"case {case.name}")
        case_names.add(case.name)
        for name in case.functions:
            if name in functions or name in REASONS:
                duplicates.append(name)
            functions[name] = case.name
    methods: dict[str, str] = {}
    for case in cases:
        for name in case.methods:
            if name in methods or name in METHOD_REASONS:
                duplicates.append(f"method {name}")
            methods[name] = case.name
    public_methods = [name for name in dir(TidyFrame) if not name.startswith("_")]
    unmapped_methods = []
    for name in public_methods:
        if name in methods or name in METHOD_REASONS or name in REASONS:
            continue
        owner = functions.get(name)
        if owner is None:
            unmapped_methods.append(name)
            continue
        methods[name] = owner
    unmapped = [
        name for name in tidy3.__all__
        if name not in functions and name not in REASONS
    ]
    return {
        "cases": cases,
        "functions": functions,
        "methods": methods,
        "unmapped": unmapped,
        "unmapped_methods": unmapped_methods,
        "duplicates": duplicates,
    }


def _describe(df: pl.DataFrame) -> pl.DataFrame:
    """Polars describe plus an n_unique row, matching tidy3 summary()."""
    desc = df.describe(percentiles=[0.25, 0.5, 0.75])
    extra_row = {"statistic": "n_unique"}
    for name in df.columns:
        extra_row[name] = df[name].n_unique()
    extra = pl.DataFrame(extra_row)
    for name in desc.columns:
        if name != "statistic" and desc.schema[name].is_numeric():
            extra = extra.with_columns(pl.col(name).cast(pl.Float64, strict=False))
    index = desc["statistic"].to_list().index("null_count") + 1
    return pl.concat([desc.slice(0, index), extra, desc.slice(index)], how="diagonal_relaxed")


def markdown_report(report: dict) -> str:
    ranked = sorted(report["comparisons"], key=lambda row: row["excess_ms"], reverse=True)
    lines = [
        "# Polars-relative public API timings",
        "",
        f"Python {report['python']}; Polars {report['polars']}; {report['platform']}.",
        "",
        f"{report['rows']:,} rows. One correctness comparison, {report['warmup']} warm-up(s), best of {report['repeat']} alternating rounds.",
        "Preparation is outside the samples. Cases are ranked by absolute excess time (tidy3 minus raw Polars).",
        "",
        "| Case | Family | Rows | Raw Polars | tidy3 | Ratio | Excess |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in ranked:
        ratio = row["tidy3_over_polars"]
        ratio_text = "" if ratio is None else f"{ratio:.2f}x"
        lines.append(
            f"| {row['name']} | {row['family']} | {row['rows']:,} | {row['raw_polars_ms']:.2f} ms | {row['tidy3_ms']:.2f} ms | {ratio_text} | {row['excess_ms']:.2f} ms |"
        )
    lines += [
        "",
        f"{len(report['reasons'])} exports and {len(report['method_reasons'])} methods have no raw Polars comparison.",
        "",
    ]
    return "\n".join(lines)


def _selected(info: dict, names: list[str] | None, family: str | None) -> list[Case]:
    chosen = []
    for case in info["cases"]:
        if names and case.name not in names:
            continue
        if family and case.family != family:
            continue
        chosen.append(case)
    return chosen


def _main() -> None:
    info = classification()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--case", action="append")
    parser.add_argument("--family")
    parser.add_argument("--output", type=Path, default=Path("api-polars.json"))
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    if info["unmapped"] or info["unmapped_methods"] or info["duplicates"]:
        parser.error(
            "incomplete classification: "
            f"unmapped={info['unmapped']} methods={info['unmapped_methods']} duplicates={info['duplicates']}"
        )
    if args.list:
        for case in info["cases"]:
            print(f"{case.family}: {case.name} [{', '.join(case.functions)}]")
        for name, reason in sorted(REASONS.items()):
            print(f"reason: {name} — {reason}")
        return
    if args.repeat < 1 or args.warmup < 0 or args.rows < 8:
        parser.error("rows must be at least 8, repeat positive, and warmup non-negative")
    chosen = _selected(info, args.case, args.family)
    if not chosen:
        parser.error("no cases matched")
    frame = work(args.rows)
    report = {
        "python": platform.python_version(),
        "polars": pl.__version__,
        "platform": f"{platform.system()} {platform.machine()}",
        "rows": args.rows,
        "repeat": args.repeat,
        "warmup": args.warmup,
        "reasons": REASONS,
        "method_reasons": METHOD_REASONS,
        "comparisons": [],
    }
    for case in chosen:
        row = run_case(case, frame, repeat=args.repeat, warmup=args.warmup)
        ratio = row["tidy3_over_polars"]
        ratio_text = "" if ratio is None else f"{ratio:.2f}x"
        print(
            f"{case.name}: raw Polars {row['raw_polars_ms']:.2f} ms, tidy3 {row['tidy3_ms']:.2f} ms ({ratio_text}, excess {row['excess_ms']:.2f} ms)",
            flush=True,
        )
        report["comparisons"].append(row)
    report["comparisons"].sort(key=lambda row: row["excess_ms"], reverse=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    args.output.with_suffix(".md").write_text(markdown_report(report))


if __name__ == "__main__":
    _main()
