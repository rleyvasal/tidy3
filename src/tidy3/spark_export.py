"""Run a tidy3 pipeline on Spark.

:meth:`TidyFrame.to_spark` translates the lazy Polars query that tidy3
built into PySpark DataFrame calls, the way :mod:`tidy3.polars_export`
prints it as Polars code: both read Polars' serialized plan. The query
already holds dplyr's rules (a ``mean`` with a missing value is missing,
``arrange`` puts missing values last), so no verb is reimplemented and the
Spark result is tidy3's result.

A step with no Spark translation raises :class:`SparkExportError` naming the
step. Columns computed by a Python function (tidy3's text parsers and
R-style ``round``, for example) are refused, as in ``to_polars_code()``.
Steps that built their query from values in the data (factor levels,
``compute()``) run: the query holds those values, read by Polars from the
same sources Spark reads.

Row order. Polars keeps row order; Spark does not. Each scan numbers its
rows, and each translated step carries the ordering of its rows. A stable
sort puts its keys in front of that ordering, so tidy3's stacked sorts
(group keys, then the ``arrange`` key) become one Spark sort, which
:func:`to_spark` applies last.
"""

from __future__ import annotations

import functools
import glob
import io
import json
import re
import sys
import warnings
from dataclasses import dataclass, field
from typing import Any

import polars as pl

from tidy3.polars_export import ExportError, _legacy_join, _udf_name

try:
    from pyspark.sql import Column, DataFrame, SparkSession, Window
    from pyspark.sql import functions as F
    from pyspark.sql import types as T
except ImportError:  # pragma: no cover - exercised without the extra
    raise ImportError("to_spark() needs PySpark: pip install 'tidy3[spark]'") from None

__all__ = ["SparkExportError", "to_spark"]


class SparkExportError(ExportError):
    """The pipeline has a step with no Spark translation."""


def to_spark(frame: Any, spark: SparkSession | None = None) -> DataFrame:
    """A Spark DataFrame that computes *frame*'s pipeline, in tidy3's row order.

    *spark* defaults to the active Spark session. Nothing runs until Spark
    is asked for rows.
    """
    if spark is None:
        spark = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
    lf = frame._lf if hasattr(frame, "_lf") else frame
    with warnings.catch_warnings():
        # Polars marks JSON plan serialization deprecated; it is the only
        # readable form of the plan.
        warnings.simplefilter("ignore")
        plan = json.loads(lf.serialize(format="json"))
    node = _Translator(spark).plan(plan)
    df = node.df.orderBy(*_sort_columns(node.order)) if node.order else node.df
    return df.select(*[_col(name) for name in node.columns])


# ── translated steps ────────────────────────────────────────────────────

# One ordering key: (column name, descending, nulls_last).
Key = tuple[str, bool, bool]


@dataclass
class _Node:
    """A translated plan step: the Spark frame, its user columns, its row order.

    Columns of ``df`` not in ``columns`` are hidden ones the translator added
    (row numbers, sort keys). They ride along until :func:`to_spark` drops them.
    """

    df: DataFrame
    columns: list[str]
    order: list[Key] = field(default_factory=list)

    @property
    def hidden(self) -> list[str]:
        return [c for c in self.df.columns if c not in self.columns]


def _true_divide(a: Column, b: Column) -> Column:
    """Polars' ``a / b``, the same in either Spark ANSI mode.

    A zero divisor gives ``inf``, ``-inf``, or ``nan`` for ``0 / 0``, where
    Spark gives null (ANSI off) or raises (ANSI on). A zero's sign counts:
    ``1 / -0.0`` is ``-inf``.
    """
    x, y = a.cast(T.DoubleType()), b.cast(T.DoubleType())
    # The sign of a zero divisor, read from its text ("-0.0"): Spark's
    # comparisons, signum() and atan2() all treat -0.0 as 0.0.
    sign = F.when(F.startswith(y.cast(T.StringType()), F.lit("-")), F.lit(-1.0)).otherwise(F.lit(1.0))
    by_zero = F.when(F.isnan(x) | (x == 0), F.lit(float("nan"))).otherwise(F.signum(x) * sign * F.lit(float("inf")))
    return F.when(y == 0, by_zero).otherwise(x / F.when(y != 0, y))


def _col(name: str) -> Column:
    return F.col("`" + name.replace("`", "``") + "`")


def _sort_columns(order: list[Key]) -> list[Column]:
    columns = []
    for name, descending, nulls_last in order:
        column = _col(name)
        if descending:
            columns.append(column.desc_nulls_last() if nulls_last else column.desc_nulls_first())
        else:
            columns.append(column.asc_nulls_last() if nulls_last else column.asc_nulls_first())
    return columns


def _floor_mod(a: Column, b: Column) -> Column:
    """Polars' remainder, which takes the divisor's sign, like Python's and R's ``%%``.

    Spark's ``%`` takes the dividend's sign, and ``pmod`` is never negative.
    """
    r = a % b
    return F.when((r != 0) & ((r < 0) != (b < 0)), r + b).otherwise(r)


def _contains_agg(node: Any) -> bool:
    """Whether a serialized expression is an aggregation (one value per group)."""
    if isinstance(node, dict):
        if "Agg" in node or "Len" in node:
            return True
        if "Over" in node:
            return False
        if "Function" in node:
            function = node["Function"].get("function")
            if isinstance(function, dict) and isinstance(function.get("Boolean"), dict):
                if set(function["Boolean"]) & {"Any", "All"}:
                    return True
        return any(_contains_agg(value) for value in node.values())
    if isinstance(node, list):
        return any(_contains_agg(item) for item in node)
    return node == "Len"


def _is_scalar(node: Any) -> bool:
    """Whether a serialized expression is one value: every column inside an aggregation.

    Literals count as one value, so ``select(sum(x), lit("Total"))`` is one row
    in Polars, while ``x - mean(x)`` and windows are one value per row.
    """
    if isinstance(node, dict):
        if "Agg" in node or "Len" in node:
            return True
        if "Column" in node or "Columns" in node or "Over" in node or "Selector" in node:
            return False
        if "Function" in node:
            function = node["Function"].get("function")
            if isinstance(function, dict) and isinstance(function.get("Boolean"), dict):
                if set(function["Boolean"]) & {"Any", "All"}:
                    return True
        return all(_is_scalar(value) for value in node.values())
    if isinstance(node, list):
        return all(_is_scalar(item) for item in node)
    return node != "Wildcard"


def _int_literal(node: Any) -> int | None:
    """The value of a serialized integer literal, else None."""
    if isinstance(node, dict) and "Literal" in node:
        (kind, value), = node["Literal"].items()
        if kind == "Dyn" and "Int" in value:
            return int(value["Int"])
        if kind == "Scalar" and isinstance(value, dict):
            (_dtype, raw), = value.items()
            if isinstance(raw, int) and not isinstance(raw, bool):
                return raw
    return None


def _str_literal(node: Any) -> str | None:
    """The value of a serialized string literal, else None."""
    if isinstance(node, dict) and "Literal" in node:
        (kind, value), = node["Literal"].items()
        if kind == "Scalar" and isinstance(value, dict) and isinstance(value.get("String"), str):
            return value["String"]
    return None


def _is_null_literal(node: Any) -> bool:
    return isinstance(node, dict) and node.get("Literal", {}).get("Scalar", {}) == {"Null": "Null"}


# Polars writes regular expressions for Rust's regex crate; Spark runs Java's.
# (?U) gives Java Rust's Unicode \d, \s, \w and case folding; (?d) makes \n
# the only line end, as in Rust.
_JAVA_FLAGS = "(?Ud)"
# Characters regex::escape escapes: before Polars 2, a pattern with none is
# replaced literally, and so is its replacement.
_RUST_META = set("\\.+*?()|[]{}^$#&-~")
_POLARS_2 = int(pl.__version__.split(".")[0]) >= 2


@functools.cache
def _titlecase_letters() -> dict[str, str]:
    """Letters whose titlecase is not their uppercase: "ß" is "Ss", "ǆ" is "ǅ"."""
    letters = (chr(c) for c in range(sys.maxunicode + 1))
    return {c: c.title() for c in letters if c.title() != c.upper()}


def _java_regex(pattern: str) -> str:
    # Java takes scripts as \p{IsHan}, and categories with or without "Is".
    pattern = re.sub(r"\\([pP])\{(?!Is|In)(\w+)\}", r"\\\1{Is\2}", pattern)
    return pattern.replace("(?P<", "(?<")


def _java_quote(text: str) -> str:
    """Java's Pattern.quote: *text* as a literal pattern."""
    return "\\Q" + text.replace("\\E", "\\E\\\\E\\Q") + "\\E"


def _java_literal_replacement(text: str) -> str:
    """Java's Matcher.quoteReplacement: *text* as a literal replacement."""
    return text.replace("\\", "\\\\").replace("$", "\\$")


def _java_replacement(rust: str, shift: int = 0) -> str | None:
    """A Rust regex replacement (``$1``, ``${name}``, ``$$``) for Java, else None.

    Group numbers move up by *shift*. Rust reads the longest name after
    ``$``, so ``$1x`` names a group "1x", which no pattern has: empty.
    """
    out, i, after_group = [], 0, False
    while i < len(rust):
        c = rust[i]
        if c == "$" and rust.startswith("$$", i):
            out.append("\\$")
            i, after_group = i + 2, False
            continue
        if c == "$":
            if rust.startswith("${", i) and "}" in rust[i:]:
                end = rust.index("}", i)
                name, i = rust[i + 2:end], end + 1
            else:
                j = i + 1
                while j < len(rust) and (rust[j].isascii() and (rust[j].isalnum() or rust[j] == "_")):
                    j += 1
                name, i = rust[i + 1:j], j
            if not name:
                out.append("\\$")
                after_group = False
            elif name.isdigit():
                out.append(f"${int(name) + shift}")
                after_group = True
            elif name[0].isdigit():
                after_group = False
            elif name.isascii() and name.isalnum() and name[0].isalpha():
                out.append("${" + name + "}")
                after_group = False
            else:
                return None
            continue
        # A digit straight after a group number would extend it in Java.
        out.append("\\" + c if c == "\\" or (after_group and c.isdigit()) else c)
        i, after_group = i + 1, False
    return "".join(out)


def _group_always_set(pattern: str) -> bool:
    """Whether every group of *pattern* is set, and not empty, when it matches.

    Spark's regexp_extract gives "" both for an empty group and for one that
    took no part, where Polars gives "" and null; with no alternation,
    optional or zero-width repetition, or empty group, neither can happen.
    """
    plain = pattern.replace("(?", "(")
    return not (set("|?*") & set(plain) or "{0" in plain or "{," in plain or "()" in plain or not pattern)


# Units of a Polars duration string such as "1mo2d" or "-14d".
_MONTHS = {"y": 12, "q": 3, "mo": 1}
_DAYS = {"w": 7, "d": 1}
_MICROS = {"h": 3_600_000_000, "m": 60_000_000, "s": 1_000_000, "ms": 1_000, "us": 1}


def _duration(text: str) -> tuple[int, int, int] | None:
    """(months, days, microseconds) of a Polars duration string, else None."""
    sign, body = (-1, text[1:]) if text.startswith("-") else (1, text)
    months = days = micros = 0
    for count, unit in re.findall(r"(\d+)([a-z]+)", body):
        n = int(count)
        if unit in _MONTHS:
            months += n * _MONTHS[unit]
        elif unit in _DAYS:
            days += n * _DAYS[unit]
        elif unit in _MICROS:
            micros += n * _MICROS[unit]
        elif unit == "ns" and n % 1000 == 0:
            micros += n // 1000
        else:
            return None
    if not body or re.sub(r"\d+[a-z]+", "", body):
        return None
    return sign * months, sign * days, sign * micros


_BINARY = {
    "Plus": lambda a, b: a + b, "Minus": lambda a, b: a - b, "Multiply": lambda a, b: a * b,
    "TrueDivide": _true_divide,
    "Gt": lambda a, b: a > b, "GtEq": lambda a, b: a >= b, "Lt": lambda a, b: a < b,
    "LtEq": lambda a, b: a <= b, "Eq": lambda a, b: a == b, "NotEq": lambda a, b: a != b,
    "EqValidity": lambda a, b: a.eqNullSafe(b), "NotEqValidity": lambda a, b: ~a.eqNullSafe(b),
    "And": lambda a, b: a & b, "Or": lambda a, b: a | b,
    "LogicalAnd": lambda a, b: a & b, "LogicalOr": lambda a, b: a | b,
}

_AGG = {
    "Mean": F.mean, "Sum": F.sum, "Min": F.min, "Max": F.max, "Median": F.median,
    "NUnique": F.count_distinct,
}

# Polars' data types, by serialized name, with a Spark type that holds every value.
_SIMPLE_TYPES = {
    "Boolean": T.BooleanType(), "String": T.StringType(), "Float64": T.DoubleType(),
    "Float32": T.FloatType(), "Int64": T.LongType(), "Int32": T.IntegerType(),
    "Int16": T.ShortType(), "Int8": T.ByteType(), "UInt8": T.ShortType(),
    "UInt16": T.IntegerType(), "UInt32": T.LongType(), "UInt64": T.DecimalType(20, 0),
    "Date": T.DateType(), "Null": T.NullType(),
}

# The window an aggregation runs over in a row context: the whole frame. A
# marker, since a Spark Window can only be built once a session exists.
_WHOLE = "whole frame"

# tidy3's collect-time checks. The summarise size check becomes a Spark
# raise_error(); the row checks are refused.
_SUMMARY_SIZE_CHECK = "_pl_check_summary_sizes"
_ROW_CHECKS = ("_pl_assert_rows.<locals>.validate", "_pl_guard_no_rows.<locals>.validate")

# CSV options a Spark read honours the same way, with their only allowed value.
_CSV_DEFAULTS = {
    "skip_rows": 0, "skip_lines": 0, "skip_rows_after_header": 0, "n_rows": None,
    "row_index": None, "ignore_errors": False,
}
_CSV_PARSE_DEFAULTS = {
    "quote_char": 34, "eol_char": 10, "encoding": "Utf8", "missing_is_null": True,
    "truncate_ragged_lines": False, "comment_prefix": None, "try_parse_dates": False,
    "decimal_comma": False,
}


def _spark_type(dtype: pl.DataType) -> T.DataType | None:
    """The Spark type for a Polars type, or None when there is no faithful one."""
    simple = _SIMPLE_TYPES.get(type(dtype).__name__ if not isinstance(dtype, type) else dtype.__name__)
    if simple is not None:
        return simple
    if isinstance(dtype, pl.Datetime) and dtype.time_zone is None and dtype.time_unit == "us":
        return T.TimestampNTZType()
    if isinstance(dtype, pl.Decimal) and dtype.precision is not None:
        return T.DecimalType(dtype.precision, dtype.scale)
    if isinstance(dtype, pl.List):
        inner = _spark_type(dtype.inner)
        return None if inner is None else T.ArrayType(inner)
    if isinstance(dtype, pl.Struct):
        fields = [(f.name, _spark_type(f.dtype)) for f in dtype.fields]
        if any(t is None for _n, t in fields):
            return None
        return T.StructType([T.StructField(n, t) for n, t in fields])
    return None


class _Translator:
    def __init__(self, spark: SparkSession) -> None:
        self.spark = spark
        self.hidden = 0
        # While a group_by's aggregations are translated: its keys' window, and
        # the columns to compute before grouping (see agg_input).
        self.group_window: Any = None
        self.pre_group: dict[str, Column] = {}

    def unsupported(self, what: str) -> SparkExportError:
        return SparkExportError(f"cannot run this pipeline on Spark yet: {what} has no Spark translation")

    def hidden_name(self, stem: str) -> str:
        self.hidden += 1
        return f"__tidy3_spark_{stem}{self.hidden}"

    # ── plan steps ──────────────────────────────────────────────────────

    # A plan node is ``{"Kind": body}``; some are wrapped in ``{"IR": {"dsl": node}}``.
    def plan(self, node: Any) -> _Node:
        if isinstance(node, dict) and set(node) == {"IR"}:
            return self.plan(node["IR"]["dsl"])
        if not isinstance(node, dict) or len(node) != 1:
            raise self.unsupported(f"the plan {str(node)[:60]}")
        (kind, body), = node.items()
        method = getattr(self, "p_" + kind, None)
        if method is None:
            raise self.unsupported(f"the Polars plan step {kind}")
        return method(body)

    def numbered(self, df: DataFrame) -> _Node:
        """Number a source's rows in the order Polars reads them."""
        row = self.hidden_name("row")
        return _Node(df.withColumn(row, F.monotonically_increasing_id()), list(df.columns), [(row, False, True)])

    def spark_schema(self, schema: pl.Schema) -> T.StructType:
        fields = []
        for name, dtype in schema.items():
            spark_type = _spark_type(dtype)
            if spark_type is None:
                raise self.unsupported(f"the column {name!r} of type {dtype}")
            fields.append(T.StructField(name, spark_type))
        return T.StructType(fields)

    def p_Scan(self, body: dict) -> _Node:
        sources = body["sources"]
        paths = sources.get("Paths") if isinstance(sources, dict) else None
        if not paths:
            raise self.unsupported("a scan of in-memory sources")
        paths = [path["inner"] if isinstance(path, dict) else path for path in paths]
        (kind, options), = body["scan_type"].items()
        args = body.get("unified_scan_args") or {}
        if args.get("row_index") or args.get("pre_slice") or args.get("include_file_paths"):
            raise self.unsupported("a scan with a row index, slice, or file-path column")
        if (args.get("hive_options") or {}).get("enabled"):
            raise self.unsupported("a hive-partitioned scan")
        # The column types Polars gives this scan, overrides and inference included.
        schema = pl.LazyFrame.deserialize(io.StringIO(json.dumps({"Scan": body})), format="json").collect_schema()
        spark_schema = self.spark_schema(schema)
        if kind == "Csv":
            reader = self.csv_reader(options["options"], spark_schema)
        elif kind == "Parquet":
            # Read, then hold each column in the type Polars gives it.
            reader = lambda path: self.spark.read.parquet(path).select(  # noqa: E731
                *[_col(f.name).cast(f.dataType) for f in spark_schema.fields]
            )
        else:
            raise self.unsupported(f"a {kind} scan")
        files = self.expand(paths, glob_on=args.get("glob", True))
        if len(files) == 1:
            return self.numbered(reader(files[0]))
        # Spark orders a scan's files by size; Polars reads them in order. Read
        # each file on its own and order by (file, row).
        part, row = self.hidden_name("file"), self.hidden_name("row")
        frames = [
            reader(path).withColumn(part, F.lit(i)).withColumn(row, F.monotonically_increasing_id())
            for i, path in enumerate(files)
        ]
        df = functools.reduce(lambda a, b: a.unionByName(b), frames)
        return _Node(df, list(schema), [(part, False, True), (row, False, True)])

    def expand(self, paths: list[str], *, glob_on: bool) -> list[str]:
        files: list[str] = []
        for path in paths:
            if glob_on and any(ch in path for ch in "*?["):
                if "://" in path:
                    raise self.unsupported("a glob over remote files")
                matched = sorted(glob.glob(path))
                if not matched:
                    raise FileNotFoundError(path)
                files.extend(matched)
            else:
                files.append(path)
        return files

    def csv_reader(self, options: dict, schema: T.StructType):
        parse = options["parse_options"]
        for key, default in _CSV_DEFAULTS.items():
            if options.get(key, default) != default:
                raise self.unsupported(f"the scan_csv option {key}")
        for key, default in _CSV_PARSE_DEFAULTS.items():
            if parse.get(key, default) != default:
                raise self.unsupported(f"the scan_csv option {key}")
        nulls = parse.get("null_values")
        null_value = None
        if nulls:
            (kind, value), = nulls.items()
            if kind == "AllColumnsSingle":
                null_value = value
            elif kind == "AllColumns" and len(value) == 1:
                null_value = value[0]
            else:
                raise self.unsupported("several or per-column null_values")
        read = {
            "header": options["has_header"], "sep": chr(parse["separator"]), "schema": schema,
            # Polars quotes like RFC 4180: a quote inside quotes is doubled.
            "quote": '"', "escape": '"', "mode": "FAILFAST", "enforceSchema": False,
        }
        if null_value is not None:
            read["nullValue"] = null_value
        return lambda path: self.spark.read.csv(path, **read)

    def p_DataFrameScan(self, body: dict) -> _Node:
        data = pl.read_ipc_stream(io.BytesIO(bytes(body["df"])))
        schema = self.spark_schema(data.schema)
        # Spark rejects unsigned Arrow integers: widen them to the signed type
        # that holds them, as the Spark schema does.
        wider = {pl.UInt8: pl.Int16, pl.UInt16: pl.Int32, pl.UInt32: pl.Int64, pl.UInt64: pl.Decimal(20, 0)}
        casts = {name: wider[dtype.base_type()] for name, dtype in data.schema.items() if dtype.base_type() in wider}
        if casts:
            data = data.cast(casts)
        table = data.to_arrow(compat_level=pl.CompatLevel.oldest())
        return self.numbered(self.spark.createDataFrame(table, schema=schema))

    def p_Filter(self, body: dict) -> _Node:
        node = self.plan(body["input"])
        predicate = self.expr(body["predicate"], node, _WHOLE)
        # Spark refuses window functions in a WHERE clause (grouped filters,
        # slice_*); a column holding the predicate is allowed.
        keep = self.hidden_name("keep")
        df = node.df.withColumn(keep, predicate).filter(F.col(keep)).drop(keep)
        return _Node(df, node.columns, node.order)

    def p_HStack(self, body: dict) -> _Node:
        node = self.plan(body["input"])
        named = self.named(body["exprs"], node, _WHOLE)
        columns = node.columns + [name for name in named if name not in node.columns]
        return _Node(node.df.withColumns(named), columns, node.order)

    def p_Select(self, body: dict) -> _Node:
        node = self.plan(body["input"])
        exprs = body["expr"]
        if exprs and all(_is_scalar(e) for e in exprs) and any(_contains_agg(e) for e in exprs):
            # summarise() without groups, or a totals row: one row.
            self.group_window, self.pre_group = Window.partitionBy(), {}
            named = self.named(exprs, node, "agg")
            pre, self.group_window, self.pre_group = self.pre_group, None, {}
            df = node.df.withColumns(pre) if pre else node.df
            return _Node(df.agg(*[c.alias(n) for n, c in named.items()]), list(named))
        named = self.named(exprs, node, _WHOLE)
        keep = [_col(name) for name in node.hidden if name not in named]
        df = node.df.select(*[c.alias(n) for n, c in named.items()], *keep)
        return _Node(df, list(named), node.order)

    def p_GroupBy(self, body: dict) -> _Node:
        if body.get("apply") is not None:
            raise self.unsupported("a Python function per group")
        options = body.get("options") or {}
        if options.get("dynamic") or options.get("rolling"):
            raise self.unsupported("a dynamic or rolling group_by")
        if body.get("maintain_order"):
            raise self.unsupported("a group_by that keeps the groups' order of appearance")
        node = self.plan(body["input"])
        keys = self.named(body["keys"], node, _WHOLE)
        self.group_window, self.pre_group = Window.partitionBy(*keys.values()), {}
        aggs = self.named(body["aggs"], node, "agg")
        pre, self.group_window, self.pre_group = self.pre_group, None, {}
        df = node.df.withColumns(pre) if pre else node.df
        df = df.groupBy(*[c.alias(n) for n, c in keys.items()]).agg(*[c.alias(n) for n, c in aggs.items()])
        return _Node(df, list(keys) + list(aggs))

    def p_Sort(self, body: dict) -> _Node:
        node = self.plan(body["input"])
        options = body["sort_options"]
        if body.get("slice") or options.get("limit"):
            raise self.unsupported("a sort with a limit")
        keys = [self.expr(e, node, _WHOLE) for e in body["by_column"]]
        n = len(keys)
        descending = options["descending"] * (n if len(options["descending"]) == 1 else 1)
        nulls_last = options["nulls_last"] * (n if len(options["nulls_last"]) == 1 else 1)
        # Sort keys become hidden columns, so a later select can't drop them.
        df, order = node.df, []
        for key, desc, last in zip(keys, descending, nulls_last):
            name = self.hidden_name("sort")
            df = df.withColumn(name, key)
            order.append((name, desc, last))
        # A stable sort breaks ties by the order it was given.
        tail = node.order if options["maintain_order"] else []
        return _Node(df, node.columns, order + tail)

    def p_Slice(self, body: dict) -> _Node:
        node = self.plan(body["input"])
        offset, length = body["offset"], body["len"]
        if offset < 0:
            raise self.unsupported("a slice from the end")
        if not node.order:
            raise self.unsupported("a slice of rows with no defined order")
        if offset == 0:
            return _Node(node.df.orderBy(*_sort_columns(node.order)).limit(length), node.columns, node.order)
        position = self.hidden_name("pos")
        df = node.df.withColumn(position, F.row_number().over(Window.orderBy(*_sort_columns(node.order))))
        df = df.filter((F.col(position) > offset) & (F.col(position) <= offset + length)).drop(position)
        return _Node(df, node.columns, node.order)

    def p_Distinct(self, body: dict) -> _Node:
        node = self.plan(body["input"])
        options = body["options"]
        subset = options["subset"]
        keys = node.columns if subset is None else [n for e in subset for n in self.selector_names(e, node)]
        strategy = options["keep_strategy"]
        if strategy == "None":
            raise self.unsupported("distinct keep='none'")
        if strategy == "Any" or not node.order:
            if strategy != "Any":
                raise self.unsupported(f"distinct keep={strategy.lower()!r} of rows with no defined order")
            return _Node(node.df.dropDuplicates(keys), node.columns, node.order)
        order = node.order if strategy == "First" else [(n, not d, not l) for n, d, l in node.order]
        rank = self.hidden_name("rank")
        window = Window.partitionBy(*map(_col, keys)).orderBy(*_sort_columns(order))
        df = node.df.withColumn(rank, F.row_number().over(window))
        return _Node(df.filter(F.col(rank) == 1).drop(rank), node.columns, node.order)

    def p_Join(self, body: dict) -> _Node:
        if "condition" in body:
            # Polars 2 writes a join's keys as condition: {Equi, NonEqui}.
            try:
                body = _legacy_join(body)
            except ExportError:
                raise self.unsupported("a join on a condition") from None
        if body.get("predicates"):
            raise self.unsupported("a join on a condition")
        left, right = self.plan(body["input_left"]), self.plan(body["input_right"])
        args = body["options"]["args"]
        how = args["how"] if isinstance(args["how"], str) else next(iter(args["how"]))
        spark_how = {"Left": "left", "Inner": "inner", "Semi": "left_semi", "Anti": "left_anti"}.get(how)
        if spark_how is None:
            raise self.unsupported(f"a {how.lower()} join")
        if args.get("validation", "ManyToMany") != "ManyToMany":
            raise self.unsupported("a join with relationship validation")
        left_keys = [self.column_name(e) for e in body["left_on"]]
        right_keys = [self.column_name(e) for e in body["right_on"]]
        # Rename every right column so nothing clashes; Polars' suffix rule follows.
        renamed = {name: self.hidden_name("r") for name in right.df.columns}
        r = right.df.select(*[_col(old).alias(new) for old, new in renamed.items()])
        equal = (lambda a, b: a.eqNullSafe(b)) if args.get("nulls_equal") else (lambda a, b: a == b)
        condition = functools.reduce(
            lambda a, b: a & b,
            [equal(left.df[lk], r[renamed[rk]]) for lk, rk in zip(left_keys, right_keys)],
        )
        df = left.df.join(r, condition, spark_how)
        if spark_how in ("left_semi", "left_anti"):
            return _Node(df, left.columns, left.order)
        if args.get("coalesce") not in ("JoinSpecific", "CoalesceColumns"):
            raise self.unsupported("a join that keeps both key columns")
        suffix = args["suffix"]
        columns, projections = list(left.columns), []
        for name in right.columns:
            if name in right_keys:
                continue
            out = name + suffix if name in columns else name
            projections.append(F.col(renamed[name]).alias(out))
            columns.append(out)
        df = df.select(*[_col(c) for c in left.df.columns], *projections, *[F.col(renamed[c]) for c in right.hidden])
        if args.get("maintain_order") not in ("Left", "LeftRight"):
            return _Node(df, columns)
        # Polars keeps the left order, then the right order within a left row.
        right_order = [(renamed[n], d, l) for n, d, l in right.order]
        return _Node(df, columns, left.order + right_order)

    def p_MapFunction(self, body: dict) -> _Node:
        function = body["function"]
        kind, args = (function, None) if isinstance(function, str) else next(iter(function.items()))
        node = self.plan(body["input"])
        if kind == "Rename":
            mapping = dict(zip(self.selector_names(args["existing"], node), args["new"]))
            df = node.df.withColumnsRenamed(mapping)
            return _Node(df, [mapping.get(c, c) for c in node.columns], node.order)
        if kind == "Drop":
            drop = set(self.selector_names(args["columns"], node))
            return _Node(node.df.drop(*drop), [c for c in node.columns if c not in drop], node.order)
        if kind == "RowIndex":
            if not node.order:
                raise self.unsupported("a row index of rows with no defined order")
            window = Window.orderBy(*_sort_columns(node.order))
            index = (F.row_number().over(window) - 1 + (args.get("offset") or 0)).cast(T.LongType())
            return _Node(node.df.withColumn(args["name"], index), [args["name"], *node.columns], node.order)
        raise self.unsupported(f"the Polars plan step {kind}")

    def p_Union(self, body: dict) -> _Node:
        options = body.get("args") or body.get("options") or {}
        nodes = [self.plan(n) for n in body["inputs"]]
        columns = list(dict.fromkeys(c for n in nodes for c in n.columns))
        if not options.get("diagonal") and any(n.columns != columns for n in nodes):
            raise self.unsupported("bind_rows of frames with different columns")
        # A diagonal concat (tidy3's bind_rows) fills a frame's missing columns
        # with nulls, as unionByName(allowMissingColumns=True) does. Polars and
        # Spark widen types differently, so each column must have one type.
        for name in columns:
            types = {str(n.df.schema[name].dataType) for n in nodes if name in n.columns}
            if len(types) > 1:
                raise self.unsupported(f"bind_rows of {name!r} with different types ({', '.join(sorted(types))})")
        if any(not n.order for n in nodes):
            raise self.unsupported("bind_rows of rows with no defined order")
        part = self.hidden_name("part")
        shapes = {tuple((d, l) for _name, d, l in n.order) for n in nodes}
        if len(shapes) == 1:
            # The inputs order their rows the same way: carry the keys over,
            # as shared columns after the part number.
            (shape,) = shapes
            keys = [self.hidden_name("key") for _ in shape]
            order_columns = lambda n: [_col(k).alias(new) for (k, _d, _l), new in zip(n.order, keys)]  # noqa: E731
            order = [(new, d, l) for new, (d, l) in zip(keys, shape)]
        else:
            # Otherwise number each input's rows: one Spark partition per input.
            row = self.hidden_name("row")
            order_columns = lambda n: [F.row_number().over(Window.orderBy(*_sort_columns(n.order))).alias(row)]  # noqa: E731
            order = [(row, False, True)]
        frames = [
            n.df.select(*[_col(c) for c in n.columns], F.lit(i).alias(part), *order_columns(n))
            for i, n in enumerate(nodes)
        ]
        df = functools.reduce(lambda a, b: a.unionByName(b, allowMissingColumns=True), frames)
        return _Node(df.select(*map(_col, columns), part, *[name for name, _d, _l in order]),
                     columns, [(part, False, True), *order])

    # ── expressions ─────────────────────────────────────────────────────

    def named(self, exprs: list, node: _Node, ctx: Any) -> dict[str, Column]:
        """Translate output expressions, expanding selectors, keyed by output name."""
        out: dict[str, Column] = {}
        for e in exprs:
            if isinstance(e, dict) and "Selector" in e:
                for name in self.selector_names(e["Selector"], node):
                    out[name] = _col(name)
                continue
            out[self.output_name(e)] = self.expr(e, node, ctx)
        return out

    def output_name(self, e: Any) -> str:
        """The name Polars gives an expression's result: its alias, else its first column."""
        if e == "Len" or (isinstance(e, dict) and "Len" in e):
            return "len"
        if not isinstance(e, dict):
            raise self.unsupported(f"an unnamed expression {str(e)[:60]}")
        if "Alias" in e:
            return e["Alias"][1]
        if "Column" in e:
            return e["Column"]
        for value in e.values():
            for child in value if isinstance(value, list) else [value]:
                if isinstance(child, dict):
                    if "input" in child:
                        inputs = child["input"]
                        return self.output_name(inputs[0] if isinstance(inputs, list) else inputs)
                    try:
                        return self.output_name(child)
                    except SparkExportError:
                        pass
        raise self.unsupported(f"an unnamed expression {str(e)[:60]}")

    def column_name(self, e: Any) -> str:
        if isinstance(e, dict) and "Column" in e:
            return e["Column"]
        raise self.unsupported(f"the join key {str(e)[:60]}")

    def selector_names(self, selector: Any, node: _Node) -> list[str]:
        if isinstance(selector, dict) and "Selector" in selector:
            selector = selector["Selector"]
        if selector == "Wildcard":
            return list(node.columns)
        if isinstance(selector, str):
            return [selector]
        if isinstance(selector, list):
            return [n for s in selector for n in self.selector_names(s, node)]
        (kind, body), = selector.items()
        if kind == "ByName":
            return list(body["names"])
        if kind == "Column":
            return [body]
        if kind == "Difference":
            drop = set(self.selector_names(body[1], node))
            return [n for n in self.selector_names(body[0], node) if n not in drop]
        if kind == "Union":
            return list(dict.fromkeys(self.selector_names(body[0], node) + self.selector_names(body[1], node)))
        raise self.unsupported(f"the column selector {kind}")

    def ordered_window(self, ctx: Any, order: list[Column]) -> Any:
        """*ctx*'s window (the whole frame, or a group's), ordered by *order*."""
        if ctx == "agg":
            raise self.unsupported("a row-position function inside summarise()")
        return (Window.partitionBy() if ctx == _WHOLE else ctx).orderBy(*order)

    def agg_input(self, arg: Any, node: _Node, ctx: Any) -> Column:
        """The input of an aggregation.

        An aggregation of an aggregation (``sum(x - min(x))``) is allowed in
        Polars, which broadcasts the inner one within the group. Spark allows
        neither that nor a window inside an aggregation, so the input is
        computed first, as a column, with the inner one over the group's window.
        """
        if ctx != "agg" or not _contains_agg(arg):
            return self.expr(arg, node, ctx)
        name = self.hidden_name("pre")
        self.pre_group[name] = self.expr(arg, node, self.group_window)
        return F.col(name)

    def type_of(self, value: Column, node: _Node) -> T.DataType | None:
        """The Spark type of a row-level *value*, or None if Spark can't say here."""
        df = node.df.withColumns(self.pre_group) if self.pre_group else node.df
        try:
            return df.select(value).schema.fields[0].dataType
        except Exception:
            return None

    def summable(self, value: Column, node: _Node) -> Column:
        """*value*, with booleans as 0 and 1: Polars sums booleans, Spark refuses to."""
        return value.cast(T.LongType()) if isinstance(self.type_of(value, node), T.BooleanType) else value

    def aggregate(self, column: Column, ctx: Any) -> Column:
        """An aggregation: per group in ``agg``, else broadcast over a window."""
        if ctx == "agg":
            return column
        return column.over(Window.partitionBy() if ctx == _WHOLE else ctx)

    def expr(self, e: Any, node: _Node, ctx: Any) -> Column:
        if e == "Len" or e == {"Len": None}:
            return self.aggregate(F.count(F.lit(1)), ctx)
        if not isinstance(e, dict) or len(e) != 1:
            raise self.unsupported(f"the expression {str(e)[:60]}")
        (kind, body), = e.items()
        method = getattr(self, "e_" + kind, None)
        if method is None:
            raise self.unsupported(f"the Polars expression {kind}")
        return method(body, node, ctx)

    def e_Column(self, body, node, ctx):
        return _col(body)

    def e_Alias(self, body, node, ctx):
        return self.expr(body[0], node, ctx)

    def e_Literal(self, body, node, ctx):
        (kind, value), = body.items()
        if kind == "Dyn":
            (_type, raw), = value.items()
            return F.lit(raw)
        if kind == "Scalar":
            if "Null" in value:
                return F.lit(None)
            (dtype, raw), = value.items()
            spark_type = _SIMPLE_TYPES.get(dtype)
            if spark_type is None or dtype in ("Date", "UInt64"):
                raise self.unsupported(f"a {dtype} literal")
            return F.lit(raw).cast(spark_type)
        raise self.unsupported(f"a {kind} literal")

    def e_BinaryExpr(self, body, node, ctx):
        op = body["op"]
        left, right = self.expr(body["left"], node, ctx), self.expr(body["right"], node, ctx)
        if op == "FloorDivide":
            return F.floor(_true_divide(left, right))
        if op == "Modulus":
            # By zero, Polars gives null for integers and NaN for floats; Spark
            # gives null or raises, by ANSI mode.
            types = (self.type_of(left, node), self.type_of(right, node))
            floating = any(isinstance(t, (T.DoubleType, T.FloatType)) for t in types)
            by_zero = F.lit(float("nan")) if floating else F.lit(None)
            return F.when(right == 0, by_zero).otherwise(_floor_mod(left, F.when(right != 0, right)))
        if op not in _BINARY:
            raise self.unsupported(f"the operator {op}")
        return _BINARY[op](left, right)

    def e_Ternary(self, body, node, ctx):
        return F.when(
            self.expr(body["predicate"], node, ctx), self.expr(body["truthy"], node, ctx)
        ).otherwise(self.expr(body["falsy"], node, ctx))

    def e_Cast(self, body, node, ctx):
        value, target = self.expr(body["expr"], node, ctx), self.dtype(body["dtype"])
        options = body.get("options")
        if isinstance(target, T.DateType) and isinstance(self.type_of(value, node), T.IntegralType):
            # An integer as a date is a count of days since 1970, as in Polars.
            return F.date_from_unix_date(value.cast(T.IntegerType()))
        if options not in ("Strict", "NonStrict"):
            # Overflowing: wraps with ANSI off; raises with it on.
            return value.cast(target)
        # try_cast gives null for a value with no such form, in either ANSI mode.
        converted = value.try_cast(target)
        if options == "NonStrict":
            return converted
        # A strict cast fails, as in Polars, rather than giving null.
        message = f"conversion failed: a value has no {target.simpleString()} form (strict cast)"
        return F.when(value.isNotNull() & converted.isNull(), F.raise_error(F.lit(message))).otherwise(converted)

    def e_Over(self, body, node, ctx):
        if body.get("order_by"):
            raise self.unsupported("a window with order_by")
        window = Window.partitionBy(*[self.expr(p, node, _WHOLE) for p in body["partition_by"]])
        return self.expr(body["function"], node, window)

    def e_Agg(self, body, node, ctx):
        (kind, arg), = body.items()
        if kind in ("First", "Last") and _is_scalar(arg):
            # first() of one value per group (an aggregation, a literal): the value.
            return self.expr(arg, node, ctx)
        if kind == "Count":
            include_nulls, arg = arg["include_nulls"], arg["input"]
            if _is_scalar(arg):
                return F.lit(1)
            value = self.expr(arg, node, ctx)
            return self.aggregate(F.count(F.lit(1)) if include_nulls else F.count(value), ctx)
        if kind in ("Std", "Var"):
            arg, ddof = arg
            fn = {("Std", 1): F.stddev_samp, ("Std", 0): F.stddev_pop,
                  ("Var", 1): F.var_samp, ("Var", 0): F.var_pop}.get((kind, ddof))
            if fn is None:
                raise self.unsupported(f"{kind.lower()}(ddof={ddof})")
            return self.aggregate(fn(self.agg_input(arg, node, ctx)), ctx)
        if isinstance(arg, dict) and "propagate_nans" in arg:
            if arg["propagate_nans"]:
                raise self.unsupported(f"{kind.lower()}() that propagates NaN")
            arg = arg["input"]
        fn = _AGG.get(kind)
        if fn is None:
            # first(), last() and the rest need row order inside groups.
            raise self.unsupported(f"the aggregation {kind}")
        value = self.agg_input(arg, node, ctx)
        if kind in ("Sum", "Mean"):
            value = self.summable(value, node)
        return self.aggregate(fn(value), ctx)

    def e_Function(self, body, node, ctx):
        function = body["function"]
        group, detail = (function, None) if isinstance(function, str) else next(iter(function.items()))
        inputs = body["input"]

        def args(context=ctx):
            return [self.expr(i, node, context) for i in inputs]

        name, options = (detail, None) if not isinstance(detail, dict) else next(iter(detail.items()))
        if group == "Boolean":
            if name in ("Any", "All"):
                (value,) = [self.agg_input(i, node, ctx) for i in inputs]
                return self.aggregate(F.bool_or(value) if name == "Any" else F.bool_and(value), ctx)
            if name == "IsBetween":
                value, low, high = args()
                closed = options["closed"]
                above = value >= low if closed in ("Both", "Left") else value > low
                below = value <= high if closed in ("Both", "Right") else value < high
                return above & below
            simple = {
                "IsNull": lambda a: a.isNull(), "IsNotNull": lambda a: a.isNotNull(),
                "IsNan": F.isnan, "IsNotNan": lambda a: ~F.isnan(a), "Not": lambda a: ~a,
                "IsFinite": lambda a: ~F.isnan(a) & (F.abs(a) != float("inf")),
                "IsInfinite": lambda a: F.abs(a) == float("inf"),
                "AllHorizontal": lambda *a: functools.reduce(lambda x, y: x & y, a),
                "AnyHorizontal": lambda *a: functools.reduce(lambda x, y: x | y, a),
            }
            if name not in simple:
                raise self.unsupported(f"the Boolean function {name}")
            return simple[name](*args())
        if group in ("Abs", "Negate", "Floor", "Ceil", "Sqrt"):
            (value,) = args()
            return {"Abs": F.abs, "Negate": lambda v: -v, "Floor": F.floor, "Ceil": F.ceil, "Sqrt": F.sqrt}[group](value)
        if group == "Round":
            (value,) = args()
            # tidy3's round() is half-to-even: bround, not Spark's round.
            mode = detail.get("mode", "HalfToEven")
            return F.bround(value, detail["decimals"]) if mode == "HalfToEven" else F.round(value, detail["decimals"])
        if group in ("FillNull", "Coalesce"):
            return F.coalesce(*args())
        if group in ("MaxHorizontal", "MinHorizontal"):
            # greatest() and least() skip missing values, as Polars does.
            return (F.greatest if group == "MaxHorizontal" else F.least)(*args())
        if group == "SumHorizontal":
            if not detail or not detail.get("ignore_nulls", True):
                raise self.unsupported("sum_horizontal() that keeps missing values")
            return functools.reduce(lambda a, b: a + b, [F.coalesce(v, F.lit(0)) for v in args()])
        if group == "Range":
            if name != "IntRange" or options["step"] != 1 or not _contains_agg(inputs[1]):
                raise self.unsupported(f"the range {name}")
            if not node.order:
                raise self.unsupported("row positions of rows with no defined order")
            # int_range(start, start + n()) over a group: row positions.
            start = self.expr(inputs[0], node, _WHOLE)
            position = F.row_number().over(self.ordered_window(ctx, _sort_columns(node.order))) - 1
            return (start + position).cast(self.dtype(options["dtype"]))
        if group == "Rank":
            rank = detail["options"]
            (value,) = args(_WHOLE)
            key = value.desc_nulls_last() if rank["descending"] else value.asc_nulls_last()
            fn = {"Min": F.rank, "Dense": F.dense_rank, "Ordinal": F.row_number}.get(rank["method"])
            if fn is None:
                raise self.unsupported(f"rank(method={rank['method'].lower()!r})")
            order = [key, *_sort_columns(node.order)] if rank["method"] == "Ordinal" else [key]
            # Polars leaves missing values unranked.
            return F.when(value.isNull(), F.lit(None)).otherwise(fn().over(self.ordered_window(ctx, order)))
        if group in ("CumSum", "CumMin", "CumMax"):
            reverse = detail.get("reverse") if isinstance(detail, dict) else bool(detail)
            if reverse:
                raise self.unsupported(f"a reversed {group}")
            if not node.order:
                raise self.unsupported(f"{group} of rows with no defined order")
            (value,) = args(_WHOLE)
            value = self.summable(value, node)
            running = self.ordered_window(ctx, _sort_columns(node.order)).rowsBetween(Window.unboundedPreceding, Window.currentRow)
            fn = {"CumSum": F.sum, "CumMin": F.min, "CumMax": F.max}[group]
            # Polars keeps a missing value missing and carries on past it.
            return F.when(value.isNull(), F.lit(None)).otherwise(fn(value).over(running))
        if group == "Shift":
            n = _int_literal(inputs[1])
            if n is None or not node.order:
                raise self.unsupported("a shift by a computed amount, or of rows with no defined order")
            value = self.expr(inputs[0], node, _WHOLE)
            window = self.ordered_window(ctx, _sort_columns(node.order))
            return F.lag(value, n).over(window) if n >= 0 else F.lead(value, -n).over(window)
        if group == "StringExpr":
            return self.string_function(name, options, inputs, node, ctx)
        if group == "TemporalExpr":
            return self.date_function(name, options, inputs, node, ctx)
        raise self.unsupported(f"the Polars function {group}")

    def regex(self, node: Any, what: str) -> str:
        """The Java form of a literal Rust pattern."""
        pattern = _str_literal(node)
        if pattern is None:
            raise self.unsupported(f"{what} with a computed pattern")
        return _JAVA_FLAGS + _java_regex(pattern)

    def string_function(self, name, options, inputs, node, ctx):
        (value, *rest) = [self.expr(i, node, ctx) for i in inputs]
        if name == "Lowercase":
            return F.lower(value)
        if name == "Uppercase":
            return F.upper(value)
        if name == "LenChars":
            return F.char_length(value)
        if name == "StartsWith":
            return F.startswith(value, rest[0])
        if name == "EndsWith":
            return F.endswith(value, rest[0])
        if name == "Contains":
            # Java regular expressions, not Rust's: most patterns agree.
            if options["literal"]:
                return F.contains(value, rest[0])
            return F.regexp(value, F.lit(self.regex(inputs[1], "a regex match")))
        if name == "ConcatHorizontal":
            delimiter = options["delimiter"]
            if options["ignore_nulls"]:
                return F.concat_ws(delimiter, value, *rest)
            # concat() is missing if any input is, as in Polars.
            parts = [value]
            for v in rest:
                parts += [F.lit(delimiter), v] if delimiter else [v]
            return F.concat(*parts)
        if name == "Slice":
            offset, length = rest
            # Polars counts from 0, Spark from 1; from the end, both count back.
            pos = F.when(offset >= 0, offset + 1).otherwise(offset).cast(T.IntegerType())
            span = F.lit(2**31 - 1) if _is_null_literal(inputs[2]) else length.cast(T.IntegerType())
            return F.substring(value, pos, span)
        if name == "Replace":
            return self.replace(value, inputs, options)
        if name == "Extract":
            pattern = _str_literal(inputs[1])
            if pattern is None or not _group_always_set(pattern):
                raise self.unsupported("str_extract() of a group that may be empty or unset")
            regex = self.regex(inputs[1], "str_extract()")
            # regexp_extract gives "" where nothing matches; Polars, null.
            return F.when(F.regexp(value, F.lit(regex)), F.regexp_extract(value, regex, options))
        if name == "CountMatches":
            pattern = _str_literal(inputs[1])
            if pattern is None:
                raise self.unsupported("a match count with a computed pattern")
            regex = _java_quote(pattern) if options else self.regex(inputs[1], "a match count")
            return F.regexp_count(value, F.lit(regex))
        if name in ("PadStart", "PadEnd"):
            width = _int_literal(inputs[1])
            if width is None:
                raise self.unsupported("padding to a computed width")
            pad = F.lpad if name == "PadStart" else F.rpad
            # lpad and rpad cut a longer string; Polars leaves it.
            return F.when(F.char_length(value) >= width, value).otherwise(pad(value, width, options["fill_char"]))
        if name in ("StripChars", "StripCharsStart", "StripCharsEnd"):
            if _is_null_literal(inputs[1]):
                # Unicode whitespace, as Rust's trim(); Spark's trim() takes spaces only.
                chars = "\\s"
            else:
                text = _str_literal(inputs[1])
                if text is None:
                    raise self.unsupported("stripping computed characters")
                if not text:
                    return value
                chars = "[" + "".join(c if c.isalnum() else "\\" + c for c in text) + "]"
            ends = {"StripChars": [f"\\A{chars}+", f"{chars}+\\z"],
                    "StripCharsStart": [f"\\A{chars}+"], "StripCharsEnd": [f"{chars}+\\z"]}[name]
            return F.regexp_replace(value, F.lit(_JAVA_FLAGS + "|".join(ends)), F.lit(""))
        if name == "Titlecase":
            # Polars capitalizes each letter that follows a non-letter and
            # lowercases the rest: "x1y" is "X1Y", where initcap() gives "X1y".
            # Polars 2 capitalizes in titlecase ("ß" is "Ss"); Polars 1, in uppercase.
            chars = F.regexp_extract_all(value, F.lit("(?s)."), 0)
            title = None
            if _POLARS_2:
                title = F.create_map(*[F.lit(s) for pair in _titlecase_letters().items() for s in pair])

            def cased(c, i):
                after_letter = F.coalesce(F.regexp(F.get(chars, i - 1), F.lit("\\p{IsAlphabetic}")), F.lit(False))
                first = F.upper(c) if title is None else F.coalesce(F.try_element_at(title, c), F.upper(c))
                return F.when((i == 0) | ~after_letter, first).otherwise(F.lower(c))

            return F.array_join(F.transform(chars, cased), "")
        raise self.unsupported(f"the string function {name}")

    def replace(self, value, inputs, options):
        """str_replace() and friends: the first match (n=1) or all (n=-1)."""
        pattern, replacement = _str_literal(inputs[1]), _str_literal(inputs[2])
        n = options["n"]
        if pattern is None or replacement is None or n not in (1, -1):
            raise self.unsupported("a replacement with a computed pattern or count")
        # Polars 1 replaces a pattern with no regex characters literally, and
        # then reads the replacement literally too.
        if options["literal"] or not (_POLARS_2 or _RUST_META & set(pattern)):
            regex, java = _java_quote(pattern), _java_literal_replacement(replacement)
        else:
            regex, java = _java_regex(pattern), _java_replacement(replacement, shift=1 if n == 1 else 0)
            if java is None:
                raise self.unsupported(f"the replacement {replacement!r}")
        if n == -1:
            return F.regexp_replace(value, F.lit(_JAVA_FLAGS + regex), F.lit(java))
        # Spark replaces every match: anchor the first, keeping the text before it.
        lead = "$1" + ("\\" if java[:1].isdigit() else "")
        return F.regexp_replace(value, F.lit(f"{_JAVA_FLAGS}\\A(?s:(.*?))(?:{regex})"), F.lit(lead + java))

    def date_function(self, name, options, inputs, node, ctx):
        parts = {"Year": F.year, "Month": F.month, "Day": F.dayofmonth, "Hour": F.hour,
                 "Minute": F.minute, "Second": F.second, "Quarter": F.quarter, "OrdinalDay": F.dayofyear,
                 # Polars numbers weekdays 1 (Monday) to 7; Spark's weekday() counts from 0.
                 "WeekDay": lambda v: F.weekday(v) + 1,
                 # ISO weeks, and the ISO year: that of the week's Thursday.
                 "Week": F.weekofyear,
                 "IsoYear": lambda v: F.year(F.date_add(F.to_date(v), 3 - F.weekday(v))),
                 "DaysInMonth": lambda v: F.dayofmonth(F.last_day(v)),
                 "Nanosecond": lambda v: (F.extract(F.lit("SECOND"), v) * 1_000_000).cast(T.LongType()) % 1_000_000 * 1000}
        (value, *rest) = [self.expr(i, node, ctx) for i in inputs]
        if name in parts:
            return parts[name](value)
        if name == "IsLeapYear":
            year = F.year(value)
            return (year % 4 == 0) & ((year % 100 != 0) | (year % 400 == 0))
        is_date = isinstance(self.type_of(value, node), T.DateType)
        if name == "Truncate":
            every = _str_literal(inputs[1])
            unit = {"1y": "year", "1q": "quarter", "1mo": "month", "1w": "week", "1d": "day",
                    "1h": "hour", "1m": "minute", "1s": "second"}.get(every)
            if unit is None:
                raise self.unsupported(f"floor_date() to {every!r}")
            if unit in ("hour", "minute", "second"):
                if is_date:
                    raise self.unsupported(f"floor_date() of a date to {every!r}")
                # Not date_trunc(): it reads a naive datetime in the session's
                # time zone, which moves a clock time daylight saving skipped.
                kept = {"hour": 1, "minute": 2, "second": 3}[unit]
                clock = [F.hour(value), F.minute(value), F.second(value)][:kept] + [F.lit(0)] * (3 - kept)
                return F.make_timestamp_ntz(F.year(value), F.month(value), F.dayofmonth(value), *clock)
            day = value if is_date else F.to_date(value)
            floored = day if unit == "day" else F.trunc(day, unit)
            return floored if is_date else floored.cast(T.TimestampNTZType())
        if name == "OffsetBy":
            text = _str_literal(inputs[1])
            span = _duration(text) if text is not None else None
            if span is None:
                raise self.unsupported(f"a date offset of {text!r}")
            months, days, micros = span
            # Polars adds months first (clamped to the month's end, as
            # add_months is), then days and time.
            if is_date:
                if micros:
                    raise self.unsupported("a date offset by part of a day")
                return F.date_add(F.add_months(value, months) if months else value, days)
            shifted = value + F.make_ym_interval(F.lit(0), F.lit(months)) if months else value
            if days or micros:
                seconds = F.lit(micros).cast(T.DecimalType(18, 6)) / 1_000_000
                shifted = shifted + F.make_dt_interval(F.lit(days), F.lit(0), F.lit(0), seconds.cast(T.DecimalType(18, 6)))
            return shifted
        raise self.unsupported(f"the date function {name}")

    def e_AnonymousFunction(self, body, node, ctx):
        text = bytes(body.get("function") or []).decode("latin-1")
        if _SUMMARY_SIZE_CHECK in text:
            return self.summary_size_check(body, node, ctx)
        if any(check in text for check in _ROW_CHECKS):
            raise self.unsupported("tidy3's row check")
        raise SparkExportError(
            "cannot run this pipeline on Spark: a column is computed by a Python function "
            f"({_udf_name(body)}), which has no Spark form"
        )

    def summary_size_check(self, body, node, ctx):
        """dplyr's 'must be size 1' summarise error, raised by Spark."""
        (size_input,) = body["input"]
        size = self.expr(size_input, node, ctx)
        name = size_input.get("Column", "") if isinstance(size_input, dict) else ""
        column = name.removeprefix("__tidy3_size_") or "a summary"
        message = f"summarise(): `{column}` must be size 1, but a group gave another size"
        return F.when(size != 1, F.raise_error(F.lit(message))).otherwise(F.lit(True))

    def dtype(self, node: Any) -> T.DataType:
        if isinstance(node, dict) and "Literal" in node:
            node = node["Literal"]
        spark_type = _SIMPLE_TYPES.get(node) if isinstance(node, str) else None
        if node == {"Datetime": ["Microseconds", None]}:
            spark_type = T.TimestampNTZType()
        if spark_type is None:
            # A parametrized type is {"Enum": ...}, {"Datetime": [...]}: name it only.
            kind = next(iter(node), "?") if isinstance(node, dict) else node
            raise self.unsupported(f"a cast to {kind}")
        return spark_type
