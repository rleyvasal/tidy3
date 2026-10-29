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
import warnings
from dataclasses import dataclass, field
from typing import Any

import polars as pl

from tidy3.polars_export import ExportError, _udf_name

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


_BINARY = {
    "Plus": lambda a, b: a + b, "Minus": lambda a, b: a - b, "Multiply": lambda a, b: a * b,
    "TrueDivide": lambda a, b: a / b, "Modulus": _floor_mod,
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
        if exprs and all(_contains_agg(e) for e in exprs):
            # summarise() without groups: one row.
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
        if "condition" in body or body.get("predicates"):
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
        if options.get("diagonal") or options.get("to_supertypes"):
            raise self.unsupported("a diagonal or type-widening bind_rows")
        nodes = [self.plan(n) for n in body["inputs"]]
        columns = nodes[0].columns
        if any(n.columns != columns for n in nodes):
            raise self.unsupported("bind_rows of frames with different columns")
        part, row = self.hidden_name("part"), self.hidden_name("row")
        frames = []
        for i, n in enumerate(nodes):
            if not n.order:
                raise self.unsupported("bind_rows of rows with no defined order")
            position = F.row_number().over(Window.orderBy(*_sort_columns(n.order)))
            frames.append(n.df.select(*map(_col, columns), F.lit(i).alias(part), position.alias(row)))
        df = functools.reduce(lambda a, b: a.unionByName(b), frames)
        return _Node(df, columns, [(part, False, True), (row, False, True)])

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

    def summable(self, value: Column, node: _Node) -> Column:
        """*value*, with booleans as 0 and 1: Polars sums booleans, Spark refuses to."""
        df = node.df.withColumns(self.pre_group) if self.pre_group else node.df
        try:
            dtype = df.select(value).schema.fields[0].dataType
        except Exception:
            return value
        return value.cast(T.LongType()) if isinstance(dtype, T.BooleanType) else value

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
            return F.floor(left / right)
        if op not in _BINARY:
            raise self.unsupported(f"the operator {op}")
        return _BINARY[op](left, right)

    def e_Ternary(self, body, node, ctx):
        return F.when(
            self.expr(body["predicate"], node, ctx), self.expr(body["truthy"], node, ctx)
        ).otherwise(self.expr(body["falsy"], node, ctx))

    def e_Cast(self, body, node, ctx):
        return self.expr(body["expr"], node, ctx).cast(self.dtype(body["dtype"]))

    def e_Over(self, body, node, ctx):
        if body.get("order_by"):
            raise self.unsupported("a window with order_by")
        window = Window.partitionBy(*[self.expr(p, node, _WHOLE) for p in body["partition_by"]])
        return self.expr(body["function"], node, window)

    def e_Agg(self, body, node, ctx):
        (kind, arg), = body.items()
        if kind in ("First", "Last") and _contains_agg(arg):
            # first() of one value per group: the value.
            return self.expr(arg, node, ctx)
        if kind == "Count":
            include_nulls, arg = arg["include_nulls"], arg["input"]
            if _contains_agg(arg):
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
            (value, *rest) = args()
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
                return F.contains(value, rest[0]) if options["literal"] else F.regexp(value, rest[0])
            raise self.unsupported(f"the string function {name}")
        if group == "TemporalExpr":
            parts = {"Year": F.year, "Month": F.month, "Day": F.dayofmonth, "Hour": F.hour,
                     "Minute": F.minute, "Quarter": F.quarter, "OrdinalDay": F.dayofyear,
                     # Polars numbers weekdays 1 (Monday) to 7; Spark's weekday() counts from 0.
                     "WeekDay": lambda v: F.weekday(v) + 1}
            if name not in parts:
                raise self.unsupported(f"the date function {name}")
            (value,) = args()
            return parts[name](value)
        raise self.unsupported(f"the Polars function {group}")

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
        if spark_type is None:
            # A parametrized type is {"Enum": ...}, {"Datetime": [...]}: name it only.
            kind = next(iter(node), "?") if isinstance(node, dict) else node
            raise self.unsupported(f"a cast to {kind}")
        return spark_type
