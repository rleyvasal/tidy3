"""Backend-neutral tidy-select helpers and column-wise specifications."""

from __future__ import annotations

import re
from contextvars import ContextVar
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, is_dataclass
from typing import Any

import polars as pl

from tidy3.expr import Expr, col, cur_group_id as _expr_cur_group_id, n_groups as _expr_n_groups


_CURRENT_COLUMN: ContextVar[str | None] = ContextVar("tidy3_current_column", default=None)
_CURRENT_GROUPS: ContextVar[tuple[str, ...] | None] = ContextVar(
    "tidy3_current_groups", default=None
)
_CURRENT_GROUP_ROWS: ContextVar[tuple[int, ...] | None] = ContextVar(
    "tidy3_current_group_rows", default=None
)


class Selector:
    """Deferred column selection resolved against a concrete frame schema."""

    __slots__ = ("_resolve", "label")

    def __init__(self, resolve: Callable[..., list[str]], label: str):
        self._resolve = resolve
        self.label = label

    def resolve(
        self, columns: list[str], schema: Mapping[str, Any], groups: list[str]
    ) -> list[str]:
        selected = self._resolve(columns, schema, groups)
        return list(dict.fromkeys(selected))

    def __or__(self, other: Any) -> Selector:
        right = as_selector(other)
        return Selector(
            lambda columns, schema, groups: [
                *self.resolve(columns, schema, groups),
                *right.resolve(columns, schema, groups),
            ],
            f"({self.label} | {right.label})",
        )

    def __and__(self, other: Any) -> Selector:
        right = as_selector(other)

        def resolve(columns, schema, groups):
            wanted = set(right.resolve(columns, schema, groups))
            return [
                name for name in self.resolve(columns, schema, groups) if name in wanted
            ]

        return Selector(resolve, f"({self.label} & {right.label})")

    def __sub__(self, other: Any) -> Selector:
        right = as_selector(other)

        def resolve(columns, schema, groups):
            unwanted = set(right.resolve(columns, schema, groups))
            return [
                name
                for name in self.resolve(columns, schema, groups)
                if name not in unwanted
            ]

        return Selector(resolve, f"({self.label} - {right.label})")

    def __invert__(self) -> Selector:
        """Selection complement (dplyr ``!`` / tidyselect ``!``)."""

        def resolve(columns, schema, groups):
            unwanted = set(self.resolve(columns, schema, groups))
            return [name for name in columns if name not in unwanted]

        return Selector(resolve, f"!{self.label}")

    def __neg__(self) -> Selector:
        """Exclude columns (dplyr ``-starts_with(...)`` / ``-col``)."""
        return self.__invert__()

    def __repr__(self) -> str:
        return f"<tidy3.Selector {self.label}>"


def _frame_schema(tf: Any) -> tuple[list[str], Mapping[str, Any]]:
    if tf._backend == "pandas":
        columns = [str(name) for name in tf._pdf.columns]
        return columns, {str(name): dtype for name, dtype in tf._pdf.dtypes.items()}
    schema = tf._lf.collect_schema()
    return schema.names(), schema


def as_selector(value: Any) -> Selector:
    if isinstance(value, Selector):
        return value
    if isinstance(value, str):
        return all_of([value])
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        return all_of(list(value))
    raise TypeError(f"cannot use {value!r} as a column selector")


def resolve_selection(
    tf: Any,
    specs: Iterable[Any],
    *,
    strict_strings: bool = True,
) -> list[str]:
    columns, schema = _frame_schema(tf)
    groups = list(tf._groups or [])
    selected: list[str] = []

    def add(spec: Any) -> None:
        if isinstance(spec, Selector):
            selected.extend(spec.resolve(columns, schema, groups))
        elif isinstance(spec, str):
            if spec not in columns:
                if strict_strings:
                    raise KeyError(f"column not found: {spec!r}")
                return
            selected.append(spec)
        elif isinstance(spec, int) and not isinstance(spec, bool):
            if spec == 0:
                raise ValueError("column positions are 1-based and cannot be zero")
            index = spec - 1 if spec > 0 else len(columns) + spec
            if index < 0 or index >= len(columns):
                raise IndexError(f"column position out of range: {spec}")
            selected.append(columns[index])
        elif isinstance(spec, Iterable) and not isinstance(spec, (str, bytes)):
            for item in spec:
                add(item)
        else:
            raise TypeError(f"unsupported column selection: {spec!r}")

    for spec in specs:
        add(spec)
    return list(dict.fromkeys(selected))


def everything() -> Selector:
    return Selector(lambda columns, schema, groups: list(columns), "everything()")


def _text_matches(match: str | Iterable[str], helper: str) -> list[str]:
    values = [match] if isinstance(match, str) else list(match)
    if any(not isinstance(value, str) for value in values):
        raise TypeError(f"{helper}() requires a string or strings")
    return values


def starts_with(
    match: str | Iterable[str], *, ignore_case: bool = True
) -> Selector:
    values = _text_matches(match, "starts_with")
    needles = [value.casefold() if ignore_case else value for value in values]
    return Selector(
        lambda columns, schema, groups: [
            name
            for name in columns
            if any(
                (name.casefold() if ignore_case else name).startswith(needle)
                for needle in needles
            )
        ],
        f"starts_with({match!r})",
    )


def ends_with(match: str | Iterable[str], *, ignore_case: bool = True) -> Selector:
    values = _text_matches(match, "ends_with")
    needles = [value.casefold() if ignore_case else value for value in values]
    return Selector(
        lambda columns, schema, groups: [
            name
            for name in columns
            if any(
                (name.casefold() if ignore_case else name).endswith(needle)
                for needle in needles
            )
        ],
        f"ends_with({match!r})",
    )


def contains(match: str | Iterable[str], *, ignore_case: bool = True) -> Selector:
    values = _text_matches(match, "contains")
    needles = [value.casefold() if ignore_case else value for value in values]
    return Selector(
        lambda columns, schema, groups: [
            name
            for name in columns
            if any(
                needle in (name.casefold() if ignore_case else name)
                for needle in needles
            )
        ],
        f"contains({match!r})",
    )


def matches(pattern: str, *, ignore_case: bool = True) -> Selector:
    flags = re.IGNORECASE if ignore_case else 0
    regex = re.compile(pattern, flags)
    return Selector(
        lambda columns, schema, groups: [
            name for name in columns if regex.search(name)
        ],
        f"matches({pattern!r})",
    )


def num_range(prefix: str, values: Iterable[int], *, width: int | None = None) -> Selector:
    names = [
        f"{prefix}{value:0{width}d}" if width is not None else f"{prefix}{value}"
        for value in values
    ]
    return any_of(names)


def col_range(first: str, last: str) -> Selector:
    """Select the inclusive schema range between two named columns.

    dplyr-ish column ranges (``mpg:hp`` in R)::

        select(col_range("mpg", "hp"))
        select(cols_between("mpg", "hp"))  # alias
    """

    def resolve(columns, schema, groups):
        missing = [name for name in (first, last) if name not in columns]
        if missing:
            raise KeyError(f"col_range(): columns not found: {missing}")
        start = columns.index(first)
        stop = columns.index(last)
        step = 1 if start <= stop else -1
        return [columns[index] for index in range(start, stop + step, step)]

    return Selector(resolve, f"col_range({first!r}, {last!r})")


def cols_between(first: str, last: str) -> Selector:
    """Alias of :func:`col_range` (inclusive column-name range for ``select``)."""
    if not isinstance(first, str) or not isinstance(last, str):
        raise TypeError("cols_between() expects two column name strings")
    return col_range(first, last)


def all_of(names: Iterable[str]) -> Selector:
    wanted = list(names)
    if any(not isinstance(name, str) for name in wanted):
        raise TypeError("all_of() requires column names")

    def resolve(columns, schema, groups):
        missing = [name for name in wanted if name not in columns]
        if missing:
            raise KeyError(f"all_of(): columns not found: {missing}")
        return wanted

    return Selector(resolve, f"all_of({wanted!r})")


def any_of(names: Iterable[str]) -> Selector:
    wanted = list(names)
    if any(not isinstance(name, str) for name in wanted):
        raise TypeError("any_of() requires column names")
    return Selector(
        lambda columns, schema, groups: [name for name in wanted if name in columns],
        f"any_of({wanted!r})",
    )


def where(predicate: Callable[[Any], bool]) -> Selector:
    if not callable(predicate):
        raise TypeError("where() predicate must be callable")
    return Selector(
        lambda columns, schema, groups: [
            name for name in columns if bool(predicate(schema[name]))
        ],
        f"where({getattr(predicate, '__name__', 'predicate')})",
    )


def last_col(offset: int = 0) -> Selector:
    if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
        raise ValueError("last_col() offset must be a non-negative integer")

    def resolve(columns, schema, groups):
        index = len(columns) - offset - 1
        if index < 0:
            raise IndexError("last_col() offset exceeds the number of columns")
        return [columns[index]]

    return Selector(resolve, f"last_col(offset={offset})")


def group_cols() -> Selector:
    return Selector(
        lambda columns, schema, groups: list(groups),
        "group_cols()",
    )


def _polars_dtype(dtype: Any) -> Any | None:
    """Return a Polars dtype when *dtype* is one (class or instance)."""
    try:
        if isinstance(dtype, pl.DataType):
            return dtype
        # Polars sometimes exposes dtype classes (e.g. pl.Int64)
        if isinstance(dtype, type) and issubclass(dtype, pl.DataType):
            return dtype()
    except TypeError:
        pass
    return None


def is_numeric(dtype: Any) -> bool:
    """True for int/float numeric columns (not boolean)."""
    try:
        numeric = getattr(dtype, "is_numeric", None)
        if callable(numeric):
            # Polars: ints/floats are numeric; booleans are not in recent Polars
            if is_boolean(dtype):
                return False
            return bool(numeric())
    except (TypeError, AttributeError):
        pass
    try:
        from pandas.api.types import is_bool_dtype, is_numeric_dtype

        return bool(is_numeric_dtype(dtype) and not is_bool_dtype(dtype))
    except TypeError:
        return False


def is_integer(dtype: Any) -> bool:
    """True for integer columns (signed/unsigned)."""
    pdt = _polars_dtype(dtype)
    if pdt is not None:
        try:
            return bool(pdt.is_integer())
        except (TypeError, AttributeError):
            pass
    try:
        from pandas.api.types import is_integer_dtype

        return bool(is_integer_dtype(dtype))
    except TypeError:
        return False


def is_float(dtype: Any) -> bool:
    """True for floating-point columns."""
    pdt = _polars_dtype(dtype)
    if pdt is not None:
        try:
            return bool(pdt.is_float())
        except (TypeError, AttributeError):
            pass
    try:
        from pandas.api.types import is_float_dtype

        return bool(is_float_dtype(dtype))
    except TypeError:
        return False


def is_string(dtype: Any) -> bool:
    """True for text / string columns."""
    if dtype == pl.String or dtype is pl.String:
        return True
    pdt = _polars_dtype(dtype)
    if pdt is not None and type(pdt) is pl.String:
        return True
    try:
        from pandas.api.types import is_string_dtype

        return bool(is_string_dtype(dtype))
    except TypeError:
        return False


def is_character(dtype: Any) -> bool:
    """Alias of :func:`is_string` (R ``is.character``)."""
    return is_string(dtype)


def is_boolean(dtype: Any) -> bool:
    """True for boolean columns."""
    if dtype == pl.Boolean or dtype is pl.Boolean:
        return True
    pdt = _polars_dtype(dtype)
    if pdt is not None and type(pdt) is pl.Boolean:
        return True
    try:
        from pandas.api.types import is_bool_dtype

        return bool(is_bool_dtype(dtype))
    except TypeError:
        return False


def is_bool(dtype: Any) -> bool:
    """Alias of :func:`is_boolean`."""
    return is_boolean(dtype)


def is_datetime(dtype: Any) -> bool:
    """True for datetime / date columns (not pure time-delta)."""
    pdt = _polars_dtype(dtype)
    if pdt is not None:
        name = type(pdt).__name__
        return name in {"Datetime", "Date"}
    try:
        from pandas.api.types import is_datetime64_any_dtype

        if is_datetime64_any_dtype(dtype):
            return True
    except TypeError:
        pass
    return False


def is_temporal(dtype: Any) -> bool:
    """True for datetime, date, time, or duration columns."""
    try:
        if isinstance(dtype, pl.DataType):
            return bool(dtype.is_temporal())
        pdt = _polars_dtype(dtype)
        if pdt is not None:
            return bool(pdt.is_temporal())
    except (TypeError, AttributeError):
        pass
    try:
        from pandas.api.types import (
            is_datetime64_any_dtype,
            is_timedelta64_dtype,
        )

        return bool(is_datetime64_any_dtype(dtype) or is_timedelta64_dtype(dtype))
    except TypeError:
        return False


def is_categorical(dtype: Any) -> bool:
    """True for categorical / factor-like columns."""
    pdt = _polars_dtype(dtype)
    if pdt is not None:
        name = type(pdt).__name__
        if name in {"Categorical", "Enum"}:
            return True
    if dtype is pl.Categorical or dtype == pl.Categorical:
        return True
    try:
        import pandas as pd

        return isinstance(dtype, pd.CategoricalDtype) or str(dtype) == "category"
    except Exception:
        return False


def _functions(fns: Any) -> list[tuple[str, Callable[[Expr], Any]]]:
    if isinstance(fns, Mapping):
        raw = list(fns.items())
    elif isinstance(fns, (list, tuple)):
        raw = [
            (getattr(fn, "__name__", fn if isinstance(fn, str) else f"fn{i}"), fn)
            for i, fn in enumerate(fns, start=1)
        ]
    else:
        raw = [(getattr(fns, "__name__", fns if isinstance(fns, str) else "fn"), fns)]
    out = []
    for label, fn in raw:
        if isinstance(fn, str):
            method = fn
            out.append((str(label), lambda value, method=method: getattr(value, method)()))
        elif callable(fn):
            out.append((str(label), fn))
        else:
            raise TypeError("column-wise functions must be callables or method names")
    return out


def _format_name(template: str, column: str, function: str) -> str:
    template = template.replace("{.col}", "{col}").replace("{.fn}", "{fn}")
    try:
        return template.format(col=column, fn=function)
    except (KeyError, ValueError) as error:
        raise ValueError("names must use {col}/{fn} (or {.col}/{.fn})") from error


def _format_unpack_name(template: str, outer: str, inner: str) -> str:
    """Format a name for a field returned by an unpacked across function."""
    template = template.replace("{.outer}", "{outer}").replace(
        "{.inner}", "{inner}"
    )
    try:
        return template.format(outer=outer, inner=inner)
    except (KeyError, ValueError) as error:
        raise ValueError(
            "across(unpack=...) names must use {outer}/{inner} "
            "(or {.outer}/{.inner})"
        ) from error


def _unpack_fields(value: Any) -> Mapping[str, Any] | None:
    """Normalize supported structured across results to named fields."""
    if isinstance(value, Mapping):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if hasattr(value, "_asdict") and callable(value._asdict):
        return value._asdict()
    return None


class AcrossSpec:
    __slots__ = ("selector", "functions", "names", "unpack")

    def __init__(
        self, selector: Any, fns: Any, names: str | None, unpack: bool | str
    ):
        self.selector = as_selector(selector)
        self.functions = _functions(fns)
        self.names = names
        if not isinstance(unpack, (bool, str)):
            raise TypeError("across() unpack must be False, True, or a name template")
        self.unpack = unpack

    def expand(self, tf: Any) -> dict[str, Any]:
        columns = resolve_selection(tf, [self.selector])
        columns = [name for name in columns if name not in (tf._groups or [])]
        template = self.names or (
            "{col}" if len(self.functions) == 1 else "{col}_{fn}"
        )
        result: dict[str, Any] = {}
        for column in columns:
            for function_name, fn in self.functions:
                name = _format_name(template, column, function_name)
                if not name:
                    raise ValueError("across() produced an empty column name")
                column_token = _CURRENT_COLUMN.set(column)
                groups_token = _CURRENT_GROUPS.set(tuple(tf._groups or ()))
                try:
                    value = fn(col(column))
                finally:
                    _CURRENT_GROUPS.reset(groups_token)
                    _CURRENT_COLUMN.reset(column_token)
                if self.unpack:
                    fields = _unpack_fields(value)
                    if fields is None:
                        raise TypeError(
                            "across(unpack=True) functions must return a mapping, "
                            "named tuple, or dataclass of expressions"
                        )
                    unpack_template = (
                        self.unpack if isinstance(self.unpack, str) else "{outer}_{inner}"
                    )
                    for inner, field_value in fields.items():
                        output_name = _format_unpack_name(
                            unpack_template, name, str(inner)
                        )
                        if not output_name:
                            raise ValueError("across() produced an empty column name")
                        if output_name in result:
                            raise ValueError(
                                f"across() produced duplicate column {output_name!r}"
                            )
                        result[output_name] = field_value
                else:
                    if _unpack_fields(value) is not None:
                        raise TypeError(
                            "across() function returned multiple fields; use unpack=True"
                        )
                    if name in result:
                        raise ValueError(f"across() produced duplicate column {name!r}")
                    result[name] = value
        return result


class ColumnwisePredicate:
    __slots__ = ("selector", "functions", "all")

    def __init__(self, selector: Any, fns: Any, *, all: bool):  # noqa: A002
        self.selector = as_selector(selector)
        self.functions = _functions(fns)
        self.all = all

    def expand(self, tf: Any) -> Any:
        columns = resolve_selection(tf, [self.selector])
        columns = [name for name in columns if name not in (tf._groups or [])]
        predicates = [fn(col(column)) for column in columns for _, fn in self.functions]
        if not predicates:
            return Expr(("lit", self.all))
        result = predicates[0]
        for predicate in predicates[1:]:
            result = result & predicate if self.all else result | predicate
        return result


class HorizontalSpec:
    __slots__ = ("selector", "operation", "requires_rowwise")

    def __init__(self, selector: Any, operation: str, *, requires_rowwise: bool):
        self.selector = as_selector(selector)
        self.operation = operation
        self.requires_rowwise = requires_rowwise

    def expand(self, tf: Any) -> Expr:
        if self.requires_rowwise and not tf._rowwise:
            raise ValueError("c_across() must be used after rowwise()")
        columns = resolve_selection(tf, [self.selector])
        columns = [name for name in columns if name not in (tf._groups or [])]
        return Expr(("horizontal", self.operation, tuple(columns)))


class ColumnSet:
    """Deferred data-frame column set used by ``pick`` and ``c_across``."""

    __slots__ = ("selector", "requires_rowwise")

    def __init__(self, selector: Any, *, requires_rowwise: bool):
        self.selector = as_selector(selector)
        self.requires_rowwise = requires_rowwise

    def _tidy3_aggregate(self, operation: str) -> HorizontalSpec:
        return HorizontalSpec(
            self.selector, operation, requires_rowwise=self.requires_rowwise
        )

    def expand(self, tf: Any) -> Expr:
        if self.requires_rowwise and not tf._rowwise:
            raise ValueError("c_across() must be used after rowwise()")
        columns = resolve_selection(tf, [self.selector])
        columns = [name for name in columns if name not in (tf._groups or [])]
        return Expr(("column_set", tuple(columns), self.requires_rowwise))

    def sum(self) -> HorizontalSpec:
        return self._tidy3_aggregate("sum")

    def mean(self) -> HorizontalSpec:
        return self._tidy3_aggregate("mean")

    def min(self) -> HorizontalSpec:
        return self._tidy3_aggregate("min")

    def max(self) -> HorizontalSpec:
        return self._tidy3_aggregate("max")

    def median(self) -> HorizontalSpec:
        return self._tidy3_aggregate("median")

    def std(self) -> HorizontalSpec:
        return self._tidy3_aggregate("std")

    def any(self) -> HorizontalSpec:
        return self._tidy3_aggregate("any")

    def all(self) -> HorizontalSpec:
        return self._tidy3_aggregate("all")

    def first(self) -> HorizontalSpec:
        return self._tidy3_aggregate("first")

    def last(self) -> HorizontalSpec:
        return self._tidy3_aggregate("last")


def across(
    selector: Any,
    fns: Any,
    *,
    names: str | None = None,
    unpack: bool | str = False,
) -> AcrossSpec:
    """Apply one or more functions to selected columns.

    When ``unpack`` is true (or a ``{outer}/{inner}`` template), each function
    may return a mapping of field names to expressions.  The fields become
    ordinary output columns, mirroring dplyr's ``across(.unpack=...)``.
    """
    return AcrossSpec(selector, fns, names, unpack)


def cur_column() -> str:
    """Return the name currently being processed by ``across``.

    This is available while an across function is being expanded, matching
    dplyr's ``cur_column()`` for column-dependent lambdas.
    """
    column = _CURRENT_COLUMN.get()
    if column is None:
        raise RuntimeError("cur_column() can only be used inside across()")
    return column


def cur_group() -> dict[str, Expr]:
    """Return grouped keys as expressions inside an ``across`` function.

    The mapping mirrors the one-row ``cur_group()`` data frame in dplyr while
    remaining vectorized: indexing a key (for example ``cur_group()["g"]``)
    yields the grouped column expression and therefore works in both mutate
    and summarise pipelines.
    """
    groups = _CURRENT_GROUPS.get()
    if groups is None:
        raise RuntimeError("cur_group() can only be used inside across()")
    return {name: col(name) for name in groups}


def cur_group_id() -> Expr:
    """Return the 1-based current group identifier inside ``across``."""
    groups = _CURRENT_GROUPS.get()
    if groups is None:
        raise RuntimeError("cur_group_id() can only be used inside across()")
    return _expr_cur_group_id(groups)


def group_vars() -> tuple[str, ...]:
    """Return grouping variable names inside an ``across`` function."""
    groups = _CURRENT_GROUPS.get()
    if groups is None:
        raise RuntimeError("group_vars() can only be used inside across()")
    return groups


def n_groups() -> Expr:
    """Return the number of groups inside an ``across`` function."""
    groups = _CURRENT_GROUPS.get()
    if groups is None:
        raise RuntimeError("n_groups() can only be used inside across()")
    return _expr_n_groups(groups)


def _group_rows_token(rows: tuple[int, ...]):
    return _CURRENT_GROUP_ROWS.set(rows)


def _reset_group_rows(token: Any) -> None:
    _CURRENT_GROUP_ROWS.reset(token)


def cur_group_rows() -> tuple[int, ...]:
    """Return zero-based source row positions in a group callback."""
    rows = _CURRENT_GROUP_ROWS.get()
    if rows is None:
        raise RuntimeError("cur_group_rows() can only be used inside group callbacks")
    return rows


def if_any(selector: Any, predicate: Any) -> ColumnwisePredicate:
    return ColumnwisePredicate(selector, predicate, all=False)


def if_all(selector: Any, predicate: Any) -> ColumnwisePredicate:
    return ColumnwisePredicate(selector, predicate, all=True)


def _combined_selector(selectors: tuple[Any, ...]) -> Selector:
    if not selectors:
        return everything()
    combined = as_selector(selectors[0])
    for selector in selectors[1:]:
        combined = combined | selector
    return combined


def pick(*selectors: Any) -> ColumnSet:
    return ColumnSet(_combined_selector(selectors), requires_rowwise=False)


def c_across(*selectors: Any) -> ColumnSet:
    return ColumnSet(_combined_selector(selectors), requires_rowwise=True)


__all__ = [
    "AcrossSpec",
    "ColumnwisePredicate",
    "ColumnSet",
    "HorizontalSpec",
    "Selector",
    "across",
    "cur_column",
    "cur_group",
    "cur_group_id",
    "group_vars",
    "n_groups",
    "cur_group_rows",
    "all_of",
    "any_of",
    "c_across",
    "col_range",
    "cols_between",
    "contains",
    "ends_with",
    "everything",
    "group_cols",
    "if_all",
    "if_any",
    "is_bool",
    "is_boolean",
    "is_categorical",
    "is_character",
    "is_datetime",
    "is_float",
    "is_integer",
    "is_numeric",
    "is_string",
    "is_temporal",
    "last_col",
    "matches",
    "num_range",
    "pick",
    "resolve_selection",
    "starts_with",
    "where",
]
