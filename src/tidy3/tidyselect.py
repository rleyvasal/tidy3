"""Backend-neutral tidy-select helpers and column-wise specifications."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import polars as pl

from tidy3.expr import Expr, col


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
        def resolve(columns, schema, groups):
            unwanted = set(self.resolve(columns, schema, groups))
            return [name for name in columns if name not in unwanted]

        return Selector(resolve, f"~{self.label}")

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
    """Select the inclusive schema range between two named columns."""

    def resolve(columns, schema, groups):
        missing = [name for name in (first, last) if name not in columns]
        if missing:
            raise KeyError(f"col_range(): columns not found: {missing}")
        start = columns.index(first)
        stop = columns.index(last)
        step = 1 if start <= stop else -1
        return [columns[index] for index in range(start, stop + step, step)]

    return Selector(resolve, f"col_range({first!r}, {last!r})")


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


def is_numeric(dtype: Any) -> bool:
    try:
        numeric = getattr(dtype, "is_numeric", None)
        if callable(numeric):
            return bool(numeric())
    except (TypeError, AttributeError):
        pass
    try:
        from pandas.api.types import is_bool_dtype, is_numeric_dtype

        return bool(is_numeric_dtype(dtype) and not is_bool_dtype(dtype))
    except TypeError:
        return False


def is_string(dtype: Any) -> bool:
    if dtype == pl.String:
        return True
    try:
        from pandas.api.types import is_string_dtype

        return bool(is_string_dtype(dtype))
    except TypeError:
        return False


def is_boolean(dtype: Any) -> bool:
    if dtype == pl.Boolean:
        return True
    try:
        from pandas.api.types import is_bool_dtype

        return bool(is_bool_dtype(dtype))
    except TypeError:
        return False


def is_temporal(dtype: Any) -> bool:
    try:
        if isinstance(dtype, pl.DataType):
            return bool(dtype.is_temporal())
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


class AcrossSpec:
    __slots__ = ("selector", "functions", "names")

    def __init__(self, selector: Any, fns: Any, names: str | None):
        self.selector = as_selector(selector)
        self.functions = _functions(fns)
        self.names = names

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
                if name in result:
                    raise ValueError(f"across() produced duplicate column {name!r}")
                result[name] = fn(col(column))
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


def across(selector: Any, fns: Any, *, names: str | None = None) -> AcrossSpec:
    return AcrossSpec(selector, fns, names)


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
    "all_of",
    "any_of",
    "c_across",
    "col_range",
    "contains",
    "ends_with",
    "everything",
    "group_cols",
    "if_all",
    "if_any",
    "is_boolean",
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
