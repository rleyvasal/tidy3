"""One conversion engine for ``as_*``, ``parse_*``, and date orders.

Strict conversion raises :class:`ConversionError` at execution, naming the
column and showing example values that did not convert. Permissive
conversion (``strict=False``, and the default for ``parse_*`` and ``ymd``)
turns those values into missing and records them for :func:`problems`.

``as_integer`` refuses ``2.5`` instead of cutting it to ``2``. ``as_double``
does not strip grouping marks; ``parse_double`` and ``parse_number`` do.
"""

from __future__ import annotations

import math
import re
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import date, datetime, timezone
from typing import Any, Iterator

import pandas as pd
import polars as pl

from tidy3.expr import Expr, _func, col

__all__ = [
    "ConversionError",
    "locale",
    "Locale",
    "as_numeric",
    "as_double",
    "as_integer",
    "as_character",
    "as_logical",
    "as_date",
    "as_datetime",
    "parse_double",
    "parse_integer",
    "parse_logical",
    "parse_date",
    "parse_datetime",
    "parse_number",
    "problems",
    "stop_for_problems",
]

_MAX_EXAMPLES = 5
_INT64_MIN = -2**63
_INT64_MAX = 2**63 - 1
_OUTPUT: ContextVar[str | None] = ContextVar("tidy3_output_column", default=None)
_SINK: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "tidy3_problem_sink", default=None
)

# kind used by the engine, default strict, expected-type label, prose
_SPECS: dict[str, tuple[str, bool, str, str]] = {
    "as_numeric": ("double", True, "double", "numbers"),
    "as_double": ("double", True, "double", "numbers"),
    "as_integer": ("integer", True, "integer", "whole numbers"),
    "as_character": ("character", True, "character", "text"),
    "as_logical": ("logical", True, "logical", "logical values"),
    "as_date": ("date", True, "date", "dates (YYYY-MM-DD)"),
    "as_datetime": ("datetime", True, "datetime", "datetimes"),
    "parse_double": ("parse_double", False, "double", "numbers"),
    "parse_integer": ("parse_integer", False, "integer", "whole numbers"),
    "parse_logical": ("parse_logical", False, "logical", "logical values"),
    "parse_date": ("parse_date", False, "date", "dates"),
    "parse_datetime": ("parse_datetime", False, "datetime", "datetimes"),
    "parse_number": ("parse_number", False, "double", "numbers"),
    "ymd": ("ymd", False, "date", "year-month-day dates"),
    "mdy": ("mdy", False, "date", "month-day-year dates"),
    "dmy": ("dmy", False, "date", "day-month-year dates"),
    "ym": ("ym", False, "date", "year-month dates"),
}

_DTYPES = {
    "double": pl.Float64,
    "integer": pl.Int64,
    "character": pl.String,
    "logical": pl.Boolean,
    "date": pl.Date,
    "datetime": pl.Datetime("us"),
    "parse_double": pl.Float64,
    "parse_integer": pl.Int64,
    "parse_logical": pl.Boolean,
    "parse_date": pl.Date,
    "parse_datetime": pl.Datetime("us"),
    "parse_number": pl.Float64,
    "ymd": pl.Date,
    "mdy": pl.Date,
    "dmy": pl.Date,
    "ym": pl.Date,
}

_AS_TRUE = frozenset({"true", "t"})
_AS_FALSE = frozenset({"false", "f"})
_PARSE_TRUE = frozenset({"true", "t", "1"})
_PARSE_FALSE = frozenset({"false", "f", "0"})
FUNCTIONS = frozenset(_SPECS)


_ORDERS = {
    "ymd": ("y", "m", "d"),
    "mdy": ("m", "d", "y"),
    "dmy": ("d", "m", "y"),
    "ym": ("y", "m"),
}
_TIME_TAIL = re.compile(
    r"\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?"
)


class _Bad(Exception):
    """A single value could not be converted."""


class ConversionError(ValueError):
    """A strict conversion found values that are not the requested type.

    The message names the column and shows example bad values, in the same
    style as :class:`tidy3.errors.ColumnNotFoundError`. On a lazy Polars
    frame this is raised when the plan runs, because the bad values are in
    the data.
    """

    def __init__(
        self,
        message: str,
        *,
        column: str | None = None,
        examples: list[str] | None = None,
    ):
        super().__init__(message)
        self.column = column
        self.examples = list(examples or [])

    @classmethod
    def from_values(
        cls,
        function: str,
        column: str,
        detail: str,
        examples: list[str],
    ) -> "ConversionError":
        shown = ", ".join(repr(item) for item in examples[:_MAX_EXAMPLES])
        extra = len(examples) - _MAX_EXAMPLES
        more = f"\n({extra} more)" if extra > 0 else ""
        message = (
            f"{function}(): column {column!r} has values that are not {detail}.\n"
            f"Examples: {shown}{more}"
        )
        return cls(message, column=column, examples=examples)


class Locale:
    """Decimal mark, grouping mark, and encoding for the parsers.

    The marks must differ. ``as_double`` ignores them; ``parse_double``,
    ``parse_integer``, and ``parse_number`` use them.
    """

    __slots__ = ("decimal_mark", "grouping_mark", "encoding")

    def __init__(
        self,
        decimal_mark: str = ".",
        grouping_mark: str = ",",
        encoding: str = "utf-8",
    ):
        if (
            not isinstance(decimal_mark, str)
            or len(decimal_mark) != 1
            or not isinstance(grouping_mark, str)
            or len(grouping_mark) != 1
        ):
            raise ValueError("decimal_mark and grouping_mark must be single characters")
        if decimal_mark == grouping_mark:
            raise ValueError("decimal_mark and grouping_mark must be different")
        if not isinstance(encoding, str) or not encoding:
            raise ValueError("encoding must be a non-empty string")
        self.decimal_mark = decimal_mark
        self.grouping_mark = grouping_mark
        self.encoding = encoding

    def __repr__(self) -> str:
        return (
            f"locale(decimal_mark={self.decimal_mark!r}, "
            f"grouping_mark={self.grouping_mark!r}, encoding={self.encoding!r})"
        )


def locale(
    decimal_mark: str = ".",
    grouping_mark: str = ",",
    encoding: str = "utf-8",
) -> Locale:
    """Locale for :func:`parse_double`, :func:`parse_integer`, and :func:`parse_number`.

    The default is a point for the decimal mark and a comma for grouping
    (``1,234.50``). A comma decimal is ``locale(decimal_mark=",", grouping_mark=".")``.
    """
    return Locale(decimal_mark, grouping_mark, encoding)


def _marks(value: Locale | None) -> dict[str, str]:
    chosen = value if value is not None else Locale()
    if not isinstance(chosen, Locale):
        raise TypeError("locale must be locale() or None")
    return {
        "decimal_mark": chosen.decimal_mark,
        "grouping_mark": chosen.grouping_mark,
        "encoding": chosen.encoding,
    }


def _value(x: Any) -> Any:
    return col(x) if isinstance(x, str) else x


def _bool(name: str, value: bool) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")


def _cutoff(name: str, value: int | None) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 99:
        raise ValueError(f"{name} cutoff must be an integer from 0 to 99, or None")


@contextmanager
def output_column(name: str) -> Iterator[None]:
    """Name the column currently being assigned, for conversion errors."""
    token = _OUTPUT.set(name)
    try:
        yield
    finally:
        _OUTPUT.reset(token)


@contextmanager
def problem_sink() -> Iterator[list[dict[str, Any]]]:
    """Collect permissive failures while the pandas backend evaluates."""
    rows: list[dict[str, Any]] = []
    token = _SINK.set(rows)
    try:
        yield rows
    finally:
        _SINK.reset(token)


def _column_name(raw_args: tuple, fallback: str | None) -> str:
    if raw_args and raw_args[0][0] in {"col", "name"}:
        return str(raw_args[0][1])
    return fallback or _OUTPUT.get() or "<expression>"


def _literal(name: str, key: str, kwargs: dict[str, Any], default: Any) -> Any:
    if key not in kwargs:
        return default
    node = kwargs[key]
    if not isinstance(node, tuple) or node[0] != "lit":
        raise TypeError(f"{name}() {key} must be a constant")
    return node[1]


# ── value conversion ────────────────────────────────────────────────────


def _is_na_sentinel(value: Any) -> bool:
    if value is None or value is pd.NA:
        return True
    try:
        return bool(pd.isna(value)) and not isinstance(value, (float, int, bool))
    except TypeError:
        return False


def _original(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, float) and math.isnan(value):
        return "NaN"
    if isinstance(value, float) and math.isinf(value):
        return "Inf" if value > 0 else "-Inf"
    if isinstance(value, bytes):
        return repr(value)
    return repr(value)


def _decode(value: Any, encoding: str) -> Any:
    if isinstance(value, (bytes, bytearray)):
        try:
            return bytes(value).decode(encoding)
        except UnicodeDecodeError as error:
            raise _Bad() from error
    return value


def _plain_float(text: str) -> float:
    token = text.strip()
    lowered = token.lower()
    if lowered in {"nan", "+nan", "-nan"}:
        return float("nan")
    if lowered in {"inf", "+inf", "-inf"}:
        return float(lowered)
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", token):
        raise _Bad()
    return float(token)


def _ungroup(whole: str, grouping_mark: str) -> str:
    if grouping_mark not in whole:
        if whole != "" and not whole.isdigit():
            raise _Bad()
        return whole
    parts = whole.split(grouping_mark)
    if not parts[0].isdigit() or not 1 <= len(parts[0]) <= 3:
        raise _Bad()
    for part in parts[1:]:
        if not (part.isdigit() and len(part) == 3):
            raise _Bad()
    return "".join(parts)


def _localized_float(text: str, decimal_mark: str, grouping_mark: str) -> float:
    token = text.strip()
    lowered = token.lower()
    if decimal_mark == "." and lowered in {"nan", "+nan", "-nan", "inf", "+inf", "-inf"}:
        return float(lowered)
    sign = 1
    if token[:1] in "+-":
        sign = -1 if token[0] == "-" else 1
        token = token[1:]
    if token == "" or token == decimal_mark:
        raise _Bad()
    if decimal_mark in token:
        whole, frac = token.split(decimal_mark, 1)
        if decimal_mark in frac or grouping_mark in frac or not frac.isdigit():
            raise _Bad()
    else:
        whole, frac = token, None
    whole = _ungroup(whole, grouping_mark)
    if whole == "":
        whole = "0"
    if not whole.isdigit():
        raise _Bad()
    number = float(f"{whole}.{frac}" if frac is not None else whole)
    return sign * number


def _first_number(text: str, decimal_mark: str, grouping_mark: str) -> float:
    decimal = re.escape(decimal_mark)
    grouping = re.escape(grouping_mark)
    pattern = re.compile(
        rf"\d(?:\d|{grouping})*(?:{decimal}\d+)?|{decimal}\d+"
    )
    found = pattern.search(text)
    if found is None:
        raise _Bad()
    prefix = text[: found.start()]
    sign = -1 if re.search(r"-(?=[^\d]*)$", prefix) else 1
    magnitude = _localized_float(found.group(0), decimal_mark, grouping_mark)
    return sign * abs(magnitude)


def _whole_int(value: Any, *, grouping_mark: str | None, decimal_mark: str | None) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        number = value
    elif isinstance(value, float):
        if not math.isfinite(value) or value != math.trunc(value):
            raise _Bad()
        number = int(value)
    elif isinstance(value, str):
        token = value.strip()
        if grouping_mark is None:
            if not re.fullmatch(r"[+-]?\d+", token):
                raise _Bad()
            number = int(token)
        else:
            if decimal_mark and decimal_mark in token:
                raise _Bad()
            sign = 1
            if token[:1] in "+-":
                sign = -1 if token[0] == "-" else 1
                token = token[1:]
            digits = _ungroup(token, grouping_mark)
            if not digits.isdigit():
                raise _Bad()
            number = sign * int(digits)
    else:
        raise _Bad()
    if number < _INT64_MIN or number > _INT64_MAX:
        raise _Bad()
    return number


def _logical(value: Any, *, readr: bool) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        if isinstance(value, float) and math.isinf(value):
            return True
        if readr:
            if value == 1:
                return True
            if value == 0:
                return False
            raise _Bad()
        return value != 0
    if isinstance(value, str):
        key = value.strip().lower()
        truth = _PARSE_TRUE if readr else _AS_TRUE
        false = _PARSE_FALSE if readr else _AS_FALSE
        if key in truth:
            return True
        if key in false:
            return False
        raise _Bad()
    raise _Bad()


def _character(value: Any) -> str:
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Inf" if value > 0 else "-Inf"
        if math.isfinite(value) and value == math.trunc(value) and abs(value) < 2**53:
            return str(int(value))
        return format(value, ".15g")
    if isinstance(value, int):
        return str(value)
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            value = value.astimezone(timezone.utc).replace(tzinfo=None)
        timespec = "seconds" if value.microsecond == 0 else "microseconds"
        return value.isoformat(sep="T", timespec=timespec)
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        return value
    return str(value)


def _as_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return datetime.strptime(value.strip(), "%Y-%m-%d").date()
        except ValueError as error:
            raise _Bad() from error
    raise _Bad()


def _as_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value
        return value.astimezone(timezone.utc).replace(tzinfo=None)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    if not isinstance(value, str):
        raise _Bad()
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2} .+", text):
        text = text.replace(" ", "T", 1)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as error:
        raise _Bad() from error
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def _with_format(text: str, fmt: str, cutoff: int | None, *, as_date: bool) -> date | datetime:
    if not isinstance(fmt, str) or not fmt:
        raise TypeError("parse_date() and parse_datetime() need a format string")
    token = text.strip()
    if token == "":
        raise _Bad()
    two_digit = "%y" in fmt and "%Y" not in fmt
    if two_digit and cutoff is None:
        raise _Bad()
    try:
        parsed = datetime.strptime(token, fmt)
    except ValueError as error:
        raise _Bad() from error
    if two_digit:
        year = parsed.year % 100
        century = 2000 if year <= cutoff else 1900
        parsed = parsed.replace(year=century + year)
    return parsed.date() if as_date else parsed


def _split_date_text(text: str) -> str:
    if "T" in text:
        head, tail = text.split("T", 1)
        if _TIME_TAIL.fullmatch(tail):
            return head
    if " " in text:
        head, tail = text.split(" ", 1)
        if _TIME_TAIL.fullmatch(tail):
            return head
    return text


def _expand_year(token: str, cutoff: int | None) -> int:
    if not token.isdigit():
        raise _Bad()
    if len(token) == 4:
        return int(token)
    if len(token) <= 2:
        if cutoff is None:
            raise _Bad()
        value = int(token)
        return 2000 + value if value <= cutoff else 1900 + value
    raise _Bad()


def _parts_for(order: str, text: str) -> list[str]:
    body = _split_date_text(text.strip())
    parts = re.findall(r"\d+", body)
    widths = {"ymd": (8, 6), "mdy": (8, 6), "dmy": (8, 6), "ym": (6, 4)}
    if len(parts) == 1 and len(parts[0]) in widths[order]:
        digits = parts[0]
        if order == "ym":
            year_width = 4 if len(digits) == 6 else 2
            return [digits[:year_width], digits[year_width:]]
        if len(digits) == 8 and order == "ymd":
            return [digits[:4], digits[4:6], digits[6:]]
        if len(digits) == 8:
            return [digits[:2], digits[2:4], digits[4:]]
        return [digits[:2], digits[2:4], digits[4:]]
    return parts


def _ordered_date(text: str, order: str, cutoff: int | None) -> date:
    if not isinstance(text, str):
        if isinstance(text, datetime):
            return text.date()
        if isinstance(text, date):
            return text
        raise _Bad()
    slots = _ORDERS[order]
    parts = _parts_for(order, text)
    if len(parts) != len(slots):
        raise _Bad()
    fields = dict(zip(slots, parts))
    year = _expand_year(fields["y"], cutoff)
    try:
        month = int(fields["m"])
        day = 1 if order == "ym" else int(fields["d"])
        return date(year, month, day)
    except (TypeError, ValueError) as error:
        raise _Bad() from error


def _convert_one(
    value: Any,
    *,
    kind: str,
    cutoff: int | None,
    fmt: str | None,
    decimal_mark: str,
    grouping_mark: str,
    encoding: str,
) -> tuple[Any, bool]:
    """Return ``(converted, failed)``. Missing input is not a failure."""
    if _is_na_sentinel(value):
        return None, False
    try:
        value = _decode(value, encoding)
    except _Bad:
        return None, True
    if isinstance(value, float) and math.isnan(value):
        if kind in {"double", "parse_double", "parse_number"}:
            return float("nan"), False
        if kind in {"logical", "parse_logical"}:
            return None, False
        if kind == "character":
            return "NaN", False
        return None, True
    try:
        if kind == "double":
            if isinstance(value, bool):
                return float(value), False
            if isinstance(value, (int, float)):
                return float(value), False
            if isinstance(value, str):
                return _plain_float(value), False
            raise _Bad()
        if kind == "parse_double":
            if isinstance(value, bool):
                return float(value), False
            if isinstance(value, (int, float)):
                return float(value), False
            if isinstance(value, str):
                return _localized_float(value, decimal_mark, grouping_mark), False
            raise _Bad()
        if kind == "parse_number":
            if isinstance(value, bool):
                return float(value), False
            if isinstance(value, (int, float)):
                return float(value), False
            if isinstance(value, str):
                return _first_number(value, decimal_mark, grouping_mark), False
            raise _Bad()
        if kind == "integer":
            return _whole_int(value, grouping_mark=None, decimal_mark=None), False
        if kind == "parse_integer":
            return _whole_int(
                value, grouping_mark=grouping_mark, decimal_mark=decimal_mark
            ), False
        if kind == "logical":
            return _logical(value, readr=False), False
        if kind == "parse_logical":
            return _logical(value, readr=True), False
        if kind == "character":
            return _character(value), False
        if kind == "date":
            return _as_date(value), False
        if kind == "datetime":
            return _as_datetime(value), False
        if kind == "parse_date":
            if isinstance(value, (date, datetime)):
                return _as_date(value), False
            if not isinstance(value, str):
                raise _Bad()
            return _with_format(value, fmt or "", cutoff, as_date=True), False
        if kind == "parse_datetime":
            if isinstance(value, (date, datetime)):
                return _as_datetime(value), False
            if not isinstance(value, str):
                raise _Bad()
            return _with_format(value, fmt or "", cutoff, as_date=False), False
        if kind in _ORDERS:
            return _ordered_date(value, kind, cutoff), False
    except _Bad:
        return None, True
    raise _Bad()


def convert_batch(
    values: list[Any],
    *,
    kind: str,
    column: str,
    expected: str,
    cutoff: int | None = None,
    fmt: str | None = None,
    decimal_mark: str = ".",
    grouping_mark: str = ",",
    encoding: str = "utf-8",
) -> tuple[list[Any], list[dict[str, Any]]]:
    """Convert *values*. Failures become ``None`` and a problem row.

    Row numbers are 1-based positions in *values*. Callers that already
    know the frame row pass that position by slicing one batch per row
    number, or replace ``row`` afterwards.
    """
    converted: list[Any] = []
    problems: list[dict[str, Any]] = []
    for index, value in enumerate(values, start=1):
        result, failed = _convert_one(
            value,
            kind=kind,
            cutoff=cutoff,
            fmt=fmt,
            decimal_mark=decimal_mark,
            grouping_mark=grouping_mark,
            encoding=encoding,
        )
        converted.append(result)
        if failed:
            problems.append(
                {
                    "row": index,
                    "column": column,
                    "original": _original(value),
                    "expected": expected,
                }
            )
    return converted, problems


def _options(name: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    kind, default_strict, expected, detail = _SPECS[name]
    strict = kwargs.get("strict", default_strict)
    _bool(f"{name}() strict", strict)
    cutoff = kwargs.get("cutoff")
    _cutoff(name, cutoff)
    return {
        "kind": kind,
        "strict": strict,
        "expected": expected,
        "detail": detail,
        "cutoff": cutoff,
        "fmt": kwargs.get("format"),
        "decimal_mark": kwargs.get("decimal_mark", "."),
        "grouping_mark": kwargs.get("grouping_mark", ","),
        "encoding": kwargs.get("encoding", "utf-8"),
    }


def _apply(values: list[Any], name: str, options: dict[str, Any], column: str):
    converted, problems = convert_batch(
        values,
        kind=options["kind"],
        column=column,
        expected=options["expected"],
        cutoff=options["cutoff"],
        fmt=options["fmt"],
        decimal_mark=options["decimal_mark"],
        grouping_mark=options["grouping_mark"],
        encoding=options["encoding"],
    )
    if options["strict"] and problems:
        raise ConversionError.from_values(
            name,
            column,
            options["detail"],
            [row["original"] for row in problems],
        )
    return converted, problems


def compile_polars(
    name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    column: str,
) -> pl.Expr:
    """Compile a conversion to a Polars expression."""
    options = _options(name, kwargs)
    dtype = _DTYPES[options["kind"]]

    def batch(series: pl.Series) -> pl.Series:
        converted, _problems = _apply(series.to_list(), name, options, column)
        return pl.Series(converted, dtype=dtype)

    return args[0].map_batches(batch, return_dtype=dtype)


def eval_pandas(
    name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    df: pd.DataFrame,
    *,
    column: str,
) -> pd.Series:
    """Evaluate a conversion against a pandas frame."""
    options = _options(name, kwargs)
    source = args[0]
    if not isinstance(source, pd.Series):
        source = pd.Series(source, index=df.index)
    converted, problems = _apply(source.tolist(), name, options, column)
    sink = _SINK.get()
    if sink is not None and problems and not options["strict"]:
        sink.extend(problems)
    kind = options["kind"]
    if kind in {"integer", "parse_integer"}:
        return pd.Series(converted, index=source.index, dtype="Int64")
    if kind in {"double", "parse_double", "parse_number"}:
        return pd.Series(converted, index=source.index, dtype="Float64")
    if kind in {"logical", "parse_logical"}:
        return pd.Series(converted, index=source.index, dtype="boolean")
    if kind == "character":
        return pd.Series(converted, index=source.index, dtype="string")
    if kind in {"datetime", "parse_datetime"}:
        return pd.Series(pd.to_datetime(converted), index=source.index)
    return pd.Series(converted, index=source.index, dtype="object")


# ── lazy problems ───────────────────────────────────────────────────────


_PROBLEM_SCHEMA = {
    "row": pl.Int64,
    "column": pl.String,
    "original": pl.String,
    "expected": pl.String,
}


def empty_problems_lazy() -> pl.LazyFrame:
    return pl.DataFrame(schema=_PROBLEM_SCHEMA).lazy()


def empty_problems_pandas() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row": pd.Series(dtype="int64"),
            "column": pd.Series(dtype="string"),
            "original": pd.Series(dtype="string"),
            "expected": pd.Series(dtype="string"),
        }
    )


def _node_options(name: str, kwargs: dict[str, Any]) -> dict[str, Any] | None:
    """Options when this call records problems, else None."""
    kind, default_strict, expected, detail = _SPECS[name]
    strict = _literal(name, "strict", kwargs, default_strict)
    if strict:
        return None
    _bool(f"{name}() strict", strict)
    return {
        "kind": kind,
        "expected": expected,
        "detail": detail,
        "cutoff": _literal(name, "cutoff", kwargs, None),
        "fmt": _literal(name, "format", kwargs, None),
        "decimal_mark": _literal(name, "decimal_mark", kwargs, "."),
        "grouping_mark": _literal(name, "grouping_mark", kwargs, ","),
        "encoding": _literal(name, "encoding", kwargs, "utf-8"),
    }


def _children(node: tuple) -> list[tuple]:
    kind = node[0]
    if kind == "bin":
        return [node[2], node[3]]
    if kind in {"neg", "not", "desc"}:
        return [node[1]]
    if kind == "func":
        return [*node[2], *node[3].values()]
    if kind == "call":
        children = [node[2]]
        children.extend(value.node for value in node[3] if isinstance(value, Expr))
        children.extend(
            value.node for value in node[4].values() if isinstance(value, Expr)
        )
        return children
    if kind == "case_when":
        return [
            child
            for condition, value in node[1]
            for child in (condition, value)
        ] + [node[2]]
    return []


def _failure_expr(
    source: pl.Expr,
    name: str,
    options: dict[str, Any],
    column: str,
) -> pl.Expr:
    def batch(series: pl.Series) -> pl.Series:
        values = series.to_list()
        _converted, problems = convert_batch(
            values,
            kind=options["kind"],
            column=column,
            expected=options["expected"],
            cutoff=options["cutoff"],
            fmt=options["fmt"],
            decimal_mark=options["decimal_mark"],
            grouping_mark=options["grouping_mark"],
            encoding=options["encoding"],
        )
        original: list[str | None] = [None] * len(values)
        for row in problems:
            original[row["row"] - 1] = row["original"]
        return pl.Series(original, dtype=pl.String)

    return source.map_batches(batch, return_dtype=pl.String)


def problem_plans(lf: pl.LazyFrame, node: Any, output_name: str) -> list[pl.LazyFrame]:
    """Lazy frames of parse failures in *node*, without collecting *lf*.

    Each plan keeps the row number from *lf* as it is now (before this
    assignment). Later verbs can keep the plan; it still describes these
    rows, and it does not scan any rows until something collects it.
    """
    from tidy3.expr import _compile_pl

    plans: list[pl.LazyFrame] = []

    def visit(current: Any) -> None:
        if not isinstance(current, tuple) or not current or not isinstance(current[0], str):
            return
        if current[0] == "func" and current[1] in _SPECS:
            name, args, kwargs = current[1], current[2], current[3]
            options = _node_options(name, kwargs)
            if options is not None and args:
                column = _column_name(args, output_name)
                source = _compile_pl(args[0])
                indexed = lf.with_columns(
                    pl.int_range(1, pl.len() + 1).alias("__tidy3_row")
                )
                plans.append(
                    indexed.select(
                        pl.col("__tidy3_row").alias("row"),
                        pl.lit(column).alias("column"),
                        _failure_expr(source, name, options, column).alias("original"),
                        pl.lit(options["expected"]).alias("expected"),
                    ).filter(pl.col("original").is_not_null())
                )
        for child in _children(current):
            visit(child)

    visit(node)
    return plans


def merge_problem_plans(
    existing: pl.LazyFrame | None, plans: list[pl.LazyFrame]
) -> pl.LazyFrame | None:
    frames = [existing] if existing is not None else []
    frames.extend(plans)
    if not frames:
        return None
    if len(frames) == 1:
        return frames[0]
    return pl.concat(frames, how="vertical")


def pandas_problem_frame(rows: list[dict[str, Any]]) -> pd.DataFrame | None:
    if not rows:
        return None
    return pd.DataFrame(rows, columns=["row", "column", "original", "expected"])


def merge_pandas_problems(
    existing: pd.DataFrame | None, rows: list[dict[str, Any]]
) -> pd.DataFrame | None:
    added = pandas_problem_frame(rows)
    if existing is None:
        return added
    if added is None:
        return existing
    return pd.concat([existing, added], ignore_index=True)


def problems_frame(data: Any) -> Any:
    """A frame of parse failures: row, column, original, expected.

    On Polars this is a lazy plan captured when the value was parsed. It
    is not collected here, and it does not collect the data frame.
    """
    from tidy3.frame import TidyFrame

    if not isinstance(data, TidyFrame):
        raise TypeError("problems() expects a TidyFrame")
    stored = getattr(data, "_problems", None)
    if data.backend == "pandas":
        frame = stored if stored is not None else empty_problems_pandas()
        return TidyFrame(frame.copy())
    plan = stored if stored is not None else empty_problems_lazy()
    return TidyFrame(plan)


def problems(data: Any) -> Any:
    """Failures from permissive parses on *data*.

    Columns are ``row`` (1-based), ``column``, ``original``, and
    ``expected``. An empty result has that same schema. On a lazy frame
    this returns another lazy frame and does not read the file.
    """
    return problems_frame(data)


def stop_for_problems(data: Any) -> Any:
    """Raise when *data* has parse failures.

    Collects the problems plan only, not the data frame. Returns *data*
    when nothing failed.
    """
    found = problems(data).collect()
    if isinstance(found, pl.DataFrame):
        rows = found.to_dicts()
    else:
        rows = found.to_dict("records")
    if not rows:
        return data
    lines = [f"stop_for_problems(): {len(rows)} value(s) failed to parse."]
    for row in rows[:_MAX_EXAMPLES]:
        lines.append(
            f"row {row['row']}, column {row['column']!r}: "
            f"{row['original']!r} is not {row['expected']}"
        )
    hidden = len(rows) - _MAX_EXAMPLES
    if hidden > 0:
        lines.append(f"({hidden} more)")
    raise ConversionError("\n".join(lines))


# ── constructors ────────────────────────────────────────────────────────


def _call(name: str, x: Any, *, suffix: dict[str, Any]) -> Expr:
    return _func(name, _value(x), **suffix)


def as_double(x: Any, *, strict: bool = True) -> Expr:
    """Convert to floating point. Strict fails on values that are not numbers.

    Grouping marks are not removed: ``"1,234"`` is a failure, as in R's
    ``as.numeric``. ``"1.5"`` converts. Missing values stay missing, and
    NaN stays NaN. ``as_numeric`` is the same function.
    """
    _bool("as_double() strict", strict)
    return _call("as_double", x, suffix={"strict": strict})


def as_numeric(x: Any, *, strict: bool = True) -> Expr:
    """Alias of :func:`as_double`."""
    return as_double(x, strict=strict)


def as_integer(x: Any, *, strict: bool = True) -> Expr:
    """Convert to an integer. ``2`` and ``2.0`` succeed; ``2.5`` is a failure.

    This does not truncate toward zero the way R's ``as.integer`` does.
    ``"2"`` succeeds; ``"2.0"`` and ``"2.5"`` do not. NaN is a failure.
    """
    _bool("as_integer() strict", strict)
    return _call("as_integer", x, suffix={"strict": strict})


def as_character(x: Any, *, strict: bool = True) -> Expr:
    """Convert to text. Logical values become ``TRUE`` and ``FALSE``.

    Whole numbers have no trailing ``.0``. Dates are ISO ``YYYY-MM-DD``.
    NaN becomes ``"NaN"``. Missing values stay missing.
    """
    _bool("as_character() strict", strict)
    return _call("as_character", x, suffix={"strict": strict})


def as_logical(x: Any, *, strict: bool = True) -> Expr:
    """Convert to logical, as R's ``as.logical``.

    ``TRUE`` / ``FALSE`` / ``T`` / ``F`` (any case) and numbers work: ``0``
    is false and any other number is true. The string ``"1"`` is a failure.
    NaN is missing, not a failure. For ``true`` / ``false`` / ``1`` / ``0``
    text, use :func:`parse_logical`.
    """
    _bool("as_logical() strict", strict)
    return _call("as_logical", x, suffix={"strict": strict})


def as_date(x: Any, *, strict: bool = True) -> Expr:
    """Convert to a date. Text must be ISO ``YYYY-MM-DD``.

    A datetime keeps its calendar date. For other orders, use :func:`ymd`,
    :func:`mdy`, or :func:`dmy`.
    """
    _bool("as_date() strict", strict)
    return _call("as_date", x, suffix={"strict": strict})


def as_datetime(x: Any, *, strict: bool = True) -> Expr:
    """Convert to a naive datetime.

    Accepts an ISO date, a space or ``T`` before the time, and a trailing
    ``Z`` or numeric offset. Offsets are converted to UTC and the zone is
    then dropped, so the result is the UTC clock time with no timezone.
    """
    _bool("as_datetime() strict", strict)
    return _call("as_datetime", x, suffix={"strict": strict})


def _parse(name: str, x: Any, *, suffix: dict[str, Any], marks: Locale | None) -> Expr:
    _bool(f"{name}() strict", suffix.get("strict", False))
    if "cutoff" in suffix:
        _cutoff(name, suffix["cutoff"])
    payload = {**suffix, **_marks(marks)}
    return _call(name, x, suffix=payload)


def parse_double(x: Any, *, locale: Locale | None = None, strict: bool = False) -> Expr:
    """Parse a whole string as a number, using *locale* grouping and decimal marks.

    ``"1,234.50"`` is ``1234.5`` with the default locale. ``"$1,234"`` fails
    because of the dollar sign; use :func:`parse_number` for that. Failures
    become missing and are listed by :func:`problems`, unless ``strict=True``.
    """
    return _parse("parse_double", x, suffix={"strict": strict}, marks=locale)


def parse_integer(x: Any, *, locale: Locale | None = None, strict: bool = False) -> Expr:
    """Parse a whole string as an integer. ``"1,234"`` works; ``"2.5"`` does not."""
    return _parse("parse_integer", x, suffix={"strict": strict}, marks=locale)


def parse_logical(x: Any, *, strict: bool = False) -> Expr:
    """Parse logical text the way readr does: ``true``/``false``/``t``/``f``/``1``/``0``.

    ``yes`` and ``no`` are failures. Case does not matter.
    """
    return _parse("parse_logical", x, suffix={"strict": strict}, marks=None)


def parse_number(x: Any, *, locale: Locale | None = None, strict: bool = False) -> Expr:
    """The first number in messy text, such as ``"$1,234.50"`` → ``1234.5``."""
    return _parse("parse_number", x, suffix={"strict": strict}, marks=locale)


def parse_date(
    x: Any,
    format: str,
    *,
    cutoff: int | None = None,
    strict: bool = False,
) -> Expr:
    """Parse dates with a ``strptime`` *format*.

    A format that uses ``%y`` needs ``cutoff``. Years up to and including
    the cutoff fall in this century (``cutoff=68`` reads ``68`` as 2068 and
    ``69`` as 1969). Without a cutoff, a two-digit year is a failure rather
    than a silent century.
    """
    if not isinstance(format, str) or not format:
        raise TypeError("parse_date() format must be a string")
    return _parse(
        "parse_date",
        x,
        suffix={"format": format, "cutoff": cutoff, "strict": strict},
        marks=None,
    )


def parse_datetime(
    x: Any,
    format: str,
    *,
    cutoff: int | None = None,
    strict: bool = False,
) -> Expr:
    """Parse datetimes with a ``strptime`` *format*. See :func:`parse_date`."""
    if not isinstance(format, str) or not format:
        raise TypeError("parse_datetime() format must be a string")
    return _parse(
        "parse_datetime",
        x,
        suffix={"format": format, "cutoff": cutoff, "strict": strict},
        marks=None,
    )


def ymd(x: Any, *, cutoff: int | None = None, strict: bool = False) -> Expr:
    """Parse year-month-day text into a date (lubridate's ``ymd``).

    ``"2010-01-31"`` and ``"20100131"`` both work. A two-digit year needs
    ``cutoff``; see :func:`parse_date`. Failures become missing and show up
    in :func:`problems`.
    """
    return _parse("ymd", x, suffix={"cutoff": cutoff, "strict": strict}, marks=None)


def mdy(x: Any, *, cutoff: int | None = None, strict: bool = False) -> Expr:
    """Parse month-day-year text into a date."""
    return _parse("mdy", x, suffix={"cutoff": cutoff, "strict": strict}, marks=None)


def dmy(x: Any, *, cutoff: int | None = None, strict: bool = False) -> Expr:
    """Parse day-month-year text into a date."""
    return _parse("dmy", x, suffix={"cutoff": cutoff, "strict": strict}, marks=None)


def ym(x: Any, *, cutoff: int | None = None, strict: bool = False) -> Expr:
    """Parse year-month text into the first day of that month."""
    return _parse("ym", x, suffix={"cutoff": cutoff, "strict": strict}, marks=None)
