"""One conversion engine for ``as_*``, ``parse_*``, and date orders.

Strict conversion raises :class:`ConversionError` at execution, naming the
column and showing example values that did not convert. Permissive
conversion (``strict=False``, and the default for ``parse_*`` and ``ymd``)
turns those values into missing and records them for :func:`problems`.

``as_integer`` refuses ``2.5`` instead of cutting it to ``2``. ``as_double``
and ``parse_double`` do not strip grouping marks; ``parse_number`` does.
"""

from __future__ import annotations

import math
import re
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from functools import lru_cache
from typing import Any, Iterator
from zoneinfo import ZoneInfo

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
    "as_string",
    "as_logical",
    "as_date",
    "as_datetime",
    "parse_double",
    "parse_integer",
    "parse_logical",
    "parse_date",
    "parse_datetime",
    "parse_number",
    "ymd_hms",
    "ymd_hm",
    "mdy_hms",
    "dmy_hms",
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
    "as_string": ("character", True, "character", "text"),
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
    "ymd_hms": ("ymd_hms", False, "datetime", "year-month-day datetimes"),
    "ymd_hm": ("ymd_hm", False, "datetime", "year-month-day datetimes"),
    "mdy_hms": ("mdy_hms", False, "datetime", "month-day-year datetimes"),
    "dmy_hms": ("dmy_hms", False, "datetime", "day-month-year datetimes"),
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

# as.logical() takes exactly these spellings: no trimming, no other case.
_AS_TRUE = frozenset({"TRUE", "true", "True", "T"})
_AS_FALSE = frozenset({"FALSE", "false", "False", "F"})
_PARSE_TRUE = frozenset({"true", "t", "1"})
_PARSE_FALSE = frozenset({"false", "f", "0"})
FUNCTIONS = frozenset(_SPECS)


_ORDERS = {
    "ymd": ("y", "m", "d"),
    "mdy": ("m", "d", "y"),
    "dmy": ("d", "m", "y"),
    "ym": ("y", "m"),
}
_MONTH_NAMES = (
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
)
_ORDINALS = frozenset({"st", "nd", "rd", "th"})
_DATETIME_ORDERS = frozenset({"ymd_hms", "ymd_hm", "mdy_hms", "dmy_hms"})


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

    The marks must differ. ``parse_double`` uses the decimal mark, and
    :func:`parse_number` uses both. ``as_double`` ignores them.
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
    """Locale for :func:`parse_double` and :func:`parse_number`.

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


def _localized_float(text: str, decimal_mark: str) -> float:
    """readr's ``parse_double``: a whole number with an optional exponent.

    Grouping marks, ``Inf``, and ``NaN`` are failures, as in readr; use
    :func:`parse_number` for ``"1,234.5"``. Only ASCII digits count.
    """
    token = text.strip()
    if not re.fullmatch(_double_pattern(re.escape(decimal_mark)), token):
        raise _Bad()
    return float(token.replace(decimal_mark, ".") if decimal_mark != "." else token)


def _double_pattern(decimal: str) -> str:
    return rf"[+-]?(?:[0-9]+(?:{decimal}[0-9]*)?|{decimal}[0-9]+)(?:[eE][+-]?[0-9]+)?"


def _first_number(text: str, decimal_mark: str, grouping_mark: str) -> float:
    """readr's ``parse_number``: the first number in *text*.

    Skip to the first digit, minus, or decimal mark. A minus must be
    followed by the number itself, so ``"$-5"`` is -5 but ``"-$5"`` and
    ``"- 5"`` fail. Grouping marks are ignored wherever they fall in the
    whole part (``"1,23"`` is 123), a second decimal mark ends the number
    (``"1.2.3"`` is 1.2), and an exponent such as ``e3`` is read.
    """
    size = len(text)
    i = 0
    while i < size and not (
        text[i] == "-" or text[i] == decimal_mark or "0" <= text[i] <= "9"
    ):
        i += 1
    if i == size:
        raise _Bad()
    sign = 1
    if text[i] == "-":
        sign = -1
        i += 1
    whole: list[str] = []
    frac: list[str] = []
    in_frac = False
    while i < size:
        char = text[i]
        if "0" <= char <= "9":
            (frac if in_frac else whole).append(char)
        elif char == decimal_mark and not in_frac:
            in_frac = True
        elif char != grouping_mark or in_frac:
            break
        i += 1
    if not whole and not frac:
        raise _Bad()
    exponent = re.match(r"[eE][+-]?[0-9]+", text[i:])
    digits = f"{''.join(whole) or '0'}.{''.join(frac) or '0'}"
    return sign * float(digits + (exponent.group(0) if exponent else ""))


def _whole_int(value: Any) -> int:
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
        if not re.fullmatch(r"[+-]?\d+", token):
            raise _Bad()
        number = int(token)
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
        key = value.strip().lower() if readr else value
        truth = _PARSE_TRUE if readr else _AS_TRUE
        false = _PARSE_FALSE if readr else _AS_FALSE
        if key in truth:
            return True
        if key in false:
            return False
        raise _Bad()
    raise _Bad()


# Text follows the column's type, as Polars writes it: a float stays a float
# ("21.0"), a float32 reads back as the same number ("22.8"), and nothing is
# rounded. Logical values use the caller's spellings (true / false by default).
_DATETIME_TEXT = "%Y-%m-%d %H:%M:%S%.f"


def _text_series(series: pl.Series, true_text: str, false_text: str) -> pl.Series | None:
    """*series* as text by its own type, or None for types written value by value."""
    dtype = series.dtype
    value = pl.col("v")
    if dtype == pl.Boolean:
        text = (
            pl.when(value.is_null()).then(None)
            .when(value).then(pl.lit(true_text))
            .otherwise(pl.lit(false_text))
        )
    elif isinstance(dtype, pl.Datetime):
        # Fractional seconds only when there are any; a zone is kept as its offset.
        text = value.dt.to_string(_DATETIME_TEXT + ("%:z" if dtype.time_zone else ""))
    elif dtype == pl.Time:
        text = value.dt.to_string("%H:%M:%S%.f")
    elif (
        dtype.is_numeric()
        or dtype in (pl.String, pl.Date, pl.Null)
        or isinstance(dtype, (pl.Categorical, pl.Enum))
    ):
        text = value.cast(pl.String)
    else:
        return None
    return series.to_frame("v").select(text.alias(series.name)).to_series()


def _character(value: Any, true_text: str = "true", false_text: str = "false") -> str:
    if isinstance(value, bool):
        return true_text if value else false_text
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, Decimal, date, time)):
        try:
            text = _text_series(pl.Series([value]), true_text, false_text)
        except (OverflowError, TypeError, ValueError):
            text = None
        if text is not None:
            return text[0]
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
    if parsed.tzinfo is not None and not as_date:
        # %z is an instant. Keep the naive-datetime column and store the
        # UTC clock, the same contract as as_datetime. ymd_hms stays zoned.
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed.date() if as_date else parsed


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


def _month_number(word: str) -> int | None:
    """``Jan``, ``jan``, ``Sept``, or ``January`` as 1-12; None for other words."""
    word = word.lower()
    if len(word) < 3:
        return None
    for number, name in enumerate(_MONTH_NAMES, start=1):
        if name.startswith(word):
            return number
    return None


def _parts_for(order: str, text: str) -> list[str]:
    """Date fields of *text* in order, as lubridate's ``ymd`` family reads them.

    Any punctuation or space separates fields, month names count as the
    month (``"Jan 31 2010"``), and ordinals such as ``1st`` drop their
    suffix. Any other word, or a time after the date, is a failure, as in
    lubridate: ``ymd("2010-01-31 10:00")`` is NA.
    """
    parts: list[str] = []
    for token in re.findall(r"[0-9]+|[A-Za-z]+", text):
        if token.isdigit():
            parts.append(token)
        elif token.lower() in _ORDINALS and parts:
            continue
        else:
            month = _month_number(token)
            if month is None:
                raise _Bad()
            parts.append(str(month))
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


@lru_cache(maxsize=None)
def _datetime_pattern(order: str, fields: int) -> re.Pattern:
    sep = r"[^0-9A-Za-z]+"
    slots = {
        "y": r"(?:[0-9]{4}|[0-9]{1,2})",
        "m": r"(?:[0-9]{1,2}|[A-Za-z]{3,9})",
        "d": r"[0-9]{1,2}(?:st|nd|rd|th)?",
    }
    calendar = sep.join(slots[slot] for slot in _ORDERS[order])
    calendar = rf"(?P<date>{calendar}|[0-9]{{8}})"
    if not fields:
        return re.compile(calendar, re.IGNORECASE)
    # Reserve signs for UTC offsets, including after a truncated clock.
    time_sep = r"[^0-9A-Za-z+-]+"
    clock = time_sep.join([r"[0-9]{1,2}"] * fields)
    compact = rf"[0-9]{{{2 * fields}}}"
    fraction = r"(?:\.(?P<fraction>[0-9]+))?" if fields == 3 else ""
    return re.compile(
        calendar + rf"(?P<boundary>T|{time_sep})?(?P<time>{clock}|{compact}){fraction}"
        + r"\s*(?P<ampm>[ap]m)?\s*(?P<offset>Z|[+-][0-9]{2}(?::?[0-9]{2})?)?",
        re.IGNORECASE,
    )


def _ordered_datetime(
    value: Any, kind: str, cutoff: int | None, tz: str, truncated: int
) -> datetime:
    zone = ZoneInfo(tz)
    if isinstance(value, datetime):
        parsed = value
    else:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            value = format(value, ".0f") if float(value).is_integer() else str(value)
            if not kind.startswith("ymd") and len(value) % 2:
                value = "0" + value
        if not isinstance(value, str):
            raise _Bad()
        order, clock_order = kind.split("_")
        for fields in range(len(clock_order), len(clock_order) - truncated - 1, -1):
            match = _datetime_pattern(order, fields).fullmatch(value.strip())
            if match is not None:
                if fields and not match["boundary"] and not match["date"].isdigit():
                    continue
                break
        else:
            raise _Bad()
        calendar = _ordered_date(match["date"], order, cutoff)
        parsed = datetime.combine(calendar, datetime.min.time())
        if fields:
            clock = match["time"]
            fraction = match["fraction"] if fields == 3 else None
            pieces = (
                [clock[i:i + 2] for i in range(0, len(clock), 2)]
                if clock.isdigit() else re.findall(r"[0-9]+", clock)
            )
            h, m, s = ([int(piece) for piece in pieces] + [0, 0])[:3]
            micro = int((fraction + "000000")[:6]) if fraction else 0
            if h > 24 or m > 59 or s > 60 or (s == 60 and micro):
                raise _Bad()
            if match["ampm"]:
                if h > 12:
                    raise _Bad()
                h = h % 12 + (12 if match["ampm"].lower() == "pm" else 0)
            parsed += timedelta(hours=h, minutes=m, seconds=s, microseconds=micro)
            offset = match["offset"]
            if offset:
                if offset.upper() == "Z":
                    source_zone = timezone.utc
                else:
                    digits = offset[1:].replace(":", "")
                    hours, minutes = int(digits[:2]), int(digits[2:] or "0")
                    if hours > 23 or minutes > 59:
                        raise _Bad()
                    sign = 1 if offset[0] == "+" else -1
                    source_zone = timezone(sign * timedelta(hours=hours, minutes=minutes))
                parsed = parsed.replace(tzinfo=source_zone)
    if parsed.tzinfo is not None:
        return parsed.astimezone(zone)
    # Choose the later occurrence of a repeated clock time. A UTC round trip
    # detects the nonexistent clocks during a spring daylight saving change.
    local = parsed.replace(tzinfo=zone, fold=1)
    if local.astimezone(timezone.utc).astimezone(zone).replace(tzinfo=None) != parsed:
        raise _Bad()
    return local


def _convert_one(
    value: Any,
    *,
    kind: str,
    cutoff: int | None,
    fmt: str | None,
    decimal_mark: str,
    grouping_mark: str,
    encoding: str,
    tz: str = "UTC",
    truncated: int = 0,
    true_text: str = "true",
    false_text: str = "false",
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
        if kind in {"logical", "parse_logical"} or kind in _DATETIME_ORDERS:
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
                return _localized_float(value, decimal_mark), False
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
            return _whole_int(value), False
        if kind == "parse_integer":
            # readr's parse_integer takes no grouping marks: "1,234" fails.
            return _whole_int(value), False
        if kind == "logical":
            return _logical(value, readr=False), False
        if kind == "parse_logical":
            return _logical(value, readr=True), False
        if kind == "character":
            return _character(value, true_text, false_text), False
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
        if kind in _DATETIME_ORDERS:
            try:
                return _ordered_datetime(value, kind, cutoff, tz, truncated), False
            except (ValueError, OverflowError) as error:
                raise _Bad() from error
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
    tz: str = "UTC",
    truncated: int = 0,
    true_text: str = "true",
    false_text: str = "false",
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
            tz=tz,
            truncated=truncated,
            true_text=true_text,
            false_text=false_text,
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
        "tz": kwargs.get("tz", "UTC"),
        "truncated": kwargs.get("truncated", 0),
        "true_text": kwargs.get("true", "true"),
        "false_text": kwargs.get("false", "false"),
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
        tz=options["tz"],
        truncated=options["truncated"],
        true_text=options["true_text"],
        false_text=options["false_text"],
    )
    if options["strict"] and problems:
        raise ConversionError.from_values(
            name,
            column,
            options["detail"],
            [row["original"] for row in problems],
        )
    return converted, problems


# ── native Polars path ──────────────────────────────────────────────────
# Conversions on text, number, logical, and date columns run as vectorized
# Polars expressions. Two kinds of pattern decide which rows Polars settles:
#
# * For numbers and logical values the pattern is the whole grammar of the
#   per-value converter above, written with ASCII digits, so text that does
#   not match is a failure.
# * For dates, datetimes, and parse_number the pattern covers the common
#   shapes only ("2024-01-07", "2024-01-07 13:05:09", "$1,234.50"); any other
#   text goes through the per-value rules.
#
# Rows Polars cannot settle (an impossible date, 24:00:00, a daylight saving
# gap, accepted text that does not cast, or a failure with Unicode digits or
# spaces, where Python's isdigit() and strip() differ from these patterns) go
# through the per-value path one by one, so the two paths never disagree.

_COMPLETE_KINDS = frozenset(
    {"double", "integer", "parse_double", "parse_integer", "logical", "parse_logical"}
)
_DATE_KINDS = frozenset({"date", "ymd", "mdy", "dmy", "ym", "parse_date"})
_SHAPE_KINDS = frozenset(
    {*_DATE_KINDS, "datetime", "parse_datetime", "parse_number", *_DATETIME_ORDERS}
)
_FAST_KINDS = _COMPLETE_KINDS | _SHAPE_KINDS
_NUMBER_KINDS = frozenset({"double", "parse_double", "parse_number"})
_INTEGER_KINDS = frozenset({"integer", "parse_integer"})
_LOGICAL_KINDS = frozenset({"logical", "parse_logical"})
# NaN and Inf in any case, with an optional sign: Polars' float cast reads
# these spellings itself, as Python's float() does.
_SPECIAL = r"(?i:[+-]?(?:nan|inf))"
_PLAIN_FLOAT = (
    rf"^(?:[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?|{_SPECIAL})$"
)
_PLAIN_INT = r"^[+-]?[0-9]+$"
# Python's \d and isdigit() take any Unicode digit, and its strip() also
# removes \x1c-\x1f; the patterns here do neither.
_UNSURE = r"[\x1c-\x1f]|[\d\s&&[^0-9\t\n\x0B\x0C\r ]]"
# Date shapes: a four-digit year from 1000 (Python's date has no year 0),
# two-digit month and day, and one of - / . between them.
_SHAPE_FIELDS = {
    "y": r"(?P<y>[1-9][0-9]{3})",
    "m": r"(?P<m>[0-9]{2})",
    "d": r"(?P<d>[0-9]{2})",
}
_CLOCK = r"(?P<H>[0-9]{2}):(?P<M>[0-9]{2})"


def _rx(mark: str) -> str:
    """*mark* as a regex literal; Polars' regex has no re.escape."""
    return "\\x{%X}" % ord(mark)


def _localized_pattern(decimal_mark: str) -> str:
    return rf"^{_double_pattern(_rx(decimal_mark))}$"


def _out_dtype(kind: str, options: dict[str, Any]) -> Any:
    if kind in _INTEGER_KINDS:
        return pl.Int64
    if kind in _NUMBER_KINDS:
        return pl.Float64
    if kind in _LOGICAL_KINDS:
        return pl.Boolean
    if kind in _DATE_KINDS:
        return pl.Date
    if kind in _DATETIME_ORDERS:
        return pl.Datetime("us", options["tz"])
    return pl.Datetime("us")


def _date_shape(order: str) -> str:
    return r"[-/.]".join(_SHAPE_FIELDS[field] for field in order)


def _iso_date(parts: pl.Expr, order: str) -> pl.Expr:
    day = parts.struct.field("d") if "d" in order else pl.lit("01")
    return pl.concat_str(
        [parts.struct.field("y"), pl.lit("-"), parts.struct.field("m"), pl.lit("-"), day]
    )


def _format_shape(fmt: str | None, *, clock: bool) -> str | None:
    """A pattern for a strptime format of %Y %m %d (and %H %M %S), else None.

    Fields have fixed widths here, so a matched row reads the same in
    Python's strptime and Polars. Other codes use the per-value rules.
    """
    if not isinstance(fmt, str) or not fmt:
        return None
    widths = {"Y": r"[1-9][0-9]{3}", "m": "[0-9]{2}", "d": "[0-9]{2}"}
    if clock:
        widths.update({"H": "[0-9]{2}", "M": "[0-9]{2}", "S": "[0-9]{2}"})
    parts = []
    index = 0
    while index < len(fmt):
        char = fmt[index]
        if char == "%":
            code = fmt[index + 1:index + 2]
            if code not in widths:
                return None
            parts.append(widths[code])
            index += 2
            continue
        if char.isspace() and char != " ":
            return None
        parts.append(_rx(char))
        index += 1
    return "^" + "".join(parts) + "$"


def _fast_text(
    raw: pl.Expr, text: pl.Expr, kind: str, options: dict[str, Any]
) -> tuple[pl.Expr, pl.Expr] | None:
    """``(value, accepted)`` for a text column, or None when no shape applies.

    *raw* is the text as given and *text* is it stripped.
    """
    if kind in _INTEGER_KINDS:
        return text.cast(pl.Int64, strict=False), text.str.contains(_PLAIN_INT)
    if kind == "double":
        return text.cast(pl.Float64, strict=False), text.str.contains(_PLAIN_FLOAT)
    if kind == "parse_double":
        decimal_mark = options["decimal_mark"]
        cleaned = text
        if decimal_mark != ".":
            cleaned = cleaned.str.replace(decimal_mark, ".", literal=True)
        accepted = text.str.contains(_localized_pattern(decimal_mark))
        return cleaned.cast(pl.Float64, strict=False), accepted
    if kind in _LOGICAL_KINDS:
        key = raw if kind == "logical" else text.str.to_lowercase()
        truth = list(_AS_TRUE if kind == "logical" else _PARSE_TRUE)
        false = list(_AS_FALSE if kind == "logical" else _PARSE_FALSE)
        value = pl.when(key.is_in(truth)).then(True).when(key.is_in(false)).then(False)
        return value, key.is_in(truth) | key.is_in(false)
    if kind == "parse_number":
        decimal, grouping = _rx(options["decimal_mark"]), _rx(options["grouping_mark"])
        shape = (
            rf"^[^\x{{2D}}{decimal}0-9]*(?P<sign>-?)(?P<whole>[0-9][0-9{grouping}]*)"
            rf"(?:{decimal}(?P<frac>[0-9]*))?(?:[^eE0-9{decimal}{grouping}][\s\S]*)?$"
        )
        parts = raw.str.extract_groups(shape)
        digits = pl.concat_str([
            parts.struct.field("whole").str.replace_all(options["grouping_mark"], "", literal=True),
            pl.lit("."),
            pl.when(parts.struct.field("frac") == "").then(pl.lit("0"))
            .otherwise(parts.struct.field("frac").fill_null("0")),
        ])
        number = digits.cast(pl.Float64, strict=False)
        value = pl.when(parts.struct.field("sign") == "-").then(-number).otherwise(number)
        return value, raw.str.contains(shape)
    if kind == "date":
        shape = rf"^{_SHAPE_FIELDS['y']}-{_SHAPE_FIELDS['m']}-{_SHAPE_FIELDS['d']}$"
        return text.str.to_date("%Y-%m-%d", strict=False), text.str.contains(shape)
    if kind in _ORDERS:
        shape = rf"^{_date_shape(_ORDERS[kind])}$"
        iso = _iso_date(text.str.extract_groups(shape), _ORDERS[kind])
        return iso.str.to_date("%Y-%m-%d", strict=False), text.str.contains(shape)
    if kind == "datetime":
        day = rf"{_SHAPE_FIELDS['y']}-{_SHAPE_FIELDS['m']}-{_SHAPE_FIELDS['d']}"
        date_only = text.str.contains(rf"^{day}$")
        shape = rf"^{day}(?:[ T]{_CLOCK}:(?P<S>[0-9]{{2}}))?$"
        stamp = text.str.replace("T", " ", literal=True)
        value = (
            pl.when(date_only)
            .then(text.str.to_date("%Y-%m-%d", strict=False).cast(pl.Datetime("us")))
            .otherwise(stamp.str.to_datetime("%Y-%m-%d %H:%M:%S", time_unit="us", strict=False))
        )
        return value, text.str.contains(shape)
    if kind in {"parse_date", "parse_datetime"}:
        clock = kind == "parse_datetime"
        shape = _format_shape(options["fmt"], clock=clock)
        if shape is None:
            return None
        if clock:
            value = text.str.to_datetime(options["fmt"], time_unit="us", strict=False)
        else:
            value = text.str.to_date(options["fmt"], strict=False)
        return value, text.str.contains(shape)
    if kind in _DATETIME_ORDERS:
        order, clock = kind.split("_")
        seconds = r":(?P<S>[0-9]{2})" if clock == "hms" else ""
        shape = rf"^{_date_shape(_ORDERS[order])}[ T]{_CLOCK}{seconds}$"
        parts = text.str.extract_groups(shape)
        stamp = pl.concat_str([
            _iso_date(parts, _ORDERS[order]), pl.lit(" "),
            parts.struct.field("H"), pl.lit(":"), parts.struct.field("M"), pl.lit(":"),
            parts.struct.field("S") if clock == "hms" else pl.lit("00"),
        ])
        # Repeated clocks take the later occurrence and skipped clocks fail,
        # as the per-value rules do; those rows are rechecked one by one.
        value = stamp.str.to_datetime(
            "%Y-%m-%d %H:%M:%S", time_unit="us", strict=False
        ).dt.replace_time_zone(options["tz"], ambiguous="latest", non_existent="null")
        return value, text.str.contains(shape)
    return None


def _per_value(values: list[Any], options: dict[str, Any]) -> tuple[list[Any], list[bool]]:
    pairs = [
        _convert_one(
            value,
            kind=options["kind"],
            cutoff=options["cutoff"],
            fmt=options["fmt"],
            decimal_mark=options["decimal_mark"],
            grouping_mark=options["grouping_mark"],
            encoding=options["encoding"],
            tz=options["tz"],
            truncated=options["truncated"],
            true_text=options.get("true_text", "true"),
            false_text=options.get("false_text", "false"),
        )
        for value in values
    ]
    return [value for value, _ in pairs], [failed for _, failed in pairs]


def _typed_values(series: pl.Series, kind: str) -> tuple[pl.Expr, pl.Expr] | None:
    """``(value, failed)`` for a column that is not text, or None for per-value."""
    dtype = series.dtype
    v = pl.col("v")
    present = v.is_not_null()
    if dtype == pl.Null:
        return pl.lit(None), pl.lit(False)
    if kind in _INTEGER_KINDS | _NUMBER_KINDS:
        if dtype == pl.Boolean:
            return v, pl.lit(False)
        if dtype.is_integer():
            if kind in _INTEGER_KINDS:
                value = v.cast(pl.Int64, strict=False)
                return value, present & value.is_null()
            return v.cast(pl.Float64), pl.lit(False)
        if dtype.is_float():
            if kind in _INTEGER_KINDS:
                whole = v.is_finite() & (v == v.floor())
                value = pl.when(whole).then(v.cast(pl.Int64, strict=False))
                return value, present & value.is_null()
            return v.cast(pl.Float64), pl.lit(False)
        return None
    if kind in _LOGICAL_KINDS:
        if dtype == pl.Boolean:
            return v, pl.lit(False)
        if not (dtype.is_integer() or dtype.is_float()):
            return None
        number = v.cast(pl.Float64)
        nan = number.is_nan().fill_null(False)
        if kind == "logical":  # R: 0 is false, any other number true, NaN is NA
            return pl.when(nan).then(None).otherwise(number != 0), pl.lit(False)
        known = number.is_infinite() | (number == 1) | (number == 0)
        value = (
            pl.when(nan).then(None)
            .when(number.is_infinite() | (number == 1)).then(True)
            .when(number == 0).then(False)
        )
        return value, present & ~nan & ~known
    if kind in _DATE_KINDS:
        if dtype == pl.Date:
            return v, pl.lit(False)
        if isinstance(dtype, pl.Datetime):
            return v.dt.date(), pl.lit(False)
        return None
    if kind in {"datetime", "parse_datetime"}:
        if dtype == pl.Date:
            return v.cast(pl.Datetime("us")), pl.lit(False)
        if isinstance(dtype, pl.Datetime):
            if dtype.time_zone:
                return v.dt.convert_time_zone("UTC").dt.replace_time_zone(None), pl.lit(False)
            return v, pl.lit(False)
    return None


_DATE_FORMATS = {
    "date": ("%Y-%m-%d",),
    "ymd": ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d"),
    "mdy": ("%m/%d/%Y", "%m-%d-%Y", "%m.%d.%Y"),
    "dmy": ("%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y"),
    "ym": ("%Y-%m", "%Y/%m", "%Y.%m"),
}
_CLOCK_FORMATS = {
    "datetime": ("%Y-%m-%d %H:%M:%S",),
    "ymd_hms": ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y.%m.%d %H:%M:%S"),
    "ymd_hm": ("%Y-%m-%d %H:%M", "%Y/%m/%d %H:%M", "%Y.%m.%d %H:%M"),
    "mdy_hms": ("%m/%d/%Y %H:%M:%S", "%m-%d-%Y %H:%M:%S", "%m.%d.%Y %H:%M:%S"),
    "dmy_hms": ("%d/%m/%Y %H:%M:%S", "%d-%m-%Y %H:%M:%S", "%d.%m.%Y %H:%M:%S"),
}


def _no_failures(length: int) -> pl.Series:
    return pl.repeat(False, length, dtype=pl.Boolean, eager=True).alias("failed")


def _every_row(frame: pl.DataFrame) -> bool:
    return bool(frame.select(pl.col("settled").fill_null(False).all()).item())


def _fixed_width(fmt: str) -> int | None:
    """Character width of a zero-padded strptime format, or None if a code varies."""
    widths = {"Y": 4, "m": 2, "d": 2, "H": 2, "M": 2, "S": 2}
    total = 0
    index = 0
    while index < len(fmt):
        if fmt[index] == "%":
            code = fmt[index + 1:index + 2]
            if code not in widths:
                return None
            total += widths[code]
            index += 2
            continue
        total += 1
        index += 1
    return total


def _uniform_fixed(
    series: pl.Series,
    formats: tuple[str, ...],
    *,
    clock: bool,
    tz: str | None,
) -> tuple[pl.Series, pl.Series] | None:
    """Parse *series* when every present value has one format's exact width and parses.

    Polars accepts some shorter spellings (``2024-1-7``, ``24-01-07``). Those
    widths miss, so the column stays on the general path. Impossible dates
    parse as null and do the same.
    """
    for fmt in formats:
        width = _fixed_width(fmt)
        if width is None:
            continue
        try:
            if clock:
                naive = pl.col("v").str.to_datetime(fmt, time_unit="us", strict=False)
                value = (
                    naive if tz is None
                    else naive.dt.replace_time_zone(tz, ambiguous="latest", non_existent="null")
                )
            else:
                value = pl.col("v").str.to_date(fmt, strict=False)
            checked = series.to_frame("v").with_columns(value=value).select(
                "value",
                settled=pl.col("v").is_null() | (
                    (pl.col("v").str.len_chars() == width) & pl.col("value").is_not_null()
                ),
            )
        except (pl.exceptions.PolarsError, ValueError):
            continue
        if _every_row(checked):
            return checked["value"], _no_failures(series.len())
    return None


def _uniform_dates(series: pl.Series, formats: tuple[str, ...]) -> tuple[pl.Series, pl.Series] | None:
    return _uniform_fixed(series, formats, clock=False, tz=None)


def _uniform_clocks(
    series: pl.Series, formats: tuple[str, ...], *, tz: str | None
) -> tuple[pl.Series, pl.Series] | None:
    found = _uniform_fixed(series, formats, clock=True, tz=tz)
    if found is not None:
        return found
    # ``2024-01-07T13:05:09`` is the same width as the space form.
    replaced = series.str.replace("T", " ", literal=True)
    if replaced.equals(series):
        return None
    return _uniform_fixed(replaced, formats, clock=True, tz=tz)


def _uniform_number(series: pl.Series, options: dict[str, Any]) -> tuple[pl.Series, pl.Series] | None:
    """``$1,234.50`` and ``$5.50`` when grouping is a comma and the decimal is a point."""
    if options["decimal_mark"] != "." or options["grouping_mark"] != ",":
        return None
    shape = r"^\$[0-9]+(?:,[0-9]+)*\.[0-9]+$"
    base = series.to_frame("v").with_columns(
        text=pl.col("v").str.strip_chars(),
        present=pl.col("v").is_not_null(),
    )
    value = (
        pl.col("text").str.replace_all(",", "", literal=True)
        .str.replace_all("$", "", literal=True)
        .cast(pl.Float64, strict=False)
    )
    checked = base.with_columns(value=value).select(
        "value",
        settled=~pl.col("present") | (pl.col("text").str.contains(shape) & pl.col("value").is_not_null()),
    )
    if _every_row(checked):
        return checked["value"], _no_failures(series.len())
    return None


def _uniform_fast(
    series: pl.Series, kind: str, options: dict[str, Any]
) -> tuple[pl.Series, pl.Series] | None:
    """A direct Polars parse when the whole column shares one common shape."""
    if kind == "parse_number":
        return _uniform_number(series, options)
    if kind in _DATE_FORMATS:
        return _uniform_dates(series, _DATE_FORMATS[kind])
    if kind == "parse_date":
        fmt = options["fmt"]
        if _format_shape(fmt, clock=False) is None:
            return None
        return _uniform_dates(series, (fmt,))
    if kind == "parse_datetime":
        fmt = options["fmt"]
        if _format_shape(fmt, clock=True) is None:
            return None
        # An explicit format is literal: no ``T`` retry, unlike the guessed orders.
        return _uniform_fixed(series, (fmt,), clock=True, tz=None)
    if kind not in _CLOCK_FORMATS:
        return None
    if kind in _DATETIME_ORDERS and options["truncated"]:
        return None
    tz = options["tz"] if kind in _DATETIME_ORDERS else None
    found = _uniform_clocks(series, _CLOCK_FORMATS[kind], tz=tz)
    if found is not None or kind != "datetime":
        return found
    dates = _uniform_dates(series, ("%Y-%m-%d",))
    if dates is None:
        return None
    value, failed = dates
    return value.cast(pl.Datetime("us")), failed


# Kinds whose text a plain Polars cast reads exactly as the rules do, for
# every value the cast accepts. Floats exclude inf and NaN spellings, which
# the cast reads and the rules may not.
_CAST_FIRST = {
    "integer": pl.Int64,
    "parse_integer": pl.Int64,
    "double": pl.Float64,
    "parse_double": pl.Float64,
}


def _cast_first(
    series: pl.Series, kind: str, options: dict[str, Any], out_dtype: Any
) -> tuple[pl.Series, pl.Series] | None:
    """A plain cast (or word lookup) when it reads every present value, else None.

    Clean columns skip the strip and the shape check. A column with any
    value the cast rejects (spaces, ``NA`` text, overflow) takes the full path.
    """
    if kind in _LOGICAL_KINDS:
        # The same words the shape check accepts, looked up in one pass.
        key = series if kind == "logical" else series.str.to_lowercase()
        truth, false = (_AS_TRUE, _AS_FALSE) if kind == "logical" else (_PARSE_TRUE, _PARSE_FALSE)
        words = {**dict.fromkeys(truth, True), **dict.fromkeys(false, False)}
        value = key.replace_strict(words, default=None, return_dtype=pl.Boolean)
        if (series.is_not_null() & value.is_null()).any():
            return None
        return value.cast(out_dtype).alias(series.name), _no_failures(series.len())
    target = _CAST_FIRST.get(kind)
    if target is None or (kind == "parse_double" and options["decimal_mark"] != "."):
        return None
    value = series.cast(target, strict=False)
    unsure = series.is_not_null() & value.is_null()
    if target == pl.Float64:
        unsure = unsure | ~value.is_finite().fill_null(True)
    if unsure.any():
        return None
    return value.cast(out_dtype).alias(series.name), _no_failures(series.len())


def _fast_convert(series: pl.Series, options: dict[str, Any]) -> tuple[pl.Series, pl.Series] | None:
    """``(converted, failed)`` computed natively, or None for the per-value path."""
    kind = options["kind"]
    if kind not in _FAST_KINDS or options["encoding"].lower().replace("_", "-") not in {"utf-8", "utf8"}:
        return None
    out_dtype = _out_dtype(kind, options)
    v = pl.col("v")
    present = v.is_not_null()
    if series.dtype == pl.String:
        direct = _cast_first(series, kind, options, out_dtype)
        if direct is not None:
            return direct
        direct = _uniform_fast(series, kind, options)
        if direct is not None:
            value, failed = direct
            return value.cast(out_dtype), failed
    if series.dtype != pl.String:
        typed = _typed_values(series, kind)
        if typed is None:
            return None
        value, failed = typed
        frame = series.to_frame("v").select(
            value=value.cast(out_dtype), failed=pl.repeat(False, pl.len()) | failed
        )
        return frame["value"], frame["failed"]
    shaped = _fast_text(v, pl.col("t"), kind, options)
    if shaped is None:
        return None
    value, accepted = shaped
    frame = (
        series.to_frame("v")
        .with_columns(t=v.str.strip_chars())
        .with_columns(value=value, accepted=accepted)
        .select(
            value=pl.when("accepted").then("value").cast(out_dtype),
            failed=present & ~pl.col("accepted"),
            miss=pl.col("accepted") & pl.col("value").is_null(),
        )
    )
    value_out, failed_out = frame["value"], frame["failed"]
    rows = failed_out.arg_true()
    if kind in _COMPLETE_KINDS:
        # Text the grammar rejects is a failure, unless Python's Unicode-aware
        # rules might still read it.
        if rows.len():
            rows = rows.filter(series.gather(rows).str.contains(_UNSURE))
    else:
        # A shape covers common text only: the rest goes to the per-value rules.
        failed_out = pl.repeat(False, failed_out.len(), dtype=pl.Boolean, eager=True).alias(failed_out.name)
    rows = pl.concat([rows, frame["miss"].arg_true()])
    if rows.len():
        redo, redo_failed = _per_value(series.gather(rows).to_list(), options)
        value_out = value_out.scatter(rows, pl.Series(redo, dtype=out_dtype))
        failed_out = failed_out.scatter(rows, pl.Series(redo_failed, dtype=pl.Boolean))
    return value_out, failed_out


def _failed_originals(series: pl.Series, failed: pl.Series) -> pl.Series:
    """Original text of failed values, null elsewhere, as the per-value path writes it."""
    if series.dtype == pl.String:
        return series.to_frame("v").select(
            pl.when(failed).then(pl.col("v")).otherwise(None)
        ).to_series()
    out = pl.repeat(None, series.len(), dtype=pl.String, eager=True)
    rows = failed.arg_true()
    if rows.len():
        out = out.scatter(rows, [_original(value) for value in series.gather(rows).to_list()])
    return out


def _fast_apply(
    series: pl.Series, name: str, options: dict[str, Any], column: str
) -> tuple[pl.Series, pl.Series] | None:
    """The native path for :func:`_apply`: raise in strict mode, else return."""
    fast = _fast_convert(series, options)
    if fast is None:
        return None
    converted, failed = fast
    if options["strict"] and failed.any():
        originals = _failed_originals(series, failed).drop_nulls().to_list()
        raise ConversionError.from_values(name, column, options["detail"], originals)
    return converted, failed


def compile_polars(
    name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    column: str,
) -> pl.Expr:
    """Compile a conversion to a Polars expression."""
    options = _options(name, kwargs)
    dtype = (
        pl.Datetime("us", options["tz"]) if name in _DATETIME_ORDERS
        else _DTYPES[options["kind"]]
    )

    if options["kind"] == "character":

        def text_batch(series: pl.Series) -> pl.Series:
            text = _text_series(series, options["true_text"], options["false_text"])
            if text is not None:
                return text
            converted, _problems = _apply(series.to_list(), name, options, column)
            return pl.Series(series.name, converted, dtype=pl.String)

        return args[0].map_batches(text_batch, return_dtype=pl.String, is_elementwise=True)

    def batch(series: pl.Series) -> pl.Series:
        fast = _fast_apply(series, name, options, column)
        if fast is not None:
            return fast[0]
        converted, _problems = _apply(series.to_list(), name, options, column)
        return pl.Series(converted, dtype=dtype)

    return args[0].map_batches(batch, return_dtype=dtype)


def _pandas_fast(
    source: pd.Series, name: str, options: dict[str, Any], column: str
) -> pd.Series | None:
    """The native Polars path for a pandas column, or None for per-value.

    The column goes to Polars with its own type (NaN becomes missing, as
    pandas means it), converts there, and comes back without Python lists.
    Failures are recorded for problems() exactly as the per-value path does.
    """
    kind = options["kind"]
    if kind not in _FAST_KINDS:
        return None
    try:
        series = pl.from_pandas(source)
    except (OverflowError, TypeError, ValueError, pl.exceptions.PolarsError):
        return None
    if isinstance(series.dtype, (pl.Categorical, pl.Enum)):
        series = series.cast(pl.String)
    fast = _fast_apply(series, name, options, column)
    if fast is None:
        return None
    converted, failed = fast
    sink = _SINK.get()
    if sink is not None and not options["strict"] and failed.any():
        originals = _failed_originals(series, failed)
        rows = failed.arg_true()
        sink.extend(
            {"row": int(row) + 1, "column": column, "original": original, "expected": options["expected"]}
            for row, original in zip(rows.to_list(), originals.gather(rows).to_list())
        )
    if kind in _DATE_KINDS:
        # Dates stay Python date objects in an object column, as before.
        return pd.Series(converted.to_list(), index=source.index, dtype="object")
    if kind in _INTEGER_KINDS:
        target: Any = "Int64"
    elif kind in _NUMBER_KINDS:
        target = "Float64"
    elif kind in _LOGICAL_KINDS:
        target = "boolean"
    else:  # datetimes keep their NumPy datetime64 type and any time zone
        out = converted.to_pandas().dt.as_unit("us")
        out.index = source.index
        return out
    out = converted.to_pandas(use_pyarrow_extension_array=True).astype(target)
    out.index = source.index
    return out


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
    if options["kind"] == "character":
        # The same text as on Polars: convert the column with its own type
        # (a float32 stays float32; NaN becomes missing, as pandas means it),
        # and fall back to value-by-value text when the values mix types.
        try:
            text = _text_series(
                pl.from_pandas(source), options["true_text"], options["false_text"]
            )
        except (OverflowError, TypeError, ValueError, pl.exceptions.PolarsError):
            text = None
        if text is not None:
            return pd.Series(text.to_list(), index=source.index, dtype="string")
    fast = _pandas_fast(source, name, options, column)
    if fast is not None:
        return fast
    # pandas marks a missing value with NaN, in float and object columns
    # alike, so NaN here is missing, not a failure: as_integer(c(2, NA)) is
    # c(2L, NA), and as_date(c("2010-01-31", NA)) is a date and NA.
    values = [
        None if isinstance(value, float) and value != value else value
        for value in source.tolist()
    ]
    converted, problems = _apply(values, name, options, column)
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
        return pd.Series(pd.to_datetime(converted), index=source.index).dt.as_unit("us")
    if kind in _DATETIME_ORDERS:
        return pd.Series(
            converted, index=source.index, dtype=pd.DatetimeTZDtype("us", options["tz"])
        )
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


def _node_options(name: str, kwargs: dict[str, Any]) -> dict[str, Any] | None:
    """Options when this call records problems, else None."""
    kind, default_strict, expected, detail = _SPECS[name]
    strict = _literal(name, "strict", kwargs, default_strict)
    if strict or kind == "character":  # every value has a text form
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
        "tz": _literal(name, "tz", kwargs, "UTC"),
        "truncated": _literal(name, "truncated", kwargs, 0),
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
        fast = _fast_convert(series, options)
        if fast is not None:
            return _failed_originals(series, fast[1])
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
            tz=options["tz"],
            truncated=options["truncated"],
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


def parse_columns_eagerly(
    frame: pl.DataFrame, assignments: dict[str, Any], *, row_offset: int = 0
) -> tuple[pl.DataFrame, pl.DataFrame | None] | None:
    """Parse in-memory text columns once, with their failures.

    *assignments* maps each column to a parse of that same column, as the
    readers build them. Each column is converted once natively, giving
    both the values and the problems() rows (numbered from 1 plus
    *row_offset*). Returns None when any parse needs the general path,
    including strict parses, which raise from the plan.
    """
    plans = []
    for output, value in assignments.items():
        node = getattr(value, "node", None)
        if not (
            isinstance(node, tuple) and node[0] == "func" and node[1] in _SPECS
            and node[2] == (("col", output),)
            and all(isinstance(v, tuple) and v[0] == "lit" for v in node[3].values())
        ):
            return None
        name, kwargs = node[1], {key: v[1] for key, v in node[3].items()}
        options = _options(name, kwargs)
        if options["strict"] or options["kind"] == "character" or frame.schema[output] != pl.String:
            return None
        plans.append((output, name, options, _node_options(name, node[3])))
    converted, problems = {}, []
    for output, name, options, recorded in plans:
        series = frame.get_column(output)
        fast = _fast_convert(series, options)
        if fast is None:
            return None
        value, failed = fast
        dtype = (
            pl.Datetime("us", options["tz"]) if name in _DATETIME_ORDERS
            else _DTYPES[options["kind"]]
        )
        converted[output] = value.cast(dtype).alias(output)
        if recorded is not None and failed.any():
            rows = failed.arg_true()
            problems.append(pl.DataFrame({
                "row": (rows.cast(pl.Int64) + 1 + row_offset),
                "column": pl.repeat(output, rows.len(), dtype=pl.String, eager=True),
                "original": series.gather(rows),
                "expected": pl.repeat(recorded["expected"], rows.len(), dtype=pl.String, eager=True),
            }, schema=_PROBLEM_SCHEMA))
    data = frame.with_columns(converted.values())
    return data, (pl.concat(problems) if problems else None)


def problems_frame(data: Any) -> Any:
    """A frame of parse failures: row, column, original, expected.

    On Polars this is a lazy plan captured when the value was parsed. It
    is not collected here, and it does not collect the data frame.
    """
    from tidy3.frame import TidyFrame

    if not isinstance(data, TidyFrame):
        raise TypeError("problems() expects a TidyFrame")
    stored = getattr(data, "_problems", None)
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

    Computes the data and its parse failures in one run, so the parses
    happen once. Returns the computed frame when nothing failed.
    """
    from tidy3.frame import TidyFrame

    if not isinstance(data, TidyFrame):
        raise TypeError("problems() expects a TidyFrame")
    if data._problems is None:
        return data
    data = data._compute(reads=("stop_for_problems() checks the parsed values",))
    rows = data._problems.collect().to_dicts()
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
    """Alias of :func:`as_double`. Errors name ``as_numeric``."""
    _bool("as_numeric() strict", strict)
    return _call("as_numeric", x, suffix={"strict": strict})


def as_integer(x: Any, *, strict: bool = True) -> Expr:
    """Convert to an integer. ``2`` and ``2.0`` succeed; ``2.5`` is a failure.

    This does not truncate toward zero the way R's ``as.integer`` does.
    ``"2"`` succeeds; ``"2.0"`` and ``"2.5"`` do not. NaN is a failure.
    """
    _bool("as_integer() strict", strict)
    return _call("as_integer", x, suffix={"strict": strict})


def _text(name: str, x: Any, strict: bool, true: str, false: str) -> Expr:
    _bool(f"{name}() strict", strict)
    for key, value in (("true", true), ("false", false)):
        if not isinstance(value, str):
            raise TypeError(f"{name}() {key} must be a string")
    return _call(name, x, suffix={"strict": strict, "true": true, "false": false})


def as_character(
    x: Any, *, true: str = "true", false: str = "false", strict: bool = True
) -> Expr:
    """Convert to text, written the way the column's type shows it.

    ============================  ==============================
    Column type and value         Text
    ============================  ==============================
    float ``21.0``                ``"21.0"``
    float ``0.1 + 0.2``           ``"0.30000000000000004"``
    float32 ``22.8``              ``"22.8"``
    integer ``21``                ``"21"``
    logical ``True`` / ``False``  ``"true"`` / ``"false"``
    date                          ``"2010-01-31"``
    datetime                      ``"2010-01-31 15:30:00"``
    NaN, infinity                 ``"NaN"``, ``"inf"``
    ============================  ==============================

    Numbers are never rounded: the text reads back as exactly the same
    number. A datetime shows fractional seconds only when it has them, and
    a datetime with a time zone ends with its offset (``"-08:00"``). Missing
    values stay missing. ``true`` and ``false`` set the text for logical
    values: ``true="TRUE", false="FALSE"`` for R, or ``true="True",
    false="False"`` for Python's ``str()``.

    Differences from R's ``as.character``: R writes ``21`` for ``21.0``,
    ``TRUE`` / ``FALSE`` for logical values, rounds to 15 significant
    digits (``0.3`` for ``0.1 + 0.2``), and switches to scientific
    notation for ``1e+05``. For an id stored as a float, convert to an
    integer first: ``as_character(as_integer(id))`` gives ``"1001"``, not
    ``"1001.0"``. Differences from Python's ``str()``: logical values are
    ``"true"`` / ``"false"`` by default, not ``"True"`` / ``"False"``, and
    NaN is ``"NaN"``, not ``"nan"``.

    :func:`as_string` is the same function. ``strict`` is accepted for the
    same signature as the other ``as_*`` functions; every value has a text
    form, so nothing fails.
    """
    return _text("as_character", x, strict, true, false)


def as_string(
    x: Any, *, true: str = "true", false: str = "false", strict: bool = True
) -> Expr:
    """Alias of :func:`as_character`: the same text, options, and rules.

    The Python name for the same conversion, as ``is_string`` is for
    ``is_character``. Errors name ``as_string``.
    """
    return _text("as_string", x, strict, true, false)


def as_logical(x: Any, *, strict: bool = True) -> Expr:
    """Convert to logical, as R's ``as.logical``.

    Text must be exactly ``TRUE``, ``true``, ``True``, or ``T`` (or the
    ``FALSE`` forms); ``"tRuE"`` and ``" TRUE"`` are failures, as in R.
    Numbers work: ``0`` is false and any other number is true. The string
    ``"1"`` is a failure.
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
    """Parse a whole string as a number, as readr's ``parse_double`` does.

    ``"1.5"``, ``" 1e3 "``, and ``"5."`` work; the decimal mark comes from
    *locale*. Grouping marks, currency, ``Inf``, and ``NaN`` are failures:
    ``"1,234.50"`` and ``"$1,234"`` fail, so use :func:`parse_number` for
    them. Failures become missing and are listed by :func:`problems`,
    unless ``strict=True``.
    """
    return _parse("parse_double", x, suffix={"strict": strict}, marks=locale)


def parse_integer(x: Any, *, locale: Locale | None = None, strict: bool = False) -> Expr:
    """Parse a whole string as an integer, as readr's ``parse_integer`` does.

    ``" 7 "``, ``"+5"``, and ``"007"`` work. ``"2.5"``, ``"1e3"``, and
    ``"1,234"`` are failures. Integers are 64-bit, so values past R's
    32-bit limit still parse. *locale* is accepted for symmetry and unused.
    """
    return _parse("parse_integer", x, suffix={"strict": strict}, marks=locale)


def parse_logical(x: Any, *, strict: bool = False) -> Expr:
    """Parse logical text the way readr does: ``true``/``false``/``t``/``f``/``1``/``0``.

    ``yes`` and ``no`` are failures. Case does not matter.
    """
    return _parse("parse_logical", x, suffix={"strict": strict}, marks=None)


def parse_number(x: Any, *, locale: Locale | None = None, strict: bool = False) -> Expr:
    """The first number in messy text, as readr's ``parse_number`` reads it.

    ``"$1,234.50"`` is 1234.5, ``"50%"`` is 50, and ``"1e3"`` is 1000.
    Grouping marks are skipped anywhere in the whole part, so ``"1,23"`` is
    123. A minus counts only right before the number: ``"$-5"`` is -5, but
    ``"-$5"`` and ``"- 5"`` are failures, as in readr.
    """
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
    """Parse datetimes with a ``strptime`` *format*. See :func:`parse_date`.

    The result is a naive datetime. A ``%z`` offset is converted to UTC and
    the zone is dropped, as in :func:`as_datetime` and readr's UTC instant:
    ``2024-01-01 12:00:00+0200`` is 10:00 and ``2024-01-01 12:00:00-0500``
    is 17:00. Mixed offsets in one column parse. For a timezone-aware
    result, use :func:`ymd_hms`.
    """
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

    ``"2010-01-31"``, ``"20100131"``, and ``"2010 Jan 31"`` all work; month
    names may be abbreviated. Text with a time, such as
    ``"2010-01-31 10:00"``, is a failure, as in lubridate. A two-digit year
    needs ``cutoff``; see :func:`parse_date`. Failures become missing and
    show up in :func:`problems`.
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


def _parse_clock(
    name: str, x: Any, tz: str, truncated: int, cutoff: int | None, strict: bool
) -> Expr:
    if not isinstance(tz, str) or not tz:
        raise TypeError(f"{name}() tz must be a timezone name")
    ZoneInfo(tz)
    maximum = len(name.split("_")[1])
    if isinstance(truncated, bool) or not isinstance(truncated, int) or not 0 <= truncated <= maximum:
        raise ValueError(f"{name}() truncated must be an integer from 0 to {maximum}")
    return _parse(
        name, x,
        suffix={"tz": tz, "truncated": truncated, "cutoff": cutoff, "strict": strict},
        marks=None,
    )


def ymd_hms(
    x: Any, *, tz: str = "UTC", truncated: int = 0,
    cutoff: int | None = None, strict: bool = False,
) -> Expr:
    """Parse year-month-day hours:minutes:seconds into a zoned datetime.

    Accepts numeric or English month names, separated or compact fields,
    AM/PM, fractional seconds, and ``Z`` or numeric offsets. Fractions are
    truncated to microseconds. Unzoned clocks use *tz* (UTC by default);
    explicit offsets preserve the instant and convert to *tz*.

    ``truncated=1`` also accepts missing seconds, 2 accepts an hour alone,
    and 3 accepts a date alone, filling omitted time fields with zero.
    Two-digit years require *cutoff*, as in :func:`ymd`. Nonexistent DST
    clocks fail; repeated clocks choose the later occurrence. Failures
    become missing and appear in :func:`problems`, unless *strict*.
    """
    return _parse_clock("ymd_hms", x, tz, truncated, cutoff, strict)


def ymd_hm(
    x: Any, *, tz: str = "UTC", truncated: int = 0,
    cutoff: int | None = None, strict: bool = False,
) -> Expr:
    """Parse year-month-day hours:minutes; see :func:`ymd_hms`.

    ``truncated=1`` permits an hour alone; 2 permits a date alone.
    Input with seconds fails rather than silently discarding them.
    """
    return _parse_clock("ymd_hm", x, tz, truncated, cutoff, strict)


def mdy_hms(
    x: Any, *, tz: str = "UTC", truncated: int = 0,
    cutoff: int | None = None, strict: bool = False,
) -> Expr:
    """Parse month-day-year hours:minutes:seconds; see :func:`ymd_hms`."""
    return _parse_clock("mdy_hms", x, tz, truncated, cutoff, strict)


def dmy_hms(
    x: Any, *, tz: str = "UTC", truncated: int = 0,
    cutoff: int | None = None, strict: bool = False,
) -> Expr:
    """Parse day-month-year hours:minutes:seconds; see :func:`ymd_hms`."""
    return _parse_clock("dmy_hms", x, tz, truncated, cutoff, strict)
