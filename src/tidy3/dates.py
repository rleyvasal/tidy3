"""Calendar helpers in the style of lubridate.

``date + months(1)`` and :func:`add_with_rollback` use the same month rule
Polars already uses: a day that does not exist in the target month rolls
back to the last real day. 31 January plus one month is 28 February, or 29
in a leap year. 29 February plus one year is 28 February. That is lubridate's
``add_with_rollback`` (``%m+%``), not lubridate's ``+``, which can return
missing instead of rolling back.

:func:`today` and :func:`now` default to UTC, not the machine's local zone.
Pass ``tzone="local"`` or a zone name such as ``"America/Los_Angeles"``.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

import numbers

import polars as pl

from tidy3.convert import dmy, dmy_hms, mdy, mdy_hms, ym, ymd, ymd_hm, ymd_hms
from tidy3.convert import dmy_h, dmy_hm, mdy_h, mdy_hm, parse_date_time, ymd_h
from tidy3.expr import Expr, _func, col

__all__ = [
    "Period",
    "Duration",
    "months",
    "years",
    "weeks",
    "days",
    "hours",
    "minutes",
    "seconds",
    "period",
    "dseconds",
    "dminutes",
    "dhours",
    "ddays",
    "dweeks",
    "dmonths",
    "dyears",
    "duration",
    "as_duration",
    "time_length",
    "add_with_rollback",
    "year",
    "month",
    "day",
    "wday",
    "yday",
    "week",
    "isoweek",
    "quarter",
    "hour",
    "minute",
    "second",
    "with_tz",
    "force_tz",
    "floor_date",
    "ceiling_date",
    "round_date",
    "days_in_month",
    "make_date",
    "make_datetime",
    "seq",
    "date_seq",
    "isoyear",
    "leap_year",
    "today",
    "now",
    "excel_numeric_to_date",
    "ymd",
    "mdy",
    "dmy",
    "ym",
    "ymd_hms",
    "ymd_hm",
    "mdy_hms",
    "dmy_hms",
    "ymd_h",
    "mdy_hm",
    "mdy_h",
    "dmy_hm",
    "dmy_h",
    "parse_date_time",
]

FUNCTIONS = frozenset(
    {
        "year",
        "month",
        "day",
        "wday",
        "yday",
        "week",
        "isoweek",
        "quarter",
        "hour",
        "minute",
        "second",
        "with_tz",
        "force_tz",
        "floor_date",
        "ceiling_date",
        "days_in_month",
        "add_with_rollback",
        "excel_numeric_to_date",
        "round_date",
        "make_date",
        "make_datetime",
        "seq_date",
        "isoyear",
        "leap_year",
        "duration",
        "as_duration",
        "time_length",
    }
)
_UNITS = ("second", "minute", "hour", "day", "week", "month", "quarter", "year")
_TRUNC = {
    "second": "1s",
    "minute": "1m",
    "hour": "1h",
    "day": "1d",
    "month": "1mo",
    "quarter": "1q",
    "year": "1y",
}
_OFFSET = {
    "second": "1s",
    "minute": "1m",
    "hour": "1h",
    "day": "1d",
    "week": "7d",
    "month": "1mo",
    "quarter": "3mo",
    "year": "1y",
}
_PANDAS_FREQ = {"second": "s", "minute": "min", "hour": "h", "day": "D"}


_PERIOD_SLOTS = ("years", "months", "days", "hours", "minutes", "seconds")
# Average lengths, as lubridate uses for durations and period comparisons.
_SECONDS = {
    "second": 1.0,
    "minute": 60.0,
    "hour": 3600.0,
    "day": 86400.0,
    "week": 604800.0,
    "month": 2629800.0,
    "year": 31557600.0,
}
_SPAN_UNITS = {
    **{name: name for name in _SECONDS},
    **{f"{name}s": name for name in _SECONDS},
    "sec": "second",
    "secs": "second",
    "min": "minute",
    "mins": "minute",
}
_SLOT_SECONDS = {
    "years": _SECONDS["year"],
    "months": _SECONDS["month"],
    "days": _SECONDS["day"],
    "hours": _SECONDS["hour"],
    "minutes": _SECONDS["minute"],
    "seconds": 1.0,
}


def _span_unit(name: str, unit: Any) -> str:
    if not isinstance(unit, str):
        raise TypeError(f"{name}() unit must be a string")
    token = _SPAN_UNITS.get(unit.strip().lower())
    if token is None:
        raise ValueError(f"{name}() unit must be one of: {', '.join(_SECONDS)}; got {unit!r}")
    return token


def _is_column(value: Any) -> bool:
    return isinstance(value, (Expr, pl.Expr))


class Period:
    """Calendar and clock units: years, months, days, hours, minutes, seconds.

    Adding a period moves the calendar or the local clock, so ``days(1)``
    can be 23 or 25 real hours. Months and years roll a missing day back to
    the end of the month (see :func:`add_with_rollback`). Weeks are stored
    as seven days. Units are not normalized: ``hours(25)`` stays 25 hours.
    """

    __slots__ = _PERIOD_SLOTS

    def __init__(
        self,
        *,
        years: Any = 0,
        months: Any = 0,
        weeks: Any = 0,
        days: Any = 0,
        hours: Any = 0,
        minutes: Any = 0,
        seconds: Any = 0,
    ):
        self.years = years
        self.months = months
        self.days = days if _zero(weeks) else _combine(_scale(weeks, 7), days)
        self.hours = hours
        self.minutes = minutes
        self.seconds = seconds

    def _slots(self) -> tuple[Any, ...]:
        return tuple(getattr(self, name) for name in _PERIOD_SLOTS)

    def __repr__(self) -> str:
        parts = [f"{name}={value!r}" for name, value in zip(_PERIOD_SLOTS, self._slots()) if not _zero(value)]
        return f"Period({', '.join(parts) or 'seconds=0'})"

    def __neg__(self) -> "Period":
        return Period(**{name: _negate(value) for name, value in zip(_PERIOD_SLOTS, self._slots())})

    def __add__(self, other: Any) -> "Period":
        if isinstance(other, timedelta):
            raise TypeError(_INCOMPATIBLE)
        if not isinstance(other, Period):
            return NotImplemented
        return Period(**{
            name: _combine(left, right)
            for name, left, right in zip(_PERIOD_SLOTS, self._slots(), other._slots())
        })

    def __radd__(self, other: Any) -> "Period":
        if isinstance(other, timedelta):
            raise TypeError(_INCOMPATIBLE)
        return NotImplemented

    def __sub__(self, other: Any) -> "Period":
        if isinstance(other, timedelta):
            raise TypeError(_INCOMPATIBLE)
        if not isinstance(other, Period):
            return NotImplemented
        return self + -other

    def __rsub__(self, other: Any) -> "Period":
        if isinstance(other, timedelta):
            raise TypeError(_INCOMPATIBLE)
        return NotImplemented

    def __mul__(self, other: Any) -> "Period":
        if isinstance(other, bool) or not isinstance(other, int):
            return NotImplemented
        return Period(**{name: _scale(value, other) for name, value in zip(_PERIOD_SLOTS, self._slots())})

    __rmul__ = __mul__

    def total_seconds(self) -> float:
        """Length using average units (a month is 30.4375 days), as lubridate compares periods."""
        if any(_is_column(value) or value is None for value in self._slots()):
            raise TypeError("a period with column counts has no single length")
        return float(sum(value * _SLOT_SECONDS[name] for name, value in zip(_PERIOD_SLOTS, self._slots())))

    def _compare(self, other: Any) -> Any:
        if isinstance(other, (Period, timedelta)) and not any(_is_column(v) for v in self._slots()):
            if isinstance(other, Period) and any(_is_column(v) for v in other._slots()):
                return None
            return self.total_seconds(), other.total_seconds()
        return None

    def __eq__(self, other: Any) -> Any:
        pair = self._compare(other)
        return NotImplemented if pair is None else pair[0] == pair[1]

    def __ne__(self, other: Any) -> Any:
        pair = self._compare(other)
        return NotImplemented if pair is None else pair[0] != pair[1]

    def __lt__(self, other: Any) -> Any:
        pair = self._compare(other)
        return NotImplemented if pair is None else pair[0] < pair[1]

    def __le__(self, other: Any) -> Any:
        pair = self._compare(other)
        return NotImplemented if pair is None else pair[0] <= pair[1]

    def __gt__(self, other: Any) -> Any:
        pair = self._compare(other)
        return NotImplemented if pair is None else pair[0] > pair[1]

    def __ge__(self, other: Any) -> Any:
        pair = self._compare(other)
        return NotImplemented if pair is None else pair[0] >= pair[1]

    def __hash__(self) -> int:
        if any(_is_column(value) for value in self._slots()):
            return id(self)
        return hash(("Period", self._slots()))

    def has_clock(self) -> bool:
        """True when hours, minutes, or seconds may be nonzero."""
        return not all(_zero(value) for value in (self.hours, self.minutes, self.seconds))


_INCOMPATIBLE = (
    "cannot combine a period with a duration: periods move the calendar, "
    "durations count seconds. Convert one with as_duration()."
)


def _zero(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value == 0


def _negate(value: Any) -> Any:
    return None if value is None else -value


def _scale(value: Any, factor: int) -> Any:
    if value is None:
        return None
    return value * factor


def _combine(left: Any, right: Any) -> Any:
    if left is None or right is None:
        return None
    if _zero(left):
        return right
    if _zero(right):
        return left
    return left + right


def _count(name: str, value: Any, *, fraction: bool = False) -> None:
    if value is None or _is_column(value):
        return
    if isinstance(value, bool) or not isinstance(value, (int, float) if fraction else int):
        kind = "a number" if fraction else "an integer"
        raise TypeError(f"{name}() needs {kind} or a column; periods must have integer values")
    if fraction and value != value:
        raise ValueError(f"{name}() count must not be NaN; use None for a missing count")


def years(n: Any = 1) -> Period:
    """A period of *n* calendar years. 29 February plus ``years(1)`` is 28 February."""
    _count("years", n)
    return Period(years=n)


def months(n: Any = 1) -> Period:
    """A period of *n* calendar months, for ``date + months(n)``.

    ``n`` may be an integer or a column. The day rolls backward when the
    destination month is shorter: 31 January plus ``months(1)`` is the last
    day of February.
    """
    _count("months", n)
    return Period(months=n)


def weeks(n: Any = 1) -> Period:
    """A period of *n* weeks, stored as ``7 * n`` calendar days."""
    _count("weeks", n)
    return Period(weeks=n)


def days(n: Any = 1) -> Period:
    """A period of *n* calendar days. Across a daylight-saving change the
    clock time stays the same, so a day can be 23 or 25 real hours.
    ``days(1.5)`` raises, as in lubridate; use :func:`ddays` for exact time.
    """
    _count("days", n)
    return Period(days=n)


def hours(n: Any = 1) -> Period:
    """A period of *n* hours on the local clock. A date becomes a datetime.

    On a zoned column, 01:30 plus ``hours(1)`` on a spring-forward day is
    missing, because 02:30 does not exist, as in lubridate. Use
    :func:`dhours` for elapsed hours.
    """
    _count("hours", n)
    return Period(hours=n)


def minutes(n: Any = 1) -> Period:
    """A period of *n* minutes on the local clock."""
    _count("minutes", n)
    return Period(minutes=n)


def seconds(n: Any = 1) -> Period:
    """A period of *n* seconds on the local clock. Fractions are allowed, as in lubridate."""
    _count("seconds", n, fraction=True)
    return Period(seconds=n)


def period(num: Any = None, units: Any = "second", **parts: Any) -> Period:
    """Build a period from counts and unit names, or from keyword units.

    ``period(hours=25)`` is 25 hours, not normalized to a day.
    ``period(2, "week")`` is 14 days. ``period([1, 2], ["hour", "minute"])``
    pairs counts with units, as R's ``period(c(1, 2), c("hour", "minute"))``.
    Keywords are ``years``, ``months``, ``weeks``, ``days``, ``hours``,
    ``minutes``, and ``seconds``.
    """
    allowed = ("years", "months", "weeks", "days", "hours", "minutes", "seconds")
    unknown = sorted(set(parts) - set(allowed))
    if unknown:
        raise TypeError(f"period() got unknown units: {', '.join(unknown)}")
    builders = {"year": years, "month": months, "week": weeks, "day": days,
                "hour": hours, "minute": minutes, "second": seconds}
    result = Period()
    for name, value in parts.items():
        result = result + builders[name[:-1]](value)
    if num is None:
        return result
    counts = list(num) if isinstance(num, (list, tuple)) else [num]
    names = list(units) if isinstance(units, (list, tuple)) else [units] * len(counts)
    if len(names) != len(counts):
        raise ValueError("period() needs one unit per count")
    for count_, unit in zip(counts, names):
        result = result + builders[_span_unit("period", unit)](count_)
    return result


class Duration(timedelta):
    """An exact number of seconds, held to the microsecond.

    A day is 86,400 seconds, a year 365.25 days, and a month a twelfth of
    a year, as in lubridate. Arithmetic with other durations stays a
    duration; adding a :class:`Period` raises, as R does.
    """

    __slots__ = ()

    def __new__(cls, seconds: float = 0.0) -> "Duration":
        if isinstance(seconds, timedelta):
            return super().__new__(cls, seconds.days, seconds.seconds, seconds.microseconds)
        return super().__new__(cls, microseconds=round(seconds * 1_000_000))

    def __repr__(self) -> str:
        total = self.total_seconds()
        text = f"{_number(total)}s"
        size = abs(total)
        for unit, length in (("years", 31557600), ("weeks", 604800), ("days", 86400),
                             ("hours", 3600), ("minutes", 60)):
            if size >= length:
                return f"{text} (~{total / length:.3g} {unit})"
        return text

    __str__ = __repr__

    def __add__(self, other: Any) -> Any:
        if isinstance(other, Period):
            raise TypeError(_INCOMPATIBLE)
        return _as_duration_value(super().__add__(other))

    def __radd__(self, other: Any) -> Any:
        if isinstance(other, Period):
            raise TypeError(_INCOMPATIBLE)
        return _as_duration_value(super().__radd__(other))

    def __sub__(self, other: Any) -> Any:
        if isinstance(other, Period):
            raise TypeError(_INCOMPATIBLE)
        return _as_duration_value(super().__sub__(other))

    def __rsub__(self, other: Any) -> Any:
        if isinstance(other, Period):
            raise TypeError(_INCOMPATIBLE)
        return _as_duration_value(super().__rsub__(other))

    def __neg__(self) -> "Duration":
        return Duration(super().__neg__())

    def __abs__(self) -> "Duration":
        return Duration(super().__abs__())

    def __mul__(self, other: Any) -> Any:
        return _as_duration_value(super().__mul__(other))

    __rmul__ = __mul__

    def __truediv__(self, other: Any) -> Any:
        if isinstance(other, Period):
            return self.total_seconds() / other.total_seconds()
        return _as_duration_value(super().__truediv__(other))

    def __eq__(self, other: Any) -> Any:
        if isinstance(other, Period):
            return NotImplemented
        return super().__eq__(other)

    def __ne__(self, other: Any) -> Any:
        if isinstance(other, Period):
            return NotImplemented
        return super().__ne__(other)

    __hash__ = timedelta.__hash__


def _as_duration_value(value: Any) -> Any:
    if type(value) is timedelta:
        return Duration(value)
    return value


def _number(value: float) -> str:
    return str(int(value)) if value == int(value) else repr(value)


def _duration_of(function: str, n: Any, unit: str) -> Any:
    if n is None or (isinstance(n, float) and n != n):
        return None
    if isinstance(n, str):
        raise TypeError(f"{function}() needs a number or a column; pass col({n!r}) for a column")
    if _is_column(n):
        return _func("duration", n, unit=unit)
    if isinstance(n, bool) or not isinstance(n, numbers.Real):
        raise TypeError(f"{function}() needs a number or a column")
    return Duration(float(n) * _SECONDS[unit])


def dseconds(x: Any = 1) -> Any:
    """An exact duration of *x* seconds. *x* may be a number or a column."""
    return _duration_of("dseconds", x, "second")


def dminutes(x: Any = 1) -> Any:
    """An exact duration of *x* minutes (60 seconds each)."""
    return _duration_of("dminutes", x, "minute")


def dhours(x: Any = 1) -> Any:
    """An exact duration of *x* hours (3,600 seconds each)."""
    return _duration_of("dhours", x, "hour")


def ddays(x: Any = 1) -> Any:
    """An exact duration of *x* days of 86,400 seconds.

    Across a daylight-saving change this is 24 real hours, so 12:00 plus
    ``ddays(1)`` can be 13:00. A date plus whole days stays a date; a
    fraction, such as ``ddays(1.5)``, gives a datetime.
    """
    return _duration_of("ddays", x, "day")


def dweeks(x: Any = 1) -> Any:
    """An exact duration of *x* weeks of seven 86,400-second days."""
    return _duration_of("dweeks", x, "week")


def dmonths(x: Any = 1) -> Any:
    """An exact duration of *x* average months: 30.4375 days (2,629,800 seconds)."""
    return _duration_of("dmonths", x, "month")


def dyears(x: Any = 1) -> Any:
    """An exact duration of *x* average years: 365.25 days (31,557,600 seconds)."""
    return _duration_of("dyears", x, "year")


def duration(num: Any = None, units: str = "second", **parts: Any) -> Any:
    """An exact duration from a count and a unit, or from keyword units.

    ``duration(90, "minutes")`` is 5,400 seconds; ``duration(hours=1,
    minutes=30)`` is the same. Units are ``second``, ``minute``, ``hour``,
    ``day``, ``week``, ``month`` (30.4375 days), and ``year`` (365.25
    days), singular or plural. Counts may be columns.
    """
    allowed = {f"{name}s": name for name in _SECONDS}
    unknown = sorted(set(parts) - set(allowed))
    if unknown:
        raise TypeError(f"duration() got unknown units: {', '.join(unknown)}")
    pieces = [_duration_of("duration", value, allowed[name]) for name, value in parts.items()]
    if num is not None or not pieces:
        pieces.insert(0, _duration_of("duration", 0 if num is None else num, _span_unit("duration", units)))
    total = pieces[0]
    for piece in pieces[1:]:
        total = None if total is None or piece is None else total + piece
    return total


def as_duration(x: Any) -> Any:
    """Convert to an exact duration.

    A number counts seconds. A :class:`Period` uses average lengths, so
    ``as_duration(months(1))`` is 2,629,800 seconds. A column of numbers
    becomes a duration column; a duration column (such as ``date - date``)
    is returned as is. A quoted string is a column name.
    """
    if x is None:
        return None
    if isinstance(x, timedelta):
        return Duration(x)
    if isinstance(x, Period):
        return Duration(x.total_seconds())
    if isinstance(x, str) or _is_column(x):
        return _func("as_duration", _value(x))
    if isinstance(x, bool) or not isinstance(x, numbers.Real):
        raise TypeError("as_duration() needs a number, a period, a duration, or a column")
    return None if x != x else Duration(float(x))


def time_length(x: Any, unit: str = "second") -> Any:
    """Length of a duration in *unit*, as a fractional number.

    ``time_length(ddays(1.5), "hour")`` is 36. Months and years are the
    average lengths (30.4375 and 365.25 days). A column of durations gives
    a column of numbers; numbers are read as seconds. A period uses the
    average lengths too.
    """
    token = _span_unit("time_length", unit)
    if x is None:
        return None
    if isinstance(x, (timedelta, Period)):
        return x.total_seconds() / _SECONDS[token]
    if isinstance(x, str) or _is_column(x):
        return _func("time_length", _value(x), unit=token)
    if isinstance(x, bool) or not isinstance(x, numbers.Real):
        raise TypeError("time_length() needs a duration, a period, or a column")
    return float(x) / _SECONDS[token]


def _value(x: Any) -> Any:
    return col(x) if isinstance(x, str) else x


def add_with_rollback(x: Any, period: Period, *, roll_to_first: bool = False) -> Expr:
    """Add a period, rolling a day that does not exist in the month backward.

    31 January plus ``months(1)`` is the last day of February; with
    ``roll_to_first=True`` it is 1 March instead, as in lubridate. ``+``
    rolls back the same way. On a zoned column, a clock time skipped by
    daylight saving lands on the transition here (02:30 becomes 03:00),
    while ``+`` gives missing. Subtracting a date from a period is an error.
    """
    if not isinstance(period, Period):
        raise TypeError("add_with_rollback() period must be a period such as months(1)")
    if not isinstance(roll_to_first, bool):
        raise TypeError("add_with_rollback() roll_to_first must be a boolean")
    return _func("add_with_rollback", _value(x), period=period, roll_to_first=roll_to_first)


def year(x: Any) -> Expr:
    """Year of a date or datetime, as an integer. A bare ``year`` in a
    comparison is still the column: ``filter(year == 2007)``."""
    return _func("year", _value(x))


def month(x: Any) -> Expr:
    """Month number of a date or datetime, from 1 to 12."""
    return _func("month", _value(x))


def day(x: Any) -> Expr:
    """Day of the month, from 1."""
    return _func("day", _value(x))


_DAY_NAMES = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
_QUARTER_TYPES = ("quarter", "year.quarter", "date_first", "date_last", "year_start/end")


def wday(x: Any, *, label: bool = False, abbr: bool = True, week_start: int = 7) -> Expr:
    """Weekday, with the first day numbered 1 (Sunday by default).

    ``label=True`` returns an ordered categorical of English day names;
    ``abbr=False`` uses full names. ``week_start=1`` starts on Monday.
    """
    if not isinstance(label, bool) or not isinstance(abbr, bool):
        raise TypeError("wday() label and abbr must be booleans")
    return _func("wday", _value(x), label=label, abbr=abbr, week_start=_week_start(week_start))


def yday(x: Any) -> Expr:
    """Day of the year, from 1 to 365 or 366."""
    return _func("yday", _value(x))


def week(x: Any) -> Expr:
    """Seven-day block since January 1, numbered from 1; not an ISO week."""
    return _func("week", _value(x))


def isoweek(x: Any) -> Expr:
    """ISO 8601 week number (1–53), which can belong to a neighboring year."""
    return _func("isoweek", _value(x))


def quarter(x: Any, type: str = "quarter", *, fiscal_start: int = 1) -> Expr:
    """Fiscal quarter, numbered from the month *fiscal_start*.

    ``type`` also accepts ``year.quarter`` (the fiscal ending year plus
    quarter/10), ``date_first``, ``date_last``, and ``year_start/end``.
    """
    if type not in _QUARTER_TYPES:
        raise ValueError(f"quarter() type must be one of: {', '.join(_QUARTER_TYPES)}")
    if isinstance(fiscal_start, bool) or not isinstance(fiscal_start, int) or not 1 <= fiscal_start <= 12:
        raise ValueError("quarter() fiscal_start must be an integer from 1 to 12")
    return _func("quarter", _value(x), type=type, fiscal_start=fiscal_start)


def hour(x: Any) -> Expr:
    """Hour of local clock time, 0–23; dates have hour 0."""
    return _func("hour", _value(x))


def minute(x: Any) -> Expr:
    """Minute of local clock time, 0–59; dates have minute 0."""
    return _func("minute", _value(x))


def second(x: Any) -> Expr:
    """Seconds of local clock time, including the fractional part."""
    return _func("second", _value(x))


def with_tz(x: Any, tzone: str = "UTC") -> Expr:
    """Display the same instant in *tzone*. Naive clocks and dates use UTC.

    Dates become datetimes at midnight UTC before changing zones. Unknown
    zone names raise instead of silently falling back to UTC.
    """
    from tidy3._timezones import validate_zone

    return _func("with_tz", _value(x), tzone=validate_zone(tzone))


def force_tz(
    x: Any, tzone: str = "UTC", *, roll_dst: str | tuple[str, str] = ("NA", "post")
) -> Expr:
    """Keep local clock fields, interpreting them in a new timezone.

    *roll_dst* controls skipped and repeated times, respectively. A string
    applies to both; a pair supplies separate policies. ``NA`` returns
    missing, ``pre``/``post`` choose before/after the transition, and
    ``boundary`` chooses the transition itself. Defaults match lubridate:
    skipped clocks become missing, and repeated clocks choose the later
    occurrence. An already resolved repeated clock keeps its occurrence.
    Dates become local midnight. Unlike :func:`with_tz`, this changes the
    instant being represented. Both helpers default to UTC.
    """
    from tidy3._timezones import validate_roll, validate_zone

    return _func(
        "force_tz", _value(x), tzone=validate_zone(tzone), roll_dst=validate_roll(roll_dst)
    )


def _weekday_labels(abbr: bool, week_start: int) -> list[str]:
    names = [name[:3] if abbr else name for name in _DAY_NAMES]
    return names[week_start - 1:] + names[:week_start - 1]


def _unit(name: str, unit: str) -> str:
    if not isinstance(unit, str):
        raise TypeError(f"{name}() unit must be a string")
    token = unit.strip().lower()
    if token.startswith("1 "):
        token = token[2:]
    if token.endswith("s") and token[:-1] in _UNITS:
        token = token[:-1]
    if token not in _UNITS:
        choices = ", ".join(_UNITS)
        raise ValueError(f"{name}() unit must be one of: {choices}")
    return token


def _week_start(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 7:
        raise ValueError("week_start must be an integer from 1 (Monday) to 7 (Sunday)")
    return value


def floor_date(x: Any, unit: str = "day", *, week_start: int = 7) -> Expr:
    """Round a date down to a calendar boundary.

    *unit* is ``second``, ``minute``, ``hour``, ``day``, ``week``,
    ``month``, ``quarter``, or ``year``. Weeks start on Sunday
    (``week_start=7``), matching lubridate, not on Monday. ``1`` is Monday
    and ``7`` is Sunday.
    """
    token = _unit("floor_date", unit)
    start = _week_start(week_start)
    return _func("floor_date", _value(x), unit=token, week_start=start)


def ceiling_date(x: Any, unit: str = "day", *, week_start: int = 7) -> Expr:
    """Round a date up to a calendar boundary.

    As in lubridate, a datetime already on the boundary stays there, but a
    date moves to the next one: ``ceiling_date`` of 2010-02-01 to a month
    is 2010-03-01, and of any date to a day is the next day.
    """
    token = _unit("ceiling_date", unit)
    start = _week_start(week_start)
    return _func("ceiling_date", _value(x), unit=token, week_start=start)


def days_in_month(x: Any) -> Expr:
    """How many days the month of *x* has, so February is 28 or 29."""
    return _func("days_in_month", _value(x))


# ── round_date, make_date, make_datetime, isoyear, leap_year ───────────


def round_date(x: Any, unit: str = "day", *, week_start: int = 7) -> Expr:
    """Round a date to the nearer of :func:`floor_date` and the next boundary.

    Units and ``week_start`` are those of :func:`floor_date`. Distance is
    elapsed time, and a tie rounds up, as in lubridate: 12:00 rounds to the
    next day, and 2010-02-15 rounds to March 1 (14 days either way). A date
    already on a boundary stays there. Across daylight saving the distance
    is real time, so on a 23-hour day 12:00 rounds down to midnight. The
    default unit is ``day``, as for :func:`floor_date`; lubridate's default
    is ``second``.
    """
    token = _unit("round_date", unit)
    start = _week_start(week_start)
    return _func("round_date", _value(x), unit=token, week_start=start)


_SEQ_STEPS = {
    "second": "s", "minute": "m", "hour": "h", "day": "d", "week": "w",
    "month": "mo", "quarter": "q", "year": "y",
}


def _split_interval(interval: str) -> tuple[int, str]:
    digits = len(interval) - len(interval.lstrip("0123456789"))
    return int(interval[:digits]), interval[digits:]


def _seq_interval(by: Any) -> str:
    """A Polars interval from ``"month"``, ``"2 weeks"``, ``"-1 year"``, or a period."""
    if isinstance(by, Period):
        used = [(n, v) for n, v in zip(_PERIOD_SLOTS, by._slots()) if not _zero(v)]
        if len(used) != 1 or not isinstance(used[0][1], int) or isinstance(used[0][1], bool):
            raise ValueError("date_seq() by must be one unit, such as months(1) or days(7)")
        unit, count = used[0]
        if count < 1:
            raise ValueError("date_seq() by must count up; use a positive period")
        return f"{count}{_SEQ_STEPS[unit.rstrip('s')]}"
    if not isinstance(by, str):
        raise TypeError('date_seq() by must be a string such as "month" or "2 weeks", or a period')
    text = by.strip().lower()
    parts = text.split()
    count = 1
    if len(parts) == 2:
        try:
            count = int(parts[0])
        except ValueError:
            raise ValueError(f"date_seq() cannot read by={by!r}") from None
        text = parts[1]
    elif len(parts) != 1:
        raise ValueError(f"date_seq() cannot read by={by!r}")
    if text.endswith("s") and text[:-1] in _SEQ_STEPS:
        text = text[:-1]
    if text not in _SEQ_STEPS or count < 1:
        raise ValueError(f"date_seq() by unit must be one of: {', '.join(_SEQ_STEPS)}, with a count of 1 or more")
    return f"{count}{_SEQ_STEPS[text]}"


def date_seq(
    from_: Any, to: Any = None, by: Any = None, total_size: int | None = None
) -> Expr:
    """Dates from *from_* by a step, as clock's ``date_seq()`` and R's ``seq.Date``.

    Give exactly two of *to*, *by*, and *total_size*:

    - ``date_seq(a, b, "month")``: every month from *a* up to *b*, both
      included when the steps land on them.
    - ``date_seq(a, by="month", total_size=12)``: 12 dates starting at *a*.

    ``by`` is ``"day"``, ``"week"``, ``"month"``, ``"quarter"``, ``"year"``,
    optionally with a count (``"2 months"``), a whole number of days, or a
    period such as ``months(1)``. Steps count up only: a negative step
    raises, where R counts down, and a *to* before *from_* gives an empty
    list. Giving *to* and *total_size* without *by* is not supported yet.

    Each row gets its own list of dates, so in ``mutate`` the result is a
    list column; ``explode`` it for one row per date. A month step from the
    29th to the 31st stays on the last real day of a short month (Polars'
    rule), where R spills into the next month. A missing *from_* or *to*
    gives a missing list.

        mutate(window = date_seq(ym(sale_month) - months(11), ym(sale_month), by="month"))
    """
    given = sum(x is not None for x in (to, by, total_size))
    if given != 2:
        raise ValueError("date_seq() needs exactly two of to, by, and total_size")
    if by is None:
        raise NotImplementedError("date_seq() with to and total_size but no by is not supported yet")
    if isinstance(by, int) and not isinstance(by, bool):
        by = f"{by} days"
    interval = _seq_interval(by)
    if total_size is None:
        return _func("seq_date", _value(from_), _value(to), by=interval)
    if isinstance(total_size, bool) or not isinstance(total_size, int) or total_size < 0:
        raise ValueError("date_seq() total_size must be a whole number of 0 or more")
    return _func("seq_date", _value(from_), by=interval, total=total_size)


def seq(start: Any, end: Any, by: Any = "day") -> Expr:
    """:func:`date_seq` with R's ``seq()`` argument order: *start*, *end*, *by*."""
    return date_seq(start, end, by)


def _part(function: str, name: str, value: Any, *, whole: bool) -> Any:
    if isinstance(value, (Expr, pl.Expr)) or value is None:
        return value
    if isinstance(value, str):
        return col(value)
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{function}() {name} must be a number, a column name, or a column")
    if whole and value == value and abs(value) != float("inf") and value != int(value):
        raise ValueError(f"{function}() {name} must be a whole number, not {value!r}")
    return value


def make_date(year: Any = 1970, month: Any = 1, day: Any = 1) -> Expr:
    """Build a date from year, month, and day columns or numbers.

    Each part is a column name, a column, or a number, recycled as in
    lubridate. A date that does not exist (30 February, month 13) or a
    missing part gives a missing date. Fractions are dropped toward zero,
    so 2023.7 is year 2023.
    """
    parts = zip(("year", "month", "day"), (year, month, day))
    return _func("make_date", *(_part("make_date", name, value, whole=False) for name, value in parts))


def make_datetime(
    year: Any = 1970,
    month: Any = 1,
    day: Any = 1,
    hour: Any = 0,
    min: Any = 0,
    sec: Any = 0,
    *,
    tz: str = "UTC",
) -> Expr:
    """Build a datetime from its parts, in *tz* (UTC by default).

    As in lubridate, parts outside their range roll over: 31 February is
    3 March, hour 25 is 01:00 the next day, and second -1 is the previous
    minute. Only ``sec`` may have a fraction; it keeps microseconds. A
    missing part gives a missing datetime. A clock skipped by daylight
    saving lands on the transition (02:30 on a spring-forward day is
    03:00), and a repeated clock takes the later occurrence. lubridate
    raises on a fractional year, month, day, hour, or minute; tidy3 raises
    for a number and gives a missing value for a column.
    """
    from tidy3._timezones import validate_zone

    names = ("year", "month", "day", "hour", "min", "sec")
    values = (year, month, day, hour, min, sec)
    parts = (
        _part("make_datetime", name, value, whole=name != "sec")
        for name, value in zip(names, values)
    )
    return _func("make_datetime", *parts, tz=validate_zone(tz))


def isoyear(x: Any) -> Expr:
    """ISO 8601 year, the year that :func:`isoweek` counts in.

    2021-01-01 is in ISO week 53 of 2020, so its ``isoyear`` is 2020.
    """
    return _func("isoyear", _value(x))


def leap_year(x: Any) -> Expr:
    """Whether the year of *x* is a leap year.

    *x* holds dates, datetimes, or numeric years. As in lubridate, a
    fractional year is cut toward zero, so 2000.5 is a leap year.
    """
    return _func("leap_year", _value(x))


def _zone(tzone: str) -> Any:
    if not isinstance(tzone, str) or not tzone:
        raise ValueError('tzone must be "UTC", "local", or a time zone name')
    if tzone.upper() == "UTC":
        return timezone.utc
    if tzone.lower() == "local":
        return datetime.now().astimezone().tzinfo or timezone.utc
    try:
        return ZoneInfo(tzone)
    except Exception as error:
        raise ValueError(
            f"unknown time zone {tzone!r}. Use 'UTC', 'local', or a name "
            "such as 'America/Los_Angeles'."
        ) from error


def today(tzone: str = "UTC") -> date:
    """Today's date in *tzone*. The default is UTC, not the local zone.

    This is the date when the function runs (when the pipe is built), not
    when a lazy frame is collected.
    """
    return datetime.now(_zone(tzone)).date()


def now(tzone: str = "UTC") -> datetime:
    """The current clock time in *tzone*, as a naive datetime.

    The default zone is UTC. ``tzone="local"`` uses the machine zone. The
    zone is documented by the argument and then dropped from the value, so
    the result lines up with :func:`as_datetime`.
    """
    current = datetime.now(_zone(tzone))
    return current.replace(tzinfo=None)


# ── polars ──────────────────────────────────────────────────────────────


_STEP_UNITS = {"months": "mo", "days": "d", "clock": "us"}
_CLOCK_US = {"hours": 3_600_000_000, "minutes": 60_000_000, "seconds": 1_000_000}


def _polars_count(value: Any) -> Any:
    """An int count as is; None, or a column, as an Int64 Polars expression."""
    if value is None:
        return pl.lit(None, dtype=pl.Int64)
    if isinstance(value, Expr):
        from tidy3.expr import to_polars

        return to_polars(value)
    return value


def _amount(value: Any, unit: str) -> Any:
    if isinstance(value, bool):
        raise TypeError("a period count must be an integer or a column")
    if isinstance(value, int):
        if value == 0:
            return None
        return f"{value}{unit}"
    compiled = _polars_count(value)
    return pl.format("{}{}", compiled.cast(pl.Int64), pl.lit(unit))


def total_months(period: Period) -> Any:
    """All of *period* in months: ``12 * years + months``.

    lubridate's ``add_with_rollback`` adds a period this way, so the day
    rolls back once, at the end. Adding years and then months would roll
    29 February back to the 28th before the month moves.
    """
    years_, months_ = period.years, period.months
    if years_ is None or months_ is None:
        return None
    if isinstance(years_, int) and isinstance(months_, int):
        return 12 * years_ + months_
    if isinstance(years_, int) and years_ == 0:
        return months_
    if isinstance(months_, int) and months_ == 0:
        return years_ * 12
    return years_ * 12 + months_


def _clock_micros(period: Period) -> Any:
    """Hours, minutes, and seconds as one count of microseconds."""
    total: Any = 0
    for name, size in _CLOCK_US.items():
        value = getattr(period, name)
        if _zero(value):
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            total = total + round(value * size)
            continue
        compiled = _polars_count(value)
        if name == "seconds":
            micros = (compiled.cast(pl.Float64) * size).round().cast(pl.Int64)
        else:
            micros = compiled.cast(pl.Int64) * size
        total = micros if isinstance(total, int) and total == 0 else total + micros
    return total


def _steps(period: Period) -> list[tuple[str, Any]]:
    """The moves of *period* in the order they apply: months, days, clock."""
    days_ = period.days
    return [
        ("months", total_months(period)),
        ("days", days_ if isinstance(days_, int) else _polars_count(days_)),
        ("clock", _clock_micros(period)),
    ]


def _offset(base: pl.Expr, steps: list[tuple[str, Any]], roll_to_first: bool) -> pl.Expr:
    """Apply each step with ``dt.offset_by``: one call when the counts allow."""
    nonzero = [(unit, count) for unit, count in steps if not (isinstance(count, int) and count == 0)]
    if not nonzero:
        return base
    plain = all(isinstance(count, int) for _, count in nonzero)
    if plain and len({count > 0 for _, count in nonzero}) == 1 and not (
        roll_to_first and nonzero[0][0] == "months"
    ):
        # Polars allows one sign, at the front: "-1mo2d" moves back a
        # month, then two days, the same order as separate steps.
        sign = "" if nonzero[0][1] > 0 else "-"
        return base.dt.offset_by(sign + "".join(f"{abs(count)}{_STEP_UNITS[unit]}" for unit, count in nonzero))
    for unit, count in nonzero:
        moved = base.dt.offset_by(_amount(count, _STEP_UNITS[unit]))
        if unit == "months" and roll_to_first:
            # A rolled-back day is the last of its month, so one more day
            # is the first of the next, as lubridate's roll_to_first gives.
            moved = pl.when(moved.dt.day() != base.dt.day()).then(moved.dt.offset_by("1d")).otherwise(moved)
        base = moved
    return base


def _resolve_clocks(source: pl.Series, moved: pl.Series, tz: str, rollback: bool) -> pl.Series:
    """Place local clocks *moved* back in *tz*.

    Most rows have one reading and stay in Polars. A repeated clock keeps
    *source*'s UTC offset when that offset is valid, otherwise it takes the
    earlier occurrence, so adding zero is a no-op. A skipped clock is
    missing, or with *rollback* lands on the transition, as ``%m+%`` does.
    """
    placed = moved.dt.replace_time_zone(tz, ambiguous="null", non_existent="null")
    rows = (placed.is_null() & moved.is_not_null()).arg_true()
    if not len(rows):
        return placed
    clocks = moved.gather(rows)
    origin = source.gather(rows)
    offset = origin.dt.replace_time_zone(None) - origin.dt.convert_time_zone("UTC").dt.replace_time_zone(None)
    kept = (clocks - offset).dt.replace_time_zone("UTC").dt.convert_time_zone(tz)
    keep = (kept.dt.replace_time_zone(None) == clocks).fill_null(False)
    fallback = clocks.dt.replace_time_zone(tz, ambiguous="earliest", non_existent="null")
    if rollback:
        from tidy3._timezones import transform

        boundary = transform(clocks, "force_tz", tzone=tz, roll_dst=("boundary", "pre"))
        fallback = fallback.zip_with(fallback.is_not_null(), boundary)
    return placed.scatter(rows, kept.zip_with(keep, fallback))


def _shift_known(
    value: pl.Expr, dtype: Any, period: Period, *, rollback: bool, roll_to_first: bool
) -> pl.Expr:
    steps = _steps(period)
    if dtype in {pl.Date, pl.Null}:
        # A clock unit makes a date a datetime, as R gives a UTC POSIXct.
        base = value.cast(pl.Datetime("us")) if period.has_clock() else value.cast(pl.Date)
        return _offset(base, steps, roll_to_first)
    if isinstance(dtype, pl.Datetime) and dtype.time_zone not in {None, "UTC"}:
        # Move the local clock, then put it back in the zone.
        zone, unit = dtype.time_zone, dtype.time_unit
        moved = _offset(value.dt.replace_time_zone(None), steps, roll_to_first)
        placed = pl.struct(x=value, m=moved).map_batches(
            lambda batch: _resolve_clocks(batch.struct.field("x"), batch.struct.field("m"), zone, rollback),
            return_dtype=pl.Datetime(unit, zone),
            is_elementwise=True,
        )
        return _named_like(placed, value)
    return _offset(value, steps, roll_to_first)


def _named_like(result: pl.Expr, value: pl.Expr) -> pl.Expr:
    """*result* under *value*'s output name, as a plain operator would keep it."""
    try:
        return result.alias(value.meta.output_name())
    except Exception:
        return result


def shift_polars(
    expr: pl.Expr, period: Period, *, rollback: bool = True, roll_to_first: bool = False
) -> pl.Expr:
    """Move *expr* by *period*: months, then days, then the local clock.

    *rollback* chooses what a zoned clock skipped by daylight saving
    becomes: the transition (:func:`add_with_rollback`) or missing (``+``).
    """
    from tidy3.expr import input_dtype

    known = input_dtype(expr)
    if known is not None:
        return _shift_known(expr, known, period, rollback=rollback, roll_to_first=roll_to_first)
    # The rule depends on the dtype, known per batch. Column counts travel
    # with the values as struct fields.
    fields: dict[str, Any] = {"v": expr}
    slots: dict[str, Any] = {}
    for name, count in zip(_PERIOD_SLOTS, period._slots()):
        if _is_column(count):
            key = f"c{len(fields)}"
            fields[key] = _polars_count(count)
            count = pl.col(key)
        slots[name] = count
    local = Period(**slots)

    def batch(series: pl.Series) -> pl.Series:
        frame = series.struct.unnest()
        shifted = _shift_known(
            pl.col("v"), frame.schema["v"], local, rollback=rollback, roll_to_first=roll_to_first
        )
        return frame.select(shifted).to_series()

    return _named_like(pl.struct(**fields).map_batches(batch, is_elementwise=True), expr)


def _exact_divide(value: pl.Expr, divisor: float, *, finite: bool = False) -> pl.Expr:
    """``value / divisor`` divided row by row.

    Polars divides by a scalar as multiplication by its reciprocal, which
    can miss R's result by one unit in the last place. A divisor of the
    value's own shape is divided exactly, in a mutate or a summary. With
    ``finite`` (a duration's microseconds), ``value * 0.0`` is that shape
    at the least cost; otherwise ``is_null()``, false (0.0) on every
    present row, keeps an infinite value from turning the divisor to NaN.
    """
    if finite:
        return value / (value * 0.0 + divisor)
    return value / (value.is_null().cast(pl.Float64) + divisor)


def _duration_polars(value: pl.Expr, unit: str) -> pl.Expr:
    from tidy3.expr import input_dtype

    size = int(_SECONDS[unit] * 1_000_000)
    known = input_dtype(value)
    if known is not None and known.is_integer():
        micros = value.cast(pl.Int64) * size
    else:
        micros = (value.cast(pl.Float64) * size).round().cast(pl.Int64)
    return micros.cast(pl.Duration("us"))


def _as_duration_of(value: pl.Expr, dtype: Any) -> pl.Expr:
    if isinstance(dtype, pl.Duration):
        return value.cast(pl.Duration("us"))
    if dtype == pl.Null or dtype.is_numeric():
        return _duration_polars(value, "second")
    raise TypeError("as_duration() needs numbers (seconds) or durations")


def _time_length_of(value: pl.Expr, dtype: Any, unit: str) -> pl.Expr:
    if isinstance(dtype, pl.Duration):
        micros = value.dt.total_microseconds().cast(pl.Float64)
        return _exact_divide(micros, _SECONDS[unit] * 1_000_000, finite=True)
    if dtype == pl.Null or dtype.is_numeric():
        return _exact_divide(value.cast(pl.Float64), _SECONDS[unit])
    raise TypeError("time_length() needs durations or numbers (seconds); intervals are not supported yet")


def _by_dtype(value: pl.Expr, rule: Any, return_dtype: Any) -> pl.Expr:
    from tidy3.expr import input_dtype

    known = input_dtype(value)
    if known is not None:
        return rule(value, known)

    def batch(series: pl.Series) -> pl.Series:
        return series.to_frame("v").select(rule(pl.col("v"), series.dtype)).to_series()

    return value.map_batches(batch, return_dtype=return_dtype, is_elementwise=True)


def _duration_literal(node: Any) -> timedelta | None:
    if isinstance(node, tuple) and node and node[0] == "lit" and isinstance(node[1], timedelta):
        return node[1]
    return None


def _datetime_if_date(value: pl.Expr) -> pl.Expr:
    from tidy3.expr import input_dtype

    return value.cast(pl.Datetime("us")) if input_dtype(value) == pl.Date else value


def _duration_bin(op: str, value: pl.Expr, span: timedelta, value_first: bool) -> pl.Expr:
    """``date + ddays(n)``: whole days keep a date; anything else is a datetime.

    Polars would keep the date and drop the fraction of a day.
    """
    from tidy3.expr import _PL_BIN, input_dtype

    literal = pl.lit(span, dtype=pl.Duration("us"))
    whole = span % timedelta(days=1) == timedelta(0)

    def apply(side: pl.Expr) -> pl.Expr:
        return _PL_BIN[op](side, literal) if value_first else _PL_BIN[op](literal, side)

    known = input_dtype(value)
    if known is not None or whole:
        return apply(value.cast(pl.Datetime("us")) if known == pl.Date and not whole else value)

    def batch(series: pl.Series) -> pl.Series:
        if series.dtype == pl.Date:
            series = series.cast(pl.Datetime("us"))
        return series.to_frame("v").select(apply(pl.col("v"))).to_series()

    return value.map_batches(batch, is_elementwise=True)


_DURATION_FUNCTIONS = frozenset({"duration", "as_duration"})
_COMPARISONS = frozenset({"==", "!=", "<", "<=", ">", ">="})


def _period_of(node: Any) -> Period | None:
    if isinstance(node, tuple) and node and node[0] == "lit" and isinstance(node[1], Period):
        return node[1]
    return None


def _is_duration_node(node: Any) -> bool:
    return isinstance(node, tuple) and len(node) > 1 and node[0] == "func" and node[1] in _DURATION_FUNCTIONS


def compile_period_bin(op: str, left: Any, right: Any, compile: Any) -> pl.Expr | None:
    """Compile span arithmetic that plain Polars operators get wrong.

    ``date + months(1)`` and other periods move the calendar and clock
    before a period would reach ``pl.lit``. A date plus a duration that is
    not whole days becomes a datetime. A period compared with a duration
    uses average lengths, as lubridate does. Anything else returns None.
    """
    from tidy3.expr import _PL_BIN

    left_period = _period_of(left)
    right_period = _period_of(right)
    if op in _COMPARISONS:
        if left_period is None and right_period is None:
            return None

        def side(node: Any, period_: Period | None) -> Any:
            return compile(node) if period_ is None else pl.lit(Duration(period_.total_seconds()), dtype=pl.Duration("us"))

        return _PL_BIN[op](side(left, left_period), side(right, right_period))
    if op not in {"+", "-"}:
        return None
    if left_period is not None or right_period is not None:
        if left_period is not None and right_period is not None:
            raise TypeError("add a period to a date, not to another period")
        if left_period is not None:
            if op == "-":
                raise TypeError("cannot subtract a date from a period")
            return shift_polars(compile(right), left_period, rollback=False)
        period_ = right_period if op == "+" else -right_period
        return shift_polars(compile(left), period_, rollback=False)
    left_span, right_span = _duration_literal(left), _duration_literal(right)
    if (left_span is None) != (right_span is None):
        if left_span is not None:
            return _duration_bin(op, compile(right), left_span, value_first=False)
        return _duration_bin(op, compile(left), right_span, value_first=True)
    if _is_duration_node(left) != _is_duration_node(right):
        # A duration column: a date side becomes a datetime first.
        first, second = compile(left), compile(right)
        if _is_duration_node(right):
            first = _datetime_if_date(first)
        elif op == "+":
            second = _datetime_if_date(second)
        return _PL_BIN[op](first, second)
    if left[0] in {"col", "name"} or right[0] in {"col", "name"}:
        # A date column plus a duration expression, or a duration column
        # plus a date. Column types come from the cached schema; the other
        # side's type is looked up only next to a date or duration column.
        from tidy3.expr import input_dtype

        first, second = compile(left), compile(right)
        types = [input_dtype(side) if node[0] in {"col", "name"} else None
                 for side, node in ((first, left), (second, right))]
        if not any(kind == pl.Date or isinstance(kind, pl.Duration) for kind in types):
            return _PL_BIN[op](first, second)
        types = [kind if kind is not None else input_dtype(side) for kind, side in zip(types, (first, second))]
        if types[0] == pl.Date and isinstance(types[1], pl.Duration):
            first = first.cast(pl.Datetime("us"))
        elif op == "+" and types[1] == pl.Date and isinstance(types[0], pl.Duration):
            second = second.cast(pl.Datetime("us"))
        return _PL_BIN[op](first, second)
    return None


def _floor_polars(expr: pl.Expr, unit: str, week_start: int) -> pl.Expr:
    if unit == "week":
        # Polars weekday is ISO: Monday 1 … Sunday 7. truncate("1w") is
        # Monday, so weeks are shifted onto week_start instead.
        shift = (expr.dt.weekday() - week_start) % 7
        return expr.dt.truncate("1d").dt.offset_by(pl.format("-{}d", shift))
    return expr.dt.truncate(_TRUNC[unit])


_SUB_DAY = frozenset({"second", "minute", "hour"})


def _ceiling_polars(expr: pl.Expr, unit: str, week_start: int) -> pl.Expr:
    from tidy3.expr import input_dtype

    dtype = input_dtype(expr)
    if dtype is not None:
        floored = _floor_polars(expr, unit, week_start)
        nudged = floored.dt.offset_by(_OFFSET[unit])
        if dtype == pl.Date and unit not in _SUB_DAY:
            return nudged
        return pl.when(expr == floored).then(expr).otherwise(nudged)

    def batch(series: pl.Series) -> pl.Series:
        value = pl.col("v")
        floored = _floor_polars(value, unit, week_start)
        nudged = floored.dt.offset_by(_OFFSET[unit])
        if series.dtype == pl.Date and unit not in _SUB_DAY:
            # lubridate rounds a Date up to the next boundary even when it
            # is on one: ceiling_date(ymd("2010-02-01"), "month") is March 1.
            result = nudged
        else:
            result = pl.when(value == floored).then(value).otherwise(nudged)
        return series.to_frame("v").select(result).to_series()

    # The rule depends on the dtype (Date or Datetime), known per batch.
    return expr.map_batches(batch, return_dtype=pl.dtype_of(expr), is_elementwise=True)


def _round_polars(expr: pl.Expr, unit: str, week_start: int) -> pl.Expr:
    # lubridate's tie rule: the later boundary wins when both are as far.
    below = _floor_polars(expr, unit, week_start)
    above = below.dt.offset_by(_OFFSET[unit])
    return pl.when((above - expr) <= (expr - below)).then(above).otherwise(below)


def _whole_polars(part: pl.Expr, *, exact: bool = False) -> pl.Expr:
    """An Int64 part. Fractions go toward zero, or to missing when *exact*."""
    from tidy3.expr import input_dtype

    whole = part.cast(pl.Int64, strict=False)
    if not exact:
        return whole
    known = input_dtype(part)
    if known is not None and (known.is_integer() or known == pl.Null):
        # Already whole. Skipping the check also lets Polars fold literal
        # parts such as a year of 2024 instead of computing them per row.
        return whole
    return pl.when(part.cast(pl.Float64) == whole.cast(pl.Float64)).then(whole)


def _civil_days(year: pl.Expr, month: pl.Expr, day: pl.Expr) -> pl.Expr:
    """Days since 1970-01-01 for a valid month, in integer arithmetic.

    Howard Hinnant's days_from_civil: no calendar range limit, so years
    past 9999 work as they do in R.
    """
    march = year - (month <= 2).cast(pl.Int64)
    era = march // 400
    of_era = march - era * 400
    of_year = (153 * (month + pl.when(month > 2).then(-3).otherwise(9)) + 2) // 5 + day - 1
    return era * 146097 + of_era * 365 + of_era // 4 - of_era // 100 + of_year - 719468


def _month_days(year: pl.Expr, month: pl.Expr) -> pl.Expr:
    leap = (year % 4 == 0) & ((year % 100 != 0) | (year % 400 == 0))
    return (
        pl.when(month == 2)
        .then(28 + leap.cast(pl.Int64))
        .otherwise(30 + (month + month // 8) % 2)
    )


def _make_date_polars(year: pl.Expr, month: pl.Expr, day: pl.Expr) -> pl.Expr:
    year, month, day = (_whole_polars(part) for part in (year, month, day))
    valid = month.is_between(1, 12) & (day >= 1) & (day <= _month_days(year, month))
    days = _civil_days(year, month, day).cast(pl.Int32, strict=False)
    return pl.when(valid).then(days).cast(pl.Date)


_US_PER = {"day": 86_400_000_000, "hour": 3_600_000_000, "min": 60_000_000}
_MAX_DAYS = 106_751_991  # the Int64 microsecond range


def _make_datetime_polars(parts: tuple[pl.Expr, ...], tz: str) -> pl.Expr:
    year, month, day, hour, minute = (_whole_polars(part, exact=True) for part in parts[:5])
    # Parts roll over as in lubridate: months into years, then a running
    # count of days, hours, minutes, and seconds on the local clock.
    months_ = year * 12 + month - 1
    days = _civil_days(months_ // 12, months_ % 12 + 1, pl.lit(1)) + day - 1
    micros = (
        days * _US_PER["day"]
        + hour * _US_PER["hour"]
        + minute * _US_PER["min"]
        + (parts[5].cast(pl.Float64) * 1e6).round().cast(pl.Int64, strict=False)
    )
    clock = pl.when(days.abs() < _MAX_DAYS).then(micros).cast(pl.Datetime("us"))
    if tz == "UTC":
        return clock.dt.replace_time_zone("UTC")
    from tidy3._timezones import transform

    # Skipped clocks land on the transition and repeated clocks take the
    # later occurrence, as lubridate's make_datetime does. Only those rows
    # leave Polars.
    return clock.map_batches(
        lambda series: transform(series, "force_tz", tzone=tz, roll_dst=("boundary", "post")),
        return_dtype=pl.Datetime("us", tz),
        is_elementwise=True,
    )


def _leap_of(value: pl.Expr, dtype: Any) -> pl.Expr:
    if dtype == pl.Date or isinstance(dtype, pl.Datetime):
        return value.dt.is_leap_year()
    if dtype == pl.Null or dtype.is_numeric():
        year = value.cast(pl.Int64, strict=False)
        return (year % 4 == 0) & ((year % 100 != 0) | (year % 400 == 0))
    raise TypeError("leap_year() needs dates, datetimes, or numeric years; parse text first")


def _leap_polars(value: pl.Expr) -> pl.Expr:
    from tidy3.expr import input_dtype

    known = input_dtype(value)
    if known is not None:
        return _leap_of(value, known)

    def batch(series: pl.Series) -> pl.Series:
        return series.to_frame("v").select(_leap_of(pl.col("v"), series.dtype)).to_series()

    return value.map_batches(batch, return_dtype=pl.Boolean, is_elementwise=True)


def _clock_polars(value: pl.Expr, name: str) -> pl.Expr:
    from tidy3.expr import input_dtype

    dtype = pl.Float64 if name == "second" else pl.Int64
    known = input_dtype(value)
    if known is not None:
        if known == pl.Date:
            return value.cast(pl.Int64).cast(dtype) * 0
        result = getattr(value.dt, name)().cast(dtype)
        return result + value.dt.nanosecond() / 1e9 if name == "second" else result

    def batch(series: pl.Series) -> pl.Series:
        if series.dtype == pl.Date:
            return series.cast(pl.Int64).cast(dtype) * 0
        result = getattr(series.dt, name)().cast(dtype)
        if name == "second":
            result = result + series.dt.nanosecond() / 1e9
        return result

    return value.map_batches(batch, return_dtype=dtype, is_elementwise=True)


def _quarter_polars(value: pl.Expr, kind: str, start: int) -> pl.Expr:
    month_ = value.dt.month().cast(pl.Int64)
    number = ((month_ - start) % 12) // 3 + 1
    if kind == "quarter":
        return number
    ending_year = value.dt.year() + ((month_ >= start) & (start != 1)).cast(pl.Int64)
    if kind == "year.quarter":
        return ending_year + number / 10
    if kind == "year_start/end":
        return pl.concat_str([
            (ending_year - 1).cast(pl.String).fill_null("NA"), pl.lit("/"),
            (ending_year % 100).cast(pl.String).str.pad_start(2, "0").fill_null("NA"),
            pl.lit(" Q"), number.cast(pl.String).fill_null("NA"),
        ])
    beginning = value.dt.date().dt.month_start().dt.offset_by(pl.format("-{}mo", (month_ - start) % 3))
    return beginning if kind == "date_first" else beginning.dt.offset_by("3mo").dt.offset_by("-1d")


_EXCEL_ORIGINS = {
    "modern": date(1899, 12, 30),
    "mac pre-2011": date(1904, 1, 1),
}


def _excel_zone(tz: str) -> ZoneInfo:
    if not isinstance(tz, str) or not tz:
        raise TypeError("excel_numeric_to_date() tz must be a time zone name")
    try:
        return ZoneInfo(tz)
    except Exception as error:
        raise ValueError(
            f"excel_numeric_to_date() unknown time zone {tz!r}"
        ) from error


def _excel_flags(
    date_system: str, include_time: bool, round_seconds: bool, tz: str
) -> tuple[str, bool, bool, str]:
    if date_system not in _EXCEL_ORIGINS:
        raise ValueError(
            "excel_numeric_to_date() date_system must be 'modern' or 'mac pre-2011'"
        )
    if not isinstance(include_time, bool) or not isinstance(round_seconds, bool):
        raise TypeError(
            "excel_numeric_to_date() include_time and round_seconds must be booleans"
        )
    _excel_zone(tz)
    return date_system, include_time, round_seconds, tz


def _excel_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and value != value:
        return True
    try:
        import pandas as pd

        if value is pd.NA or value is pd.NaT:
            return True
    except ImportError:
        pass
    return False


def _excel_numbers(values: Any) -> Any:
    import numpy as np

    if isinstance(values, pl.Series):
        values = values.to_list()
    elif hasattr(values, "tolist") and not isinstance(values, (str, bytes)):
        values = values.tolist()
    numeric = (int, float, np.integer, np.floating)
    numbers = np.empty(len(values), dtype=np.float64)
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, numeric):
            if _excel_missing(value):
                numbers[index] = np.nan
                continue
            raise TypeError("excel_numeric_to_date() date_num must be numeric")
        numbers[index] = float(value)
    return numbers


def _excel_convert(
    values: Any,
    *,
    date_system: str,
    include_time: bool,
    round_seconds: bool,
    tz: str,
) -> list[Any]:
    """Janitor's serial conversion, including the 1900 leap-day adjustment.

    The adjustment runs for both date systems, as in janitor: serial 60 is
    missing, and serials before 60 move forward one day. ``59`` is
    1900-02-28, ``60`` is missing, and ``61`` is 1900-03-01.
    """
    import math

    import numpy as np

    origin = _EXCEL_ORIGINS[date_system]
    zone = _excel_zone(tz)
    numbers = _excel_numbers(values)
    days = np.floor((numbers * 86400.0 + 0.001) / 86400.0)
    plain = np.floor(numbers)
    seconds = (numbers - days) * 86400.0
    seconds = np.where(np.isfinite(numbers) & (days > plain), 0.0, seconds)
    if round_seconds:
        seconds = np.round(seconds)
    leap = np.isfinite(days) & (np.floor(days) == 60)
    before = np.isfinite(days) & (np.floor(days) < 60)
    days = days.copy()
    days[leap] = np.nan
    days[before] = days[before] + 1
    converted: list[Any] = []
    for day, second in zip(days, seconds):
        if not math.isfinite(float(day)):
            converted.append(None)
            continue
        stamp = origin + timedelta(days=int(day))
        if not include_time:
            converted.append(stamp)
            continue
        clock = datetime(stamp.year, stamp.month, stamp.day, tzinfo=zone)
        extra = float(second) if math.isfinite(float(second)) else 0.0
        converted.append(clock + timedelta(seconds=extra))
    return converted


def _excel_scalar(value: Any) -> bool:
    if isinstance(value, (str, bytes, list, tuple, range, pl.Series, pl.Expr, Expr)):
        return False
    try:
        import pandas as pd

        if isinstance(value, (pd.Series, pd.Index)):
            return False
    except ImportError:
        pass
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return False
    except ImportError:
        pass
    return True


def excel_numeric_to_date(
    date_num: Any,
    *,
    date_system: str = "modern",
    include_time: bool = False,
    round_seconds: bool = True,
    tz: str = "UTC",
) -> Any:
    """Convert Excel serial numbers to dates (R **janitor::excel_numeric_to_date**).

    ``42370`` is 2016-01-01 in the modern system. Serial 60 is missing on
    both systems: Excel treated 1900 as a leap year, and janitor applies
    that adjustment to the Mac system as well, so Mac serial 1 is
    1904-01-03. ``date_system="mac pre-2011"`` uses the 1904 origin;
    ``40908`` is 2016-01-01 there. ``include_time=False`` ignores the
    fraction. ``round_seconds=True`` rounds the seconds to an integer the
    way R's ``round`` does. The default zone is UTC, not the machine zone.
    A quoted string is a column name. A number or list is converted
    immediately. Leap seconds are not applied, and no coercion warning is
    printed.
    """
    date_system, include_time, round_seconds, tz = _excel_flags(
        date_system, include_time, round_seconds, tz
    )
    if isinstance(date_num, (str, Expr, pl.Expr)):
        return _func(
            "excel_numeric_to_date",
            _value(date_num),
            date_system=date_system,
            include_time=include_time,
            round_seconds=round_seconds,
            tz=tz,
        )
    scalar = _excel_scalar(date_num)
    converted = _excel_convert(
        [date_num] if scalar else date_num,
        date_system=date_system,
        include_time=include_time,
        round_seconds=round_seconds,
        tz=tz,
    )
    if scalar:
        return converted[0]
    try:
        import pandas as pd

        if isinstance(date_num, pd.Series):
            if include_time:
                return pd.Series(
                    converted,
                    index=date_num.index,
                    dtype=pd.DatetimeTZDtype("us", tz),
                )
            return pd.Series(converted, index=date_num.index, dtype="object")
    except ImportError:
        pass
    if isinstance(date_num, pl.Series):
        dtype = pl.Datetime("us", tz) if include_time else pl.Date
        return pl.Series(converted, dtype=dtype)
    return converted


# Inclusive serials that land in years 1..9999 after the leap-day shift.
# Outside this range the Python converter raises OverflowError.
_EXCEL_SERIAL_LIMITS = {
    "modern": (-693594, 2958465),
    "mac pre-2011": (-695056, 2957003),
}


def _excel_needs_python(series: pl.Series, date_system: str, include_time: bool) -> bool:
    """True when the per-value converter must run.

    Integers and floats inside the calendar range become dates in Polars.
    Clocks, other dtypes, and serials outside year 1..9999 stay on the Python
    converter so the errors and the time-of-day rules stay the same.
    """
    if include_time or not (series.dtype.is_integer() or series.dtype.is_float()):
        return True
    if series.len() == 0:
        return False
    number = series.cast(pl.Float64)
    finite = number.filter(number.is_finite())
    if finite.len() == 0:
        return False
    low, high = _EXCEL_SERIAL_LIMITS[date_system]
    return finite.min() < low or finite.max() > high


def _excel_vector_dates(series: pl.Series, date_system: str) -> pl.Series:
    """Calendar dates for in-range serials, including the 1900 leap-day gap."""
    import numpy as np

    number = series.cast(pl.Float64).to_numpy()  # missing values become NaN
    with np.errstate(invalid="ignore"):
        # The 0.001 second bias matches janitor's day boundary. NumPy divides
        # exactly, as R does; Polars multiplies by 1/86400, which moves
        # serials such as -194.0000000115741 to the next day.
        days = np.floor((number * 86400.0 + 0.001) / 86400.0)
    valid = np.isfinite(days) & (days != 60)
    days = np.where(days < 60, days + 1, days)
    epoch = (_EXCEL_ORIGINS[date_system] - date(1970, 1, 1)).days
    since_epoch = np.where(valid, days + epoch, 0).astype(np.int32)
    return (
        pl.Series("d", since_epoch)
        .to_frame()
        .select(pl.when(pl.Series(valid)).then(pl.col("d")).cast(pl.Date))
        .to_series()
    )


def _excel_polars(
    value: pl.Expr,
    *,
    date_system: str,
    include_time: bool,
    round_seconds: bool,
    tz: str,
) -> pl.Expr:
    dtype = pl.Datetime("us", tz) if include_time else pl.Date

    def batch(series: pl.Series) -> pl.Series:
        if not _excel_needs_python(series, date_system, include_time):
            return _excel_vector_dates(series, date_system)
        converted = _excel_convert(
            series,
            date_system=date_system,
            include_time=include_time,
            round_seconds=round_seconds,
            tz=tz,
        )
        return pl.Series(converted, dtype=dtype)

    return value.map_batches(batch, return_dtype=dtype, is_elementwise=True)


def compile_polars(name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> pl.Expr:
    """Compile a date helper to a Polars expression."""
    value = args[0]
    if name == "excel_numeric_to_date":
        return _excel_polars(value, **kwargs)
    if name == "with_tz":
        from tidy3.expr import input_dtype

        # with_tz() only changes the zone a datetime is shown in.
        known = input_dtype(value)
        if known is not None and (known in {pl.Date, pl.Null} or isinstance(known, pl.Datetime)):
            stamp = value.cast(pl.Datetime("us")) if known in {pl.Date, pl.Null} else value
            return stamp.dt.convert_time_zone(kwargs["tzone"])
    if name == "force_tz":
        from tidy3.expr import input_dtype

        # A naive or UTC source has no repeated clocks of its own, so the
        # roll policies map onto replace_time_zone() when skipped clocks
        # become missing.
        known = input_dtype(value)
        skipped, repeated = kwargs["roll_dst"]
        ambiguous = {"pre": "earliest", "post": "latest", "NA": "null"}.get(repeated)
        if (
            known is not None
            and skipped == "NA"
            and ambiguous is not None
            and (known in {pl.Date, pl.Null} or (isinstance(known, pl.Datetime) and known.time_zone in {None, "UTC"}))
        ):
            stamp = value.cast(pl.Datetime("us")) if known in {pl.Date, pl.Null} else value
            if isinstance(known, pl.Datetime) and known.time_zone is not None:
                stamp = stamp.dt.replace_time_zone(None)
            return stamp.dt.replace_time_zone(
                kwargs["tzone"], ambiguous=ambiguous, non_existent="null"
            )
    if name in {"with_tz", "force_tz"}:
        from tidy3._timezones import transform

        # Polars infers the output on a dummy Series, preserving the input
        # time unit while changing its zone. No source rows are collected.
        return value.map_batches(
            lambda series: transform(series, name, **kwargs), is_elementwise=True
        )
    if name == "year":
        return value.dt.year().cast(pl.Int64)
    if name == "month":
        return value.dt.month().cast(pl.Int64)
    if name == "day":
        return value.dt.day().cast(pl.Int64)
    if name in {"hour", "minute", "second"}:
        return _clock_polars(value, name)
    if name in {"yday", "week"}:
        day_ = value.dt.ordinal_day().cast(pl.Int64)
        return day_ if name == "yday" else (day_ - 1) // 7 + 1
    if name == "isoweek":
        return value.dt.week().cast(pl.Int64)
    if name == "wday":
        number = (value.dt.weekday().cast(pl.Int64) - kwargs["week_start"]) % 7 + 1
        if not kwargs["label"]:
            return number
        labels = _weekday_labels(kwargs["abbr"], kwargs["week_start"])
        return number.replace_strict(dict(enumerate(labels, 1)), return_dtype=pl.String).cast(pl.Enum(labels))
    if name == "quarter":
        return _quarter_polars(value, kwargs["type"], kwargs["fiscal_start"])
    if name == "days_in_month":
        return value.dt.days_in_month().cast(pl.Int64)
    if name == "floor_date":
        return _floor_polars(value, kwargs["unit"], int(kwargs["week_start"]))
    if name == "ceiling_date":
        return _ceiling_polars(value, kwargs["unit"], int(kwargs["week_start"]))
    if name == "add_with_rollback":
        return shift_polars(value, kwargs["period"], roll_to_first=kwargs.get("roll_to_first", False))
    if name == "duration":
        return _duration_polars(value, kwargs["unit"])
    if name == "as_duration":
        return _by_dtype(value, _as_duration_of, pl.Duration("us"))
    if name == "time_length":
        return _by_dtype(value, lambda v, dtype: _time_length_of(v, dtype, kwargs["unit"]), pl.Float64)
    if name == "round_date":
        return _round_polars(value, kwargs["unit"], int(kwargs["week_start"]))
    if name == "seq_date":
        if "total" in kwargs:
            count, unit = _split_interval(kwargs["by"])
            end = value.dt.offset_by(f"{count * (kwargs['total'] + 1)}{unit}")
            return pl.date_ranges(value, end, interval=kwargs["by"], closed="both").list.head(kwargs["total"])
        return pl.date_ranges(args[0], args[1], interval=kwargs["by"], closed="both")
    if name == "make_date":
        return _make_date_polars(*args)
    if name == "make_datetime":
        return _make_datetime_polars(args, kwargs["tz"])
    if name == "isoyear":
        return value.dt.iso_year().cast(pl.Int64)
    if name == "leap_year":
        return _leap_polars(value)
    raise ValueError(f"unknown date helper: {name}")
