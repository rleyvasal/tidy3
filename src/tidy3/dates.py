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

import pandas as pd
import polars as pl

from tidy3.convert import dmy, dmy_hms, mdy, mdy_hms, ym, ymd, ymd_hm, ymd_hms
from tidy3.expr import Expr, _func, col

__all__ = [
    "Period",
    "months",
    "years",
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
    "days_in_month",
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


class Period:
    """A number of calendar months, years, or both.

    Adding one to a date rolls the day backward when the target month is
    shorter. See :func:`add_with_rollback`.
    """

    __slots__ = ("months", "years")

    def __init__(self, *, months: Any = 0, years: Any = 0):
        self.months = months
        self.years = years

    def __repr__(self) -> str:
        return f"Period(months={self.months!r}, years={self.years!r})"

    def __neg__(self) -> "Period":
        return Period(months=_negate(self.months), years=_negate(self.years))

    def __add__(self, other: Any) -> "Period":
        if not isinstance(other, Period):
            return NotImplemented
        return Period(
            months=_combine(self.months, other.months),
            years=_combine(self.years, other.years),
        )


def _negate(value: Any) -> Any:
    if isinstance(value, bool) or not isinstance(value, int):
        return -value
    return -value


def _combine(left: Any, right: Any) -> Any:
    if isinstance(left, int) and isinstance(right, int):
        return left + right
    return left + right


def _count(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        if isinstance(value, (Expr, pl.Expr)):
            return
        raise TypeError(f"{name}() needs an integer or a column")


def months(n: Any = 1) -> Period:
    """A period of *n* calendar months, for ``date + months(n)``.

    ``n`` may be an integer or a column. The day rolls backward when the
    destination month is shorter: 31 January plus ``months(1)`` is the last
    day of February.
    """
    _count("months", n)
    return Period(months=n)


def years(n: Any = 1) -> Period:
    """A period of *n* calendar years. 29 February plus ``years(1)`` is 28 February."""
    _count("years", n)
    return Period(years=n)


def _value(x: Any) -> Any:
    return col(x) if isinstance(x, str) else x


def add_with_rollback(x: Any, period: Period) -> Expr:
    """Add a :func:`months` or :func:`years` period, rolling invalid days backward.

    ``+`` and ``-`` with those periods do the same thing. Subtracting a date
    from a period is an error.
    """
    if not isinstance(period, Period):
        raise TypeError("add_with_rollback() period must be months() or years()")
    return _func("add_with_rollback", _value(x), period=period)


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


def _amount(value: Any, unit: str) -> Any:
    if isinstance(value, bool):
        raise TypeError("a period count must be an integer or a column")
    if isinstance(value, int):
        if value == 0:
            return None
        return f"{value}{unit}"
    from tidy3.expr import to_polars

    compiled = value if isinstance(value, pl.Expr) else to_polars(value)
    return pl.format("{}{}", compiled.cast(pl.Int64), pl.lit(unit))


def total_months(period: Period) -> Any:
    """All of *period* in months: ``12 * years + months``.

    lubridate's ``add_with_rollback`` adds a period this way, so the day
    rolls back once, at the end. Adding years and then months would roll
    29 February back to the 28th before the month moves.
    """
    years, months_ = period.years, period.months
    if isinstance(years, int) and isinstance(months_, int):
        return 12 * years + months_
    if isinstance(years, int) and years == 0:
        return months_
    if isinstance(months_, int) and months_ == 0:
        return years * 12
    return years * 12 + months_


def shift_polars(expr: pl.Expr, period: Period) -> pl.Expr:
    """Move *expr* by *period* as one count of months."""
    by = _amount(total_months(period), "mo")
    return expr if by is None else expr.dt.offset_by(by)


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


def _period_of(node: Any) -> Period | None:
    if isinstance(node, tuple) and node and node[0] == "lit" and isinstance(node[1], Period):
        return node[1]
    return None


def compile_period_bin(op: str, left: Any, right: Any, compile: Any) -> pl.Expr | None:
    """Compile ``date + months(1)`` before a period would be sent to ``pl.lit``."""
    left_period = _period_of(left)
    right_period = _period_of(right)
    if left_period is None and right_period is None:
        return None
    if op not in {"+", "-"}:
        return None
    if left_period is not None and right_period is not None:
        raise TypeError("add months() or years() to a date, not to another period")
    if left_period is not None:
        if op == "-":
            raise TypeError("cannot subtract a date from months() or years()")
        return shift_polars(compile(right), left_period)
    period = right_period if op == "+" else -right_period
    return shift_polars(compile(left), period)


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


def _excel_pandas(source: pd.Series, **kwargs: Any) -> pd.Series:
    if not kwargs["include_time"]:
        series = pl.from_pandas(source)
        if not _excel_needs_python(series, kwargs["date_system"], False):
            converted = _excel_vector_dates(series, kwargs["date_system"]).to_list()
            return pd.Series(converted, index=source.index, dtype="object")
    converted = _excel_convert(source, **kwargs)
    if kwargs["include_time"]:
        return pd.Series(
            converted,
            index=source.index,
            dtype=pd.DatetimeTZDtype("us", kwargs["tz"]),
        )
    return pd.Series(converted, index=source.index, dtype="object")


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
        return shift_polars(value, kwargs["period"])
    raise ValueError(f"unknown date helper: {name}")


# ── pandas ──────────────────────────────────────────────────────────────


def _date_only(series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return False
    for value in series.tolist():
        if value is None or value is pd.NaT:
            continue
        try:
            if bool(pd.isna(value)):
                continue
        except TypeError:
            pass
        return isinstance(value, date) and not isinstance(value, datetime)
    return False


def _temporal(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    return pd.to_datetime(series)


def _restore(series: pd.Series, date_only: bool) -> pd.Series:
    if not date_only:
        return series
    return pd.Series(
        [None if pd.isna(value) else value.date() for value in series.tolist()],
        index=series.index,
        dtype="object",
    )


def _component(series: pd.Series, name: str) -> pd.Series:
    stamped = _temporal(series)
    values = getattr(stamped.dt, name)
    return values.astype("Int64")


def _quarter_pandas(source: pd.Series, kind: str, start: int) -> pd.Series:
    stamped = _temporal(source)
    month_ = stamped.dt.month.astype("Int64")
    number = ((month_ - start) % 12) // 3 + 1
    if kind == "quarter":
        return number
    ending_year = stamped.dt.year.astype("Int64") + ((month_ >= start) & (start != 1)).astype("Int64")
    if kind == "year.quarter":
        return ending_year + number / 10
    if kind == "year_start/end":
        return ((ending_year - 1).astype("string").fillna("NA") + "/"
                + (ending_year % 100).astype("string").str.zfill(2).fillna("NA")
                + " Q" + number.astype("string").fillna("NA"))
    values = []
    for value in stamped:
        if pd.isna(value):
            values.append(None)
            continue
        beginning = date(value.year, value.month, 1) - pd.DateOffset(months=(value.month - start) % 3)
        result = beginning if kind == "date_first" else beginning + pd.DateOffset(months=3) - pd.Timedelta(days=1)
        values.append(result.date())
    import pyarrow as pa

    return pd.Series(values, index=source.index, dtype=pd.ArrowDtype(pa.date32()))


def shift_pandas(series: pd.Series, period: Period) -> pd.Series:
    """Move *series* by *period* as one count of months.

    Column counts must already be evaluated to Series (see
    :func:`tidy3.pandas_engine`). A missing count gives a missing date.
    """
    date_only = _date_only(series)
    result = _temporal(series)
    count = total_months(period)
    if isinstance(count, (Expr, pl.Expr)):
        raise TypeError(
            "months() or years() of a column must be evaluated before shifting"
        )
    if isinstance(count, int):
        if count != 0:
            result = result + pd.DateOffset(months=count)
        return _restore(result, date_only)
    counts = count if isinstance(count, pd.Series) else pd.Series(count, index=result.index)
    moved = [
        None if pd.isna(step) or pd.isna(value) else value + pd.DateOffset(months=int(step))
        for value, step in zip(result.tolist(), counts.reindex(result.index).tolist())
    ]
    result = pd.Series(moved, index=result.index, dtype=result.dtype)
    return _restore(result, date_only)


def _dtype_zone(series: pd.Series):
    return getattr(series.dtype, "tz", None)


def _relocalize(wall: pd.Series, template: pd.Series) -> pd.Series:
    """Attach *template*'s zone to naive wall-clock times.

    Rounding runs on the local clock, so a day is a calendar date rather
    than 24 hours. An ambiguous result keeps the source row's occurrence:
    a true flag is the daylight-saving one. Unambiguous clocks ignore it.
    """
    zone = _dtype_zone(template)
    if zone is None:
        return wall
    # pandas 3 has no Series.dt.dst(). Each instant's own dst() still says
    # which repeated clock it is; True keeps the daylight-saving one.
    ambiguous = [
        isinstance(value, datetime) and not pd.isna(value) and bool(value.dst())
        for value in template.dt.to_pydatetime()
    ]
    located = wall.dt.tz_localize(zone, ambiguous=ambiguous)
    return located.astype(template.dtype)


def _floor_wall(wall: pd.Series, unit: str) -> pd.Series:
    if unit in _PANDAS_FREQ:
        return wall.dt.floor(_PANDAS_FREQ[unit])
    if unit == "month":
        return wall.dt.to_period("M").dt.to_timestamp()
    if unit == "quarter":
        return wall.dt.to_period("Q").dt.to_timestamp()
    if unit == "year":
        return wall.dt.to_period("Y").dt.to_timestamp()
    raise ValueError(f"unknown unit: {unit}")


def _floor_pandas(series: pd.Series, unit: str) -> pd.Series:
    stamped = _temporal(series)
    if unit == "week":
        return stamped
    if _dtype_zone(stamped) is not None:
        return _relocalize(_floor_wall(stamped.dt.tz_localize(None), unit), stamped)
    if unit in _PANDAS_FREQ:
        return stamped.dt.floor(_PANDAS_FREQ[unit])
    if unit == "month":
        return stamped.dt.to_period("M").dt.to_timestamp()
    if unit == "quarter":
        return stamped.dt.to_period("Q").dt.to_timestamp()
    if unit == "year":
        return stamped.dt.to_period("Y").dt.to_timestamp()
    raise ValueError(f"unknown unit: {unit}")


def _week_floor(series: pd.Series, week_start: int) -> pd.Series:
    stamped = _temporal(series)
    zone = _dtype_zone(stamped)
    wall = stamped.dt.tz_localize(None) if zone is not None else stamped
    wall = wall.dt.normalize()
    iso = wall.dt.dayofweek + 1
    shift = (iso - week_start) % 7
    floored = wall - pd.to_timedelta(shift.fillna(0), unit="D")
    if zone is None:
        return floored
    return _relocalize(floored, stamped)


def _calendar_days(floored: pd.Series, days: int) -> pd.Series:
    """Move *days* on the calendar. A fixed duration misses local midnight across DST."""
    if _dtype_zone(floored) is None:
        return floored + pd.Timedelta(days=days)
    return floored + pd.DateOffset(days=days)


def _ceiling_pandas(
    series: pd.Series, unit: str, week_start: int, date_only: bool = False
) -> pd.Series:
    if unit == "week":
        floored = _week_floor(series, week_start)
        nudged = _calendar_days(floored, 7)
    else:
        floored = _floor_pandas(series, unit)
        if unit == "second":
            nudged = floored + pd.Timedelta(seconds=1)
        elif unit == "minute":
            nudged = floored + pd.Timedelta(minutes=1)
        elif unit == "hour":
            nudged = floored + pd.Timedelta(hours=1)
        elif unit == "day":
            nudged = _calendar_days(floored, 1)
        elif unit == "month":
            nudged = floored + pd.DateOffset(months=1)
        elif unit == "quarter":
            nudged = floored + pd.DateOffset(months=3)
        else:
            nudged = floored + pd.DateOffset(years=1)
    if _dtype_zone(floored) is not None:
        origin = _temporal(series).dtype
        floored = floored.astype(origin)
        nudged = nudged.astype(origin)
    if date_only and unit not in _SUB_DAY:
        # A date moves to the next boundary even when it is on one; see
        # _ceiling_polars.
        return nudged
    stamped = _temporal(series)
    same = stamped.eq(floored) | stamped.isna()
    return stamped.where(same, nudged)


def eval_pandas(
    name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    df: pd.DataFrame,
) -> pd.Series:
    """Evaluate a date helper against a pandas frame."""
    source = args[0]
    if not isinstance(source, pd.Series):
        source = pd.Series(source, index=df.index)
    if name == "excel_numeric_to_date":
        return _excel_pandas(source, **kwargs)
    if name in {"with_tz", "force_tz"}:
        from tidy3._timezones import transform

        # Use the same native timezone kernels and DST rules on both
        # backends; Arrow carries the original datetime precision through.
        result = transform(pl.from_pandas(source), name, **kwargs).to_pandas()
        result.index = source.index
        return result
    if name == "year":
        return _component(source, "year")
    if name == "month":
        return _component(source, "month")
    if name == "day":
        return _component(source, "day")
    if name in {"hour", "minute"}:
        return _component(source, name)
    if name == "second":
        stamped = _temporal(source)
        return (stamped.dt.second + stamped.dt.microsecond / 1e6 + stamped.dt.nanosecond / 1e9).astype("Float64")
    if name in {"yday", "week"}:
        day_ = _component(source, "dayofyear")
        return day_ if name == "yday" else (day_ - 1) // 7 + 1
    if name == "isoweek":
        return _temporal(source).dt.isocalendar().week.astype("Int64")
    if name == "wday":
        number = (_component(source, "dayofweek") + 1 - kwargs["week_start"]) % 7 + 1
        if not kwargs["label"]:
            return number
        labels = _weekday_labels(kwargs["abbr"], kwargs["week_start"])
        text = number.map(dict(enumerate(labels, 1)))
        return pd.Series(pd.Categorical(text, categories=labels, ordered=True), index=source.index)
    if name == "quarter":
        return _quarter_pandas(source, kwargs["type"], kwargs["fiscal_start"])
    if name == "days_in_month":
        return _component(source, "days_in_month")
    date_only = _date_only(source)
    if name == "floor_date":
        unit = kwargs["unit"]
        floored = (
            _week_floor(source, int(kwargs["week_start"]))
            if unit == "week"
            else _floor_pandas(source, unit)
        )
        return _restore(floored, date_only)
    if name == "ceiling_date":
        ceiling = _ceiling_pandas(
            source, kwargs["unit"], int(kwargs["week_start"]), date_only
        )
        return _restore(ceiling, date_only)
    if name == "add_with_rollback":
        return shift_pandas(source, kwargs["period"])
    raise ValueError(f"unknown date helper: {name}")
