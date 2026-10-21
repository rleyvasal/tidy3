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

from datetime import date, datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import polars as pl

from tidy3.convert import dmy, mdy, ym, ymd
from tidy3.expr import Expr, _func, col

__all__ = [
    "Period",
    "months",
    "years",
    "add_with_rollback",
    "year",
    "month",
    "day",
    "floor_date",
    "ceiling_date",
    "days_in_month",
    "today",
    "now",
    "ymd",
    "mdy",
    "dmy",
    "ym",
]

FUNCTIONS = frozenset(
    {
        "year",
        "month",
        "day",
        "floor_date",
        "ceiling_date",
        "days_in_month",
        "add_with_rollback",
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

    A value that is already on the boundary stays there: the first of a
    month ceiling-rounded to a month is that same day.
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
    return pl.format("{}{}", compiled, pl.lit(unit))


def shift_polars(expr: pl.Expr, period: Period) -> pl.Expr:
    """Move *expr* by *period*, years first, then months."""
    result = expr
    years_by = _amount(period.years, "y")
    if years_by is not None:
        result = result.dt.offset_by(years_by)
    months_by = _amount(period.months, "mo")
    if months_by is not None:
        result = result.dt.offset_by(months_by)
    return result


def _floor_polars(expr: pl.Expr, unit: str, week_start: int) -> pl.Expr:
    if unit == "week":
        # Polars weekday is ISO: Monday 1 … Sunday 7. truncate("1w") is
        # Monday, so weeks are shifted onto week_start instead.
        shift = (expr.dt.weekday() - week_start) % 7
        return expr.dt.offset_by(pl.format("-{}d", shift))
    return expr.dt.truncate(_TRUNC[unit])


def _ceiling_polars(expr: pl.Expr, unit: str, week_start: int) -> pl.Expr:
    floored = _floor_polars(expr, unit, week_start)
    nudged = floored.dt.offset_by(_OFFSET[unit])
    return pl.when(expr == floored).then(expr).otherwise(nudged)


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


def compile_polars(name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> pl.Expr:
    """Compile a date helper to a Polars expression."""
    value = args[0]
    if name == "year":
        return value.dt.year().cast(pl.Int64)
    if name == "month":
        return value.dt.month().cast(pl.Int64)
    if name == "day":
        return value.dt.day().cast(pl.Int64)
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


def shift_pandas(series: pd.Series, period: Period) -> pd.Series:
    """Move *series* by *period*, years first, then months."""
    date_only = _date_only(series)
    result = _temporal(series)

    def apply_offset(current: pd.Series, count: Any, field: str) -> pd.Series:
        if isinstance(count, int):
            if count == 0:
                return current
            return current + pd.DateOffset(**{field: count})
        counts = count if isinstance(count, pd.Series) else pd.Series(count, index=current.index)
        moved = [
            value + pd.DateOffset(**{field: 0 if pd.isna(step) else int(step)})
            for value, step in zip(current.tolist(), counts.tolist())
        ]
        return pd.Series(moved, index=current.index)

    result = apply_offset(result, period.years, "years")
    result = apply_offset(result, period.months, "months")
    return _restore(result, date_only)


def _floor_pandas(series: pd.Series, unit: str) -> pd.Series:
    stamped = _temporal(series)
    if unit in _PANDAS_FREQ:
        return stamped.dt.floor(_PANDAS_FREQ[unit])
    if unit == "week":
        return stamped
    if unit == "month":
        return stamped.dt.to_period("M").dt.to_timestamp()
    if unit == "quarter":
        return stamped.dt.to_period("Q").dt.to_timestamp()
    if unit == "year":
        return stamped.dt.to_period("Y").dt.to_timestamp()
    raise ValueError(f"unknown unit: {unit}")


def _week_floor(series: pd.Series, week_start: int) -> pd.Series:
    stamped = _temporal(series)
    iso = stamped.dt.dayofweek + 1
    shift = (iso - week_start) % 7
    return stamped - pd.to_timedelta(shift.fillna(0), unit="D")


def _ceiling_pandas(series: pd.Series, unit: str, week_start: int) -> pd.Series:
    if unit == "week":
        floored = _week_floor(series, week_start)
        nudged = floored + pd.Timedelta(days=7)
    else:
        floored = _floor_pandas(series, unit)
        if unit == "second":
            nudged = floored + pd.Timedelta(seconds=1)
        elif unit == "minute":
            nudged = floored + pd.Timedelta(minutes=1)
        elif unit == "hour":
            nudged = floored + pd.Timedelta(hours=1)
        elif unit == "day":
            nudged = floored + pd.Timedelta(days=1)
        elif unit == "month":
            nudged = floored + pd.DateOffset(months=1)
        elif unit == "quarter":
            nudged = floored + pd.DateOffset(months=3)
        else:
            nudged = floored + pd.DateOffset(years=1)
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
    if name == "year":
        return _component(source, "year")
    if name == "month":
        return _component(source, "month")
    if name == "day":
        return _component(source, "day")
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
        ceiling = _ceiling_pandas(source, kwargs["unit"], int(kwargs["week_start"]))
        return _restore(ceiling, date_only)
    if name == "add_with_rollback":
        return shift_pandas(source, kwargs["period"])
    raise ValueError(f"unknown date helper: {name}")
