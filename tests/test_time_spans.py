"""Durations and periods (phase 1 of docs/design-time-spans.md)."""

from __future__ import annotations

import ast
from datetime import date, datetime, timedelta

import pandas as pd
import polars as pl
import pytest

import tidy3 as t
from tidy3 import col, mutate, summarise, tidy
from tidy3.dates import Duration, Period
from tidy3.expr import to_polars
from tidy3.masking import COL_NAME, apply_masking

NY = "America/New_York"


def _zoned(*texts):
    return tidy({"x": list(texts)}) >> mutate(x=t.ymd_hms("x", tz=NY))


def _clocks(frame, expr):
    out = (frame >> mutate(at=expr)).collect()["at"]
    return out.dt.strftime("%Y-%m-%d %H:%M %Z").to_list()


# ── durations ───────────────────────────────────────────────────────────


def test_duration_constructors_use_lubridate_lengths():
    assert t.ddays(1) == timedelta(days=1)
    assert t.dweeks(1.5).total_seconds() == 907200
    assert t.dmonths(1).total_seconds() == 2629800
    assert t.dyears(1).total_seconds() == 31557600
    assert t.dseconds(None) is None
    assert t.duration(90, "minutes") == t.duration(hours=1, minutes=30) == t.dseconds(5400)
    assert t.duration(1, "month") == t.dmonths(1)
    assert t.as_duration(90) == t.dseconds(90)
    assert t.as_duration(t.months(1)) == t.dmonths(1)
    assert t.as_duration(t.days(1) + t.hours(1)).total_seconds() == 90000


def test_duration_prints_like_lubridate_and_stays_a_duration():
    assert repr(t.ddays(1)) == "86400s (~1 days)"
    assert repr(t.dmonths(1)) == "2629800s (~4.35 weeks)"
    assert repr(t.dseconds(1.5)) == "1.5s"
    total = t.ddays(1) + t.dhours(12)
    assert isinstance(total, Duration) and repr(total) == "129600s (~1.5 days)"
    assert isinstance(-t.ddays(1), Duration)
    assert isinstance(t.ddays(1) * 2, Duration)


def test_durations_and_periods_do_not_add():
    with pytest.raises(TypeError, match="period with a duration"):
        t.ddays(1) + t.months(1)
    with pytest.raises(TypeError, match="period with a duration"):
        t.months(1) + t.ddays(1)
    with pytest.raises(TypeError, match="col"):
        t.dseconds("qsec")
    with pytest.raises(ValueError, match="unit"):
        t.duration(1, "fortnight")


def test_durations_compare_with_periods_by_average_length():
    assert t.ddays(1) == t.days(1)
    assert t.days(1) == t.hours(24)
    assert t.dhours(1) < t.minutes(61)
    assert t.months(1) != t.days(30)
    assert t.months(1) == t.dmonths(1)


def test_date_plus_duration_keeps_dates_only_for_whole_days():
    frame = tidy({"d": [date(2012, 2, 29)], "s": [129600.0]})
    out = (frame >> mutate(
        whole=col("d") + t.ddays(365),
        part=col("d") + t.ddays(1.5),
        back=col("d") - t.ddays(1.5),
        timedelta_part=col("d") + timedelta(hours=36),
        column=col("d") + t.dseconds(col("s")),
    )).collect()
    assert out.schema["whole"] == pl.Date and out["whole"][0] == date(2013, 2, 28)
    assert out["part"][0] == datetime(2012, 3, 1, 12)
    assert out["back"][0] == datetime(2012, 2, 27, 12)
    assert out["timedelta_part"][0] == datetime(2012, 3, 1, 12)
    assert out["column"][0] == datetime(2012, 3, 1, 12)


def test_date_plus_a_duration_column_by_name_is_a_datetime():
    frame = tidy({"d": [date(2012, 2, 29)], "gap": [timedelta(hours=36)],
                  "a": [datetime(2024, 1, 1)], "b": [datetime(2024, 1, 2, 12)]})
    out = (frame >> mutate(
        at=col("d") + col("gap"),
        back=col("gap") + col("d"),
        span=col("d") + (col("b") - col("a")),
    )).collect()
    assert out["at"][0] == datetime(2012, 3, 1, 12)
    assert out["back"][0] == datetime(2012, 3, 1, 12)
    assert out["span"][0] == datetime(2012, 3, 1, 12)


def test_datetime_plus_ddays_is_exact_time():
    assert _clocks(_zoned("2024-03-09 12:00:00", "2024-11-02 12:00:00"), col("x") + t.ddays(1)) == [
        "2024-03-10 13:00 EDT", "2024-11-03 11:00 EST",
    ]


def test_date_differences_are_durations_with_lengths():
    frame = tidy({"a": [date(2012, 1, 31)], "b": [date(2012, 3, 1)]})
    out = (frame >> mutate(
        gap=col("b") - col("a"),
        days_=t.time_length(col("b") - col("a"), "day"),
        weeks_=t.as_numeric(col("b") - col("a"), "week"),
    )).collect()
    assert out.schema["gap"] == pl.Duration("us")
    assert out["days_"][0] == 30.0
    assert out["weeks_"][0] == 30 / 7


def test_time_length_divides_exactly():
    frame = tidy({"s": [3.0, 1.0, None]}) >> mutate(d=t.dseconds(col("s")))
    out = (frame >> mutate(x=t.time_length("d", "sec"))).collect()["x"].to_list()
    assert out == [3.0, 1.0, None]
    # Multiplying by 1 / 3600 would miss 3 / 3600 by one unit in the last place.
    hours_ = (frame >> mutate(x=t.time_length("d", "hour"))).collect()["x"].to_list()
    assert hours_[:2] == [3 / 3600, 1 / 3600]
    assert t.time_length(t.ddays(1.5), "hour") == 36
    assert t.time_length(t.days(1), "week") == 1 / 7
    assert t.as_numeric(t.ddays(1.5), "hours") == 36


def test_time_length_in_a_summary():
    frame = tidy({"g": [1, 1, 2], "s": [3.0, 0.0, 7.0]}) >> mutate(d=t.dseconds(col("s")))
    out = (frame >> t.group_by("g") >> summarise(x=t.time_length(t.sum("d"), "second"))).collect()
    assert out.sort("g")["x"].to_list() == [3.0, 7.0]


def test_duration_summaries_stay_durations():
    frame = tidy({"k": [1, 2, 4]}) >> mutate(d=t.ddays(col("k")))
    out = (frame >> summarise(
        s=t.sum("d"), m=t.mean("d"), med=t.median("d"), lo=t.min("d"), hi=t.max("d"),
    )).collect()
    assert all(isinstance(out.schema[name], pl.Duration) for name in out.columns)
    assert out.row(0) == (timedelta(days=7), timedelta(days=7 / 3), timedelta(days=2),
                          timedelta(days=1), timedelta(days=4))


def test_duration_comparisons_in_filter():
    frame = tidy({"qsec": [16.5, 17.0, 18.9]})
    kept = (frame >> t.filter(t.dseconds(col("qsec")) < t.dseconds(17))).collect()
    assert kept["qsec"].to_list() == [16.5]
    longer = (frame >> t.filter(t.dseconds(col("qsec")) > t.seconds(17))).collect()
    assert longer["qsec"].to_list() == [18.9]


def test_as_duration_columns():
    frame = tidy({"n": [90, None], "gap": [timedelta(seconds=1), None]})
    out = (frame >> mutate(a=t.as_duration("n"), b=t.as_duration("gap"))).collect()
    assert out["a"].to_list() == [timedelta(seconds=90), None]
    assert out.schema["b"] == pl.Duration("us")
    with pytest.raises(Exception, match="as_duration"):
        (tidy({"s": ["a"]}) >> mutate(a=t.as_duration("s"))).collect()


# ── periods ─────────────────────────────────────────────────────────────


def test_periods_keep_their_units():
    assert repr(t.hours(25)) == "Period(hours=25)"
    assert repr(t.days(1) + t.hours(25)) == "Period(days=1, hours=25)"
    assert repr(t.weeks(2)) == "Period(days=14)"
    assert repr(-t.days(2)) == "Period(days=-2)"
    assert repr(t.days(2) * 3) == "Period(days=6)"
    assert repr(t.period(hours=25)) == "Period(hours=25)"
    assert repr(t.period(2, "week")) == "Period(days=14)"
    assert repr(t.period([1, 2], ["hour", "minute"])) == "Period(hours=1, minutes=2)"
    assert repr(t.seconds(1.5)) == "Period(seconds=1.5)"
    with pytest.raises(TypeError, match="integer"):
        t.days(1.5)
    with pytest.raises(TypeError, match="integer"):
        t.minutes(1.5)
    with pytest.raises(ValueError, match="unit"):
        t.period(1, "fortnight")
    with pytest.raises(TypeError, match="unknown units"):
        t.period(fortnights=1)


def test_date_plus_periods():
    frame = tidy({"d": [date(2012, 2, 29), date(2010, 1, 31), None], "k": [1, None, -1]})
    out = (frame >> mutate(
        day=col("d") + t.days(1),
        week=col("d") + t.weeks(2),
        hour=col("d") + t.hours(1),
        zero=col("d") + t.hours(0),
        mixed=col("d") + (t.months(1) + t.days(1)),
        signs=col("d") + (t.days(1) - t.hours(2)),
        counted=col("d") + t.days(col("k")),
        clock=col("d") + t.hours(col("k")),
    )).collect()
    assert out.schema["day"] == pl.Date and out.schema["zero"] == pl.Date
    assert out.schema["hour"] == pl.Datetime("us")
    assert out["day"].to_list() == [date(2012, 3, 1), date(2010, 2, 1), None]
    assert out["week"][0] == date(2012, 3, 14)
    assert out["hour"][0] == datetime(2012, 2, 29, 1)
    # Months first, rolled back, then the day: 31 Jan → 28 Feb → 1 Mar.
    assert out["mixed"].to_list()[:2] == [date(2012, 3, 30), date(2010, 3, 1)]
    assert out["signs"][0] == datetime(2012, 2, 29, 22)
    assert out["counted"].to_list() == [date(2012, 3, 1), None, None]
    assert out["clock"].to_list() == [datetime(2012, 2, 29, 1), None, None]


def test_subtracting_a_date_from_a_period_raises():
    frame = tidy({"d": [date(2024, 1, 1)]})
    with pytest.raises(TypeError, match="subtract a date from a period"):
        (frame >> mutate(x=t.days(1) - col("d"))).collect()


def test_clock_periods_on_a_zoned_column_follow_lubridate():
    frame = _zoned("2024-03-09 12:00:00", "2024-03-09 02:30:00", "2024-03-10 01:30:00", "2024-11-03 00:30:00")
    assert _clocks(frame, col("x") + t.days(1)) == [
        "2024-03-10 12:00 EDT", None, "2024-03-11 01:30 EDT", "2024-11-04 00:30 EST",
    ]
    assert _clocks(frame, col("x") + t.hours(1)) == [
        "2024-03-09 13:00 EST", "2024-03-09 03:30 EST", None, "2024-11-03 01:30 EDT",
    ]
    assert _clocks(frame, col("x") + t.hours(2))[3] == "2024-11-03 02:30 EST"


def test_repeated_clock_keeps_the_input_offset():
    later = tidy({"x": ["2024-11-03 06:30:00"]}) >> mutate(x=t.with_tz(t.ymd_hms("x"), NY))
    assert _clocks(later, col("x")) == ["2024-11-03 01:30 EST"]
    assert _clocks(later, col("x") + t.seconds(0)) == ["2024-11-03 01:30 EST"]
    assert _clocks(later, col("x") + t.minutes(1)) == ["2024-11-03 01:31 EST"]
    # From the day after, the EST offset is still valid for 01:30.
    assert _clocks(_zoned("2024-11-04 01:30:00"), col("x") - t.days(1)) == ["2024-11-03 01:30 EST"]
    # From the day before, EDT is kept.
    assert _clocks(_zoned("2024-11-02 01:30:00"), col("x") + t.days(1)) == ["2024-11-03 01:30 EDT"]


def test_month_into_a_skipped_hour():
    frame = _zoned("2024-02-10 02:30:00")
    assert _clocks(frame, col("x") + t.months(1)) == [None]
    assert _clocks(frame, t.add_with_rollback("x", t.months(1))) == ["2024-03-10 03:00 EDT"]
    assert _clocks(frame, t.add_with_rollback("x", t.days(29))) == ["2024-03-10 03:00 EDT"]


def test_add_with_rollback_roll_to_first():
    frame = tidy({"d": [date(2010, 1, 31), date(2012, 2, 29), date(2010, 3, 31), date(2010, 1, 15)],
                  "k": [1, 12, -1, 1]})
    out = (frame >> mutate(
        first=t.add_with_rollback("d", t.months(col("k")), roll_to_first=True),
        back=t.add_with_rollback("d", t.months(col("k"))),
        mixed=t.add_with_rollback("d", t.months(1) + t.days(1), roll_to_first=True),
    )).collect()
    assert out["first"].to_list() == [date(2010, 3, 1), date(2013, 3, 1), date(2010, 3, 1), date(2010, 2, 15)]
    assert out["back"].to_list() == [date(2010, 2, 28), date(2013, 2, 28), date(2010, 2, 28), date(2010, 2, 15)]
    assert out["mixed"][0] == date(2010, 3, 2)
    with pytest.raises(TypeError, match="roll_to_first"):
        t.add_with_rollback("d", t.months(1), roll_to_first="yes")


def test_span_plans_stay_lazy():
    lazy = pl.LazyFrame({"x": ["2024-03-09 12:00:00"], "d": [date(2024, 1, 31)]})
    frame = tidy(lazy) >> mutate(x=t.ymd_hms("x", tz=NY)) >> mutate(
        a=col("x") + t.hours(1), b=col("d") + t.ddays(1.5), c=col("d") + t.months(1),
    )
    schema = frame.lazy().collect_schema()
    assert schema["a"] == pl.Datetime("us", NY)
    assert schema["b"] == pl.Datetime("us")
    assert schema["c"] == pl.Date


def test_spans_compile_without_a_frame():
    # No frame schema: the rule is chosen per batch from the dtype.
    zoned = pl.DataFrame({"x": [datetime(2024, 3, 10, 1, 30)], "k": [1]}).with_columns(
        pl.col("x").dt.replace_time_zone(NY))
    assert zoned.select(to_polars(col("x") + t.hours(col("k"))))["x"].to_list() == [None]
    dated = pl.DataFrame({"x": [date(2012, 2, 29)]})
    assert dated.select(to_polars(col("x") + t.hours(1)))["x"][0] == datetime(2012, 2, 29, 1)
    assert dated.select(to_polars(col("x") + t.ddays(1.5)))["x"][0] == datetime(2012, 3, 1, 12)
    seconds_ = pl.DataFrame({"x": [90.0]}).select(to_polars(t.time_length(t.as_duration("x"), "minute")))
    assert seconds_.to_series()[0] == 1.5


def test_period_arithmetic_keeps_dates_and_missing_counts():
    df = pl.DataFrame({
        "d": [date(2012, 2, 29), date(2010, 1, 31)],
        "x": pl.Series([datetime(2024, 3, 10, 1, 30), datetime(2024, 3, 9, 12, 0)]).dt.replace_time_zone(NY),
        "k": [1, None],
    })
    out = df.select(
        to_polars(col("d") + t.days(1)).alias("days"),
        to_polars(col("d") + t.months(col("k"))).alias("months"),
        to_polars(t.add_with_rollback("d", t.months(1), roll_to_first=True)).alias("rolled"),
        to_polars(col("x") + t.hours(1)).alias("clock"),
    )
    assert out["days"].to_list() == [date(2012, 3, 1), date(2010, 2, 1)]
    assert out["months"].to_list() == [date(2012, 3, 29), None]
    assert out["rolled"].to_list() == [date(2012, 3, 29), date(2010, 3, 1)]
    assert out["clock"].is_null().to_list() == [True, False]

def test_notebook_bare_columns_in_span_helpers():
    for helper in ("dseconds", "ddays", "days", "hours", "months", "as_duration", "time_length"):
        out = ast.unparse(ast.parse(apply_masking(f"mutate(x = {helper}(qsec))")))
        assert f"{helper}({COL_NAME}" in out, out


def test_period_is_hashable_and_column_counts_compare_lazily():
    assert hash(t.days(1)) == hash(t.days(1))
    assert isinstance(t.days(col("k")), Period)
    with pytest.raises(TypeError, match="column counts"):
        t.days(col("k")).total_seconds()
