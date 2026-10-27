"""round_date, make_date, make_datetime, isoyear, and leap_year.

The R oracle case ``date_helpers`` checks the same rules against
lubridate; these tests pin the edges and tidy3's own choices.
"""

from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import polars as pl
import pytest

import tidy3 as t
from tidy3.expr import Expr, to_polars
from tidy3.masking import apply_masking


def clocks(frame, column="at"):
    return frame.collect()[column].dt.strftime("%Y-%m-%d %H:%M:%S %z").to_list()


def new_york(values):
    return t.tidy({"x": values}) >> t.mutate(x=t.ymd_hms("x", tz="America/New_York"))


# ── round_date ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("unit, expected", [
    ("day", [date(2010, 2, 15), date(2010, 2, 16), None]),
    ("week", [date(2010, 2, 14), date(2010, 2, 14), None]),
    # 2010-02-15 is 14 days from February 1 and from March 1: a tie goes up.
    ("month", [date(2010, 3, 1), date(2010, 3, 1), None]),
    ("quarter", [date(2010, 4, 1), date(2010, 4, 1), None]),
    ("year", [date(2010, 1, 1), date(2010, 1, 1), None]),
])
def test_round_date_on_dates(unit, expected):
    frame = t.tidy({"d": [date(2010, 2, 15), date(2010, 2, 16), None]})
    out = (frame >> t.mutate(at=t.round_date("d", unit))).collect()
    assert out.schema["at"] == pl.Date
    assert out["at"].to_list() == expected


def test_round_date_keeps_a_date_on_its_boundary_unlike_ceiling():
    frame = t.tidy({"d": [date(2010, 2, 1)]})
    out = (frame >> t.mutate(r=t.round_date("d", "month"), c=t.ceiling_date("d", "month"))).collect()
    assert out.row(0) == (date(2010, 2, 1), date(2010, 2, 1), date(2010, 3, 1))


def test_round_date_sub_day_units_leave_dates_alone():
    frame = t.tidy({"d": [date(2010, 2, 1)]})
    out = (frame >> t.mutate(at=t.round_date("d", "hour"))).collect()
    assert out["at"].to_list() == [date(2010, 2, 1)]


def test_round_date_week_start():
    frame = t.tidy({"d": [date(2010, 2, 1), date(2010, 2, 4), date(2010, 2, 5)]})
    sunday = (frame >> t.mutate(at=t.round_date("d", "week"))).collect()["at"].to_list()
    monday = (frame >> t.mutate(at=t.round_date("d", "1 week", week_start=1))).collect()["at"].to_list()
    assert sunday == [date(2010, 1, 31), date(2010, 2, 7), date(2010, 2, 7)]
    assert monday == [date(2010, 2, 1), date(2010, 2, 1), date(2010, 2, 8)]


def test_round_date_ties_go_up_on_datetimes():
    frame = t.tidy({"x": [
        datetime(2024, 1, 3, 12), datetime(2024, 1, 3, 11, 59, 59),
        datetime(2024, 7, 2, 0, 0, 0, 500000), datetime(2024, 7, 2, 0, 0, 0, 499999),
    ]})
    out = (frame >> t.mutate(
        day=t.round_date("x"), week=t.round_date("x", "week"), second=t.round_date("x", "seconds"),
    )).collect()
    assert out["day"].to_list()[:2] == [datetime(2024, 1, 4), datetime(2024, 1, 3)]
    # Wednesday noon is the middle of a Sunday week.
    assert out["week"].to_list()[:2] == [datetime(2024, 1, 7), datetime(2023, 12, 31)]
    assert out["second"].to_list()[2:] == [datetime(2024, 7, 2, 0, 0, 1), datetime(2024, 7, 2)]


def test_round_date_measures_elapsed_time_across_daylight_saving():
    frame = new_york(["2024-03-10 12:00:00", "2024-03-10 12:30:00", "2024-11-03 12:00:00", "2024-11-03 11:00:00", None])
    # 10 March has 23 hours and 3 November has 25.
    assert clocks(frame >> t.mutate(at=t.round_date("x", "day"))) == [
        "2024-03-10 00:00:00 -0500",
        "2024-03-11 00:00:00 -0400",
        "2024-11-04 00:00:00 -0500",
        "2024-11-03 00:00:00 -0400",
        None,
    ]


def test_round_date_hour_in_daylight_saving_gaps_and_repeats():
    frame = new_york(["2024-03-10 01:30:00", "2024-11-03T01:30:00-05:00", "2024-11-03T01:30:00-04:00"])
    # The first two match lubridate. The first 01:30 of a fallback is as far
    # from 01:00 EDT as from the next hour, 01:00 EST, so the tie goes up;
    # lubridate takes 02:00 EST as the next hour and rounds down.
    assert clocks(frame >> t.mutate(at=t.round_date("x", "hour"))) == [
        "2024-03-10 03:00:00 -0400",
        "2024-11-03 02:00:00 -0500",
        "2024-11-03 01:00:00 -0500",
    ]


def test_round_date_keeps_the_zone_and_stays_lazy():
    stamp = pl.Series([datetime(2024, 2, 15, 17)]).dt.replace_time_zone("UTC").dt.convert_time_zone("America/New_York")
    frame = t.tidy(pl.DataFrame({"x": stamp})) >> t.mutate(at=t.round_date("x", "month"))
    assert frame.collect().schema["at"] == pl.Datetime("us", "America/New_York")
    assert clocks(frame) == ["2024-03-01 00:00:00 -0500"]
    assert "python" not in frame._lf.explain().lower()


def test_round_date_validates_its_arguments():
    with pytest.raises(ValueError, match="round_date\\(\\) unit"):
        t.round_date("x", "fortnight")
    with pytest.raises(TypeError, match="unit must be a string"):
        t.round_date("x", 1)
    with pytest.raises(ValueError, match="week_start"):
        t.round_date("x", "week", week_start=0)


# ── make_date ───────────────────────────────────────────────────────────


def test_make_date_from_columns_and_numbers():
    frame = t.tidy({"y": [2024, 2023, 2023, 2023, None], "m": [2, 2, 13, 1, 1], "d": [29, 29, 1, 0, 1]})
    out = (frame >> t.mutate(at=t.make_date("y", "m", "d"), first=t.make_date("y"), fixed=t.make_date(2020, "m"))).collect()
    assert out["at"].to_list() == [date(2024, 2, 29), None, None, None, None]
    assert out["first"].to_list() == [date(2024, 1, 1), date(2023, 1, 1), date(2023, 1, 1), date(2023, 1, 1), None]
    assert out["fixed"].to_list() == [date(2020, 2, 1), date(2020, 2, 1), None, date(2020, 1, 1), date(2020, 1, 1)]


def test_make_date_defaults_and_fractions():
    frame = t.tidy({"y": [2023.7, -2023.5]})
    out = (frame >> t.mutate(at=t.make_date("y", 1.9, 1.5), epoch=t.make_date())).collect()
    # Fractions go toward zero, as R's as.integer does.
    assert out["at"].dt.to_string().to_list() == ["2023-01-01", "-2023-01-01"]
    assert out["epoch"].to_list() == [date(1970, 1, 1)] * 2


def test_make_date_beyond_year_9999():
    # Days since 1970 from lubridate; year 0 is a leap year and -5 is not.
    frame = t.tidy({"y": [10000, -5, 0, -5], "m": [2, 1, 2, 2]})
    out = (frame >> t.mutate(at=t.make_date("y", "m", 29))).collect()
    assert out["at"].cast(pl.Int32).to_list() == [2932956, -721326, -719469, None]


def test_make_date_validates_scalar_parts():
    with pytest.raises(TypeError, match="make_date\\(\\) month"):
        t.make_date(2020, True)
    with pytest.raises(TypeError, match="make_date\\(\\) day"):
        t.make_date(2020, 1, [1, 2])


# ── make_datetime ───────────────────────────────────────────────────────


def test_make_datetime_rolls_parts_over():
    frame = t.tidy({"d": [31, 1, 1], "h": [0, 25, 23], "s": [0.0, -0.5, 60.0]})
    out = (frame >> t.mutate(at=t.make_datetime(2023, 2, "d", "h", 0, "s"))).collect()
    assert out.schema["at"] == pl.Datetime("us", "UTC")
    assert out["at"].to_list() == [
        datetime(2023, 3, 3, tzinfo=timezone.utc),
        datetime(2023, 2, 2, 0, 59, 59, 500000, tzinfo=timezone.utc),
        datetime(2023, 2, 1, 23, 1, tzinfo=timezone.utc),
    ]


def test_make_datetime_months_roll_into_years():
    frame = t.tidy({"m": [13, 0, -1]})
    out = (frame >> t.mutate(at=t.make_datetime(2023, "m"))).collect()["at"]
    assert out.dt.date().to_list() == [date(2024, 1, 1), date(2022, 12, 1), date(2022, 11, 1)]


def test_make_datetime_daylight_saving_in_new_york():
    frame = t.tidy({"m": [3, 11, 3], "d": [10, 3, 9], "h": [2, 1, 26], "mi": [30, 30, 30]})
    out = frame >> t.mutate(at=t.make_datetime(2024, "m", "d", "h", "mi", tz="America/New_York"))
    # A skipped clock lands on the transition, a repeated one on its later
    # occurrence, and the clock rolls over before either rule applies.
    assert clocks(out) == ["2024-03-10 03:00:00 -0400", "2024-11-03 01:30:00 -0500", "2024-03-10 03:00:00 -0400"]


def test_make_datetime_missing_and_fractional_column_parts():
    frame = t.tidy({"h": [1.0, 1.5, None], "s": [0.0, 1.25, 0.0]})
    out = (frame >> t.mutate(at=t.make_datetime(2023, 1, 1, "h", 0, "s"))).collect()["at"]
    # lubridate raises on hour 1.5; a column gives missing instead.
    assert out.to_list() == [datetime(2023, 1, 1, 1, tzinfo=timezone.utc), None, None]
    inf = t.tidy({"s": [float("inf")]}) >> t.mutate(at=t.make_datetime(2023, 1, 1, 0, 0, "s"))
    assert inf.collect()["at"].to_list() == [None]


def test_make_datetime_validates_scalars_and_zone():
    with pytest.raises(ValueError, match="hour must be a whole number"):
        t.make_datetime(2023, 1, 1, 1.5)
    with pytest.raises(ValueError, match="year must be a whole number"):
        t.make_datetime(2023.5)
    with pytest.raises(TypeError, match="tzone"):
        t.make_datetime(2023, tz="")
    with pytest.raises(Exception):
        t.make_datetime(2023, tz="Not/AZone")
    assert isinstance(t.make_datetime(2023.0, 1, 1, 2.0), Expr)


# ── isoyear and leap_year ───────────────────────────────────────────────


def test_isoyear_matches_isoweek():
    frame = t.tidy({"d": [date(2021, 1, 1), date(2024, 12, 30), date(2026, 1, 1), date(2027, 1, 1), None]})
    out = (frame >> t.mutate(y=t.isoyear("d"), w=t.isoweek("d"))).collect()
    assert out["y"].to_list() == [2020, 2025, 2026, 2026, None]
    assert out["w"].to_list() == [53, 1, 1, 53, None]
    assert out.schema["y"] == pl.Int64


def test_leap_year_on_dates_datetimes_and_numbers():
    frame = t.tidy({
        "d": [date(2024, 6, 1), date(1900, 1, 1), None],
        "x": [datetime(2000, 1, 1), datetime(2023, 1, 1), None],
        "n": [2000.5, -1.5, None],
        "i": [2100, 2400, None],
    })
    out = (frame >> t.mutate(d=t.leap_year("d"), x=t.leap_year("x"), n=t.leap_year("n"), i=t.leap_year("i"))).collect()
    assert out.rows() == [(True, True, True, False), (False, False, False, True), (None, None, None, None)]


def test_leap_year_rejects_text():
    with pytest.raises(Exception, match="parse text first"):
        (t.tidy({"x": ["2024"]}) >> t.mutate(at=t.leap_year("x"))).collect()


def test_leap_year_without_a_known_type():
    # Compiled outside a frame, the type is read from each batch.
    frame = pl.DataFrame({"d": [date(2024, 1, 1)], "n": [2023]})
    out = frame.select(to_polars(t.leap_year("d")).alias("d"), to_polars(t.leap_year("n")).alias("n"))
    assert out.row(0) == (True, False)


# ── shared ──────────────────────────────────────────────────────────────


def test_date_kernels_keep_missing_values():
    frame = pl.DataFrame({"d": [date(2010, 2, 15), None], "y": [2024, None]})
    out = frame.select(
        to_polars(t.round_date("d", unit="month")).alias("rounded"),
        to_polars(t.make_date("y", 2, 29)).alias("made"),
        to_polars(t.leap_year("y")).alias("leap"),
        to_polars(t.make_datetime("y", 1, 1, 0, 0, 0, tz="UTC")).alias("clock"),
        to_polars(t.isoyear("d")).alias("isoyear"),
    )
    assert out["rounded"].to_list() == [date(2010, 3, 1), None]
    assert out["made"].to_list() == [date(2024, 2, 29), None]
    assert out["leap"].dtype == pl.Boolean and out["leap"].to_list() == [True, None]
    assert out["clock"].dtype == pl.Datetime("us", "UTC")
    assert out["clock"].to_list() == [datetime(2024, 1, 1, tzinfo=ZoneInfo("UTC")), None]
    assert out["isoyear"].to_list()[0] == 2010


def test_bare_column_names_in_notebooks():
    code = apply_masking("penguins >> mutate(d=make_date(year, month, 1), l=leap_year(year), r=round_date(day))")
    assert "make_date(__tidy3_col__('year'), __tidy3_col__('month'), 1)" in code
    assert "leap_year(__tidy3_col__('year'))" in code
    assert "round_date(__tidy3_col__('day'))" in code
