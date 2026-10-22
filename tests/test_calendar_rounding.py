"""Timezone-aware floor and ceiling keep the local boundary on both backends."""

from datetime import date

import polars as pl
import pytest

import tidy3 as t


VALUES = [
    "2024-03-11 12:00:00",
    "2024-03-10 01:30:00",
    "2024-11-03 01:30:00.125",
    "2024-11-03T01:30:00.125-04:00",
    "2024-11-03T01:00:00-04:00",
    None,
]
UNITS = ["second", "minute", "hour", "day", "week", "month", "quarter", "year"]


def zoned(backend, empty=False):
    frame = (t.tidy({"g": ["a", "a", "b", "b", "b", "b"], "x": VALUES}, backend=backend)
             >> t.mutate(x=t.ymd_hms("x", tz="America/New_York")))
    return frame >> t.head(0) if empty else frame


def clocks(frame, column):
    return frame.collect()[column].dt.strftime("%Y-%m-%d %H:%M:%S %z").to_list()


def instants(frame, column):
    values = frame.collect()[column].to_list()
    return [None if value is None else (round(value.timestamp(), 6), value.strftime("%z")) for value in values]


@pytest.mark.parametrize("backend", ["polars", "pandas"])
@pytest.mark.parametrize("unit,expected", [
    ("month", "2024-03-01 00:00:00 -0500"),
    ("quarter", "2024-01-01 00:00:00 -0500"),
    ("year", "2024-01-01 00:00:00 -0500"),
    ("week", "2024-03-10 00:00:00 -0500"),
])
def test_march_floor_stays_in_new_york(backend, unit, expected):
    frame = zoned(backend) >> t.group_by("g") >> t.mutate(at=t.floor_date("x", unit))
    assert frame._groups == ["g"]
    assert "America/New_York" in str(frame.collect()["at"].dtype)
    assert clocks(frame, "at")[0] == expected


@pytest.mark.parametrize("backend", ["polars", "pandas"])
def test_ceiling_month_and_repeated_hour_second(backend):
    frame = zoned(backend)
    assert clocks(frame >> t.mutate(at=t.ceiling_date("x", "month")), "at")[0] == "2024-04-01 00:00:00 -0400"
    # 2024-11-03 01:30:00.125-04:00 is the first fallback hour. Flooring to a
    # second keeps that occurrence and drops the fraction.
    assert clocks(frame >> t.mutate(at=t.floor_date("x", "second")), "at")[3] == "2024-11-03 01:30:00 -0400"


@pytest.mark.parametrize("backend", ["polars", "pandas"])
def test_hour_boundaries_across_daylight_saving(backend):
    frame = zoned(backend) >> t.mutate(floored=t.floor_date("x", "hour"), ceiling=t.ceiling_date("x", "hour"))
    assert clocks(frame, "floored")[1:5] == [
        "2024-03-10 01:00:00 -0500",
        "2024-11-03 01:00:00 -0500",
        "2024-11-03 01:00:00 -0400",
        "2024-11-03 01:00:00 -0400",
    ]
    assert clocks(frame, "ceiling")[1:5] == [
        "2024-03-10 03:00:00 -0400",
        "2024-11-03 02:00:00 -0500",
        "2024-11-03 01:00:00 -0500",
        "2024-11-03 01:00:00 -0400",
    ]


@pytest.mark.parametrize("unit", UNITS)
@pytest.mark.parametrize("empty", [False, True])
def test_backends_round_the_same_instants(unit, empty):
    def signature(backend):
        frame = zoned(backend, empty) >> t.mutate(
            floored=t.floor_date("x", unit),
            ceiling=t.ceiling_date("x", unit),
            monday=t.floor_date("x", "week", week_start=1),
        )
        collected = frame.collect()
        assert collected["floored"].dtype == pl.Datetime("us", "America/New_York")
        return {name: instants(frame, name) for name in ("floored", "ceiling", "monday")}

    assert signature("pandas") == signature("polars")


@pytest.mark.parametrize("make", ["frame", "lazy", "tidy"])
def test_polars_dates_stay_dates_on_the_pandas_backend(make):
    # lubridate: ceiling_date(as.Date(c("2024-01-01", "2024-01-15")), "month") is Feb 1 twice.
    source = pl.DataFrame({"d": pl.Series(["2024-01-01", "2024-01-15", None]).str.to_date()})
    data = {"frame": source, "lazy": source.lazy(), "tidy": t.tidy(source)}[make]
    out = (
        t.tidy(data, backend="pandas")
        >> t.mutate(up=t.ceiling_date("d", "month"), down=t.floor_date("d", "month"))
    ).collect()
    assert out.schema == pl.Schema({"d": pl.Date, "up": pl.Date, "down": pl.Date})
    feb, jan = date(2024, 2, 1), date(2024, 1, 1)
    assert out["up"].to_list() == [feb, feb, None]
    assert out["down"].to_list() == [jan, jan, None]
