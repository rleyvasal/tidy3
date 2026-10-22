"""Datetime orders share conversion diagnostics and preserve timezone dtypes."""

from datetime import datetime, timezone
from zoneinfo import ZoneInfoNotFoundError

import pandas as pd
import polars as pl
import pytest

import tidy3 as t


BACKENDS = ["polars", "pandas"]
ORDERS = [
    (t.ymd_hms, "2024-02-29", "2024 February 29", "20240229", "12:34:56", "123456"),
    (t.ymd_hm, "2024-02-29", "2024 February 29", "20240229", "12:34", "1234"),
    (t.mdy_hms, "02/29/2024", "February 29th 2024", "02292024", "12:34:56", "123456"),
    (t.dmy_hms, "29/02/2024", "29th February 2024", "29022024", "12:34:56", "123456"),
]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("parser,calendar,named,compact,clock,digits", ORDERS)
def test_datetime_orders_formats_and_offsets(backend, parser, calendar, named, compact, clock, digits):
    values = [f"{calendar} {clock}", f"{named} {clock}", compact + digits,
              f"{calendar}T{clock}Z", f"{calendar} {clock}+05:30", None]
    frame = t.tidy({"x": values}, backend=backend) >> t.mutate(at=parser("x"))
    out = frame.collect()["at"]
    seconds = 0 if parser is t.ymd_hm else 56
    expected = datetime(2024, 2, 29, 12, 34, seconds, tzinfo=timezone.utc)
    assert out.dtype == pl.Datetime("us", "UTC")
    assert out.to_list() == [expected] * 4 + [expected.replace(hour=7, minute=4), None]
    assert t.problems(frame).collect().height == 0


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("parser,calendar,named,compact,clock,digits", ORDERS)
def test_numeric_compact_datetimes(backend, parser, calendar, named, compact, clock, digits):
    out = (t.tidy({"x": [int(compact + digits), None]}, backend=backend)
           >> t.mutate(at=parser("x"))).collect()["at"]
    assert out[0] == datetime(2024, 2, 29, 12, 34, 0 if parser is t.ymd_hm else 56, tzinfo=timezone.utc)
    assert out[1] is None


@pytest.mark.parametrize("backend", BACKENDS)
def test_fraction_am_pm_rollover_and_bad_clocks(backend):
    values = ["2024-02-29 12:34:56.123456789", "2024-02-29 12.34.56.125",
              "2024-02-29 12:00:00 AM", "2024-02-29 1:00:00 pm",
              "2024-02-29 24:00:00", "2024-02-29 23:59:60",
              "2024-02-30 12:00:00", "2024-02-29 25:00:00",
              "2024-02-29 12:60:00", "2024-02-29 12:00:61", "bad", None]
    frame = t.tidy({"x": values}, backend=backend) >> t.mutate(at=t.ymd_hms("x"))
    out = frame.collect()["at"]
    assert out[0].microsecond == 123456
    assert out[1].microsecond == 125000
    assert [value.hour for value in out[:6]] == [12, 12, 0, 13, 0, 0]
    assert out[4].day == out[5].day == 1
    assert out[4].month == out[5].month == 3
    assert out[6:].to_list() == [None] * 6
    assert t.problems(frame).collect()["row"].to_list() == [7, 8, 9, 10, 11]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("parser,calendar,compact,maximum", [
    (t.ymd_hms, "2024-01-02", "20240102", 3), (t.ymd_hm, "2024-01-02", "20240102", 2),
    (t.mdy_hms, "01/02/2024", "01022024", 3), (t.dmy_hms, "02/01/2024", "02012024", 3),
])
def test_truncation_is_explicit_and_does_not_discard_seconds(backend, parser, calendar, compact, maximum):
    values = [f"{calendar} 12:34:56", f"{calendar} 12:34", f"{calendar} 12", calendar, compact]
    frame = t.tidy({"x": values}, backend=backend)
    for truncated in range(maximum + 1):
        parsed = frame >> t.mutate(at=parser("x", truncated=truncated))
        result = parsed.collect()["at"].to_list()
        if parser is t.ymd_hm:
            assert result[0] is None
        first_valid = 1 if parser is t.ymd_hm else 0
        for index in range(first_valid, 5):
            missing = min(index, 3) - first_valid
            assert (result[index] is not None) == (missing <= truncated)


@pytest.mark.parametrize("backend", BACKENDS)
def test_timezone_dst_and_offsets_preserve_instants(backend):
    zone = "America/New_York"
    values = ["2024-03-10 02:30:00", "2024-03-10 03:30:00", "2024-11-03 01:30:00",
              "2024-11-03T01:30:00-04:00", "2024-11-03T01:30:00-05", "2024-01-01T12:00:00Z"]
    frame = t.tidy({"x": values}, backend=backend) >> t.mutate(at=t.ymd_hms("x", tz=zone))
    out = frame.collect()["at"]
    assert out.dtype == pl.Datetime("us", zone)
    utc = [value.astimezone(timezone.utc).isoformat() if value else None for value in out]
    assert utc == [None, "2024-03-10T07:30:00+00:00", "2024-11-03T06:30:00+00:00",
                   "2024-11-03T05:30:00+00:00", "2024-11-03T06:30:00+00:00", "2024-01-01T12:00:00+00:00"]
    assert t.problems(frame).collect()["row"].to_list() == [1]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("parser", [t.ymd_hms, t.ymd_hm])
def test_offsets_after_truncated_clocks_and_invalid_offsets(backend, parser):
    values = ["2024-01-02 12:34-05", "2024-01-02 12:34-0500", "2024-01-02 12:34-05:00",
              "2024-01-02 12:34+25", "2024-01-02 12:34+05:99"]
    frame = t.tidy({"x": values}, backend=backend) >> t.mutate(at=parser("x", truncated=1))
    expected = datetime(2024, 1, 2, 17, 34, tzinfo=timezone.utc)
    assert frame.collect()["at"].to_list() == [expected, expected, expected, None, None]
    assert t.problems(frame).collect()["row"].to_list() == [4, 5]


@pytest.mark.parametrize("backend", BACKENDS)
def test_cutoff_and_conversion_diagnostics(backend):
    frame = t.tidy({"x": ["68-01-02 12:00:00", "69-01-02 12:00:00", "bad", None]}, backend=backend)
    missing = frame >> t.mutate(at=t.ymd_hms("x"))
    assert missing.collect()["at"].null_count() == 4
    assert t.problems(missing).collect()["row"].to_list() == [1, 2, 3]
    parsed = (frame >> t.mutate(at=t.ymd_hms("x", cutoff=68))).compute()
    assert [value.year if value else None for value in parsed.collect()["at"]] == [2068, 1969, None, None]
    assert t.problems(parsed).collect()["original"].to_list() == ["bad"]
    with pytest.raises(t.ConversionError, match="bad"):
        t.stop_for_problems(parsed)
    with pytest.raises(t.ConversionError, match="bad"):
        (frame >> t.mutate(at=t.ymd_hms("x", cutoff=68, strict=True))).collect()


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("values", [[], [None, None], ["bad", "bad"]])
def test_empty_and_missing_outputs_keep_datetime_dtype(backend, values):
    source = pd.DataFrame({"x": pd.Series(values, dtype="string")})
    out = (t.tidy(source, backend=backend) >> t.mutate(at=t.ymd_hms("x", tz="Asia/Kolkata"))).collect()
    assert out["at"].dtype == pl.Datetime("us", "Asia/Kolkata")
    assert out["at"].null_count() == len(values)


@pytest.mark.parametrize("backend", BACKENDS)
def test_numeric_nan_is_missing_and_does_not_record_a_problem(backend):
    frame = (t.tidy({"x": [float("nan"), None, 20240102123456.]}, backend=backend)
             >> t.mutate(at=t.ymd_hms("x")))
    assert frame.collect()["at"].to_list() == [None, None, datetime(2024, 1, 2, 12, 34, 56, tzinfo=timezone.utc)]
    assert t.problems(frame).collect().height == 0


@pytest.mark.parametrize("backend", BACKENDS)
def test_grouped_nested_and_across_expressions(backend):
    source = pd.DataFrame({"g": ["a", "a", "b"], "x": ["2024-01-02 12:34:56", "bad", "2024-02-03 01:02:03"]}, index=[1, 1, 0])
    frame = t.tidy(source, backend=backend)
    grouped = frame >> t.group_by("g") >> t.summarise(at=t.ymd_hms(t.first("x")))
    assert grouped.collect()["at"].to_list() == [datetime(2024, 1, 2, 12, 34, 56, tzinfo=timezone.utc), datetime(2024, 2, 3, 1, 2, 3, tzinfo=timezone.utc)]
    replaced = frame >> t.mutate(t.across("x", t.ymd_hms))
    assert replaced.collect()["x"].null_count() == 1
    assert t.problems(replaced).collect()["row"].to_list() == [2]
    nested = frame >> t.mutate(h=t.hour(t.ymd_hms("x")))
    assert nested.collect()["h"].to_list() == [12, None, 1]


def test_datetime_plans_stay_lazy(monkeypatch):
    frame = t.tidy({"x": ["2024-01-02 12:34:56"]})
    def no_collect(*args, **kwargs):
        raise AssertionError("building datetime parsers must not collect rows")
    monkeypatch.setattr(pl.LazyFrame, "collect", no_collect)
    out = frame >> t.mutate(at=t.ymd_hms("x", tz="America/New_York"))
    assert out._lf.collect_schema()["at"] == pl.Datetime("us", "America/New_York")
    assert t.problems(out).columns == ["row", "column", "original", "expected"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_parse_datetime_offsets_are_naive_utc_clocks(backend):
    values = ["2024-01-01 12:00:00+0200", "2024-01-01 12:00:00-0500",
              "2024-01-01 12:00:00+02:00", "2024-01-01 12:00:00Z",
              "2024-06-01 00:30:00-0530", "bad", None]
    frame = (t.tidy({"g": ["a", "a", "a", "b", "b", "b", "b"], "x": values}, backend=backend)
             >> t.group_by("g")
             >> t.mutate(at=t.parse_datetime("x", "%Y-%m-%d %H:%M:%S%z")))
    assert frame._groups == ["g"]
    parsed = frame.collect()["at"]
    assert parsed.dtype == pl.Datetime("us")
    assert parsed.to_list() == [
        datetime(2024, 1, 1, 10, 0), datetime(2024, 1, 1, 17, 0),
        datetime(2024, 1, 1, 10, 0), datetime(2024, 1, 1, 12, 0),
        datetime(2024, 6, 1, 6, 0), None, None,
    ]
    assert parsed[0].tzinfo is None
    assert t.problems(frame).collect()["row"].to_list() == [6]
    plain = t.tidy({"x": ["2024-01-01 12:00:00", None]}, backend=backend)
    kept = (plain >> t.mutate(at=t.parse_datetime("x", "%Y-%m-%d %H:%M:%S"))).collect()["at"]
    assert kept.dtype == pl.Datetime("us")
    assert kept.to_list() == [datetime(2024, 1, 1, 12, 0), None]
    with pytest.raises(t.ConversionError, match="bad"):
        (t.tidy({"x": ["bad"]}, backend=backend)
         >> t.mutate(at=t.parse_datetime("x", "%Y-%m-%d %H:%M:%S%z", strict=True))).collect()


@pytest.mark.parametrize("backend", BACKENDS)
def test_parse_datetime_explicit_format_does_not_read_a_t_separator(backend):
    # Only the guessed orders (ymd_hms, ...) accept ``T`` for the space.
    frame = t.tidy({"x": ["2024-01-07 13:05:09", "2024-01-07T13:05:09"]}, backend=backend)
    out = frame >> t.mutate(
        at=t.parse_datetime("x", "%Y-%m-%d %H:%M:%S"), guessed=t.ymd_hms("x"),
    )
    collected = out.collect()
    assert collected["at"].to_list() == [datetime(2024, 1, 7, 13, 5, 9), None]
    assert collected["guessed"].dt.replace_time_zone(None).to_list() == [datetime(2024, 1, 7, 13, 5, 9)] * 2
    assert t.problems(out).collect()["row"].to_list() == [2]


def test_parser_options_validate_before_execution():
    for value in [-1, 4, True, 1.5]:
        with pytest.raises(ValueError, match="truncated"):
            t.ymd_hms("x", truncated=value)
    with pytest.raises(ValueError, match="truncated"):
        t.ymd_hm("x", truncated=3)
    with pytest.raises(TypeError, match="timezone"):
        t.ymd_hms("x", tz=None)
    with pytest.raises(ZoneInfoNotFoundError):
        t.ymd_hms("x", tz="Not/AZone")
    with pytest.raises(ValueError, match="cutoff"):
        t.ymd_hms("x", cutoff=100)
    with pytest.raises(TypeError, match="boolean"):
        t.ymd_hms("x", strict="yes")
