"""Hour orders, parse_date_time's ranked orders, and readr's parse_time."""

from datetime import date, datetime, time, timezone
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import polars as pl
import pytest

import tidy3 as t

UTC = timezone.utc


# ── hour and hour-minute orders ─────────────────────────────────────────

HOURS = [
    (t.ymd_h, "2024-02-29", "13", "2024022913"),
    (t.mdy_h, "02/29/2024", "13", "0229202413"),
    (t.dmy_h, "29/02/2024", "13", "2902202413"),
    (t.mdy_hm, "02/29/2024", "13:05", "022920241305"),
    (t.dmy_hm, "29/02/2024", "13:05", "290220241305"),
]


@pytest.mark.parametrize("parser,calendar,clock,compact", HOURS)
def test_hour_orders_read_separators_compact_text_and_am_pm(parser, calendar, clock, compact):
    minute = 5 if ":" in clock else 0
    pm = "1:05 PM" if minute else "1 PM"
    values = [f"{calendar} {clock}", f"{calendar}T{clock}", compact, f"{calendar} {pm}",
              f"{calendar} {clock}+05:30", f"{calendar} {clock}:09", "bad", None]
    frame = t.tidy({"x": values}) >> t.mutate(at=parser("x"))
    out = frame.collect()["at"]
    expected = datetime(2024, 2, 29, 13, minute, tzinfo=UTC)
    assert out.dtype == pl.Datetime("us", "UTC")
    assert out.to_list() == [expected] * 4 + [expected.replace(hour=7, minute=30 + minute), None, None, None]
    assert t.problems(frame).collect()["row"].to_list() == [6, 7]


@pytest.mark.parametrize("parser,calendar,clock,compact", HOURS)
def test_hour_orders_truncate_to_a_date_and_keep_the_zone(parser, calendar, clock, compact):
    zone = "America/New_York"
    frame = t.tidy({"x": [f"{calendar} {clock}", calendar]})
    exact = (frame >> t.mutate(at=parser("x", tz=zone))).collect()["at"]
    assert exact.dtype == pl.Datetime("us", zone)
    assert exact[1] is None
    whole = (frame >> t.mutate(at=parser("x", tz=zone, truncated=len(clock.split(":"))))).collect()["at"]
    assert whole[1] == datetime(2024, 2, 29, tzinfo=ZoneInfo(zone))
    with pytest.raises(ValueError, match="truncated"):
        parser("x", truncated=len(clock.split(":")) + 1)


def test_ymd_h_reads_24_as_the_next_day_and_rejects_impossible_dates():
    frame = t.tidy({"x": ["2024-02-29 24", "2023-02-29 12", "2024-02-29 25"]})
    out = (frame >> t.mutate(at=t.ymd_h("x"))).collect()["at"]
    assert out.to_list() == [datetime(2024, 3, 1, tzinfo=UTC), None, None]


# ── parse_date_time ─────────────────────────────────────────────────────

def _parse(values: list[Any], orders: Any, **kwargs: Any) -> tuple[pl.Series, list[int]]:
    frame = t.tidy({"x": values}) >> t.mutate(at=t.parse_date_time("x", orders, **kwargs))
    return frame.collect()["at"], t.problems(frame).collect()["row"].to_list()


def test_parse_date_time_reads_loose_separators_and_surrounding_text():
    out, rows = _parse(
        ["2024-02-29 12:34:56", "2024/2/29 1:2:3", "20240229123456", "on 2024-02-29 12:34:56",
         "Thursday 2024-02-29 12:34:56", "2024-02-29 12:34:56.125", "x2024-02-29 12:34:56",
         "2024-02-29 12:34", "", None],
        "ymd HMS",
    )
    noon = datetime(2024, 2, 29, 12, 34, 56, tzinfo=UTC)
    assert out.to_list() == [
        noon, datetime(2024, 2, 29, 1, 2, 3, tzinfo=UTC), noon, noon, noon,
        noon.replace(microsecond=125000), None, None, None, None,
    ]
    assert rows == [7, 8, 9]


def test_parse_date_time_orders_letters_and_separators_are_one_spelling():
    values = ["2024-02-29 12:34:56"]
    expected = datetime(2024, 2, 29, 12, 34, 56, tzinfo=UTC)
    for orders in ["ymd HMS", "Ymd_HMS", "%Y-%m-%d %H:%M:%S", "ymdT", "YmdHMOS", ["dmy", "ymd HMS"]]:
        assert _parse(values, orders)[0].to_list() == [expected]


def test_parse_date_time_ranks_orders_by_the_column():
    # lubridate's training: the format that reads more values goes first.
    assert _parse(["01/02/2024", "13/01/2024", "14/01/2024"], ["mdy", "dmy"])[0][0] == datetime(2024, 2, 1, tzinfo=UTC)
    assert _parse(["01/02/2024", "01/13/2024", "01/14/2024"], ["dmy", "mdy"])[0][0] == datetime(2024, 1, 2, tzinfo=UTC)
    # On a tie the listed order wins.
    assert _parse(["01/02/2024"], ["dmy", "mdy"])[0][0] == datetime(2024, 2, 1, tzinfo=UTC)
    # More fields beat more values.
    out, _ = _parse(["2024-02-29 1:02:03 PM", "2024-02-29 13:02:03", "2024-02-29 13:02:04"], ["ymd HMS", "ymd IMS p"])
    assert out[0] == datetime(2024, 2, 29, 13, 2, 3, tzinfo=UTC)


def test_parse_date_time_trains_long_columns_on_prime_rows():
    # lubridate samples rows 1, 2, 3, 5, 7, ... of a column of 100 or more:
    # when none of those parse, nothing is guessed and nothing parses.
    primes = [1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    values = ["garbage" if row in primes else "2024-01-15" for row in range(1, 101)]
    out, rows = _parse(values, "ymd")
    assert out.null_count() == 100
    assert len(rows) == 100
    values[1] = "2024-01-15"
    assert _parse(values, "ymd")[0].null_count() == len(primes) - 1


def test_parse_date_time_truncation_am_pm_and_clock_letters():
    out, _ = _parse(["2024-02-29 13:02:03", "2024-02-29 13:02", "2024-02-29 13", "2024-02-29", "2024-02"],
                    "ymd HMS", truncated=3)
    assert [value.isoformat() if value else None for value in out] == [
        "2024-02-29T13:02:03+00:00", "2024-02-29T13:02:00+00:00", "2024-02-29T13:00:00+00:00",
        "2024-02-29T00:00:00+00:00", None,
    ]
    out, _ = _parse(["2024-02-29 1:02:03 PM", "2024-02-29 12:02:03 AM", "2024-02-29 1:02:03"], "ymd IMS p")
    assert [value.hour for value in out] == [13, 0, 1]  # a final AM/PM may be missing
    assert _parse(["2024-02-29 1:02:03"], "ymd IMS p")[0][0] is None  # but some value must have one
    out, _ = _parse(["2024-02-29 1:02:03 PM", "2024-02-29 13:02:03", "2024-02-29 13:02"], ["ymd T", "ymd R"])
    assert [(value.hour, value.minute) for value in out] == [(13, 2), (13, 2), (13, 2)]


def test_parse_date_time_offsets_zones_and_daylight_saving():
    zone = "America/New_York"
    out, rows = _parse(["2024-02-29 12:34:56Z", "2024-02-29 12:34:56+05:30", "2024-02-29 12:34:56-05",
                        "2024-02-29 12:34:56"], "ymd HMS z", tz=zone)
    assert out.dtype == pl.Datetime("us", zone)
    assert [value.astimezone(UTC).strftime("%H:%M") if value else None for value in out] == ["12:34", "07:04", "17:34", None]
    assert rows == [4]
    out, rows = _parse(["2024-03-10 02:30:00", "2024-11-03 01:30:00"], "ymd HMS", tz=zone)
    assert out[0] is None and out[1].utcoffset().total_seconds() == -5 * 3600
    assert rows == [1]


def test_parse_date_time_names_cutoff_numbers_and_dates():
    out, _ = _parse(["Feb 29 2024", "february 29 2024", "2024 Sept 29"], ["bdY", "Ybd"])
    assert [value.date() for value in out] == [date(2024, 2, 29), date(2024, 2, 29), date(2024, 9, 29)]
    assert _parse(["24-02-29"], "ymd")[0][0] is None
    assert _parse(["24-02-29", "69-02-28"], "ymd", cutoff=68)[0].dt.year().to_list() == [2024, 1969]
    numbers = (t.tidy({"x": [20240229, 2024022913, None]})
               >> t.mutate(at=t.parse_date_time("x", ["ymd", "ymd H"]))).collect()["at"]
    assert numbers.to_list() == [datetime(2024, 2, 29, tzinfo=UTC), datetime(2024, 2, 29, 13, tzinfo=UTC), None]
    days = (t.tidy({"x": [date(2024, 2, 29)]}) >> t.mutate(at=t.parse_date_time("x", "ymd"))).collect()["at"]
    assert days.to_list() == [datetime(2024, 2, 29, tzinfo=UTC)]


def test_parse_date_time_exact_formats_strict_and_stop_for_problems():
    out, rows = _parse(["2024-02-29 12:34:56", "29/02/2024", "bad"], ["%Y-%m-%d %H:%M:%S", "%d/%m/%Y"], exact=True)
    assert out.to_list() == [datetime(2024, 2, 29, 12, 34, 56, tzinfo=UTC), datetime(2024, 2, 29, tzinfo=UTC), None]
    assert rows == [3]
    frame = t.tidy({"x": ["2024-02-29", "bad"]})
    with pytest.raises(t.ConversionError, match="bad"):
        (frame >> t.mutate(at=t.parse_date_time("x", "ymd", strict=True))).collect()
    with pytest.raises(t.ConversionError, match="'bad' is not datetime"):
        t.stop_for_problems(frame >> t.mutate(at=t.parse_date_time("x", "ymd")))


def test_parse_date_time_validates_orders_before_execution():
    with pytest.raises(ValueError, match="'j'"):
        t.parse_date_time("x", "Yj")
    with pytest.raises(ValueError, match="no year"):
        t.parse_date_time("x", "HMS")
    with pytest.raises(ValueError, match="truncated"):
        t.parse_date_time("x", "ymd HMS", truncated=-1)
    with pytest.raises(TypeError, match="orders"):
        t.parse_date_time("x", [])
    with pytest.raises(TypeError, match="timezone"):
        t.parse_date_time("x", "ymd", tz="")


def test_parse_date_time_plans_stay_lazy(monkeypatch):
    frame = t.tidy({"x": ["2024-01-02 12:34:56"]})

    def no_collect(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("building parse_date_time must not collect rows")

    monkeypatch.setattr(pl.LazyFrame, "collect", no_collect)
    out = frame >> t.mutate(at=t.parse_date_time("x", ["ymd HMS", "dmy HMS"], tz="Asia/Kolkata"),
                            clock=t.parse_time("x", "%Y-%m-%d %H:%M:%S"))
    schema = out._lf.collect_schema()
    assert schema["at"] == pl.Datetime("us", "Asia/Kolkata")
    assert schema["clock"] == pl.Time


def test_common_shapes_skip_the_per_value_rules(monkeypatch):
    from tidy3 import convert

    def per_value(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("used the per-value Python rules")

    for name in ("_per_value", "convert_batch", "_order_candidate", "_time_of_day"):
        monkeypatch.setattr(convert, name, per_value)
    frame = t.tidy({
        "when": ["2024-01-07 13:05:09", "2024/12/31T00:00:01", None],
        "hour": ["2024-01-07 13", "2024-12-31 00", None],
        "clock": ["13:05:09", "1:05", None],
    }) >> t.mutate(
        a=t.parse_date_time("when", ["ymd HMS", "dmy HMS", "ymd IMS p"]),
        b=t.ymd_h("hour"),
        c=t.parse_time("clock"),
    )
    out = frame.collect()
    assert out["a"].null_count() == out["b"].null_count() == out["c"].null_count() == 1
    assert t.problems(frame).collect().height == 0


# ── parse_time ──────────────────────────────────────────────────────────

def _times(values: list[Any], fmt: str = "") -> tuple[list[Any], list[int]]:
    frame = t.tidy({"x": values}) >> t.mutate(at=t.parse_time("x", fmt))
    out = frame.collect()["at"]
    assert out.dtype == pl.Time
    return out.to_list(), t.problems(frame).collect()["row"].to_list()


def test_parse_time_default_format_reads_readr_clocks():
    values, rows = _times(["12:34", "1:02:03", "01:02:03.25", "12:34 PM", "12:00 am", " 7:05pm ",
                           "12:345", "12:34:056", "23:60", "12", "12:34 X", "", None])
    assert values == [time(12, 34), time(1, 2, 3), time(1, 2, 3), time(12, 34), time(0, 0), time(19, 5),
                      time(12, 34, 5), time(12, 34, 56), None, None, None, None, None]
    assert rows == [9, 10, 11, 12]


def test_parse_time_beyond_one_day_is_a_problem():
    # readr returns an hms of a day or more; a time of day cannot hold it.
    values, rows = _times(["24:00", "13:00 PM", "23:59:59"])
    assert values == [None, None, time(23, 59, 59)]
    assert rows == [1, 2]


def test_parse_time_formats():
    assert _times(["07:05:03.25", "7:5:3"], "%H:%M:%OS")[0] == [time(7, 5, 3, 250000), None]
    assert _times(["07:05:03.5", "07:05"], "%H:%M:%S")[0] == [time(7, 5, 3), None]
    assert _times(["07:05", "07:05:03"], "%T")[0] == [time(7, 5), time(7, 5, 3)]
    assert _times(["7:05 pm", "0:05 AM", "12:05 am"], "%I:%M %p")[0] == [time(19, 5), None, time(0, 5)]
    assert _times(["07h05", "0705"], "%H%.%M")[0] == [time(7, 5), None]
    assert _times(["07abc05", "0705"], "%H%*%M")[0] == [time(7, 5), time(7, 5)]
    assert _times(["2024-01-01T07:05"], "%Y-%m-%dT%H:%M")[0] == [time(7, 5)]
    assert _times(["12:34:56 +0100", "12:34:56 UTC"], "%H:%M:%S %z")[0] == [time(12, 34, 56), None]
    assert _times(["24:00", "23:59"], "%H:%M")[0] == [None, time(23, 59)]


def test_parse_time_strict_stop_for_problems_and_validation():
    frame = t.tidy({"x": ["12:34", "noon"]})
    with pytest.raises(t.ConversionError, match="noon"):
        (frame >> t.mutate(at=t.parse_time("x", strict=True))).collect()
    with pytest.raises(t.ConversionError, match="'noon' is not time"):
        t.stop_for_problems(frame >> t.mutate(at=t.parse_time("x")))
    with pytest.raises(ValueError, match="%Q"):
        t.parse_time("x", "%H%Q")
    kept = (t.tidy({"x": [time(1, 2), None]}) >> t.mutate(at=t.parse_time("x"))).collect()["at"]
    assert kept.to_list() == [time(1, 2), None]


def test_pandas_columns_use_the_same_rules():
    from tidy3.convert import eval_pandas

    df = pd.DataFrame({"x": ["12:34", "bad", None]})
    times = eval_pandas("parse_time", (df["x"],), {"format": ""}, df, column="x")
    assert times.tolist() == [time(12, 34), None, None]
    df = pd.DataFrame({"x": ["2024-02-29 12", 20240229, None]}, dtype=object)
    parsed = eval_pandas("parse_date_time", (df["x"],), {"orders": ["ymd H", "ymd"]}, df, column="x")
    assert str(parsed.dtype) == "datetime64[us, UTC]"
    assert parsed.tolist()[:2] == [pd.Timestamp("2024-02-29 12:00", tz="UTC"), pd.Timestamp("2024-02-29", tz="UTC")]
