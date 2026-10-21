"""Lubridate component accessors on dates, datetimes, and grouped data."""

from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd
import polars as pl
import pytest

import tidy3 as t

BACKENDS = ["polars", "pandas"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_calendar_components_and_iso_year_boundaries(backend):
    frame = t.tidy({"d": [date(2020, 12, 31), date(2021, 1, 1), date(2021, 1, 4), date(2024, 2, 29), None]}, backend=backend)
    out = (frame >> t.mutate(
        day=t.yday("d"), week=t.week("d"), iso=t.isoweek("d"),
        sunday=t.wday("d"), monday=t.wday("d", week_start=1), q=t.quarter("d"),
    )).collect()
    assert out["day"].to_list() == [366, 1, 4, 60, None]
    assert out["week"].to_list() == [53, 1, 1, 9, None]
    assert out["iso"].to_list() == [53, 53, 1, 9, None]
    assert out["sunday"].to_list() == [5, 6, 2, 5, None]
    assert out["monday"].to_list() == [4, 5, 1, 4, None]
    assert out["q"].to_list() == [4, 1, 1, 1, None]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("start", range(1, 8))
def test_weekday_labels_order_and_unused_levels(backend, start):
    frame = t.tidy({"d": [date(2024, 1, 1), None]}, backend=backend)
    frame = frame >> t.mutate(day=t.wday("d", label=True, abbr=False, week_start=start))
    day = frame.collect(as_="pandas")["day"]
    names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    assert day.cat.categories.tolist() == names[start - 1:] + names[:start - 1]
    assert day.iloc[0] == "Monday"
    assert pd.isna(day.iloc[1])
    assert day.cat.ordered
    assert (frame >> t.count("day", drop=False)).collect().height == 8


@pytest.mark.parametrize("backend", BACKENDS)
def test_weekday_labels_replace_previous_factor_metadata(backend):
    frame = t.tidy(pd.DataFrame({
        "day": pd.Categorical(["old"], categories=["old", "unused"]),
        "date": [date(2024, 1, 1)],
    }), backend=backend)
    frame = frame >> t.mutate(day=t.wday("date", label=True))
    labels = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
    assert frame._category_levels["day"] == labels
    counted = (frame >> t.count("day", drop=False)).collect()
    assert counted["day"].to_list() == labels
    assert counted["n"].to_list() == [0, 1, 0, 0, 0, 0, 0]


@pytest.mark.parametrize("backend", BACKENDS)
def test_quarter_types_and_fiscal_ending_year(backend):
    frame = t.tidy({"d": [date(2024, 1, 1), date(2024, 11, 1), date(2024, 12, 31), None]}, backend=backend)
    out = (frame >> t.mutate(
        q=t.quarter("d", fiscal_start=11),
        year=t.quarter("d", type="year.quarter", fiscal_start=11),
        first=t.quarter("d", type="date_first", fiscal_start=11),
        last=t.quarter("d", type="date_last", fiscal_start=11),
        description=t.quarter("d", type="year_start/end", fiscal_start=11),
    )).collect()
    assert out["q"].to_list() == [1, 1, 1, None]
    assert out["year"].to_list() == [2024.1, 2025.1, 2025.1, None]
    assert out["first"].to_list() == [date(2023, 11, 1), date(2024, 11, 1), date(2024, 11, 1), None]
    assert out["last"].to_list() == [date(2024, 1, 31), date(2025, 1, 31), date(2025, 1, 31), None]
    assert out["description"].to_list() == ["2023/24 Q1", "2024/25 Q1", "2024/25 Q1", "NA/NA QNA"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_clock_fields_preserve_local_time_and_fractional_seconds(backend):
    zone = ZoneInfo("America/Los_Angeles")
    frame = t.tidy({"d": [
        datetime(2024, 3, 10, 3, 4, 56, 123456, tzinfo=zone),
        datetime(2024, 1, 1, 23, 59, 59, 500000, tzinfo=zone), None,
    ]}, backend=backend)
    out = (frame >> t.mutate(h=t.hour("d"), m=t.minute("d"), s=t.second("d"), weekday=t.wday("d"))).collect()
    assert out["h"].to_list() == [3, 23, None]
    assert out["m"].to_list() == [4, 59, None]
    assert out["s"].to_list() == pytest.approx([56.123456, 59.5, None])
    assert out["weekday"].to_list() == [1, 2, None]


@pytest.mark.parametrize("backend", BACKENDS)
def test_date_clock_fields_are_zero_and_missing_stays_missing(backend):
    frame = t.tidy({"d": [date(2024, 1, 1), None]}, backend=backend)
    out = (frame >> t.mutate(h=t.hour("d"), m=t.minute("d"), s=t.second("d"))).collect()
    assert out["h"].to_list() == [0, None]
    assert out["m"].to_list() == [0, None]
    assert out["s"].to_list() == [0., None]


@pytest.mark.parametrize("backend", BACKENDS)
def test_components_in_grouped_summaries_across_and_filter(backend):
    frame = t.tidy({"g": ["a", "a", "b"], "d": [date(2024, 1, 1), date(2024, 2, 29), date(2024, 7, 1)]}, backend=backend)
    grouped = frame >> t.group_by("g") >> t.summarise(day=t.yday(t.first("d")), weekday=t.wday(t.first("d"), label=True))
    assert grouped.collect()["day"].to_list() == [1, 183]
    assert grouped.collect()["weekday"].to_list() == ["Mon", "Mon"]
    out = frame >> t.mutate(t.across("d", t.quarter))
    assert out.collect()["d"].to_list() == [1, 1, 3]
    assert (frame >> t.filter(t.yday("d") > 100)).collect()["g"].to_list() == ["b"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_empty_temporal_columns(backend):
    frame = t.tidy(pl.DataFrame({"d": pl.Series([], dtype=pl.Datetime("ns"))}), backend=backend)
    out = frame >> t.mutate(w=t.wday("d", label=True), s=t.second("d"), q=t.quarter("d"))
    assert out.collect(as_="pandas").empty
    assert len(out.collect(as_="pandas")["w"].cat.categories) == 7


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("values", [[], [None, None]])
def test_empty_and_missing_quarter_boundaries_stay_dates(backend, values):
    frame = t.tidy(pl.DataFrame({"d": pl.Series(values, dtype=pl.Date)}), backend=backend)
    out = (frame >> t.mutate(first=t.quarter("d", type="date_first"), last=t.quarter("d", type="date_last"))).collect()
    assert out.schema["first"] == pl.Date
    assert out.schema["last"] == pl.Date
    assert out["first"].to_list() == values


@pytest.mark.parametrize("backend", BACKENDS)
def test_fractional_seconds_include_nanoseconds(backend):
    source = pd.DataFrame({"d": pd.to_datetime(["2024-01-01 12:34:56.123456789", None])})
    out = (t.tidy(source, backend=backend) >> t.mutate(s=t.second("d"))).collect()
    assert out["s"][0] == pytest.approx(56.123456789, abs=1e-12)
    assert out["s"][1] is None


def test_component_expressions_stay_lazy(monkeypatch):
    frame = t.tidy({"d": [datetime(2024, 1, 1, 12, 34, 56)]})
    def no_collect(*args, **kwargs):
        raise AssertionError("building date components must not collect rows")
    monkeypatch.setattr(pl.LazyFrame, "collect", no_collect)
    out = frame >> t.mutate(w=t.wday("d", label=True), y=t.yday("d"), week=t.week("d"), iso=t.isoweek("d"), q=t.quarter("d", type="date_last"), h=t.hour("d"), m=t.minute("d"), s=t.second("d"))
    assert out._lf.collect_schema()["w"] == pl.Enum(["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"])


def test_invalid_component_options():
    with pytest.raises(ValueError, match="week_start"):
        t.wday("d", week_start=0)
    with pytest.raises(TypeError, match="boolean"):
        t.wday("d", label="yes")
    with pytest.raises(ValueError, match="fiscal_start"):
        t.quarter("d", fiscal_start=13)
    with pytest.raises(ValueError, match="type"):
        t.quarter("d", type="unknown")


def test_component_names_still_work_as_notebook_columns():
    source = "frame >> mutate(copy=hour, extracted=hour(stamp))"
    namespace = {name: getattr(t, name) for name in t.__all__}
    namespace["frame"] = t.tidy({"hour": [99], "stamp": [datetime(2024, 1, 1, 12)]})
    exec("result = " + t.transform_source(source).strip(), namespace)
    assert namespace["result"].collect()["copy"].to_list() == [99]
    assert namespace["result"].collect()["extracted"].to_list() == [12]
