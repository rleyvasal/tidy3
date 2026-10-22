"""Timezone expressions preserve instants or clocks, including DST policies."""

from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd
import polars as pl
import pytest

import tidy3 as t




def test_with_tz_preserves_instant_while_force_tz_preserves_clock():
    zone = ZoneInfo("America/New_York")
    values = [datetime(2024, 1, 2, 12, 34, 56, 125000, tzinfo=zone),
              datetime(2024, 7, 2, 12, 34, 56, 125000, tzinfo=zone), None]
    out = (t.tidy({"x": values}) >> t.mutate(
        displayed=t.with_tz("x", "Asia/Kathmandu"),
        forced=t.force_tz("x", "Asia/Kathmandu"),
    )).collect()
    assert out["displayed"].cast(pl.Int64).to_list() == out["x"].cast(pl.Int64).to_list()
    assert [v.hour if v else None for v in out["displayed"]] == [23, 22, None]
    assert [v.hour if v else None for v in out["forced"]] == [12, 12, None]
    assert out["forced"][0].astimezone(timezone.utc) == datetime(2024, 1, 2, 6, 49, 56, 125000, tzinfo=timezone.utc)
    assert out["forced"].dtype.time_zone == out["displayed"].dtype.time_zone == "Asia/Kathmandu"


def test_naive_and_date_inputs_have_explicit_utc_rules():
    values = [datetime(2024, 1, 2, 12, 34, 56), None]
    out = (t.tidy({"x": values}) >> t.mutate(
        displayed=t.with_tz("x", "America/New_York"), forced=t.force_tz("x", "America/New_York")
    )).collect()
    assert out["displayed"][0].hour == 7
    assert out["forced"][0].hour == 12
    dates = (t.tidy({"x": [date(2024, 1, 2), None]}) >> t.mutate(
        displayed=t.with_tz("x", "America/New_York"), forced=t.force_tz("x", "America/New_York")
    )).collect()
    assert dates["displayed"][0].isoformat() == "2024-01-01T19:00:00-05:00"
    assert dates["forced"][0].isoformat() == "2024-01-02T00:00:00-05:00"


@pytest.mark.parametrize("policy,expected", [
    ("NA", [None, None]),
    ("pre", ["2024-03-10T01:30:00-05:00", "2024-11-03T01:30:00-04:00"]),
    ("post", ["2024-03-10T03:30:00-04:00", "2024-11-03T01:30:00-05:00"]),
    ("boundary", ["2024-03-10T03:00:00-04:00", "2024-11-03T01:00:00-05:00"]),
    (("NA", "post"), [None, "2024-11-03T01:30:00-05:00"]),
    (("post", "pre"), ["2024-03-10T03:30:00-04:00", "2024-11-03T01:30:00-04:00"]),
])
def test_force_tz_dst_policies(policy, expected):
    frame = t.tidy({"x": [datetime(2024, 3, 10, 2, 30), datetime(2024, 11, 3, 1, 30), None]})
    out = (frame >> t.mutate(at=t.force_tz("x", "America/New_York", roll_dst=policy))).collect()["at"]
    assert [v.isoformat() if v else None for v in out] == expected + [None]


def test_resolved_repeats_preserve_occurrence_even_with_a_roll_policy():
    zone = ZoneInfo("America/Toronto")
    values = [datetime(2024, 11, 3, 1, 30, 0, 125000, tzinfo=zone, fold=fold) for fold in [0, 1]]
    frame = t.tidy({"x": values})
    for policy in ["NA", "pre", "post", "boundary"]:
        for target in ["America/Toronto", "America/New_York"]:
            out = (frame >> t.mutate(at=t.force_tz("x", target, roll_dst=policy))).collect()
            assert out["at"].cast(pl.Int64).to_list() == out["x"].cast(pl.Int64).to_list()


@pytest.mark.parametrize("unit,scale,remainder", [("ms", 1000, 125), ("us", 1_000_000, 123456), ("ns", 1_000_000_000, 123456789)])
def test_datetime_precision_survives_conversion_and_dst_rolls(unit, scale, remainder):
    base = int(datetime(2024, 3, 10, 2, 30, tzinfo=timezone.utc).timestamp()) * scale
    source = pl.Series("x", [base + remainder, None], dtype=pl.Datetime(unit, "UTC")).to_frame()
    out = (t.tidy(source) >> t.mutate(
        shown=t.with_tz("x", "America/New_York"),
        pre=t.force_tz("x", "America/New_York", roll_dst="pre"),
        post=t.force_tz("x", "America/New_York", roll_dst="post"),
        boundary=t.force_tz("x", "America/New_York", roll_dst="boundary"),
        mixed=t.force_tz("x", "America/New_York", roll_dst=("boundary", "pre")),
    )).collect()
    assert out["shown"].dtype == pl.Datetime(unit, "America/New_York")
    assert out["shown"].cast(pl.Int64).to_list() == [base + remainder, None]
    assert out["pre"].cast(pl.Int64).to_list() == [base + remainder + 4 * 3600 * scale, None]
    assert out["post"].cast(pl.Int64).to_list() == [base + remainder + 5 * 3600 * scale, None]
    assert out["boundary"].cast(pl.Int64).to_list() == [base + (4 * 3600 + 30 * 60) * scale, None]
    assert out["mixed"].cast(pl.Int64).to_list() == out["boundary"].cast(pl.Int64).to_list()


@pytest.mark.parametrize("values", [[], [None, None]])
def test_empty_and_null_columns_keep_the_target_zone(values):
    frame = t.tidy(pl.Series("x", values, dtype=pl.Datetime("us", "UTC")).to_frame())
    out = (frame >> t.mutate(shown=t.with_tz("x", "Asia/Kathmandu"), forced=t.force_tz("x", "Asia/Kathmandu"))).collect()
    for name in ["shown", "forced"]:
        assert out[name].dtype == pl.Datetime("us", "Asia/Kathmandu")
        assert out[name].null_count() == len(values)


def test_grouped_nested_across_and_duplicate_indexes():
    source = pd.DataFrame({"g": ["a", "a", "b"], "x": pd.to_datetime(["2024-01-02 12:34:56", None, "2024-07-02 12:34:56"], utc=True)}, index=[1, 1, 0])
    frame = t.tidy(source)
    summary = frame >> t.group_by("g") >> t.summarise(at=t.with_tz(t.first("x"), "America/New_York"))
    assert [v.hour for v in summary.collect()["at"]] == [7, 8]
    across = frame >> t.mutate(t.across("x", lambda x: t.force_tz(x, "Asia/Kathmandu")))
    assert [v.hour if v else None for v in across.collect()["x"]] == [12, None, 12]
    nested = frame >> t.mutate(h=t.hour(t.with_tz("x", "America/New_York")))
    assert nested.collect()["h"].to_list() == [7, None, 8]


def test_timezone_plans_do_not_read_source_rows_to_discover_dtype():
    calls = []
    def source(batch):
        calls.append(batch.height)
        return batch
    lazy = pl.DataFrame({"x": [datetime(2024, 1, 2)]}).lazy().map_batches(source, schema={"x": pl.Datetime("us")})
    out = t.tidy(lazy) >> t.mutate(shown=t.with_tz("x", "Asia/Kathmandu"), forced=t.force_tz("x", "America/New_York"))
    assert out._lf.collect_schema()["shown"] == pl.Datetime("us", "Asia/Kathmandu")
    assert calls == []
    assert out.collect()["shown"][0].hour == 5
    assert calls == [1]


def test_timezone_options_are_validated_at_construction():
    for helper in [t.with_tz, t.force_tz]:
        for invalid in [None, "", 1]:
            with pytest.raises(TypeError, match="timezone"):
                helper("x", invalid)
        with pytest.raises(ZoneInfoNotFoundError):
            helper("x", "Not/AZone")
    for invalid in ["xfirst", "xlast", (), ("NA", "post", "pre"), ("NA", None)]:
        with pytest.raises(ValueError, match="roll_dst"):
            t.force_tz("x", roll_dst=invalid)
    with pytest.raises(TypeError, match="roll_dst"):
        t.force_tz("x", roll_dst=None)


def test_timezone_helpers_require_temporal_columns():
    for helper in [t.with_tz, t.force_tz]:
        with pytest.raises((TypeError, pl.exceptions.InvalidOperationError)):
            (t.tidy({"x": ["2024-01-02"]}) >> t.mutate(at=helper("x"))).collect()
