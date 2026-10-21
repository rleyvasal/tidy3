"""Helpers, conversion, dates, parsers, and readers."""

from __future__ import annotations

import zipfile
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl
import pytest

from tidy3 import (
    ConversionError,
    abs,
    across,
    add_with_rollback,
    as_character,
    as_date,
    as_datetime,
    as_double,
    as_integer,
    as_logical,
    ceiling,
    ceiling_date,
    col,
    days_in_month,
    dmy,
    excel_sheets,
    filter,
    floor,
    floor_date,
    group_by,
    is_finite,
    is_infinite,
    is_na,
    is_nan,
    is_numeric,
    locale,
    mdy,
    month,
    months,
    mutate,
    now,
    parse_date,
    parse_double,
    parse_integer,
    parse_logical,
    parse_number,
    pmin,
    problems,
    read_csv,
    read_delim,
    read_excel,
    read_tsv,
    round,
    row_means,
    row_sums,
    scan_csv,
    select,
    stop_for_problems,
    sum,
    summarise,
    tidy,
    today,
    where,
    year,
    years,
    ym,
    ymd,
)

BACKENDS = ("polars", "pandas")


def _col(frame, name):
    data = frame.collect()
    return data[name].to_list()


@pytest.mark.parametrize("backend", BACKENDS)
def test_missing_checks_and_math(backend):
    frame = tidy(
        {"x": [1.0, None, float("nan"), float("inf"), -2.5], "hp": [100, 220, 180, None, 90]},
        backend=backend,
    )
    out = frame >> mutate(
        missing=is_na("x"),
        finite=is_finite("x"),
        infinite=is_infinite("x"),
        nan=is_nan("x"),
        magnitude=abs("x"),
        whole=floor("x"),
        up=ceiling("x"),
        capped=pmin("hp", 200),
    )
    assert _col(out, "missing") == [False, True, True, False, False]
    assert _col(out, "finite") == [True, False, False, False, True]
    assert _col(out, "infinite") == [False, False, False, True, False]
    if backend == "polars":
        # Polars keeps null and NaN apart. pandas float64 stores null as NaN.
        assert _col(out, "nan") == [False, False, True, False, False]
        assert _col(frame >> filter(is_na("x")), "x")[0] is None
    else:
        assert _col(out, "nan") == [False, True, True, False, False]
    assert _col(out, "capped") == [100, 200, 180, None, 90]
    assert _col(out, "whole")[0] == 1
    assert _col(out, "up")[0] == 1


@pytest.mark.parametrize("backend", BACKENDS)
def test_round_is_half_to_even(backend):
    frame = tidy({"x": [0.5, 1.5, -1.5, 1.25, 1.35, 123.0]}, backend=backend)
    out = frame >> mutate(z=round("x"), z1=round("x", 1), tens=round("x", -1))
    assert _col(out, "z") == [0.0, 2.0, -2.0, 1.0, 1.0, 123.0]
    assert _col(out, "z1") == [0.5, 1.5, -1.5, 1.2, 1.4, 123.0]
    tens = _col(out, "tens")
    assert tens[:5] == [0.0, 0.0, 0.0, 0.0, 0.0]
    assert tens[5] == 120


@pytest.mark.parametrize("backend", BACKENDS)
def test_row_sums_and_means(backend):
    frame = tidy(
        {"a": [1.0, None, None], "b": [None, None, 4.0], "c": [2.0, None, 6.0]},
        backend=backend,
    )
    out = frame >> mutate(
        total=row_sums("a", "b", "c"),
        dropped=row_sums("a", "b", "c", na_rm=True),
        counted=row_sums("a", "b", "c", na_rm=True, min_count=1),
        mean=row_means("a", "b", "c", na_rm=True),
    )
    assert _col(out, "total") == [None, None, None]
    assert _col(out, "dropped") == [3.0, 0.0, 10.0]
    assert _col(out, "counted") == [3.0, None, 10.0]
    assert _col(out, "mean") == [1.5, None, 5.0]


@pytest.mark.parametrize("backend", BACKENDS)
def test_sum_min_count_is_per_group(backend):
    frame = tidy({"g": ["a", "a", "b"], "x": [1.0, None, 3.0]}, backend=backend)
    out = frame >> group_by("g") >> summarise(
        s=sum("x", na_rm=True, min_count=2),
        s1=sum("x", na_rm=True, min_count=1),
    )
    found = {row["g"]: row["s"] for row in out.collect().iter_rows(named=True)}
    assert found == {"a": None, "b": None}
    found = {row["g"]: row["s1"] for row in out.collect().iter_rows(named=True)}
    assert found == {"a": 1.0, "b": 3.0}


@pytest.mark.parametrize("backend", BACKENDS)
def test_across_abs(backend):
    frame = tidy({"a": [-1.0, 2.0], "b": [-3.0, 4.0]}, backend=backend)
    out = frame >> mutate(across(where(is_numeric), abs))
    assert _col(out, "a") == [1.0, 2.0]
    assert _col(out, "b") == [3.0, 4.0]


@pytest.mark.parametrize("backend", BACKENDS)
def test_as_integer_refuses_fractions(backend):
    frame = tidy({"x": [2.0, 2.5, None]}, backend=backend)
    with pytest.raises(ConversionError, match="2.5") as caught:
        (frame >> mutate(n=as_integer("x"))).collect()
    assert "x" in str(caught.value)
    permissive = frame >> mutate(n=as_integer("x", strict=False))
    assert _col(permissive, "n") == [2, None, None]
    text = tidy({"x": ["2", "2.0", "2.5"]}, backend=backend)
    assert _col(text >> mutate(n=as_integer("x", strict=False)), "n") == [2, None, None]
    assert "2.0" in problems(text >> mutate(n=as_integer("x", strict=False))).collect()["original"].to_list()


@pytest.mark.parametrize("backend", BACKENDS)
def test_as_double_logical_and_character(backend):
    numbers = tidy({"x": ["1.5", "1,234", ""]}, backend=backend)
    assert _col(numbers >> mutate(n=as_double("x", strict=False)), "n")[0] == 1.5
    assert _col(numbers >> mutate(n=as_double("x", strict=False)), "n")[1:] == [None, None]
    flags = tidy({"x": ["TRUE", "T", "1", "yes", "false"]}, backend=backend)
    assert _col(flags >> mutate(n=as_logical("x", strict=False)), "n") == [
        True, True, None, None, False
    ]
    parsed = tidy({"x": ["true", "1", "0", "yes"]}, backend=backend)
    assert _col(parsed >> mutate(n=parse_logical("x")), "n") == [True, True, False, None]
    logical = tidy({"x": [True, False]}, backend=backend)
    assert _col(logical >> mutate(c=as_character("x")), "c") == ["TRUE", "FALSE"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_parsers_locale_and_problems_are_lazy(backend, monkeypatch):
    frame = tidy({"x": ["$1,234.50", "nope", "1,5"]}, backend=backend)
    euro = locale(decimal_mark=",", grouping_mark=".")
    out = frame >> mutate(
        n=parse_number("x"),
        eu=parse_double("x", locale=euro),
    )
    assert _col(out, "n") == [1234.5, None, None]
    assert _col(out, "eu")[2] == 1.5
    if backend == "polars":
        calls = []
        real = pl.LazyFrame.collect

        def spy(self, *args, **kwargs):
            calls.append(self.collect_schema().names())
            return real(self, *args, **kwargs)

        monkeypatch.setattr(pl.LazyFrame, "collect", spy)
        assert problems(out).columns == ["row", "column", "original", "expected"]
        assert calls == []
    found = problems(out).collect()
    assert set(found.columns) == {"row", "column", "original", "expected"}
    assert "nope" in found["original"].to_list()
    with pytest.raises(ConversionError, match="nope"):
        stop_for_problems(out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_dates_orders_cutoff_and_rollback(backend):
    text = tidy(
        {
            "a": ["2010-01-31", "20100131"],
            "b": ["01/31/2010", "01/31/2010"],
            "c": ["31-01-2010", "31-01-2010"],
            "d": ["2010-02", "2010-02"],
        },
        backend=backend,
    )
    parsed = text >> mutate(y=ymd("a"), m=mdy("b"), day=dmy("c"), month=ym("d"))
    assert _col(parsed, "y") == [date(2010, 1, 31), date(2010, 1, 31)]
    assert _col(parsed, "m") == [date(2010, 1, 31), date(2010, 1, 31)]
    assert _col(parsed, "day") == [date(2010, 1, 31), date(2010, 1, 31)]
    assert _col(parsed, "month") == [date(2010, 2, 1), date(2010, 2, 1)]
    short = tidy({"x": ["68-01-01", "69-01-01"]}, backend=backend)
    assert _col(short >> mutate(d=ymd("x")), "d") == [None, None]
    assert _col(short >> mutate(d=ymd("x", cutoff=68)), "d") == [
        date(2068, 1, 1),
        date(1969, 1, 1),
    ]
    dated = tidy({"d": [date(2010, 1, 31), date(2012, 2, 29), date(2010, 3, 31)]}, backend=backend)
    moved = dated >> mutate(
        plus=col("d") + months(1),
        rolled=add_with_rollback("d", years(1)),
        minus=col("d") - months(1),
        y=year("d"),
        m=month("d"),
        n=days_in_month("d"),
    )
    assert _col(moved, "plus") == [date(2010, 2, 28), date(2012, 3, 29), date(2010, 4, 30)]
    assert _col(moved, "rolled")[1] == date(2013, 2, 28)
    assert _col(moved, "minus")[2] == date(2010, 2, 28)
    assert _col(moved, "y") == [2010, 2012, 2010]
    assert _col(moved, "n")[1] == 29
    with pytest.raises(TypeError, match="cannot subtract"):
        (dated >> mutate(bad=months(1) - col("d"))).collect()


@pytest.mark.parametrize("backend", BACKENDS)
def test_floor_and_ceiling_dates(backend):
    frame = tidy(
        {"d": [date(2010, 1, 31), date(2010, 1, 25), date(2010, 2, 3), date(2010, 1, 1)]},
        backend=backend,
    )
    out = frame >> mutate(
        fw=floor_date("d", "week"),
        cw=ceiling_date("d", "week"),
        cm=ceiling_date("d", "month"),
    )
    assert _col(out, "fw") == [
        date(2010, 1, 31),
        date(2010, 1, 24),
        date(2010, 1, 31),
        date(2009, 12, 27),
    ]
    assert _col(out, "cw") == [
        date(2010, 1, 31),
        date(2010, 1, 31),
        date(2010, 2, 7),
        date(2010, 1, 3),
    ]
    assert _col(out, "cm")[0] == date(2010, 2, 1)
    assert _col(out, "cm")[3] == date(2010, 1, 1)


def test_today_and_now_default_to_utc():
    assert today() == datetime.now(timezone.utc).date()
    local = now("America/Los_Angeles")
    utc = now("UTC")
    delta = (utc - local).total_seconds()
    assert 6 * 3600 < delta < 8 * 3600
    with pytest.raises(ValueError, match="time zone"):
        now("Not/Azone")


@pytest.mark.parametrize("backend", BACKENDS)
def test_parse_date_cutoff(backend):
    frame = tidy({"x": ["68/01/02", "69/01/02"]}, backend=backend)
    assert _col(frame >> mutate(d=parse_date("x", "%y/%m/%d")), "d") == [None, None]
    assert _col(frame >> mutate(d=parse_date("x", "%y/%m/%d", cutoff=68)), "d") == [
        date(2068, 1, 2),
        date(1969, 1, 2),
    ]
    stamped = tidy({"x": ["2010-01-31T15:00:00Z"]}, backend=backend)
    assert _col(stamped >> mutate(d=as_datetime("x")), "d") == [datetime(2010, 1, 31, 15, 0)]
    assert _col(stamped >> mutate(d=as_date("x", strict=False)), "d") == [None]


def test_read_csv_types_id_and_problems(tmp_path: Path, monkeypatch):
    folder = tmp_path / "files"
    folder.mkdir()
    (folder / "a.csv").write_text("#c\nid,v,when\n001,2.5,2010-01-31\n 002 ,3,bad\n")
    (folder / "b.csv").write_text("#c\nid,v,when\n003,4,2010-02-01\n")
    frame = read_csv(
        str(folder / "*.csv"),
        col_types={"id": "character", "v": "integer", "when": "date"},
        comment="#",
        id="source",
    )
    collected = frame.collect()
    assert collected["id"].to_list() == ["001", "002", "003"]
    assert collected["v"].to_list() == [None, 3, 4]
    assert collected["when"].to_list()[0] == date(2010, 1, 31)
    assert collected["source"].to_list()[0].endswith("a.csv")
    calls = []
    real = pl.LazyFrame.collect

    def spy(self, *args, **kwargs):
        calls.append(self.collect_schema().names())
        return real(self, *args, **kwargs)

    monkeypatch.setattr(pl.LazyFrame, "collect", spy)
    assert problems(frame).columns == ["row", "column", "original", "expected"]
    assert calls == []
    monkeypatch.setattr(pl.LazyFrame, "collect", real)
    found = problems(frame).collect()
    assert found["original"].to_list() == ["2.5", "bad"]
    with pytest.raises(TypeError, match="not both"):
        scan_csv(folder / "a.csv", na="NA", null_values="NA")
    with pytest.raises(ValueError, match="read_csv"):
        scan_csv(folder / "a.csv", encoding="latin-1")


def test_read_delim_tsv_and_latin1(tmp_path: Path):
    path = tmp_path / "wide.txt"
    path.write_text("a|b\n1|2\n")
    assert read_delim(path, "|").collect()["b"].to_list() == [2]
    tsv = tmp_path / "tab.tsv"
    tsv.write_text("a\tb\n3\t4\n")
    assert read_tsv(tsv).collect()["a"].to_list() == [3]
    encoded = tmp_path / "latin.csv"
    encoded.write_bytes("name\ncafé\n".encode("latin-1"))
    assert read_csv(encoded, encoding="latin-1").collect()["name"].to_list() == ["café"]


def test_excel_extra_is_optional(tmp_path: Path):
    path = tmp_path / "book.xlsx"
    tidy({"note": ["=1+1"]}).write_excel(path)
    try:
        import fastexcel  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match=r"tidy3\[excel\]"):
            read_excel(path)
        with pytest.raises(ImportError, match=r"tidy3\[excel\]"):
            excel_sheets(path)
        return
    assert "Sheet1" in excel_sheets(path) or excel_sheets(path)
    assert read_excel(path).collect()["note"].to_list() == ["=1+1"]
