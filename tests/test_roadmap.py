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
    as_numeric,
    as_string,
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
    mean,
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
    round_half_up,
    round_to_fraction,
    row_means,
    row_sums,
    scan_csv,
    select,
    signif_half_up,
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



def _col(frame, name):
    data = frame.collect()
    return data[name].to_list()


def test_missing_checks_and_math():
    frame = tidy(
        {"x": [1.0, None, float("nan"), float("inf"), -2.5], "hp": [100, 220, 180, None, 90]},
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
    assert _col(out, "nan") == [False, False, True, False, False]
    assert _col(frame >> filter(is_na("x")), "x")[0] is None
    assert _col(out, "capped") == [100, 200, 180, None, 90]
    assert _col(out, "whole")[0] == 1
    assert _col(out, "up")[0] == 1


def test_missing_checks_on_integer_text_and_logical_columns():
    frame = tidy(pl.DataFrame({
        "g": [1, 1, 2],
        "n": [1, None, 3],
        "s": ["a", None, "c"],
        "b": [True, None, False],
    }))
    for name, finite in (("n", [True, False, True]), ("s", [False] * 3), ("b", [False] * 3)):
        out = (frame >> group_by("g") >> mutate(
            missing=is_na(name), nan=is_nan(name), finite=is_finite(name), infinite=is_infinite(name),
        )).collect()
        assert out["missing"].to_list() == [False, True, False]
        assert out["nan"].to_list() == [False] * 3
        assert out["finite"].to_list() == finite
        assert out["infinite"].to_list() == [False] * 3
        assert {out[flag].dtype for flag in ("missing", "nan", "finite", "infinite")} == {pl.Boolean}
    assert (frame >> filter(is_na("s"))).collect()["n"].to_list() == [None]


def test_round_is_half_to_even():
    frame = tidy({"x": [0.5, 1.5, -1.5, 1.25, 1.35, 123.0]})
    out = frame >> mutate(z=round("x"), z1=round("x", 1), tens=round("x", -1))
    assert _col(out, "z") == [0.0, 2.0, -2.0, 1.0, 1.0, 123.0]
    assert _col(out, "z1") == [0.5, 1.5, -1.5, 1.2, 1.4, 123.0]
    tens = _col(out, "tens")
    assert tens[:5] == [0.0, 0.0, 0.0, 0.0, 0.0]
    assert tens[5] == 120


def test_round_half_up_sends_ties_away_from_zero():
    frame = tidy({"x": [0.5, 1.5, 2.5, -2.5, 1.25, 2.675, 25.0, None, float("nan")]})
    out = frame >> mutate(
        z=round_half_up("x"), z2=round_half_up("x", 2), tens=round_half_up("x", -1)
    )
    assert _col(out, "z")[:7] == [1.0, 2.0, 3.0, -3.0, 1.0, 3.0, 25.0]
    # 2.675 is stored as 2.67499999...; janitor's nudge still rounds it up.
    assert _col(out, "z2")[:7] == [0.5, 1.5, 2.5, -2.5, 1.25, 2.68, 25.0]
    assert _col(out, "tens")[:7] == [0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 30.0]
    z = _col(out, "z")
    assert z[7] is None and z[8] != z[8]


def test_round_half_up_keeps_the_column_type():
    frame = tidy({"n": pl.Series([15, 25, -25], dtype=pl.Int32), "f": pl.Series([0.5, 1.5, 2.5], dtype=pl.Float32)})
    out = (frame >> mutate(n0=round_half_up("n"), tens=round_half_up("n", -1), f=round_half_up("f"))).collect()
    assert out["n0"].dtype == pl.Int32 and out["n0"].to_list() == [15, 25, -25]
    assert out["tens"].dtype == pl.Int32 and out["tens"].to_list() == [20, 30, -30]
    assert out["f"].dtype == pl.Float32 and out["f"].to_list() == [1.0, 2.0, 3.0]


def test_round_half_up_around_a_summary_and_its_digits():
    cars = tidy({"cyl": [4, 4, 6], "hp": [93.0, 62.0, 110.0]})
    grouped = cars >> group_by("cyl") >> summarise(m=round_half_up(mean("hp"), -1))
    assert _col(grouped, "m") == [80.0, 110.0]
    for digits in (1.0, True, "1"):
        with pytest.raises(TypeError, match="digits must be an integer"):
            round_half_up("hp", digits)


def test_signif_half_up_keeps_significant_digits_with_ties_up():
    frame = tidy({"x": [12.5, -12.5, 0.00125, 123456.0, 0.0, None, float("inf")],
                  "n": pl.Series([1250, -1250, 7, 0, 99, 5, 1], dtype=pl.Int32)})
    out = (frame >> mutate(s=signif_half_up("x", 2), d=signif_half_up("x"), n2=signif_half_up("n", 2))).collect()
    assert out["s"].to_list() == [13.0, -13.0, 0.0013, 120000.0, 0.0, None, float("inf")]
    assert out["d"].to_list()[:4] == [12.5, -12.5, 0.00125, 123456.0]
    assert out["n2"].dtype == pl.Int32 and out["n2"].to_list() == [1300, -1300, 7, 0, 99, 5, 1]


def test_round_to_fraction_rounds_to_the_nearest_part():
    frame = tidy({"x": [1.3, 2.6, 0.125, 0.375, -1.1, 1 / 3, None]})
    out = (frame >> mutate(
        q=round_to_fraction("x", 4),
        third=round_to_fraction("x", 3, "auto"),
        tenth=round_to_fraction("x", 3, 1),
        inf=round_to_fraction("x", 3, float("inf")),
    )).collect()
    # Exact ties go to the even number of quarters, as R's round() does.
    assert out["q"].to_list() == [1.25, 2.5, 0.0, 0.5, -1.0, 0.25, None]
    assert out["third"].to_list()[:6] == [1.33, 2.67, 0.0, 0.33, -1.0, 0.33]
    assert out["tenth"].to_list()[:6] == [1.3, 2.7, 0.0, 0.3, -1.0, 0.3]
    assert out["inf"].to_list() == (frame >> mutate(r=round_to_fraction("x", 3))).collect()["r"].to_list()


def test_signif_and_fraction_check_their_arguments():
    with pytest.raises(TypeError, match="digits must be an integer"):
        signif_half_up("x", 2.0)
    with pytest.raises(ValueError, match="at least 1"):
        round_to_fraction("x", 0.5)
    with pytest.raises(TypeError, match="denominator must be a number"):
        round_to_fraction("x", True)
    with pytest.raises(TypeError, match="digits must be an integer"):
        round_to_fraction("x", 4, "two")


def test_single_value_carries_a_constant_and_fails_on_a_conflict():
    import tidy3 as t3

    frame = tidy({"g": ["a", "a", "b"], "site": ["N", None, "S"], "code": ["", "x", ""]})
    out = frame >> t3.group_by("g") >> t3.summarise(site=t3.single_value("site"), code=t3.single_value("code", [None, ""]))
    assert _col(out, "site") == ["N", "S"]
    # A group with only missing values gets the first of them: None here.
    assert _col(out, "code") == ["x", None]
    filled = frame >> t3.group_by("g") >> mutate(s=t3.single_value("site"))
    assert _col(filled, "s") == ["N", "N", "S"]
    with pytest.raises(ValueError, match=r"More than one \(2\) value found \(N, S\): sites"):
        (frame >> t3.summarise(s=t3.single_value("site", info="sites"))).collect()
    with pytest.warns(UserWarning, match="All values are missing"):
        (tidy({"v": [None, None]}) >> t3.summarise(v=t3.single_value("v", warn_if_all_missing=True))).collect()
    with pytest.raises(TypeError, match="share one type"):
        t3.single_value("v", [None, "", 0])


def test_row_sums_and_means():
    frame = tidy(
        {"a": [1.0, None, None], "b": [None, None, 4.0], "c": [2.0, None, 6.0]},
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


def test_sum_min_count_is_per_group():
    frame = tidy({"g": ["a", "a", "b"], "x": [1.0, None, 3.0]})
    out = frame >> group_by("g") >> summarise(
        s=sum("x", na_rm=True, min_count=2),
        s1=sum("x", na_rm=True, min_count=1),
    )
    found = {row["g"]: row["s"] for row in out.collect().iter_rows(named=True)}
    assert found == {"a": None, "b": None}
    found = {row["g"]: row["s1"] for row in out.collect().iter_rows(named=True)}
    assert found == {"a": 1.0, "b": 3.0}


def test_across_abs():
    frame = tidy({"a": [-1.0, 2.0], "b": [-3.0, 4.0]})
    out = frame >> mutate(across(where(is_numeric), abs))
    assert _col(out, "a") == [1.0, 2.0]
    assert _col(out, "b") == [3.0, 4.0]


def test_as_integer_refuses_fractions():
    frame = tidy({"x": [2.0, 2.5, None]})
    with pytest.raises(ConversionError, match="2.5") as caught:
        (frame >> mutate(n=as_integer("x"))).collect()
    assert "x" in str(caught.value)
    permissive = frame >> mutate(n=as_integer("x", strict=False))
    assert _col(permissive, "n") == [2, None, None]
    text = tidy({"x": ["2", "2.0", "2.5"]})
    assert _col(text >> mutate(n=as_integer("x", strict=False)), "n") == [2, None, None]
    assert "2.0" in problems(text >> mutate(n=as_integer("x", strict=False))).collect()["original"].to_list()


def test_as_double_logical_and_character():
    numbers = tidy({"x": ["1.5", "1,234", ""]})
    assert _col(numbers >> mutate(n=as_double("x", strict=False)), "n")[0] == 1.5
    assert _col(numbers >> mutate(n=as_double("x", strict=False)), "n")[1:] == [None, None]
    flags = tidy({"x": ["TRUE", "T", "1", "yes", "false"]})
    assert _col(flags >> mutate(n=as_logical("x", strict=False)), "n") == [
        True, True, None, None, False
    ]
    parsed = tidy({"x": ["true", "1", "0", "yes"]})
    assert _col(parsed >> mutate(n=parse_logical("x")), "n") == [True, True, False, None]
    logical = tidy({"x": [True, False]})
    assert _col(logical >> mutate(c=as_character("x")), "c") == ["true", "false"]


def test_parsers_locale_and_problems_are_lazy(monkeypatch):
    frame = tidy({"x": ["$1,234.50", "nope", "1,5"]})
    euro = locale(decimal_mark=",", grouping_mark=".")
    out = frame >> mutate(
        n=parse_number("x"),
        eu=parse_double("x", locale=euro),
    )
    # readr skips grouping marks anywhere in parse_number: "1,5" is 15.
    assert _col(out, "n") == [1234.5, None, 15.0]
    assert _col(out, "eu")[2] == 1.5
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


def test_dates_orders_cutoff_and_rollback():
    text = tidy(
        {
            "a": ["2010-01-31", "20100131"],
            "b": ["01/31/2010", "01/31/2010"],
            "c": ["31-01-2010", "31-01-2010"],
            "d": ["2010-02", "2010-02"],
        },
    )
    parsed = text >> mutate(y=ymd("a"), m=mdy("b"), day=dmy("c"), month=ym("d"))
    assert _col(parsed, "y") == [date(2010, 1, 31), date(2010, 1, 31)]
    assert _col(parsed, "m") == [date(2010, 1, 31), date(2010, 1, 31)]
    assert _col(parsed, "day") == [date(2010, 1, 31), date(2010, 1, 31)]
    assert _col(parsed, "month") == [date(2010, 2, 1), date(2010, 2, 1)]
    short = tidy({"x": ["68-01-01", "69-01-01"]})
    assert _col(short >> mutate(d=ymd("x")), "d") == [None, None]
    assert _col(short >> mutate(d=ymd("x", cutoff=68)), "d") == [
        date(2068, 1, 1),
        date(1969, 1, 1),
    ]
    dated = tidy({"d": [date(2010, 1, 31), date(2012, 2, 29), date(2010, 3, 31)]})
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


def test_floor_and_ceiling_dates():
    frame = tidy(
        {"d": [date(2010, 1, 31), date(2010, 1, 25), date(2010, 2, 3), date(2010, 1, 1)]},
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
    # lubridate rounds a Date up to the next boundary, even from one.
    assert _col(out, "cw") == [
        date(2010, 2, 7),
        date(2010, 1, 31),
        date(2010, 2, 7),
        date(2010, 1, 3),
    ]
    assert _col(out, "cm")[0] == date(2010, 2, 1)
    assert _col(out, "cm")[3] == date(2010, 2, 1)


def test_today_and_now_default_to_utc():
    assert today() == datetime.now(timezone.utc).date()
    local = now("America/Los_Angeles")
    utc = now("UTC")
    delta = (utc - local).total_seconds()
    assert 6 * 3600 < delta < 8 * 3600
    with pytest.raises(ValueError, match="time zone"):
        now("Not/Azone")


def test_parse_date_cutoff():
    frame = tidy({"x": ["68/01/02", "69/01/02"]})
    assert _col(frame >> mutate(d=parse_date("x", "%y/%m/%d")), "d") == [None, None]
    assert _col(frame >> mutate(d=parse_date("x", "%y/%m/%d", cutoff=68)), "d") == [
        date(2068, 1, 2),
        date(1969, 1, 2),
    ]
    stamped = tidy({"x": ["2010-01-31T15:00:00Z"]})
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


# ── regressions from the 0.7.9 review ───────────────────────────────────


def test_parse_number_reads_a_minus_as_readr_does():
    # Checked against readr: a minus counts only right before the number.
    frame = tidy(
        {"price": ["-$1,234", "$-5", "USD -7.5", "Price - 12", "1,23", "1e3", "1.2.3"]},
    )
    assert _col(frame >> mutate(n=parse_number("price")), "n") == [
        None, -5.0, -7.5, None, 123.0, 1000.0, 1.2
    ]


def test_read_csv_trims_before_guessing_types_and_reads_na(tmp_path: Path):
    path = tmp_path / "cars.csv"
    path.write_text("model,mpg,cyl,am\nMazda RX4,21.0, 6,TRUE\nDatsun 710,NA,4 ,false\nValiant, NA , 6,\n")
    cars = read_csv(path).collect()
    assert cars.schema == {"model": pl.String, "mpg": pl.Float64, "cyl": pl.Int64, "am": pl.Boolean}
    assert cars["mpg"].to_list() == [21.0, None, None]
    assert cars["cyl"].to_list() == [6, 4, 6]
    assert cars["am"].to_list() == [True, False, None]
    kept = read_csv(path, col_types={"cyl": "character"}, na=[]).collect()
    assert kept["cyl"].to_list() == ["6", "4", "6"]
    assert kept["mpg"].to_list() == ["21.0", "NA", "NA"]
    # scan_csv keeps Polars' own guessing, as before.
    assert scan_csv(path).collect()["cyl"].dtype == pl.String


def test_years_and_months_roll_back_once():
    # lubridate's add_with_rollback adds 12 * years + months in one step.
    frame = tidy({"d": [date(2024, 2, 29), date(2024, 1, 31)]})
    out = frame >> mutate(
        both=col("d") + (years(1) + months(1)),
        back=col("d") - (years(1) + months(1)),
        rolled=add_with_rollback("d", years(1) + months(1)),
    )
    assert _col(out, "both") == [date(2025, 3, 29), date(2025, 2, 28)]
    assert _col(out, "back") == [date(2023, 1, 29), date(2022, 12, 31)]
    assert _col(out, "rolled") == [date(2025, 3, 29), date(2025, 2, 28)]


def test_months_of_a_column():
    frame = tidy(
        {"d": [date(2024, 1, 31), date(2024, 1, 31), date(2010, 3, 31)], "k": [1, None, -1]},
    )
    out = frame >> mutate(
        plus=col("d") + months(col("k")),
        rolled=add_with_rollback("d", months(col("k"))),
        both=col("d") + (years(col("k")) + months(1)),
    )
    assert _col(out, "plus") == [date(2024, 2, 29), None, date(2010, 2, 28)]
    assert _col(out, "rolled") == [date(2024, 2, 29), None, date(2010, 2, 28)]
    assert _col(out, "both") == [date(2025, 2, 28), None, date(2009, 4, 30)]


def test_week_floor_and_ceiling_start_at_midnight():
    frame = tidy(
        {
            "t": [
                datetime(2026, 10, 21, 15, 30),
                datetime(2026, 10, 18, 9, 0),
                datetime(2026, 10, 18, 0, 0),
            ]
        },
    )
    out = frame >> mutate(f=floor_date("t", "week"), c=ceiling_date("t", "week"))
    assert _col(out, "f") == [datetime(2026, 10, 18)] * 3
    assert _col(out, "c") == [datetime(2026, 10, 25), datetime(2026, 10, 25), datetime(2026, 10, 18)]


_TRICKY = [
    "1", " 2 ", "+3", "-4", "007", "1.", ".5", "-.5", "1e5", "1E-2", ".", "+", "", "  ",
    "nan", "-NAN", "Inf", "+inf", "infinity", "1,234", "1,234.50", "12,34", "1.234,5",
    "0,123", "1 234", "$5", "2.5", "2.0", "9223372036854775807", "9223372036854775808",
    "1e400", "-0", "0x10", "1_000", "²", "abc", "1.2.3", ",5", None,
]


@pytest.mark.parametrize(
    "name, marks",
    [
        ("as_double", None),
        ("as_integer", None),
        ("parse_double", None),
        ("parse_double", (",", ".")),
        ("parse_double", (".", " ")),
    ],
)
@pytest.mark.parametrize(
    "values, dtype",
    [
        (_TRICKY, pl.String),
        ([1.0, 2.5, -0.0, float("nan"), float("inf"), 9.3e18, 2.0**63, None], pl.Float64),
        ([1, -5, 0, None, 2**62], pl.Int64),
        ([True, False, None], pl.Boolean),
    ],
)
def test_native_conversion_matches_the_per_value_path(name, marks, values, dtype):
    from tidy3 import convert

    kwargs = {} if marks is None else {"decimal_mark": marks[0], "grouping_mark": marks[1]}
    options = convert._options(name, kwargs)
    series = pl.Series(values, dtype=dtype)
    fast = convert._fast_convert(series, options)
    assert fast is not None
    expected, problems_found = convert.convert_batch(
        values,
        kind=options["kind"],
        column="x",
        expected="e",
        decimal_mark=options["decimal_mark"],
        grouping_mark=options["grouping_mark"],
    )
    failed = [False] * len(values)
    for row in problems_found:
        failed[row["row"] - 1] = True
    assert fast[1].to_list() == failed
    for got, want in zip(fast[0].to_list(), expected):
        if isinstance(want, float) and want != want:
            assert got != got
        else:
            assert got == want


def test_native_conversion_skips_the_python_loop(monkeypatch):
    from tidy3 import convert

    def boom(*args, **kwargs):
        raise AssertionError("per-value path used")

    monkeypatch.setattr(convert, "convert_batch", boom)
    cars = tidy({"mpg": ["21.0", "22.8", "n/a"], "cyl": ["6", "4", "8"], "disp": ["1234.5", "160", "x"]})
    out = cars >> mutate(
        mpg=as_double("mpg", strict=False),
        cyl=as_integer("cyl"),
        disp=parse_double("disp"),
    )
    data = out.collect()
    assert data["mpg"].to_list() == [21.0, 22.8, None]
    assert data["cyl"].to_list() == [6, 4, 8]
    assert data["disp"].to_list() == [1234.5, 160.0, None]
    found = problems(out).collect()
    assert found["original"].to_list() == ["n/a", "x"]
    assert found["row"].to_list() == [3, 3]
    with pytest.raises(ConversionError, match="'n/a'"):
        (cars >> mutate(mpg=as_double("mpg"))).collect()


def test_unicode_digits_fall_back_to_the_per_value_path():
    # Python reads Arabic-Indic digits; the native casts do not.
    frame = tidy({"x": ["١٢", "3", "bad"]})
    assert _col(frame >> mutate(n=as_double("x", strict=False)), "n") == [12.0, 3.0, None]
    assert _col(frame >> mutate(n=as_integer("x", strict=False)), "n") == [12, 3, None]


def test_parsers_treat_empty_text_as_a_failure_not_a_crash():
    frame = tidy({"x": ["", "  ", "5"]})
    out = frame >> mutate(d=parse_double("x"), i=parse_integer("x"))
    assert _col(out, "d") == [None, None, 5.0]
    assert _col(out, "i") == [None, None, 5]


def test_problems_after_read_csv_do_not_read_the_file_again(tmp_path: Path):
    path = tmp_path / "cars.csv"
    path.write_text("model,hp\nMazda RX4,110\nValiant,abc\n")
    frame = read_csv(path, col_types={"hp": "integer"})
    path.unlink()
    found = problems(frame).collect()
    # readr counts the header line: Valiant, the second data row, is row 3.
    assert found["row"].to_list() == [3]
    assert found["original"].to_list() == ["abc"]


def test_as_integer_treats_a_missing_float_as_missing():
    # pandas stores the missing hp as NaN; that is NA, not a failure.
    frame = tidy({"hp": [110.0, None, 93.0]})
    assert _col(frame >> mutate(n=as_integer("hp")), "n") == [110, None, 93]


def test_helpers_around_a_summary_give_one_value_per_group():
    cars = tidy({"cyl": [4, 4, 6], "hp": [93.0, 62.0, 110.0]})
    whole = cars >> summarise(m=round(mean("hp"), -1), low=pmin(mean("hp"), 80))
    assert _col(whole, "m") == [90.0]
    assert _col(whole, "low") == [80.0]
    grouped = cars >> group_by("cyl") >> summarise(m=round(mean("hp"), -1), a=abs(mean("hp")))
    assert _col(grouped, "m") == [80.0, 110.0]
    assert _col(grouped, "a") == [77.5, 110.0]


def test_helpers_take_constants():
    cars = tidy({"hp": [110.0, 250.0], "cyl": [6, 8]})
    out = cars >> mutate(r=round(2.675, 2), capped=pmin("hp", 200), tens=round("cyl", -1))
    assert _col(out, "r") == [2.67, 2.67]
    assert _col(out, "capped") == [110.0, 200.0]
    assert _col(out, "tens") == [10, 10]
    assert out.collect()["tens"].dtype == pl.Int64


@pytest.mark.parametrize("convert", [as_character, as_string])
def test_text_follows_the_column_type(convert):
    cars = tidy(
        {
            "mpg": [21.0, 0.1 + 0.2, None],
            "cyl": [6, 4, 8],
            "am": [True, False, None],
            "day": [date(2010, 1, 31), None, date(2010, 2, 1)],
            "at": [datetime(2010, 1, 31, 15, 30), datetime(2010, 1, 31, 15, 30, 0, 250000), None],
        },
    )
    out = cars >> mutate(
        mpg=convert("mpg"),
        cyl=convert("cyl"),
        am_py=convert("am", true="True", false="False"),
        am_r=convert("am", true="TRUE", false="FALSE"),
        am=convert("am"),
        day=convert("day"),
        at=convert("at"),
    )
    assert _col(out, "mpg") == ["21.0", "0.30000000000000004", None]
    assert _col(out, "cyl") == ["6", "4", "8"]
    assert _col(out, "am") == ["true", "false", None]
    assert _col(out, "am_py") == ["True", "False", None]
    assert _col(out, "am_r") == ["TRUE", "FALSE", None]
    assert _col(out, "day") == ["2010-01-31", None, "2010-02-01"]
    assert _col(out, "at") == ["2010-01-31 15:30:00", "2010-01-31 15:30:00.250", None]


def test_text_keeps_float32_and_reads_back_exactly():
    data = pl.DataFrame({"wt": pl.Series([22.8, 0.1], dtype=pl.Float32)})
    frame = tidy(data)
    assert _col(frame >> mutate(wt=as_string("wt")), "wt") == ["22.8", "0.1"]
    # An id stored as a float: convert to an integer first for "1001".
    ids = tidy({"id": [1001.0, 1002.0]})
    assert _col(ids >> mutate(id=as_string(as_integer("id"))), "id") == ["1001", "1002"]
    assert _col(ids >> mutate(id=as_string("id")), "id") == ["1001.0", "1002.0"]


def test_text_nan_and_time_zone_on_polars():
    frame = tidy({"x": [float("nan"), float("inf")]})
    assert _col(frame >> mutate(x=as_string("x")), "x") == ["NaN", "inf"]
    zoned = pl.DataFrame({"at": [datetime(2010, 1, 31, 15, 30, tzinfo=timezone.utc)]}).with_columns(
        pl.col("at").dt.convert_time_zone("America/Los_Angeles")
    )
    assert _col(tidy(zoned) >> mutate(at=as_string("at")), "at") == ["2010-01-31 07:30:00-08:00"]


def test_text_of_constants_and_summaries():
    cars = tidy({"cyl": [4, 4, 6], "hp": [93.0, 62.0, 110.0]})
    assert _col(cars >> mutate(k=as_string(5.0)), "k") == ["5.0"] * 3
    grouped = cars >> group_by("cyl") >> summarise(m=as_string(mean("hp")))
    assert _col(grouped, "m") == ["77.5", "110.0"]


def test_aliases_name_themselves_in_errors():
    frame = tidy({"x": ["1.5", "abc"]})
    with pytest.raises(ConversionError, match=r"^as_numeric\(\)"):
        (frame >> mutate(y=as_numeric("x"))).collect()
    with pytest.raises(ConversionError, match=r"^as_double\(\)"):
        (frame >> mutate(y=as_double("x"))).collect()
    with pytest.raises(TypeError, match=r"as_string\(\) true must be a string"):
        as_string("x", true=1)
    with pytest.raises(TypeError, match=r"as_character\(\) false must be a string"):
        as_character("x", false=None)


def test_rowwise_mutate_keeps_aggregates_per_row():
    from tidy3 import c_across, mean, n, rowwise, sum

    frame = tidy(pl.DataFrame({"x": [1.0, 3.0, None], "y": [10, 20, 30]})) >> rowwise()
    out = (frame >> mutate(m=mean("x"), k=n(), total=sum(c_across(["x", "y"])), twice=col("y") * 2)).collect()
    assert out["m"].to_list() == [1.0, 3.0, None]
    assert out["k"].to_list() == [1, 1, 1]
    assert out["total"].to_list() == [11.0, 23.0, 30.0]
    assert out["twice"].to_list() == [20, 40, 60]
