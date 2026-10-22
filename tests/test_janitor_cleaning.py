"""Janitor cleaning helpers on both backends.

Row-only ``remove_empty`` stays lazy. Dropping columns, constant checks, and
``row_to_names`` read the current values. Blank strings are kept. Nothing
here prints janitor's status lines.
"""

from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd
import polars as pl
import pytest

import tidy3 as t
from tidy3.expr import Expr


UTC = ZoneInfo("UTC")


def _clock(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    return value


def test_remove_empty_keeps_blanks_and_drops_missing(capsys):
    frame = t.tidy(
        {
            "a": [1.0, None, 2.0, None],
            "b": [None, None, None, None],
            "c": ["x", None, "", "y"],
            "d": [None, None, 1.0, None],
        },
    )
    out = (frame >> t.remove_empty()).collect()
    assert capsys.readouterr().out == ""
    assert out.columns == ["a", "c", "d"]
    assert out["a"].to_list() == [1.0, 2.0, None]
    assert out["c"].to_list() == ["x", "", "y"]
    assert out["d"].to_list() == [None, 1.0, None]
    rows = frame >> t.remove_empty("rows", cutoff=0.5)
    assert rows.collect()["a"].to_list() == [2.0]
    cols = (frame >> t.remove_empty("cols", cutoff=0.5)).collect()
    assert cols.columns == ["c"]
    assert cols["c"].to_list() == ["x", None, "", "y"]
    assert frame.remove_empty("cols").columns == ["a", "c", "d"]


def test_remove_empty_cutoff_rules_and_zero_frames():
    with pytest.raises(ValueError, match="only one"):
        t.remove_empty(cutoff=0.5)
    with pytest.raises(ValueError, match="cutoff"):
        t.remove_empty(cutoff=0)
    empty = t.tidy(pd.DataFrame())
    with pytest.raises(ValueError, match="at least one column"):
        empty >> t.remove_empty("rows", cutoff=0.5)
    no_rows = t.tidy(
        pd.DataFrame({"a": pd.Series([], dtype="float64")})
    )
    with pytest.raises(ValueError, match="at least one row"):
        no_rows >> t.remove_empty("cols", cutoff=0.5)
    dropped = no_rows >> t.remove_empty("cols")
    assert dropped.shape == (0, 0)
    blank = t.tidy({"a": ["", ""], "b": [None, None]})
    assert (blank >> t.remove_empty()).collect()["a"].to_list() == ["", ""]


def test_remove_constant_na_and_factors():
    frame = t.tidy(
        pd.DataFrame(
            {
                "mix": pd.Series([1.0, 1.0, None], dtype="float64"),
                "vary": pd.Series([1.0, 2.0, 1.0], dtype="float64"),
                "const": pd.Series([5.0, 5.0, 5.0], dtype="float64"),
                "allna": pd.Series([None, None, None], dtype="float64"),
            }
        ),
    )
    assert (frame >> t.remove_constant()).columns == ["mix", "vary"]
    assert frame.remove_constant(na_rm=True).columns == ["vary"]
    source = pd.DataFrame(
        {
            "only": pd.Categorical(["a", "a"], categories=["a", "b", "c"]),
            "both": pd.Categorical(
                ["a", "b"], categories=["a", "b", "c"], ordered=True
            ),
        }
    )
    kept = t.tidy(source) >> t.remove_constant()
    assert kept.columns == ["both"]
    assert kept._category_levels["both"] == ["a", "b", "c"]
    assert kept._category_levels["both"].ordered is True
    assert "only" not in kept._category_levels


def test_remove_constant_all_missing_nan_follows_the_backend():
    frame = t.tidy(pl.DataFrame({"a": [float("nan"), None], "b": [1.0, 2.0]}))
    assert (frame >> t.remove_constant()).columns == ["a", "b"]
    assert (frame >> t.remove_constant(na_rm=True)).columns == ["b"]


def test_dropping_every_column_keeps_pandas_rows_only():
    emptied = t.tidy({"a": [1, 1], "b": ["x", "x"]}) >> t.remove_constant()
    removed = (
        t.tidy({"a": [None, None], "b": [None, None]})
        >> t.remove_empty("cols")
    )
    assert emptied.shape == (0, 0)
    assert removed.shape == (0, 0)


def test_cleaning_preserves_groups_levels_and_problems():
    source = pd.DataFrame(
        {
            "g": pd.Categorical(
                ["b", "a"], categories=["a", "b", "unused"], ordered=True
            ),
            "x": [1.0, 2.0],
            "empty": [None, None],
            "raw": ["1", "bad"],
        }
    )
    frame = (
        t.tidy(source)
        >> t.group_by("g", drop=False)
        >> t.mutate(parsed=t.parse_integer("raw"))
    )
    out = frame >> t.remove_empty("cols")
    assert out.columns == ["g", "x", "raw", "parsed"]
    assert out._groups == ["g"]
    assert out._group_drop is False
    assert out._rowwise is False
    assert out._category_levels["g"] == ["a", "b", "unused"]
    assert out._category_levels["g"].ordered is True
    assert t.problems(out).collect()["original"].to_list() == ["bad"]
    grouped = t.tidy({"g": ["a", "a"], "x": [1, 2]}) >> t.group_by(
        "g", "x", drop=False
    )
    dropped = grouped >> t.remove_constant()
    assert dropped.columns == ["x"]
    assert dropped._groups == ["x"]
    assert dropped._group_drop is False
    rowwise = t.tidy({"a": [1.0, None], "b": ["x", "y"]}) >> t.rowwise()
    assert (rowwise >> t.remove_empty("rows"))._rowwise is True


def test_remove_empty_rows_stays_lazy(monkeypatch):
    frame = t.tidy({"a": [1.0, None], "b": [None, "x"]})

    def no_collect(*args, **kwargs):
        raise AssertionError("remove_empty('rows') must not collect")

    monkeypatch.setattr(pl.LazyFrame, "collect", no_collect)
    out = frame >> t.remove_empty("rows")
    assert out.columns == ["a", "b"]


def test_compare_df_cols_match_rules_and_order(capsys):
    old = t.tidy(
        pd.DataFrame(
            {
                "Z": pd.Series([1.5], dtype="float64"),
                "A": pd.Series([2.5], dtype="float64"),
                "B": pd.Series([1], dtype="int64"),
            }
        ),
    )
    new = t.tidy(
        pd.DataFrame(
            {
                "M": pd.Series([3.5], dtype="float64"),
                "A": pd.Series([4.5], dtype="float64"),
                "B": pd.Series([5.0], dtype="float64"),
                "C": pd.Series(["x"], dtype="string"),
            }
        ),
    )
    compared = t.compare_df_cols(old=old, new=new).collect()
    assert capsys.readouterr().out == ""
    assert compared.columns == ["column_name", "old", "new"]
    assert compared["column_name"].to_list() == ["A", "B", "C", "M", "Z"]
    assert compared["old"].to_list() == ["numeric", "integer", None, None, "numeric"]
    assert compared["new"].to_list() == ["numeric", "numeric", "character", "numeric", None]
    matched = t.compare_df_cols(old=old, new=new, return_="match").collect()
    assert matched["column_name"].to_list() == ["A", "C", "M", "Z"]
    mismatched = t.compare_df_cols(
        old=old, new=new, return_="mismatch", bind_method="rbind"
    ).collect()
    assert mismatched["column_name"].to_list() == ["B", "C", "M", "Z"]
    one = old.compare_df_cols().collect()
    assert one.columns == ["column_name", "..1"]
    assert one["column_name"].to_list() == ["Z", "A", "B"]
    listed = t.compare_df_cols(pair=[old, new])
    assert listed.columns == ["column_name", "pair_1", "pair_2"]
    with pytest.raises(ValueError, match="column_name"):
        t.compare_df_cols(column_name=old)
    with pytest.warns(UserWarning, match="zero columns"):
        empty = t.compare_df_cols(old=t.tidy(pd.DataFrame()), new=new)
    assert empty.columns == ["column_name", "new"]
    assert t.compare_df_cols_same(
        t.tidy({"A": [1.0]}),
        t.tidy({"B": [2.0]}),
        verbose=True,
    ) is True
    assert capsys.readouterr().out == ""
    assert (
        t.compare_df_cols_same(
            t.tidy({"A": [1.0]}),
            t.tidy({"B": [2.0]}),
            bind_method="rbind",
            verbose=False,
        )
        is False
    )
    with pytest.warns(UserWarning, match="Only one data frame"):
        assert t.compare_df_cols_same(one=old, verbose=False) is False
    with pytest.warns(UserWarning, match="zero columns"):
        assert (
            t.compare_df_cols_same(
                t.tidy(pd.DataFrame()),
                t.tidy(pd.DataFrame()),
                verbose=False,
            )
            is True
        )


def test_compare_df_cols_strict_factors_and_describe_class():
    fac = t.tidy(
        pd.DataFrame(
            {
                "f": pd.Categorical(
                    ["y", "x"], categories=["y", "x"], ordered=True
                )
            }
        ),
    )
    strict = t.compare_df_cols(fac=fac, strict_description=True).collect()
    loose = t.compare_df_cols(fac=fac).collect()
    assert strict["fac"].to_list() == ['ordered, factor(levels=c("y", "x"))']
    assert loose["fac"].to_list() == ["factor"]
    dates = t.tidy({"d": [date(2020, 1, 2), None]})
    assert t.compare_df_cols(dates=dates).collect()["dates"].to_list() == ["Date"]
    assert t.describe_class(1) == "integer"
    assert t.describe_class(1.0) == "numeric"
    assert t.describe_class(True) == "logical"
    assert t.describe_class("a") == "character"
    assert t.describe_class(date(2020, 1, 2)) == "Date"
    assert t.describe_class(datetime(2020, 1, 2, 3, 4)) == "POSIXct, POSIXt"
    assert t.describe_class(None) == "logical"
    with pytest.raises(TypeError, match="not a frame"):
        t.describe_class(fac)


def test_describe_class_polars_enum_without_recorded_levels():
    series = pl.Series("f", ["y"], dtype=pl.Enum(["y", "x"]))
    assert t.describe_class(series) == 'ordered, factor(levels=c("y", "x"))'
    assert t.describe_class(series, strict_description=False) == "factor"


def test_compare_df_cols_does_not_collect(monkeypatch):
    def no_collect(*args, **kwargs):
        raise AssertionError("compare_df_cols must not collect")

    monkeypatch.setattr(pl.LazyFrame, "collect", no_collect)
    old = t.tidy({"Z": [1.0], "A": [1]})
    new = t.tidy({"A": [1.0]})
    out = t.compare_df_cols(old=old, new=new)
    assert out.columns == ["column_name", "old", "new"]


def test_row_to_names_and_find_header():
    headers = t.tidy(
        pd.DataFrame(
            {
                "a": pd.Series(["id", "1"], dtype="string"),
                "b": pd.Series([None, "x"], dtype="string"),
            }
        ),
    )
    named = headers >> t.row_to_names(1)
    assert named.columns == ["id", "NA"]
    assert named.collect()["id"].to_list() == ["1"]
    assert named.collect()["NA"].to_list() == ["x"]
    numbers = t.tidy(
        pd.DataFrame({"n": pd.Series([1.0, 9.0], dtype="float64")})
    )
    assert (numbers >> t.row_to_names(1)).columns == ["1"]
    flags = t.tidy(pd.DataFrame({"flag": [True, False]}))
    assert (flags >> t.row_to_names(1)).columns == ["TRUE"]
    assert (flags >> t.row_to_names(1)).collect()["TRUE"].to_list() == [False]
    duplicated = t.tidy({"a": ["id", "v"], "b": ["id", "w"]})
    with pytest.warns(UserWarning, match="unique"):
        repaired = duplicated.row_to_names(1)
    assert repaired.columns == ["id", "id_2"]
    sheet = t.tidy(
        {"a": [None, "h1", "1"], "b": ["a", "h2", "2"]}
    )
    assert sheet.find_header() == 2
    assert t.find_header(sheet, value="h2", column="b") == 2
    assert t.find_header(sheet, value="h2", column=2) == 2
    found = sheet >> t.row_to_names("find_header")
    assert found.columns == ["h1", "h2"]
    assert found.collect()["h1"].to_list() == ["1"]
    kept = t.tidy({"a": ["skip", "id", "v"]}) >> t.row_to_names(
        2, remove_rows_above=False
    )
    assert kept.collect()["id"].to_list() == ["skip", "v"]
    with pytest.raises(ValueError, match="outside"):
        numbers >> t.row_to_names(3)
    with pytest.raises(TypeError, match="find_header"):
        t.row_to_names(1, value="id")
    with pytest.raises(ValueError, match="No complete rows"):
        t.find_header(t.tidy({"a": [None, None]}))
    with pytest.warns(UserWarning, match="found 2 times"):
        assert t.find_header(t.tidy({"a": ["q", "q", "z"]}), value="q") == 1


def test_excel_numeric_to_date_serials_and_mutate():
    assert t.excel_numeric_to_date(1) == date(1900, 1, 1)
    assert t.excel_numeric_to_date(59) == date(1900, 2, 28)
    assert t.excel_numeric_to_date(60) is None
    assert t.excel_numeric_to_date(61) == date(1900, 3, 1)
    assert t.excel_numeric_to_date(42370) == date(2016, 1, 1)
    assert t.excel_numeric_to_date(40000) == date(2009, 7, 6)
    assert t.excel_numeric_to_date(40000.5) == date(2009, 7, 6)
    assert t.excel_numeric_to_date(0) == date(1899, 12, 31)
    assert t.excel_numeric_to_date(0, date_system="mac pre-2011") == date(1904, 1, 2)
    assert t.excel_numeric_to_date(1, date_system="mac pre-2011") == date(1904, 1, 3)
    assert t.excel_numeric_to_date(60, date_system="mac pre-2011") is None
    assert t.excel_numeric_to_date(40908, date_system="mac pre-2011") == date(2016, 1, 1)
    assert _clock(t.excel_numeric_to_date(40000.5, include_time=True)) == datetime(
        2009, 7, 6, 12, 0
    )
    assert _clock(t.excel_numeric_to_date(40000.521, include_time=True)) == datetime(
        2009, 7, 6, 12, 30, 14
    )
    assert _clock(
        t.excel_numeric_to_date(40000.521, include_time=True, round_seconds=False)
    ) == datetime(2009, 7, 6, 12, 30, 14, 400000)
    assert t.excel_numeric_to_date(60.5, include_time=True) is None
    assert _clock(t.excel_numeric_to_date(1.999999999, include_time=True)) == datetime(
        1900, 1, 2
    )
    zoned = t.excel_numeric_to_date(40000.5, include_time=True)
    assert zoned.tzinfo == UTC
    with pytest.raises(TypeError, match="numeric"):
        t.excel_numeric_to_date(True)
    with pytest.raises(TypeError, match="numeric"):
        t.excel_numeric_to_date([1, "2"])
    assert isinstance(t.excel_numeric_to_date("serial"), Expr)
    with pytest.raises(ValueError, match="date_system"):
        t.excel_numeric_to_date(1, date_system="windows")


def test_excel_numeric_to_date_in_mutate():
    frame = t.tidy(
        {"g": ["a", "a"], "serial": [42370.0, 60.0]}
    ) >> t.group_by("g")
    out = frame >> t.mutate(d=t.excel_numeric_to_date("serial"))
    assert out._groups == ["g"]
    assert out.collect()["d"].to_list() == [date(2016, 1, 1), None]
    timed = (
        t.tidy({"serial": [40000.5]})
        >> t.mutate(d=t.excel_numeric_to_date("serial", include_time=True))
    ).collect()
    assert _clock(timed["d"].to_list()[0]) == datetime(2009, 7, 6, 12, 0)
    mac = (
        t.tidy({"serial": [1.0]})
        >> t.mutate(
            d=t.excel_numeric_to_date("serial", date_system="mac pre-2011")
        )
    ).collect()
    assert mac["d"].to_list() == [date(1904, 1, 3)]

    def convert():
        return (
            t.tidy({"label": ["1"]})
            >> t.mutate(d=t.excel_numeric_to_date("label"))
        ).collect()

    with pytest.raises(TypeError, match="numeric"):
        convert()


def test_excel_numeric_column_skips_the_per_value_loop(monkeypatch):
    """In-range numeric columns use the Polars date offset, not one Python date each."""

    def explode(*_args, **_kwargs):
        raise AssertionError("per-value excel conversion")

    monkeypatch.setattr("tidy3.dates._excel_convert", explode)
    out = (
        t.tidy({"serial": [42370, 60, None]})
        >> t.mutate(d=t.excel_numeric_to_date("serial"))
    ).collect()
    assert out["d"].to_list() == [date(2016, 1, 1), None, None]
    fractional = (
        t.tidy({"serial": [40000.5, 1.999999999]})
        >> t.mutate(d=t.excel_numeric_to_date("serial"))
    ).collect()
    assert fractional["d"].to_list() == [date(2009, 7, 6), date(1900, 1, 2)]
    mac = (
        t.tidy({"serial": [1, 0, 60]})
        >> t.mutate(d=t.excel_numeric_to_date("serial", date_system="mac pre-2011"))
    ).collect()
    assert mac["d"].to_list() == [date(1904, 1, 3), date(1904, 1, 2), None]


def test_excel_numeric_column_still_rejects_out_of_range_serials():
    frame = t.tidy({"serial": [2958466]})
    with pytest.raises(OverflowError):
        (frame >> t.mutate(d=t.excel_numeric_to_date("serial"))).collect()


def test_excel_numeric_to_date_notebook_bare_name():
    namespace = {name: getattr(t, name) for name in t.__all__}
    namespace["frame"] = t.tidy({"min": [42370.0], "label": ["x"]})
    exec(
        "out = " + t.transform_source(
            "frame >> mutate(d=excel_numeric_to_date(min))"
        ).strip(),
        namespace,
    )
    assert namespace["out"].collect()["d"].to_list() == [date(2016, 1, 1)]
    exec(
        "cleaned = " + t.transform_source("frame >> remove_empty('rows')").strip(),
        namespace,
    )
    assert namespace["cleaned"].columns == ["min", "label"]


def test_excel_vector_path_divides_exactly_at_day_boundaries():
    # janitor: excel_numeric_to_date(-194.0000000115741) is 1899-06-19. The
    # serial is just under a whole day; dividing by 86400 through its
    # reciprocal rounded it up to -194 and gave the next day.
    serials = [-194.0000000115741, -193.0000000115741, 100.0000000115741 - 2e-8]
    out = (
        t.tidy({"serial": serials * 4000})
        >> t.mutate(d=t.excel_numeric_to_date("serial"))
    ).collect()["d"].to_list()
    assert out[:3] == [date(1899, 6, 19), date(1899, 6, 20), date(1900, 4, 9)]
    assert out == out[:3] * 4000
