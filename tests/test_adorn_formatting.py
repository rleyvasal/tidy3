"""Percentage text, counts beside it, and three-variable tabyls."""

import warnings

import pandas as pd
import polars as pl
import pytest

import tidy3 as t


def shares():
    return t.tidy({"label": ["a", "b", "c", "d"], "x": [0.125, None, float("nan"), -0.0001],
                   "y": [0.5, float("inf"), 1.0, 0.0], "note": ["p", "q", "r", "s"]})


def report():
    return t.tidy({"g": ["b", "a", "b", "a", "b", "c"],
                   "answer": ["yes", "no", "yes", "yes", "no", "no"]}) >> t.tabyl("g", "answer")


def test_pct_formatting_rounds_like_r_and_marks_missing_values():
    out = (shares() >> t.adorn_pct_formatting(0)).collect()
    assert out["x"].to_list() == ["12%", "-", "-", "0%"]
    assert out["y"].to_list() == ["50%", "Inf%", "100%", "0%"]
    assert out["note"].to_list() == ["p", "q", "r", "s"]
    assert out.schema["x"] == pl.String
    assert (shares() >> t.adorn_pct_formatting(0, "half up")).collect()["x"][0] == "13%"
    assert (shares() >> t.adorn_pct_formatting(2, affix_sign=False)).collect()["x"].to_list() == [
        "12.50", "-", "-", "-0.01",
    ]
    assert (shares() >> t.adorn_pct_formatting()).collect()["x"][-1] == "0.0%"


def test_pct_formatting_selects_numeric_columns():
    out = (shares() >> t.adorn_pct_formatting(cols=["x", "note"])).collect()
    assert out.schema["y"] == pl.Float64
    assert out["note"].to_list() == ["p", "q", "r", "s"]
    first = (t.tidy({"p": [0.5], "q": [1.0]}) >> t.adorn_pct_formatting(cols=t.everything())).collect()
    assert first.rows() == [("50.0%", "100.0%")]


def test_pct_formatting_on_a_one_way_tabyl_leaves_counts():
    one_way = t.tidy({"g": ["a", "b", None, "a"]}) >> t.tabyl("g")
    out = (one_way >> t.adorn_pct_formatting(cols="n")).collect()
    assert out["n"].to_list() == [2.0, 1.0, 1.0]
    assert out["percent"].to_list() == ["50.0%", "25.0%", "25.0%"]
    assert out["valid_percent"].to_list() == ["66.7%", "33.3%", "-"]


def test_pct_formatting_keeps_the_totals_record():
    formatted = report() >> t.adorn_totals("both") >> t.adorn_percentages() >> t.adorn_pct_formatting()
    assert formatted.collect()["Total"].to_list() == ["100.0%"] * 4
    with pytest.raises(ValueError, match="re-add"):
        formatted >> t.adorn_totals()


@pytest.mark.parametrize(("make", "message"), [
    (lambda: t.adorn_pct_formatting(1.5), "integer"),
    (lambda: t.adorn_pct_formatting(True), "integer"),
    (lambda: t.adorn_pct_formatting(-1), "0 or more"),
    (lambda: t.adorn_pct_formatting(rounding="up"), "rounding"),
    (lambda: t.adorn_pct_formatting(affix_sign="yes"), "boolean"),
    (lambda: t.tidy({"label": ["a"], "n": ["x"]}) >> t.adorn_pct_formatting(), "numeric"),
    (lambda: t.adorn_ns("middle"), "position"),
    (lambda: t.adorn_ns(format_func="{:,}"), "function"),
])
def test_formatting_argument_errors(make, message):
    with pytest.raises((TypeError, ValueError), match=message):
        make()


def test_adorn_ns_adds_the_tabyl_counts_with_totals():
    totals = report() >> t.adorn_totals("both")
    out = (totals >> t.adorn_percentages("col") >> t.adorn_pct_formatting() >> t.adorn_ns()).collect()
    assert out.rows() == [
        ("a", "33.3% (1)", "33.3% (1)", "33.3% (2)"),
        ("b", "33.3% (1)", "66.7% (2)", "50.0% (3)"),
        ("c", "33.3% (1)", "0.0% (0)", "16.7% (1)"),
        ("Total", "100.0% (3)", "100.0% (3)", "100.0% (6)"),
    ]
    front = (totals >> t.adorn_percentages("col") >> t.adorn_pct_formatting() >> t.adorn_ns("front")).collect()
    assert front["yes"].to_list() == ["1  (33.3%)", "2  (66.7%)", "0   (0.0%)", "3 (100.0%)"]


def test_adorn_ns_writes_numbers_as_janitor_does():
    # A text label column makes R format each numeric column together.
    out = (report() >> t.adorn_percentages() >> t.adorn_ns()).collect()
    assert out["no"].to_list() == ["0.5000000 (1)", "0.3333333 (1)", "1.0000000 (1)"]
    front = (report() >> t.adorn_percentages() >> t.adorn_ns("front")).collect()
    assert front["no"].to_list() == ["1               (0.5)", "1 (0.333333333333333)", "1                 (1)"]
    big = t.tidy({"g": ["a"] * 1500 + ["b"], "h": ["x"] * 1501}) >> t.tabyl("g", "h")
    assert (big >> t.adorn_ns("front")).collect()["x"].to_list() == ["1,500 (1500)", "1    (1)"]
    custom = t.tidy({"g": ["a", "b"], "n": [1 / 3, 12345678.0]})
    out = (t.tidy({"g": ["a", "b"], "n": ["x", "y"]}) >> t.adorn_ns("front", ns=custom)).collect()
    assert out["n"].to_list() == ["3.333333e-01 (x)", "1.234568e+07 (y)"]


def test_adorn_ns_format_func_and_cols():
    formatted = report() >> t.adorn_percentages() >> t.adorn_pct_formatting()
    out = (formatted >> t.adorn_ns(format_func=lambda n: f"n={n:g}", cols="yes")).collect()
    assert out["yes"].to_list() == ["50.0% (n=1)", "66.7% (n=2)", "0.0% (n=0)"]
    assert out["no"].to_list() == ["50.0%", "33.3%", "100.0%"]


def test_adorn_ns_needs_counts_of_the_same_shape():
    with pytest.raises(ValueError, match="pass ns="):
        report() >> t.filter(t.col("g") != "c") >> t.adorn_ns()
    with pytest.raises(ValueError, match="shape"):
        report() >> t.adorn_ns(ns=t.tidy({"g": ["a"], "n": [1]}))
    one_way = t.tidy({"g": ["a", "b"]}) >> t.tabyl("g")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = (one_way >> t.adorn_ns()).collect()
    assert "two-variable" in str(caught[0].message)
    assert out["n"].to_list() == ["1 (1)", "1 (1)"]


def test_three_way_tabyl_returns_a_table_per_value():
    data = t.tidy({"g": ["b", "a", "b", "a", "c"], "h": ["x", "y", "x", None, "y"],
                   "z": [10.0, 2.0, 2.0, None, 10.0]})
    tables = data >> t.tabyl("g", "h", "z")
    assert list(tables) == ["2", "10", "NA_"]
    assert tables["2"].collect().rows() == [("a", 0.0, 1.0), ("b", 1.0, 0.0), ("c", 0.0, 0.0)]
    assert tables["NA_"].collect().columns == ["g", "x", "y", "NA_"]
    assert list(data >> t.tabyl("g", "h", "z", show_na=False)) == ["2", "10"]
    present = data >> t.tabyl("g", "h", "z", show_missing_levels=False)
    assert present["10"].collect().rows() == [("b", 1.0, 0.0), ("c", 0.0, 1.0)]


def test_three_way_tabyl_orders_factor_tables_by_level():
    data = t.tidy(pd.DataFrame({"g": ["a", "b", "c"], "h": ["x", "x", "y"],
                                "z": pd.Categorical(["c", "a", "b"], categories=["b", "c", "a", "unused"])}))
    # janitor 2.2.1 gives c, a, b here.
    assert list(data >> t.tabyl("g", "h", "z")) == ["b", "c", "a"]
    text = t.tidy({"g": ["a", "b"], "h": ["x", "y"], "w": ["k", ""]})
    assert list(text >> t.tabyl("g", "h", "w")) == ["", "k"]


def test_report_verbs_map_over_a_three_way_tabyl():
    data = t.tidy({"g": ["b", "a", "b"], "h": ["x", "y", "x"], "z": ["p", "p", "q"]})
    tables = (data >> t.tabyl("g", "h", "z") >> t.adorn_totals("both") >> t.adorn_percentages()
              >> t.adorn_pct_formatting(0) >> t.adorn_ns())
    assert list(tables) == ["p", "q"]
    assert tables["q"].collect().rows() == [
        ("a", "- (0)", "- (0)", "100% (0)"),
        ("b", "100% (1)", "0% (0)", "100% (1)"),
        ("Total", "100% (1)", "0% (0)", "100% (1)"),
    ]
    empty = t.tidy(pl.DataFrame(schema={"g": pl.String, "h": pl.String, "z": pl.String}))
    assert (empty >> t.tabyl("g", "h", "z")) == {}


def test_three_way_tabyl_argument_errors():
    data = t.tidy({"g": ["a"], "h": ["x"], "z": ["p"]})
    with pytest.raises(TypeError, match="second"):
        t.tabyl("g", column3="z")
    with pytest.raises(TypeError, match="frame column"):
        t.tabyl(pd.Series(["a"]), None, "z")
    with pytest.raises(ValueError, match="distinct"):
        data >> t.tabyl("g", "h", "g")


@pytest.mark.parametrize(("cell", "expected"), [
    ('df = tidy({"id": ["a", "b"], "n": [0.25, 0.5], "other": [0.5, 1.0]})\n'
     'out = df >> adorn_pct_formatting(cols=n)', ("a", "25.0%", 0.5)),
    ('df = tidy({"g": ["a", "b"], "n": ["x", "y"], "other": ["u", "v"]})\n'
     'counts = tidy({"g": ["a", "b"], "n": [1, 2], "other": [3, 4]})\n'
     'out = df >> adorn_ns(ns=counts, cols=n)', ("a", "x (1)", "u")),
])
def test_notebook_cols_accepts_bare_column_names(cell, expected):
    from IPython.core.interactiveshell import InteractiveShell

    shell = InteractiveShell.instance()
    try:
        shell.run_line_magic("load_ext", "tidy3.jupyter")
        result = shell.run_cell(cell)
        assert result.success, result.error_before_exec or result.error_in_exec
        assert shell.user_ns["out"].collect().rows()[0] == expected
    finally:
        shell.run_line_magic("unload_ext", "tidy3.jupyter")
        InteractiveShell.clear_instance()
