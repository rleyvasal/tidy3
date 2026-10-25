"""Report percentages by row, column, or grand total, with and without totals."""

import math

import pandas as pd
import polars as pl
import pytest

import tidy3 as t


def table():
    return t.tidy({"label": ["a", "b", "c"], "yes": [1, None, 0], "no": [2, 3, 0], "note": ["x", "y", "z"]})


def rows(frame):
    return [tuple(None if isinstance(v, float) and math.isnan(v) else v for v in row)
            for row in frame.collect().rows()]


def test_row_column_and_grand_denominators():
    assert rows(table() >> t.adorn_percentages()) == [
        ("a", 1 / 3, 2 / 3, "x"), ("b", None, 1.0, "y"), ("c", None, None, "z")]
    assert rows(table() >> t.adorn_percentages("col")) == [
        ("a", 1.0, 2 / 5, "x"), ("b", None, 3 / 5, "y"), ("c", 0.0, 0.0, "z")]
    assert rows(table() >> t.adorn_percentages("all")) == [
        ("a", 1 / 6, 2 / 6, "x"), ("b", None, 3 / 6, "y"), ("c", 0.0, 0.0, "z")]
    out = (table() >> t.adorn_percentages()).collect()
    assert out.schema["yes"] == pl.Float64 and out.schema["note"] == pl.String
    # A zero row sum is NaN (0 / 0), not missing.
    assert math.isnan(out["no"][2])


def test_division_is_exact_as_in_r():
    values = [3, 7, 11, 13, 1]
    out = (t.tidy({"g": list("abcde"), "n": values}) >> t.adorn_percentages("col")).collect()
    assert out["n"].to_list() == [value / sum(values) for value in values]
    out = (t.tidy({"g": list("abcde"), "n": values}) >> t.adorn_percentages("all")).collect()
    assert out["n"].to_list() == [value / sum(values) for value in values]


def test_na_rm_false_propagates_missing_sums():
    assert rows(table() >> t.adorn_percentages(na_rm=False))[1] == ("b", None, None, "y")
    assert rows(table() >> t.adorn_percentages("col", na_rm=False))[0] == ("a", None, 0.4, "x")
    # The grand total always skips missing values, as in janitor.
    assert rows(table() >> t.adorn_percentages("all", na_rm=False))[1] == ("b", None, 0.5, "y")
    nan = t.tidy({"g": ["a", "b"], "x": [1.0, float("nan")], "y": [1.0, 1.0]})
    assert rows(nan >> t.adorn_percentages())[1] == ("b", None, 1.0)


def test_totals_are_not_counted_twice():
    totals = t.tidy({"label": ["a", "b"], "yes": [1, 2], "no": [3, 2]}) >> t.adorn_totals("both")
    assert rows(totals >> t.adorn_percentages()) == [
        ("a", .25, .75, 1.0), ("b", .5, .5, 1.0), ("Total", .375, .625, 1.0)]
    assert rows(totals >> t.adorn_percentages("col")) == [
        ("a", 1 / 3, .6, .5), ("b", 2 / 3, .4, .5), ("Total", 1.0, 1.0, 1.0)]
    assert rows(totals >> t.adorn_percentages("all")) == [
        ("a", .125, .375, .5), ("b", .25, .25, .5), ("Total", .375, .625, 1.0)]
    # Choosing columns without the totals column leaves it as counts.
    assert rows(totals >> t.adorn_percentages(cols="yes"))[0] == ("a", 1.0, 3, 4.0)
    out = totals >> t.adorn_percentages()
    assert out._totals == {"row": "Total", "col": "Total"}
    with pytest.raises(ValueError, match="re-add"):
        out >> t.adorn_totals()


def test_selection_skips_text_and_can_include_the_first_column():
    out = table() >> t.adorn_percentages(cols=["yes", "note"])
    assert rows(out)[0] == ("a", 1.0, 2, "x")
    numeric = t.tidy({"g": [1, 3], "n": [3, 1]})
    assert rows(numeric >> t.adorn_percentages("col", cols=t.everything())) == [(.25, .75), (.75, .25)]
    assert rows(numeric >> t.adorn_percentages("col")) == [(1, .75), (3, .25)]
    unchanged = table() >> t.adorn_percentages(cols="note")
    assert rows(unchanged) == rows(table())


def test_tabyl_report_factor_labels_and_empty_tables():
    report = t.tidy({"g": ["a", "a", "b"], "answer": ["yes", "no", "yes"]}) >> t.tabyl("g", "answer")
    out = report >> t.adorn_totals("both") >> t.adorn_percentages("all")
    assert rows(out) == [("a", 1 / 3, 1 / 3, 2 / 3), ("b", 0.0, 1 / 3, 1 / 3), ("Total", 1 / 3, 2 / 3, 1.0)]
    factor = t.tidy(pd.DataFrame({
        "g": pd.Categorical(["b", "a"], categories=["b", "a", "unused"], ordered=True), "n": [1, 3],
    })) >> t.adorn_totals() >> t.adorn_percentages("col")
    assert factor.factor_levels()["g"] == ["b", "a", "unused", "Total"]
    assert rows(factor) == [("b", .25), ("a", .75), ("Total", 1.0)]
    empty = t.tidy(pl.DataFrame(schema={"label": pl.String, "n": pl.Int64})) >> t.adorn_percentages("col")
    assert empty.collect().schema["n"] == pl.Float64 and empty.collect().is_empty()


def test_percentages_stay_lazy_ungroup_and_keep_diagnostics(tmp_path):
    path = tmp_path / "source.csv"
    path.write_text("g,n\na,1\nb,oops\nc,3\n")
    source = t.scan_csv(path) >> t.mutate(n=t.parse_integer("n")) >> t.group_by("g")
    out = source >> t.adorn_percentages("col")
    assert t.group_vars(out) == []
    assert out._data_reads == source._data_reads
    computed = out >> t.compute()
    path.unlink()
    assert computed.collect()["n"].to_list() == [.25, None, .75]
    assert t.problems(computed).collect()["original"].to_list() == ["oops"]


def test_notebook_cols_accepts_bare_column_names():
    from IPython.core.interactiveshell import InteractiveShell

    shell = InteractiveShell.instance()
    try:
        shell.run_line_magic("load_ext", "tidy3.jupyter")
        result = shell.run_cell(
            'df = tidy({"id": ["a", "b"], "n": [1, 3], "other": [3, 4]})\n'
            'out = df >> adorn_percentages("col", cols=n)'
        )
        assert result.success, result.error_before_exec or result.error_in_exec
        assert shell.user_ns["out"].collect().rows() == [("a", .25, 3), ("b", .75, 4)]
    finally:
        shell.run_line_magic("unload_ext", "tidy3.jupyter")
        InteractiveShell.clear_instance()


@pytest.mark.parametrize("call, message", [
    (lambda: t.adorn_percentages("rows"), "denominator"),
    (lambda: t.adorn_percentages(na_rm=1), "boolean"),
    (lambda: t.tidy({"label": ["a"], "note": ["x"]}) >> t.adorn_percentages(), "numeric"),
    (lambda: t.tidy({"n": [1]}) >> t.adorn_percentages(), "numeric"),
])
def test_invalid_arguments(call, message):
    with pytest.raises((ValueError, TypeError), match=message):
        call()
