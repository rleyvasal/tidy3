"""Numeric report totals, type preservation, and report composition."""

from datetime import date

import pandas as pd
import polars as pl
from polars.testing import assert_frame_equal
import pytest

import tidy3 as t


def table():
    return t.tidy({"label": ["a", "b"], "yes": [1, None], "no": [2, 3], "note": ["x", "y"]})


@pytest.mark.parametrize("where", ["both", ("row", "col"), ("col", "row")])
def test_both_totals_and_grand_total(where):
    out = (table() >> t.adorn_totals(where)).collect()
    assert out.rows() == [("a", 1, 2, "x", 3.), ("b", None, 3, "y", 3.), ("Total", 1, 5, "-", 6.)]
    assert out.schema["yes"] == pl.Int64 and out.schema["Total"] == pl.Float64


def test_default_excludes_numeric_first_column_and_explicit_selection_can_total_it():
    source = t.tidy({"id": [1, 2], "n": [3, 4]})
    out = (source >> t.adorn_totals()).collect()
    assert out.rows() == [("1", 3), ("2", 4), ("Total", 7)]
    out = (source >> t.adorn_totals("both", cols=t.everything())).collect()
    assert out.rows() == [(1, 3, 4.), (2, 4, 6.), (3, 7, 10.)]


def test_selected_columns_custom_labels_and_fill():
    out = (table() >> t.adorn_totals("both", cols="yes", name=("All", "Grand"), fill="")).collect()
    assert out.rows() == [("a", 1, "2", "x", 1.), ("b", None, "3", "y", 0.), ("All", 1, "", "", 1.)]
    out = (table() >> t.adorn_totals(cols="yes", fill=None)).collect()
    assert out.rows()[-1] == ("Total", 1, None, None)
    assert out.schema["no"] == pl.Int64


def test_none_fill_preserves_boolean_and_date_columns():
    source = t.tidy({"id": ["a", "b"], "n": [1, 2], "active": [True, False],
                     "when": [date(2024, 1, 1), date(2024, 1, 2)]})
    out = (source >> t.adorn_totals(fill=None)).collect()
    assert out.schema["active"] == pl.Boolean and out.schema["when"] == pl.Date
    assert out.rows()[-1] == ("Total", 3, None, None)
    out = (source >> t.adorn_totals()).collect()
    assert out["active"].to_list() == ["TRUE", "FALSE", "-"]
    assert out["when"].to_list() == ["2024-01-01", "2024-01-02", "-"]


@pytest.mark.parametrize("na_rm", [False, True])
def test_null_nan_and_all_missing_values(na_rm):
    source = t.tidy({"label": ["a", "b", "c"], "x": [None, float("nan"), 2.], "y": [1., 2., None]})
    out = (source >> t.adorn_totals("both", na_rm=na_rm)).collect()
    if na_rm:
        assert out["Total"].to_list() == [1., 2., 2., 5.]
        assert out["x"][-1] == 2. and out["y"][-1] == 3.
    else:
        assert out["x"][-1] is None and out["y"][-1] is None
        assert out["Total"][0] is None and out["Total"][-1] is None
        assert out["Total"].is_nan()[1]
    missing = t.tidy(pl.DataFrame({"label": ["a", "b"], "n": pl.Series([None, None], dtype=pl.Float64)}))
    assert (missing >> t.adorn_totals()).collect()["n"][-1] == 0.


def test_empty_table_adds_one_zero_totals_row():
    source = t.tidy(pl.DataFrame(schema={"label": pl.String, "n": pl.Int64}))
    out = (source >> t.adorn_totals("both")).collect()
    assert out.rows() == [("Total", 0, 0.)]
    assert (source >> t.adorn_totals("col")).collect().shape == (0, 3)


@pytest.mark.parametrize("ordered", [False, True])
def test_factor_label_keeps_levels_order_and_ordered_flag(ordered):
    source = t.tidy(pd.DataFrame({
        "g": pd.Categorical(["b", "a"], categories=["b", "a", "unused"], ordered=ordered),
        "n": [1, 2],
    }))
    out = source >> t.adorn_totals()
    categorical = out.collect(as_="pandas")["g"]
    assert categorical.tolist() == ["b", "a", "Total"]
    assert categorical.cat.categories.tolist() == ["b", "a", "unused", "Total"]
    assert categorical.cat.ordered == ordered
    assert source.factor_levels()["g"] == ["b", "a", "unused"]


def test_tabyl_pipeline_and_separate_dimensions_match_both():
    source = t.tidy({"g": ["a", "a", "b"], "answer": ["yes", "no", "yes"]}) >> t.tabyl("g", "answer")
    expected = (source >> t.adorn_totals("both")).collect()
    for first, second in (("row", "col"), ("col", "row")):
        out = source >> t.adorn_totals(first) >> t.adorn_totals(second)
        assert_frame_equal(out.collect(), expected)
    assert expected.rows()[-1] == ("Total", 1., 2., 3.)
    one_way = t.tidy({"g": ["a", "a", "b"]}) >> t.tabyl("g") >> t.adorn_totals(cols="n", fill=None)
    assert one_way.collect().rows()[-1] == ("Total", 3, None)


def test_totals_markers_prevent_readding_and_survive_compute():
    rows = table() >> t.adorn_totals()
    with pytest.raises(ValueError, match="re-add"):
        rows >> t.adorn_totals()
    with pytest.raises(ValueError, match="re-add"):
        rows >> t.compute() >> t.adorn_totals()
    both = rows >> t.adorn_totals("col")
    with pytest.raises(ValueError, match="re-add"):
        both >> t.adorn_totals("col", name="Again")


def test_totals_are_global_lazy_and_keep_existing_diagnostics(tmp_path):
    path = tmp_path / "source.csv"
    path.write_text("g,n\na,1\nb,oops\n")
    source = t.scan_csv(path) >> t.mutate(n=t.parse_integer("n")) >> t.group_by("g")
    out = source >> t.adorn_totals("both")
    assert t.group_vars(out) == [] and t.group_vars(source) == ["g"]
    # No new row data was read while constructing the totals query.
    assert out._data_reads == source._data_reads
    computed = out >> t.compute()
    path.unlink()
    assert computed.collect().rows()[-1] == ("Total", 1, 1.)
    assert t.problems(computed).collect()["original"].to_list() == ["oops"]
    with pytest.raises(ValueError, match="re-add"):
        computed >> t.adorn_totals("col")


def test_notebook_cols_accepts_bare_column_names():
    from IPython.core.interactiveshell import InteractiveShell

    shell = InteractiveShell.instance()
    try:
        shell.run_line_magic("load_ext", "tidy3.jupyter")
        shell.run_cell('placement = "both"')
        result = shell.run_cell(
            'df = tidy({"id": ["a", "b"], "n": [1, 2], "other": [3, 4]})\n'
            'out = df >> adorn_totals(where=placement, cols=n)'
        )
        assert result.success, result.error_before_exec or result.error_in_exec
        assert shell.user_ns["out"].collect().rows()[-1] == ("Total", 3, "-", 3.)
    finally:
        shell.run_line_magic("unload_ext", "tidy3.jupyter")
        InteractiveShell.clear_instance()


@pytest.mark.parametrize("call, message", [
    (lambda: t.adorn_totals("invalid"), "where"),
    (lambda: t.adorn_totals([]), "where"),
    (lambda: t.adorn_totals(name=["one", "two", "three"]), "name"),
    (lambda: t.adorn_totals(fill=0), "fill"),
    (lambda: t.adorn_totals(na_rm=1), "boolean"),
    (lambda: t.tidy({"label": ["a"]}) >> t.adorn_totals(), "numeric"),
    (lambda: table() >> t.adorn_totals(cols="note"), "numeric"),
    (lambda: t.tidy({"label": ["a"], "Total": [1]}) >> t.adorn_totals("col"), "already exists"),
])
def test_invalid_arguments_and_selections(call, message):
    with pytest.raises((ValueError, TypeError), match=message):
        call()
