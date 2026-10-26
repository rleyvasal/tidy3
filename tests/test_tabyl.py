"""Numeric reporting tables and their downstream behavior."""

from datetime import date

import pandas as pd
import polars as pl
import pytest

import tidy3 as t


def test_counts_percentages_and_missing_denominators():
    source = t.tidy({"x": ["b", "a", "a", None]})
    out = (source >> t.tabyl("x")).collect()
    assert out.columns == ["x", "n", "percent", "valid_percent"]
    assert out["x"].to_list() == ["a", "b", None]
    assert out["n"].to_list() == [2, 1, 1]
    assert out["percent"].to_list() == [.5, .25, .25]
    assert out["valid_percent"].to_list()[:2] == pytest.approx([2 / 3, 1 / 3])
    assert out["valid_percent"][-1] is None
    out = (source >> t.tabyl(t.col("x"), show_na=False)).collect()
    assert out.columns == ["x", "n", "percent"]
    assert out["n"].to_list() == [2, 1]
    assert out["percent"].to_list() == pytest.approx([2 / 3, 1 / 3])


@pytest.mark.parametrize("show_missing_levels", [False, True])
@pytest.mark.parametrize("show_na", [False, True])
def test_factor_order_unused_levels_and_ordered_flag(show_missing_levels, show_na):
    source = t.tidy(pd.DataFrame({"x": pd.Categorical(
        ["b", None, "b"], categories=["c", "b", "a"], ordered=True,
    )}))
    out = source >> t.tabyl("x", show_na=show_na, show_missing_levels=show_missing_levels)
    expected = ["c", "b", "a"] if show_missing_levels else ["b"]
    assert out.collect()["x"].to_list() == expected + ([None] if show_na else [])
    assert out.collect()["n"].sum() == (3 if show_na else 2)
    categorical = out.collect(as_="pandas")["x"]
    assert categorical.cat.categories.tolist() == ["c", "b", "a"]
    assert categorical.cat.ordered
    # Percentages remain usable in calculations and plotting.
    counts = out >> t.mutate(percentage=t.col("percent") * 100)
    assert counts.collect()["percentage"].sum() == pytest.approx(100)


def test_empty_and_all_missing_inputs():
    empty = t.tabyl(pd.Series([], dtype="string", name="label")).collect()
    assert empty.shape == (0, 3)
    assert empty.schema["percent"] == pl.Float64
    out = t.tabyl(pd.Categorical([], categories=["b", "a"])).collect()
    assert out["x"].to_list() == ["b", "a"]
    assert out["n"].to_list() == [0, 0]
    assert out["percent"].is_nan().all()
    out = t.tabyl([None, None]).collect()
    assert out["n"].to_list() == [2]
    assert out["percent"].to_list() == [1.]
    assert out["valid_percent"].to_list() == [None]
    assert t.tabyl([None, None], show_na=False).collect().shape == (0, 3)


def test_nan_and_null_are_separate_missing_rows():
    source = t.tidy(pl.DataFrame({"x": [None, float("nan"), 1., float("nan")]}))
    out = (source >> t.tabyl("x")).collect()
    assert out["x"][0] == 1.
    assert out["x"].is_nan().fill_null(False).to_list() == [False, True, False]
    assert out["n"].to_list() == [1, 2, 1]
    assert out["valid_percent"].to_list() == [1., None, None]
    out = (source >> t.tabyl("x", show_na=False)).collect()
    assert out["x"].to_list() == [1.] and out["percent"].to_list() == [1.]


@pytest.mark.parametrize("values", [
    [False, True, False], [date(2024, 1, 2), date(2024, 1, 1), None], [3, 1, 3],
])
def test_scalar_types_and_polars_vectors(values):
    source = pl.Series("value", values)
    out = t.tabyl(source).collect()
    assert out.columns[0] == "value"
    assert out["n"].sum() == 3
    assert out.schema["value"] == source.dtype


def test_groups_are_ignored_and_input_is_unchanged():
    source = t.tidy({"g": [1, 2, 2], "x": ["a", "a", "b"]}) >> t.group_by("g")
    out = source >> t.tabyl("x")
    assert t.group_vars(out) == []
    assert out.collect()["n"].to_list() == [2, 1]
    assert t.group_vars(source) == ["g"]
    assert source.columns == ["g", "x"]


def test_snapshot_survives_source_removal_and_keeps_parse_problems(tmp_path):
    path = tmp_path / "source.csv"
    path.write_text("x,y\na,1\na,oops\nb,3\n")
    source = t.scan_csv(path) >> t.mutate(y=t.parse_integer("y"))
    out = source >> t.tabyl("x")
    path.unlink()
    assert out.collect()["n"].to_list() == [2, 1]
    assert t.problems(out).collect()["original"].to_list() == ["oops"]
    with pytest.raises(t.ExportError, match="tabyl"):
        out.to_polars_code()


def test_notebook_bare_column_names():
    from IPython.core.interactiveshell import InteractiveShell

    shell = InteractiveShell.instance()
    try:
        shell.run_line_magic("load_ext", "tidy3.jupyter")
        result = shell.run_cell(
            'df = tidy({"id": ["a", "a", "b"]})\n'
            'out = df >> tabyl(id)'
        )
        assert result.success, result.error_before_exec or result.error_in_exec
        assert shell.user_ns["out"].collect()["n"].to_list() == [2, 1]
    finally:
        shell.run_line_magic("unload_ext", "tidy3.jupyter")
        InteractiveShell.clear_instance()


def test_invalid_selections_flags_and_output_name_collisions():
    source = t.tidy({"a": [1], "b": [2], "n": [3], "nested": [[1, 2]]})
    with pytest.raises(ValueError, match="exactly one"):
        source >> t.tabyl(t.starts_with(""))
    with pytest.raises(ValueError, match="rename"):
        source >> t.tabyl("n")
    with pytest.raises(TypeError, match="scalar"):
        source >> t.tabyl("nested")
    with pytest.raises(TypeError, match="boolean"):
        t.tabyl("a", show_na=1)
    with pytest.raises((KeyError, ValueError), match="absent"):
        source >> t.tabyl("absent")


def test_two_way_counts_blank_and_missing_columns():
    source = t.tidy({"x": ["b", "a", "a", None, "b"], "y": ["", "z", None, "m", "z"]})
    out = (source >> t.tabyl("x", "y")).collect()
    assert out.columns == ["x", "m", "z", "emptystring_", "NA_"]
    assert out.rows() == [("a", 0, 1, 0, 1), ("b", 0, 1, 1, 0), (None, 1, 0, 0, 0)]
    assert all(dtype == pl.Float64 for dtype in list(out.schema.values())[1:])
    without_na = (source >> t.tabyl(t.col("x"), pl.col("y"), show_na=False)).collect()
    assert without_na.columns == ["x", "z", "emptystring_"]
    assert without_na.rows() == [("a", 1, 0), ("b", 1, 1)]


@pytest.mark.parametrize("show_missing_levels", [False, True])
@pytest.mark.parametrize("show_na", [False, True])
def test_two_way_factor_axes_keep_order_and_unused_levels(show_missing_levels, show_na):
    source = t.tidy(pd.DataFrame({
        "x": pd.Categorical(["b", None, "b"], categories=["c", "b", "a"], ordered=True),
        "y": pd.Categorical(["z", "m", None], categories=["q", "z", "m"]),
    }))
    out = source >> t.tabyl("x", "y", show_na=show_na, show_missing_levels=show_missing_levels)
    expected_rows = (["c", "b", "a"] if show_missing_levels else ["b"]) + ([None] if show_na else [])
    expected_columns = (["q", "z", "m"] if show_missing_levels else ["z", "m"] if show_na else ["z"])
    if show_na:
        expected_columns.append("NA_")
    data = out.collect()
    assert data["x"].to_list() == expected_rows
    assert data.columns == ["x", *expected_columns]
    assert data.select(pl.sum_horizontal(expected_columns).sum()).item() == (3 if show_na else 1)
    restored = out.collect(as_="pandas")["x"]
    assert restored.cat.categories.tolist() == ["c", "b", "a"] and restored.cat.ordered
    assert out.factor_levels() == {"x": ["c", "b", "a"]}


@pytest.mark.parametrize("show_missing_levels", [False, True])
def test_two_way_empty_input_has_only_the_row_column(show_missing_levels):
    source = t.tidy(pd.DataFrame({
        "x": pd.Categorical([], categories=["b", "a"]),
        "y": pd.Categorical([], categories=["z", "m"]),
    }))
    out = (source >> t.tabyl("x", "y", show_missing_levels=show_missing_levels)).collect()
    assert out.shape == (0, 1) and out.columns == ["x"]
    source = t.tidy({"x": ["a", None], "y": [None, "b"]})
    out = (source >> t.tabyl("x", "y", show_na=False)).collect()
    assert out.shape == (0, 1) and out.schema["x"] == pl.String


@pytest.mark.parametrize("values, labels", [
    ([10, 2, 1], ["1", "2", "10"]),
    ([True, False, True], ["FALSE", "TRUE"]),
    ([date(2024, 2, 1), date(2024, 1, 1), date(2024, 2, 1)], ["2024-01-01", "2024-02-01"]),
    ([1., float("nan"), None], ["1", "NaN", "NA_"]),
])
def test_two_way_scalar_column_labels_and_order(values, labels):
    source = t.tidy({"x": ["b", "a", "b"], "y": values})
    out = (source >> t.tabyl("x", "y")).collect()
    assert out.columns == ["x", *labels]
    assert out.select(pl.sum_horizontal(labels).sum()).item() == 3


def test_two_way_all_null_axes():
    out = (t.tidy({"x": [None, None], "y": [None, None]}) >> t.tabyl("x", "y")).collect()
    assert out.columns == ["x", "NA_"] and out.rows() == [(None, 2)]


def test_two_way_ignores_grouping_and_preserves_snapshot_diagnostics(tmp_path):
    path = tmp_path / "source.csv"
    path.write_text("x,y,z\na,yes,1\na,no,oops\nb,yes,3\n")
    source = t.scan_csv(path) >> t.mutate(z=t.parse_integer("z")) >> t.group_by("z")
    out = source >> t.tabyl("x", "y")
    path.unlink()
    assert t.group_vars(out) == [] and t.group_vars(source) == ["z"]
    assert out.collect().rows() == [("a", 1, 1), ("b", 0, 1)]
    assert t.problems(out).collect()["original"].to_list() == ["oops"]
    assert (out >> t.mutate(total=t.col("no") + t.col("yes"))).collect()["total"].to_list() == [2, 1]
    with pytest.raises(t.ExportError, match="tabyl"):
        out.to_polars_code()


def test_two_way_notebook_bare_names():
    from IPython.core.interactiveshell import InteractiveShell

    shell = InteractiveShell.instance()
    try:
        shell.run_line_magic("load_ext", "tidy3.jupyter")
        result = shell.run_cell(
            'df = tidy({"id": ["a", "a", "b"], "answer": ["yes", "no", "yes"]})\n'
            'out = df >> tabyl(id, answer)'
        )
        assert result.success, result.error_before_exec or result.error_in_exec
        assert shell.user_ns["out"].collect().rows() == [("a", 1, 1), ("b", 0, 1)]
    finally:
        shell.run_line_magic("unload_ext", "tidy3.jupyter")
        InteractiveShell.clear_instance()


@pytest.mark.parametrize("column_values", [["x", "other"], ["NA_", None], ["emptystring_", ""]])
def test_two_way_rejects_ambiguous_column_labels(column_values):
    source = t.tidy({"x": ["a", "b"], "y": column_values})
    with pytest.raises(ValueError, match="labels conflict"):
        source >> t.tabyl("x", "y")


def test_two_way_rejects_invalid_selections():
    source = t.tidy({"x": ["a"], "y": ["b"], "nested": [[1]]})
    with pytest.raises(ValueError, match="distinct"):
        source >> t.tabyl("x", "x")
    with pytest.raises(ValueError, match="exactly one"):
        source >> t.tabyl("x", t.starts_with(""))
    with pytest.raises(TypeError, match="scalar"):
        source >> t.tabyl("x", "nested")
    with pytest.raises(TypeError, match="frame column"):
        t.tabyl(["a", "b"], "y")


def test_percentages_divide_exactly_as_in_r():
    counts = {"a": 3, "b": 2, "c": 7, "d": 11}
    values = [name for name, n in counts.items() for _ in range(n)]
    out = (t.tidy({"x": values + [None]}) >> t.tabyl("x")).collect()
    total, valid = sum(counts.values()) + 1, sum(counts.values())
    assert out["percent"].to_list() == [n / total for n in [*counts.values(), 1]]
    assert out["valid_percent"].to_list() == [n / valid for n in counts.values()] + [None]
    out = (t.tidy({"x": list("aaabb")}) >> t.tabyl("x")).collect()
    assert out["percent"].to_list() == [0.6, 0.4]
