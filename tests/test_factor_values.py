"""Factor value changes, level metadata, and composition in real pipelines."""

import warnings

import numpy as np
import pandas as pd
import polars as pl
import pytest

import tidy3 as t


def frame():
    return t.tidy(pd.DataFrame({
        "f": pd.Categorical(["b", "a", "a", "c", None],
                            categories=["b", "a", "c", "unused"], ordered=True),
        "g": [1, 1, 2, 2, 2],
    }, index=[0, 0, 1, 1, 1]))


def factor(expr):
    return (frame() >> t.mutate(f=expr)).collect(as_="pandas")["f"]


def test_recode_changes_values_and_preserves_ordered_unused_levels():
    out = factor(t.fct_recode("f", renamed="a", spare="unused"))
    assert out.dropna().tolist() == ["b", "renamed", "renamed", "c"]
    assert out.cat.categories.tolist() == ["b", "renamed", "c", "spare"]
    assert out.cat.ordered and out.isna().sum() == 1
    merged = factor(t.fct_recode("f", b="a", NULL="c"))
    assert merged.cat.categories.tolist() == ["b", "unused"]
    assert merged.iloc[:3].tolist() == ["b", "b", "b"]
    assert merged.isna().sum() == 2 and merged.cat.ordered


def test_collapse_merges_levels_and_puts_other_last():
    out = factor(t.fct_collapse("f", pair=["a", "b"], other_level="rest"))
    assert out.cat.categories.tolist() == ["pair", "rest"]
    assert out.dropna().tolist() == ["pair", "pair", "pair", "rest"]
    assert out.cat.ordered
    # An unmentioned old level that already has the new name is kept too.
    out = factor(t.fct_collapse("f", b="a", other_level="rest"))
    assert out.dropna().tolist() == ["b", "b", "b", "rest"]
    overlapping = factor(t.fct_collapse("f", first=["a", "b"], last="a"))
    assert overlapping.cat.categories.tolist() == ["first", "last", "c", "unused"]


@pytest.mark.parametrize("fun, kwargs", [
    (t.fct_recode, {"new": "absent"}),
    (t.fct_collapse, {"new": ["a", "absent"]}),
])
def test_unknown_levels_warn(fun, kwargs):
    with pytest.warns(UserWarning, match="unknown levels.*absent"):
        factor(fun("f", **kwargs))


@pytest.mark.parametrize("method, kept", [
    ("min", ["a", "b"]), ("max", []), ("average", []),
    ("first", ["a"]), ("last", ["b"]),
])
def test_lump_n_tie_contract(method, kept):
    out = t.fct_lump_n(["a", "a", "b", "b", "c", None], 1, ties_method=method)
    assert out.categories.tolist() == kept + ["Other"]
    assert pd.isna(out[-1])


def test_lump_negative_thresholds_weights_and_existing_other():
    values = ["a", "a", "a", "b", "b", "c", None]
    assert t.fct_lump_n(values, -1).categories.tolist() == ["c", "Other"]
    assert t.fct_lump_n(values, 0).categories.tolist() == ["Other"]
    assert t.fct_lump_n(values, 20).categories.tolist() == ["a", "b", "c"]
    out = t.fct_lump_n(["a", "a", "b", "c"], 1, w=[1, 1, 5, 1])
    assert out.tolist() == ["Other", "Other", "b", "Other"]
    assert out.categories.tolist() == ["b", "Other"]
    out = t.fct_lump_min(["Other", "a", "a", "b"], 2)
    assert out.categories.tolist() == ["a", "Other"]
    assert out.tolist() == ["Other", "a", "a", "Other"]


def test_proportions_include_missing_rows_and_weights_at_exact_cutoff():
    values = ["a", "a", "b", None]
    assert t.fct_lump_prop(values, .25).tolist()[:3] == ["a", "a", "Other"]
    assert t.fct_lump_prop(values, -.25).tolist()[:3] == ["Other", "Other", "b"]
    out = t.fct_lump_prop(values, .2, w=[1, 1, 1, 7])
    assert out.categories.tolist() == ["Other"]
    assert t.fct_lump_min(values, 2).categories.tolist() == ["a", "Other"]


@pytest.mark.parametrize("values", [[], [None, None]])
def test_empty_missing_and_unused_levels(values):
    original = pd.Categorical(values, categories=["b", "a"], ordered=True)
    for fun in [lambda f: t.fct_recode(f, B="b"),
                lambda f: t.fct_collapse(f, pair=["a", "b"]),
                lambda f: t.fct_lump_min(f, 1),
                lambda f: t.fct_lump_n(f, 1)]:
        out = fun(original)
        assert len(out) == len(values) and out.ordered
        assert pd.isna(out).all()
    assert t.fct_lump_min(original, 0).categories.tolist() == ["b", "a"]
    empty = t.tidy(pd.DataFrame({"f": original[:0]}))
    out = (empty >> t.mutate(f=t.fct_collapse("f", pair=["a", "b"]))).collect()
    assert out.schema["f"] == pl.Enum(["pair"])


@pytest.mark.parametrize("transient", [False, True])
def test_grouped_lumping_changes_each_groups_values(transient):
    source = t.tidy({"g": [2, 2, 2, 1, 1, 1], "f": ["a", "a", "b", "b", "b", "a"]})
    expr = t.fct_lump_n("f", 1)
    out = (source >> t.mutate(f=expr, by="g") if transient else
           source >> t.group_by("g") >> t.mutate(f=expr))
    assert out.collect()["f"].to_list() == ["a", "a", "Other", "b", "b", "Other"]
    assert t.group_vars(out) == ([] if transient else ["g"])
    assert out.factor_levels()["f"] == (["a", "Other", "b"] if transient else ["b", "Other", "a"])


def test_nested_calls_across_and_sequential_assignments_use_changed_values():
    out = frame() >> t.mutate(
        f=t.fct_rev(t.fct_collapse("f", merged=["a", "b"])),
        renamed=t.fct_recode("f", X="merged"),
    )
    assert out.collect()["renamed"].to_list() == ["X", "X", "X", "c", None]
    assert out.factor_levels()["renamed"] == ["unused", "c", "X"]
    out = frame() >> t.mutate(t.across("f", lambda x: t.fct_lump_min(x, 2)))
    assert out.collect()["f"].to_list() == ["Other", "a", "a", "Other", None]


@pytest.mark.parametrize("ordered", [False, True])
def test_sequential_factor_stages_keep_the_updated_ordered_flag(ordered):
    source = t.tidy(pd.DataFrame({"f": pd.Categorical(["a", "b"], ordered=not ordered)}))
    out = source >> t.mutate(
        f=t.fct_infreq("f", ordered=ordered), renamed=t.fct_recode("f", A="a"),
    )
    assert out.collect(as_="pandas")["renamed"].cat.ordered == ordered
    assert out.collect()["renamed"].to_list() == ["A", "b"]


def test_snapshot_keeps_parse_problems_and_unrelated_dtypes(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("f,x\na,1\nb,oops\n")
    out = t.scan_csv(path) >> t.mutate(x=t.parse_integer("x"))
    out = out >> t.mutate(f=t.fct_recode("f", A="a"))
    path.unlink()
    assert out.collect()["f"].to_list() == ["A", "b"]
    assert out.collect().schema["x"] == pl.Int64
    assert t.problems(out).collect()["original"].to_list() == ["oops"]


def test_notebook_syntax_and_builtin_column_name():
    from IPython.core.interactiveshell import InteractiveShell

    shell = InteractiveShell.instance()
    try:
        shell.run_line_magic("load_ext", "tidy3.jupyter")
        result = shell.run_cell(
            'df = tidy({"id": ["a", "a", "b"], "w": [1, 1, 4]})\n'
            'out = df >> mutate(id=fct_recode(id, A="a"), lump=fct_lump_n(id, 1, w=w))'
        )
        assert result.success, result.error_before_exec or result.error_in_exec
        assert shell.user_ns["out"].collect()["lump"].to_list() == ["Other", "Other", "b"]
    finally:
        shell.run_line_magic("unload_ext", "tidy3.jupyter")
        InteractiveShell.clear_instance()


def test_random_ties_keep_one_tied_level():
    np.random.seed(42)
    out = t.fct_lump_n(["a", "a", "b", "b", "c"], 1, ties_method="random")
    assert len(out.categories) == 2 and out.categories[-1] == "Other"
    assert out.categories[0] in {"a", "b"}


@pytest.mark.parametrize("call, message", [
    (lambda: t.fct_recode(["a"], new=["a"]), "named strings"),
    (lambda: t.fct_collapse(["a"], new=[1]), "strings"),
    (lambda: t.fct_lump_n(["a"], float("nan")), "finite"),
    (lambda: t.fct_lump_n(["a"], 1, ties_method="dense"), "ties_method"),
    (lambda: t.fct_lump_min(["a"], -1), "non-negative"),
    (lambda: t.fct_lump_n(["a"], 1, w=[-1]), "non-negative"),
    (lambda: t.fct_lump_n(["a"], 1, w=[None]), "non-missing"),
    (lambda: t.fct_lump_n(["a"], 1, w=[float("inf")]), "finite"),
    (lambda: t.fct_lump_n(["a"], 1, w=[1, 2]), "same length"),
    (lambda: t.fct_lump_prop(["a"], .2, w=[0]), "positive total"),
])
def test_invalid_arguments(call, message):
    with pytest.raises((TypeError, ValueError), match=message):
        call()


def test_level_helpers_on_vectors_keep_values_and_ordered_flag():
    f = pd.Categorical(["b", "a", None, "b"], categories=["c", "b", "a", "d"], ordered=True)
    assert t.fct_drop(f).categories.tolist() == ["b", "a"]
    assert t.fct_drop(f, only="d").categories.tolist() == ["c", "b", "a"]
    assert t.fct_expand(f, ["e", "e"], "a", after=2).categories.tolist() == ["c", "b", "e", "a", "d"]
    assert t.fct_inorder(f).categories.tolist() == ["b", "a", "c", "d"]
    assert t.fct_inorder(f, ordered=False).ordered is False
    for out in (t.fct_drop(f), t.fct_expand(f, "e"), t.fct_inorder(f)):
        assert out.ordered and [None if pd.isna(v) else v for v in out] == ["b", "a", None, "b"]
    series = t.fct_inorder(pl.Series(["y", "x", "y"]))
    assert series.categories.tolist() == ["y", "x"]


def test_missing_value_conversions():
    f = pd.Categorical(["b", None, "a"], categories=["b", "a"])
    out = t.fct_na_value_to_level(f)
    assert out.tolist() == ["b", "(Missing)", "a"] and out.categories.tolist() == ["b", "a", "(Missing)"]
    assert t.fct_na_value_to_level(f, "a").tolist() == ["b", "a", "a"]
    assert t.fct_na_value_to_level(["x"], "M").categories.tolist() == ["x", "M"]
    back = t.fct_na_level_to_value(out, "(Missing)")
    assert back.categories.tolist() == ["b", "a"] and pd.isna(back[1])
    assert t.fct_na_level_to_value(f).categories.tolist() == ["b", "a"]


@pytest.mark.parametrize("values, levels, text", [
    (["z", "y", None, "z"], ["z", "y"], ["z", "y", None, "z"]),
    ([3, 1.5, None, 10], ["1.5", "3", "10"], ["3", "1.5", None, "10"]),
    ([2, 1], ["1", "2"], ["2", "1"]),
    ([True, None], ["FALSE", "TRUE"], ["TRUE", None]),
    ([], [], []),
])
def test_as_factor_levels(values, levels, text):
    out = t.as_factor(values)
    assert out.categories.tolist() == levels and not out.ordered
    assert [None if pd.isna(v) else v for v in out] == text


def test_as_factor_in_frames_converts_numbers_and_logical_columns():
    frame = t.tidy({"n": [2, None, 1], "b": [True, None, False], "s": ["q", "p", "q"]})
    out = (frame >> t.mutate(n=t.as_factor("n"), b=t.as_factor("b"), s=t.as_factor("s"))).collect()
    assert out["n"].to_list() == ["2", None, "1"] and out.schema["n"] == pl.Enum(["1", "2"])
    assert out.schema["b"] == pl.Enum(["FALSE", "TRUE"])
    assert out.schema["s"] == pl.Enum(["q", "p"])
    with pytest.raises(TypeError, match="as_character"):
        t.tidy({"d": [pd.Timestamp("2024-01-01")]}) >> t.mutate(d=t.as_factor("d"))


def test_reorder2_summaries_missing_values_and_custom_functions():
    f = pd.Categorical(["a", "a", "b", "b", "c"], categories=list("abcd"))
    x, y = [1, 2, 1, 2, 5], [10, 1, 3, 4, 2]
    assert t.fct_reorder2(f, x, y).categories.tolist() == ["b", "c", "a", "d"]
    assert t.fct_reorder2(f, x, y, desc=False).categories.tolist() == ["d", "a", "c", "b"]
    assert t.fct_reorder2(f, x, y, t.first2).categories.tolist() == ["a", "b", "c", "d"]
    total = t.fct_reorder2(f, x, y, lambda x, y, scale: scale * (x * y).sum(), scale=-1)
    assert total.categories.tolist() == ["c", "b", "a", "d"]
    with pytest.warns(UserWarning, match="removing 1 missing value;"):
        out = t.fct_reorder2(f, [1, None, 1, 2, 5], y)
    assert len(out) == 5 and out.categories.tolist() == ["a", "b", "c", "d"]
    kept = t.fct_reorder2(f, [1, None, 1, 2, 5], y, lambda x, y: float(x.isna().sum()), na_rm=False)
    assert kept.categories.tolist() == ["a", "b", "c", "d"]
    assert t.last2([None, 3, 3], [1, 2, 5]) == 2 and t.first2([None], [1]) is None


def test_chained_level_helpers_stay_lazy_casts_of_the_source():
    frame = t.tidy({"g": [1, 1, 2], "s": ["b", "a", None], "x": [1.0, 2.0, 3.0]})
    out = frame >> t.mutate(s=t.fct_expand(t.fct_inorder(t.as_factor("s")), "z"), x=t.col("x") * 2)
    data = out.collect()
    assert data.schema["s"] == pl.Enum(["b", "a", "z"]) and data["x"].to_list() == [2.0, 4.0, 6.0]
    grouped = frame >> t.group_by("g") >> t.mutate(s=t.fct_inorder("s"))
    assert grouped.factor_levels()["s"] == ["b", "a"] and t.group_vars(grouped) == ["g"]


def test_new_helpers_need_mutate_on_polars():
    with pytest.raises(TypeError, match="assigned with mutate"):
        (t.tidy({"s": ["a"]}) >> t.filter(t.fct_drop("s") == "a")).collect()


def test_notebook_bare_columns_in_level_helpers():
    from IPython.core.interactiveshell import InteractiveShell

    shell = InteractiveShell.instance()
    try:
        shell.run_line_magic("load_ext", "tidy3.jupyter")
        result = shell.run_cell(
            'df = tidy({"id": [2, 1, 2], "type": ["b", None, "a"], "x": [1, 2, 3]})\n'
            'out = df >> mutate(id=as_factor(id), type=fct_na_value_to_level(type),'
            ' r=fct_reorder2(type, x, x))'
        )
        assert result.success, result.error_before_exec or result.error_in_exec
        data = shell.user_ns["out"].collect()
        assert data["type"].to_list() == ["b", "(Missing)", "a"]
        assert data.schema["id"] == pl.Enum(["1", "2"])
    finally:
        shell.run_line_magic("unload_ext", "tidy3.jupyter")
        InteractiveShell.clear_instance()


@pytest.mark.parametrize("call, message", [
    (lambda: t.fct_expand(["a"], 1), "strings"),
    (lambda: t.fct_expand(["a"], "b", after=-1), "after"),
    (lambda: t.fct_drop(["a"], only=[1]), "strings"),
    (lambda: t.fct_inorder(["a"], ordered="yes"), "ordered"),
    (lambda: t.fct_na_value_to_level(["a"], None), "string"),
    (lambda: t.fct_na_level_to_value(["a"], 3), "strings"),
    (lambda: t.fct_reorder2(["a"], [1], [1], fun=None), "callable"),
    (lambda: t.fct_reorder2(["a"], [1], [1], desc=None), "desc"),
    (lambda: t.fct_reorder2(["a"], [1], [1], lambda x, y: [1, 2]), "one value"),
    (lambda: t.fct_reorder2(["a"], [1, 2], [1]), "same length"),
    (lambda: t.as_factor([1, "a"]), "as_character"),
])
def test_level_helper_invalid_arguments(call, message):
    with pytest.raises((TypeError, ValueError), match=message):
        call()


def _native_cases():
    return {
        "recode": lambda s, w: t.fct_recode(s, A="a", NULL="c"),
        "recode_merge": lambda s, w: t.fct_recode(s, b="a"),
        "collapse": lambda s, w: t.fct_collapse(s, ab=["a", "b"]),
        "collapse_other": lambda s, w: t.fct_collapse(s, ab=["a", "b"], other_level="rest"),
        "lump_n": lambda s, w: t.fct_lump_n(s, 1),
        "lump_n_ties": lambda s, w: t.fct_lump_n(s, 2, ties_method="first"),
        "lump_n_negative": lambda s, w: t.fct_lump_n(s, -1, w=w),
        "lump_prop": lambda s, w: t.fct_lump_prop(s, .25),
        "lump_prop_weighted": lambda s, w: t.fct_lump_prop(s, .3, w=w),
        "lump_min": lambda s, w: t.fct_lump_min(s, 2, other_level="b"),
        "drop": lambda s, w: t.fct_drop(s),
        "expand": lambda s, w: t.fct_expand(s, "z", "a", after=1),
        "inorder": lambda s, w: t.fct_inorder(s, ordered=True),
        "value_to_level": lambda s, w: t.fct_na_value_to_level(s),
        "value_to_existing": lambda s, w: t.fct_na_value_to_level(s, "b"),
        "level_to_value": lambda s, w: t.fct_na_level_to_value(s, ["a", "zz"]),
        "as_factor": lambda s, w: t.as_factor(s),
    }


@pytest.mark.parametrize("case", sorted(_native_cases()))
def test_string_columns_compute_levels_in_polars_with_the_same_results(case, monkeypatch):
    s = ["b", "a", None, "c", "a", "b", "a", None]
    w = [1., 5., 2., 1., 1., 1., 1., 3.]
    make = _native_cases()[case]
    expected = make(s, w)
    import tidy3.pandas_engine as engine

    monkeypatch.setattr(engine, "eval_expr", lambda *a, **k: pytest.fail("used the pandas path"))
    out = t.tidy({"s": s, "w": w}) >> t.mutate(f=make("s", "w"))
    data = out.collect()
    assert data["f"].to_list() == [None if pd.isna(v) else v for v in expected]
    assert data.schema["f"] == pl.Enum(expected.categories.tolist())
    assert out.collect(as_="pandas")["f"].cat.ordered == expected.ordered
    assert data["w"].to_list() == w


@pytest.mark.parametrize("values, levels", [
    ([3, None, 1, 3], ["1", "3"]),
    ([2.5, 1.0, None], ["1", "2.5"]),
    ([True, None, True], ["FALSE", "TRUE"]),
])
def test_as_factor_numbers_and_logical_values_in_polars(values, levels, monkeypatch):
    import tidy3.pandas_engine as engine

    monkeypatch.setattr(engine, "eval_expr", lambda *a, **k: pytest.fail("used the pandas path"))
    data = (t.tidy({"x": values}) >> t.mutate(x=t.as_factor("x"))).collect()
    assert data.schema["x"] == pl.Enum(levels)
    assert data["x"].to_list() == [None if pd.isna(v) else v for v in t.as_factor(values)]


@pytest.mark.parametrize("fun, na_rm, desc", [
    (t.last2, True, True), (t.first2, True, False), (t.last2, False, True), (t.first2, None, True),
])
def test_reorder2_with_last2_or_first2_runs_in_polars(fun, na_rm, desc, monkeypatch):
    s = ["a", "a", "b", "b", "c", None, "d"]
    x = [1., 2., 1., None, 5., 3., None]
    y = [10., 1., 3., 4., 2., 9., 1.]
    with warnings.catch_warnings(record=True) as expected_warnings:
        warnings.simplefilter("always")
        expected = t.fct_reorder2(s, x, y, fun, na_rm=na_rm, desc=desc)
    import tidy3.pandas_engine as engine

    monkeypatch.setattr(engine, "eval_expr", lambda *a, **k: pytest.fail("used the pandas path"))
    with warnings.catch_warnings(record=True) as found:
        warnings.simplefilter("always")
        out = (t.tidy({"s": s, "x": x, "y": y}) >> t.mutate(f=t.fct_reorder2("s", "x", "y", fun, na_rm=na_rm, desc=desc))).collect()
    assert out.schema["f"] == pl.Enum(expected.categories.tolist())
    assert out["f"].to_list() == s
    assert [str(w.message) for w in found] == [str(w.message) for w in expected_warnings]


def test_native_path_warns_and_falls_back_like_pandas():
    frame = t.tidy({"s": ["a", "b"], "w": [1.0, float("nan")], "g": [1, 2]})
    with pytest.warns(UserWarning, match="unknown levels.*zz"):
        frame >> t.mutate(f=t.fct_recode("s", A="zz"))
    with pytest.raises(ValueError, match="weights"):
        frame >> t.mutate(f=t.fct_lump_n("s", 1, w="w"))
    grouped = frame >> t.group_by("g") >> t.mutate(f=t.fct_lump_n("s", 1))
    assert grouped.collect()["f"].to_list() == ["a", "b"]
    nan = t.tidy({"x": [1.0, float("nan"), None]}) >> t.mutate(x=t.as_factor("x"))
    assert nan.collect()["x"].to_list() == ["1", None, None]
