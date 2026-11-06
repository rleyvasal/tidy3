"""dplyr's data mask in notebooks: a column beats a same-named variable.

Runs real IPython cells with ``%load_ext tidy3.jupyter`` on both backends.
"""

from __future__ import annotations

import pytest

pytest.importorskip("IPython")



@pytest.fixture
def shell():
    from IPython.core.interactiveshell import InteractiveShell

    InteractiveShell.clear_instance()
    ip = InteractiveShell.instance()
    ip.run_line_magic("load_ext", "tidy3.jupyter")
    yield ip
    ip.run_line_magic("unload_ext", "tidy3.jupyter")
    InteractiveShell.clear_instance()


def run(ip, source):
    result = ip.run_cell("_out = " + source, silent=True)
    error = result.error_before_exec or result.error_in_exec
    if error is not None:
        raise error
    out = ip.user_ns.pop("_out")
    return out.collect(as_="pandas").to_dict("list") if hasattr(out, "collect") else out


def setup(ip, variables):
    ip.run_cell(
        f'd = tidy({{"g": ["a", "b", "a"], "x": [1.0, 2.0, 3.0], "n": [1, 2, 1]}})\n'
        + variables,
        silent=True,
    )


def test_column_beats_variable_and_variable_fills_in(shell):
    setup(shell, "x = 100\nlimit = 1.5\nfactor = 10")
    assert run(shell, "d >> filter(x > limit)")["x"] == [2.0, 3.0]
    assert run(shell, "d >> mutate(z = x * factor)")["z"] == [10.0, 20.0, 30.0]


def test_env_forces_the_notebook_variable(shell):
    setup(shell, "x = 2.5")
    assert run(shell, "d >> filter(x > env.x)")["x"] == [3.0]


def test_mutate_sees_columns_created_earlier_in_the_same_call(shell):
    setup(shell, "y = 1000")
    out = run(shell, "d >> mutate(y = x * 2, w = y + 1)")
    assert out["w"] == [3.0, 5.0, 7.0]
    assert run(shell, "d >> mutate(w = y + 1)")["w"] == [1001, 1001, 1001]


def test_selections_prefer_columns_and_fall_back_to_values(shell):
    setup(shell, "g = 'zz'\nx = 'zz'\ncols = ['g', 'x']")
    assert list(run(shell, "d >> select(g, x)")) == ["g", "x"]
    assert list(run(shell, "d >> select([g, n])")) == ["g", "n"]
    assert list(run(shell, "d >> select(cols)")) == ["g", "x"]
    assert run(shell, "d >> count(g)")["nn"] == [2, 1]  # "nn": the frame has an n


def test_rebinding_a_tidy3_name_keeps_its_function(shell):
    setup(shell, "n = 7")
    out = run(shell, "d >> mutate(k = n(), m = x + n)")
    assert out["k"] == [3, 3, 3]  # n() still counts rows, as in R
    assert out["m"] == [2.0, 4.0, 4.0]  # bare n is the column


def test_local_names_and_function_arguments_are_untouched(shell):
    setup(shell, "score = 99\ndigits = 0")
    out = run(shell, "d >> filter(if_any(starts_with('x'), lambda score: score > 2))")
    assert out["x"] == [3.0]
    assert run(shell, "d >> mutate(r = (x / 3).round(digits))")["r"] == [0.0, 1.0, 1.0]


def test_all_of_takes_a_variable_of_names(shell):
    # tidyselect: all_of(vars) reads the variable, even one the same cell
    # defines (the mask runs before the cell, so it cannot see it yet).
    shell.run_cell(
        f'df = tidy({{"a": [1.0, 3.0], "b": [2.0, 4.0], "c": ["x", "y"]}})',
        silent=True,
    )
    result = shell.run_cell(
        'features = ["a", "b"]\n'
        "picked = df >> select(all_of(features))\n"
        "scaled = df >> mutate(across(all_of(features), lambda x: x * 10))\n"
        "kept = df >> select(-any_of(features))",
        silent=True,
    )
    assert result.error_in_exec is None and result.error_before_exec is None
    ns = shell.user_ns
    assert list(ns["picked"].columns) == ["a", "b"]
    assert ns["scaled"].collect(as_="pandas")["b"].tolist() == [20.0, 40.0]
    assert list(ns["kept"].columns) == ["c"]


def test_a_variable_made_earlier_in_the_cell_stays_a_variable(shell):
    # kmeans = KMeans(...).fit(X) then mutate(cluster = kmeans.labels_ + 1)
    # in one cell: the mask runs before the cell, so it must read the cell's
    # own assignments, as a script's export does.
    shell.run_cell(
        f'df = tidy({{"x": [1.0, 2.0, 3.0]}})', silent=True
    )
    result = shell.run_cell(
        "import types, numpy as np\n"
        "model = types.SimpleNamespace(labels_=np.array([0, 1, 0]))\n"
        "limit = 1.5\n"
        "out = df >> mutate(cluster = model.labels_) >> filter(x > limit)",
        silent=True,
    )
    assert result.error_in_exec is None and result.error_before_exec is None
    out = shell.user_ns["out"].collect(as_="pandas")
    assert out["cluster"].tolist() == [1, 0]
    assert out["x"].tolist() == [2.0, 3.0]


def test_a_local_variable_in_another_function_leaves_the_column_alone(shell):
    # One cell, two functions: helper's local ``period`` must not turn the
    # bare ``period`` column in report into a Python name. period(...) is
    # still tidy3's function, and running report alone gives the same answer.
    shell.run_cell('df = tidy({"period": ["2024-02", "2024-05"]})', silent=True)
    report = (
        "def report(df):\n"
        "    return df >> mutate(\n"
        "        start = ym(period),\n"
        "        prev = start - months(1),\n"
        "        later = start + period(2, 'month'),\n"
        "        label = str_c(period, '!'),\n"
        "    )\n"
    )
    helper = (
        "def helper():\n"
        "    period = 'quarter'\n"
        "    return period\n"
    )
    for cell in (report, report + helper):
        result = shell.run_cell(cell + "out = report(df)", silent=True)
        assert result.error_in_exec is None and result.error_before_exec is None
        out = shell.user_ns["out"].collect(as_="pandas")
        assert [str(d)[:10] for d in out["start"]] == ["2024-02-01", "2024-05-01"]
        assert [str(d)[:10] for d in out["prev"]] == ["2024-01-01", "2024-04-01"]
        assert [str(d)[:10] for d in out["later"]] == ["2024-04-01", "2024-07-01"]
        assert out["label"].tolist() == ["2024-02!", "2024-05!"]
    assert shell.user_ns["helper"]() == "quarter"


def _cell(ip, source):
    result = ip.run_cell(source, silent=True)
    error = result.error_before_exec or result.error_in_exec
    if error is not None:
        raise error


def test_naming_a_result_after_a_column_it_uses(shell):
    # The right-hand side runs before ``x`` exists, so bare x is the column.
    setup(shell, "")
    _cell(shell, "x = d >> filter(x > 1.5)")
    assert shell.user_ns["x"].collect(as_="pandas")["x"].tolist() == [2.0, 3.0]
    # Re-running the cell: the stale x from the first run is not the column.
    _cell(shell, "x = d >> filter(x > 1.5)")
    assert shell.user_ns["x"].collect(as_="pandas")["x"].tolist() == [2.0, 3.0]
    _cell(shell, "x = 100")  # a stale scalar from an earlier cell too
    _cell(shell, "x = d >> filter(x > 1.5)")
    assert shell.user_ns["x"].collect(as_="pandas")["x"].tolist() == [2.0, 3.0]
    _cell(shell, "n = d >> mutate(k = n * 2)\nk = n >> select(k)")
    assert shell.user_ns["k"].collect(as_="pandas")["k"].tolist() == [2, 4, 2]


def test_a_name_assigned_later_in_the_cell_is_still_a_column_before(shell):
    setup(shell, "")
    _cell(shell, "out = d >> filter(x > 1.5)\nx = 'later'")
    assert shell.user_ns["out"].collect(as_="pandas")["x"].tolist() == [2.0, 3.0]
    assert shell.user_ns["x"] == "later"


def test_a_loop_body_sees_names_the_loop_binds_as_variables(shell):
    # A compound statement can run its body more than once: its own names
    # stay variables throughout, as before.
    setup(shell, "")
    _cell(
        shell,
        "kept = []\n"
        "for limit in (1.5, 2.5):\n"
        "    kept.append((d >> filter(x > limit)).collect(as_='pandas')['x'].tolist())",
    )
    assert shell.user_ns["kept"] == [[2.0, 3.0], [3.0]]


@pytest.mark.parametrize(
    "value",
    [
        "tidy({'x': [7.0, 8.0, 9.0]})",
        "[7.0, 8.0, 9.0]",
        "np.array([7.0, 8.0, 9.0])",
        "pl.Series([7.0, 8.0, 9.0])",
    ],
)
def test_column_beats_a_frame_list_array_or_series_variable(shell, value):
    setup(shell, "import numpy as np, polars as pl\nx = " + value)
    assert run(shell, "d >> filter(x > 1.5)")["x"] == [2.0, 3.0]
    assert run(shell, "d >> mutate(z = x * 10)")["z"] == [10.0, 20.0, 30.0]
    assert run(shell, "d >> arrange(-x)")["x"] == [3.0, 2.0, 1.0]
    assert run(shell, "d >> summarise(m = mean(x))")["m"] == [2.0]


def test_a_list_array_or_series_variable_fills_in_without_a_column(shell):
    setup(
        shell,
        "import numpy as np, polars as pl\n"
        "codes = np.array([7, 8, 9])\nweights = pl.Series([1, 0, 1])\nkeep = ['a']",
    )
    assert run(shell, "d >> mutate(z = codes + 1)")["z"] == [8, 9, 10]
    assert run(shell, "d >> mutate(z = x * weights)")["z"] == [1.0, 0.0, 3.0]
    assert run(shell, "d >> filter(g.is_in(keep))")["x"] == [1.0, 3.0]
    assert run(shell, "d >> mutate(z = codes)")["z"] == [7, 8, 9]


def test_a_function_variable_stays_a_function(shell):
    setup(shell, "def double(v):\n    return v * 2")
    assert run(shell, "d >> mutate(across(starts_with('x'), double))")["x"] == [2.0, 4.0, 6.0]
