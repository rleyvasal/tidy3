"""dplyr's data mask in notebooks: a column beats a same-named variable.

Runs real IPython cells with ``%load_ext tidy3.jupyter`` on both backends.
"""

from __future__ import annotations

import pytest

pytest.importorskip("IPython")

BACKENDS = ["polars", "pandas"]


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


def setup(ip, backend, variables):
    ip.run_cell(
        f'd = tidy({{"g": ["a", "b", "a"], "x": [1.0, 2.0, 3.0], "n": [1, 2, 1]}}, backend="{backend}")\n'
        + variables,
        silent=True,
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_column_beats_variable_and_variable_fills_in(shell, backend):
    setup(shell, backend, "x = 100\nlimit = 1.5\nfactor = 10")
    assert run(shell, "d >> filter(x > limit)")["x"] == [2.0, 3.0]
    assert run(shell, "d >> mutate(z = x * factor)")["z"] == [10.0, 20.0, 30.0]


@pytest.mark.parametrize("backend", BACKENDS)
def test_env_forces_the_notebook_variable(shell, backend):
    setup(shell, backend, "x = 2.5")
    assert run(shell, "d >> filter(x > env.x)")["x"] == [3.0]


@pytest.mark.parametrize("backend", BACKENDS)
def test_mutate_sees_columns_created_earlier_in_the_same_call(shell, backend):
    setup(shell, backend, "y = 1000")
    out = run(shell, "d >> mutate(y = x * 2, w = y + 1)")
    assert out["w"] == [3.0, 5.0, 7.0]
    assert run(shell, "d >> mutate(w = y + 1)")["w"] == [1001, 1001, 1001]


@pytest.mark.parametrize("backend", BACKENDS)
def test_selections_prefer_columns_and_fall_back_to_values(shell, backend):
    setup(shell, backend, "g = 'zz'\nx = 'zz'\ncols = ['g', 'x']")
    assert list(run(shell, "d >> select(g, x)")) == ["g", "x"]
    assert list(run(shell, "d >> select([g, n])")) == ["g", "n"]
    assert list(run(shell, "d >> select(cols)")) == ["g", "x"]
    assert run(shell, "d >> count(g)")["nn"] == [2, 1]  # "nn": the frame has an n


@pytest.mark.parametrize("backend", BACKENDS)
def test_rebinding_a_tidy3_name_keeps_its_function(shell, backend):
    setup(shell, backend, "n = 7")
    out = run(shell, "d >> mutate(k = n(), m = x + n)")
    assert out["k"] == [3, 3, 3]  # n() still counts rows, as in R
    assert out["m"] == [2.0, 4.0, 4.0]  # bare n is the column


@pytest.mark.parametrize("backend", BACKENDS)
def test_local_names_and_function_arguments_are_untouched(shell, backend):
    setup(shell, backend, "score = 99\ndigits = 0")
    out = run(shell, "d >> filter(if_any(starts_with('x'), lambda score: score > 2))")
    assert out["x"] == [3.0]
    assert run(shell, "d >> mutate(r = (x / 3).round(digits))")["r"] == [0.0, 1.0, 1.0]


@pytest.mark.parametrize("backend", BACKENDS)
def test_all_of_takes_a_variable_of_names(shell, backend):
    # tidyselect: all_of(vars) reads the variable, even one the same cell
    # defines (the mask runs before the cell, so it cannot see it yet).
    shell.run_cell(
        f'df = tidy({{"a": [1.0, 3.0], "b": [2.0, 4.0], "c": ["x", "y"]}}, backend="{backend}")',
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


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_variable_made_earlier_in_the_cell_stays_a_variable(shell, backend):
    # kmeans = KMeans(...).fit(X) then mutate(cluster = kmeans.labels_ + 1)
    # in one cell: the mask runs before the cell, so it must read the cell's
    # own assignments, as a script's export does.
    shell.run_cell(
        f'df = tidy({{"x": [1.0, 2.0, 3.0]}}, backend="{backend}")', silent=True
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
