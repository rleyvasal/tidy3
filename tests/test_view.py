"""view(): a data explorer, as R's View(), through table3."""

import json
import re

import IPython
import IPython.display
import polars as pl
import pytest

from tidy3 import TidyFrame, arrange, distinct, group_by, options, tidy, view


@pytest.fixture
def islands():
    # 29 distinct values, three times over
    return tidy(island=[f"island {i:02d}" for i in range(29)] * 3)


@pytest.fixture
def notebook(monkeypatch):
    """Pretend to run in a notebook; collect what view() displays."""
    shown = []
    monkeypatch.setattr(IPython, "get_ipython", lambda: object())
    monkeypatch.setattr(IPython.display, "display", lambda obj: shown.append(obj.data))
    return shown


def _data(html: str) -> dict:
    block = re.search(r"<script type='application/json'>(.*?)</script>", html, re.S).group(1)
    return json.loads(block)


def test_explorer_holds_every_row(islands, notebook):
    out = islands >> distinct() >> view()
    assert isinstance(out, TidyFrame)
    (html,) = notebook
    assert "table3-explore-" in html
    data = _data(html)
    assert len(data["rows"]) == 29 and data["total"] == 29
    assert data["columns"] == [
        {"name": "island", "type": "str", "kind": "choice",
         "choices": [f"island {i:02d}" for i in range(29)]}
    ]
    assert ">TidyFrame (view)<" in html


def test_n_height_digits_and_groups(notebook):
    tf = tidy(g=["a", "b"] * 30, x=[0.000123456789] * 60) >> group_by("g")
    tf >> view(n=10, height=300, digits=6)
    (html,) = notebook
    data = _data(html)
    assert len(data["rows"]) == 10 and data["total"] == 60 and data["n"] == 10
    assert data["shown"][1][0] == "0.000123457"
    assert "data-height='300'" in html
    assert ">TidyFrame (view groups=[&#x27;g&#x27;])<" in html


def test_uncounted_rows(notebook):
    old = options().preview_count
    options(preview_count=False)
    try:
        tidy(x=list(range(50))) >> view(n=10)
    finally:
        options(preview_count=old)
    assert _data(notebook[0])["total"] is None


def test_pipe_passes_the_frame_and_leaves_the_data(capsys):
    tf = tidy(x=[3, 1, 2], y=["a", "b", None])
    before = tf.collect()
    out = tf >> view() >> arrange("x")
    assert out.pull("x").to_list() == [1, 2, 3]
    assert tf.collect().equals(before)
    assert "shape: (3, 2)" in capsys.readouterr().out  # rows as text outside a notebook


def test_called_on_data_method_and_series(notebook):
    tf = tidy(x=[1, 2])
    assert view(tf) is tf
    assert tf.view(height=200) is tf
    out = view(tf.pull("x"))
    assert isinstance(out, TidyFrame) and out.collect().columns == ["x"]
    assert isinstance(view({"y": [1.5]}), TidyFrame)
    assert isinstance(view(pl.DataFrame({"z": [1]})), TidyFrame)
    assert len(notebook) == 5


def test_hide_reaches_table3(monkeypatch, notebook):
    import table3

    seen = []
    monkeypatch.setattr(table3._solveit, "hide_calling_cell", lambda: seen.append(1))
    tidy(x=[1]) >> view()
    tidy(x=[1]) >> view(hide=False)
    tidy(x=[1]).view(hide=True)
    assert seen == [1, 1]


def test_arguments_checked():
    with pytest.raises(TypeError):
        view(300)
    for bad in ({"n": -1}, {"height": 0}, {"height": None}, {"digits": 0}):
        with pytest.raises(ValueError):
            view(**bad)
    with pytest.raises(TypeError, match="hide"):
        view(hide="yes")


def test_cell_ending_in_view_shows_the_frame_once(capsys):
    """Jupyter would print view()'s returned frame after the explorer."""
    from IPython.core.interactiveshell import InteractiveShell

    shell = InteractiveShell.instance()

    def run(code: str) -> str:
        capsys.readouterr()
        result = shell.run_cell(code)
        assert result.success, result.error_before_exec or result.error_in_exec
        return capsys.readouterr().out

    try:
        run("from tidy3 import arrange, tidy, view\ndf = tidy(x=[3, 1, 2])")
        assert "TidyFrame (preview" not in run("df >> view()")
        assert "TidyFrame (preview" not in run("df.view()")
        # Piped on, the result displays as usual.
        assert "TidyFrame (preview" in run("df >> view() >> arrange('x')")
        # Kept quiet only in the cell that viewed it.
        assert "TidyFrame (preview" not in run("v = df >> view()")
        assert "TidyFrame (preview" in run("v")
        assert "TidyFrame (preview" in run("v")
        assert shell.user_ns["v"].pull("x").to_list() == [3, 1, 2]
    finally:
        InteractiveShell.clear_instance()
