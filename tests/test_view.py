import IPython
import IPython.display
import polars as pl
import pytest
import view3

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


def test_view3_gets_rows_count_and_caption(islands):
    rows, total, caption = (islands >> distinct()).__view3__(5)
    assert isinstance(rows, pl.DataFrame) and rows.height == 5
    assert total == 29
    assert caption == "TidyFrame (view)"


def test_caption_keeps_groups():
    tf = tidy(g=["a", "b"], x=[1, 2]) >> group_by("g")
    assert tf.__view3__(10)[2] == "TidyFrame (view groups=['g'])"


def test_uncounted_preview_shows_question_mark():
    old = options().preview_count
    options(preview_count=False)
    try:
        rows, total, _ = tidy(x=list(range(50))).__view3__(10)
    finally:
        options(preview_count=old)
    assert rows.height == 10 and total is None


def test_every_distinct_row_in_the_table(islands, notebook):
    out = islands >> distinct() >> view()
    assert isinstance(out, TidyFrame)
    (html,) = notebook
    assert html.split("<tbody>")[1].count("<tr") == 29
    assert "island 28" in html and "data-label='29 rows'" in html
    assert ">TidyFrame (view)<" in html


def test_options_reach_the_table(islands, notebook):
    islands >> view(n=10, page_size=5, pages=True)
    islands >> view(pages=False, height=200)
    paged, box = notebook
    assert "data-page-size='5'" in paged and "(first 10 of 87, n=10)" in paged
    assert "<script" not in box and "max-height:200px" in box


def test_digits_reach_the_table(notebook):
    tidy(x=[0.000123456789, 1.9015544041450778]) >> view(digits=6)
    tidy(x=[0.000123456789]).view(digits="full")
    tidy(x=[0.000123456789]) >> view()
    sig, full, default = notebook
    assert ">0.000123457<" in sig and ">1.90155<" in sig
    assert ">0.000123456789<" in full
    assert ">0.000123<" in default


def test_pipe_passes_the_frame_and_leaves_the_data(capsys):
    tf = tidy(x=[3, 1, 2], y=["a", "b", None])
    before = tf.collect()
    out = tf >> view() >> arrange("x")
    assert out.pull("x").to_list() == [1, 2, 3]
    assert tf.collect().equals(before)
    text = capsys.readouterr().out  # rows as text outside a notebook
    assert text.startswith("TidyFrame (view)\nshape: (3, 2)")


def test_called_on_data_method_and_series(capsys):
    tf = tidy(x=[1, 2])
    assert view(tf) is tf
    assert tf.view(page_size=1) is tf
    out = view(tf.pull("x"))
    assert isinstance(out, TidyFrame) and out.collect().columns == ["x"]
    assert isinstance(view({"y": [1.5]}), TidyFrame)


def test_same_table_as_view3(islands):
    tf = islands >> distinct()
    html = view3.html(tf)
    assert html.split("<tbody>")[1].count("<tr") == 29


def test_arguments_checked():
    with pytest.raises(TypeError):
        view(300)
    with pytest.raises(ValueError):
        view(n=-1)
    with pytest.raises(ValueError):
        view(page_size=0)
    with pytest.raises(ValueError):
        view(height=0)
    with pytest.raises(TypeError):
        view(pages="yes")
    with pytest.raises(ValueError, match="digits"):
        view(digits=0)
