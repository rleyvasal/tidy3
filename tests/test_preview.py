"""Preview / display never full-scans large frames."""

from __future__ import annotations

import re

import polars as pl

from tidy3 import col, filter, options, tidy


def test_preview_respects_limit():
    # Large-ish frame: preview must only take head(n)
    n = 50_000
    df = pl.DataFrame({"x": list(range(n)), "y": list(range(n))})
    tf = tidy(df) >> filter(col("x") >= 0)
    options(preview_rows=7)
    try:
        prev = tf.preview()
        assert prev.height == 7
        html = tf._repr_html_()
        assert "TidyFrame" in html
        # repr should not materialize all rows as text dump of 50k
        r = repr(tf)
        assert "preview" in r.lower() or "shape" in r.lower()
    finally:
        options(preview_rows=10)


def test_partial_run_preview_on_large(tmp_path=None):
    from tidy3 import partial_run

    n = 20_000
    cars = pl.DataFrame({"mpg": list(range(n)), "cyl": [4] * n})
    tf = partial_run(
        """
        tidy(cars)
        >> filter(col("mpg") > 100)
        """,
        namespace={"cars": cars},
    )
    prev = tf.preview(5)
    assert prev.height == 5
    assert (prev["mpg"] > 100).all()


def test_preview_shape_is_the_whole_frame_like_a_tibble():
    # A filtered lazy frame shows 10 rows, but its shape is all 30 matches.
    data = {"name": [f"chevrolet {i}" for i in range(30)] + ["ford"] * 5}
    for backend in ("polars", "pandas"):
        tf = tidy(data) >> filter(col("name") != "ford")
        text = repr(tf)
        assert "shape: (30, 1)" in text
        assert text.endswith("… 20 more rows")
        html = tf._repr_html_()
        assert "shape: (30, 1)" in html
        assert "… 20 more rows" in html


def test_preview_without_more_rows_has_no_footer():
    tf = tidy({"x": [1, 2, 3]})
    assert "shape: (3, 1)" in repr(tf)
    assert "more row" not in repr(tf)
    assert "more row" not in tf._repr_html_()
    tf = tidy({"x": list(range(11))})
    assert repr(tf).endswith("… 1 more row")


def test_preview_count_off_shows_unknown_rows():
    tf = tidy({"x": list(range(30))}) >> filter(col("x") >= 0)
    options(preview_count=False)
    try:
        assert "shape: (?, 1)" in repr(tf)
        assert repr(tf).endswith("… more rows")
        assert "shape: (?, 1)" in tf._repr_html_()
        # Few enough rows to see them all: the count is known without a pass.
        assert "shape: (3, 1)" in repr(tidy({"x": [1, 2, 3]}))
    finally:
        options(preview_count=True)


def _shown_rows(text: str) -> int:
    # A table row holds one state code; pandas puts its index in front.
    return len(re.findall(r"\bS\d\d\b", text))


def test_show_and_preview_rows_print_every_requested_row(capsys):
    # 50 state codes: Polars' own printer stops at 10, the HTML table used to stop at 25.
    states = {"abb": [f"S{i:02d}" for i in range(50)], "x": list(range(50))}
    for backend in ("polars", "pandas"):
        tf = tidy(states)
        tf.show(50)
        assert _shown_rows(capsys.readouterr().out) == 50
        options(preview_rows=50)
        try:
            assert _shown_rows(repr(tf)) == 50
            assert tf._repr_html_().count("<tr>") == 2 + 50  # names, dtypes, rows
        finally:
            options(preview_rows=10)
        assert _shown_rows(repr(tf)) == 10
        assert repr(tf).splitlines()[-1] == "… 40 more rows"
