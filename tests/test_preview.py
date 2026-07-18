"""Preview / display never full-scans large frames."""

from __future__ import annotations

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
