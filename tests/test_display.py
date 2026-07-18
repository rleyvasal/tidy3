"""Polars-style display (not pandas HTML tables)."""

from __future__ import annotations

import polars as pl

from tidy3 import tidy


def test_repr_looks_like_polars():
    df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    tf = tidy(df)
    r = repr(tf)
    assert "TidyFrame" in r
    assert "shape:" in r or "a" in r
    # should embed polars table text, not pandas Index dump
    assert "Index(" not in r


def test_html_uses_polars_table():
    df = pl.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
    html = tidy(df)._repr_html_()
    assert "TidyFrame" in html
    # Polars HTML includes a dtype header row (i64/f64) — pandas to_html does not
    assert "i64" in html or "f64" in html or "<pre" in html


def test_no_mimebundle_but_plain_and_html_reprs():
    # SolveIt shows text/plain when _repr_mimebundle_ exists; polars-style
    # separate __repr__ + _repr_html_ makes it render the HTML table.
    df = pl.DataFrame({"x": [1]})
    tf = tidy(df)
    assert not hasattr(tf, "_repr_mimebundle_")
    assert "TidyFrame" in repr(tf)
    assert "<table" in tf._repr_html_()


def test_html_is_self_contained_inline_styles():
    # CRAFT republishes remote output where <style> blocks / CSS classes are
    # lost — the table must carry all styling inline and escape content.
    df = pl.DataFrame({"a": [1, 2], "b": ["<x>", "y"]})
    html = tidy(df)._repr_html_()
    assert "<style" not in html
    assert "class=" not in html
    assert "style='" in html
    assert "&lt;x&gt;" in html
    assert "shape: (2, 2)" in html


def test_df_to_html_truncates_rows_and_cols():
    from tidy3.display import df_to_html

    df = pl.DataFrame({f"c{i}": list(range(40)) for i in range(30)})
    html = df_to_html(df, max_rows=5, max_cols=10)
    assert "shape: (40, 30)" in html
    assert "…" in html
    # only 5 body rows + ellipsis row rendered
    assert html.count("<tr>") <= 2 + 5 + 1
