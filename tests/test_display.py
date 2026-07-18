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


def test_mimebundle_has_plain_and_html():
    df = pl.DataFrame({"x": [1]})
    bundle = tidy(df)._repr_mimebundle_()
    assert "text/plain" in bundle
    assert "text/html" in bundle
