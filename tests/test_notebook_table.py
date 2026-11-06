"""A frame in a notebook prints as a paged table, as R notebooks print a data frame."""

import json
import re

import polars as pl
import pytest

from tidy3 import col, distinct, filter, group_by, options, tidy


@pytest.fixture
def islands():
    # 29 distinct values, three times over
    return tidy(island=[f"island {i:02d}" for i in range(29)] * 3)


@pytest.fixture
def opts():
    """Restore tidy3's options after a test changes them."""
    saved = options()
    keep = {k: getattr(saved, k) for k in ("preview_rows", "preview_max", "preview_digits", "preview_count")}
    yield options
    options(**keep)


def _body_rows(html: str) -> int:
    """Rows written as cells: the first page."""
    return html.split("<tbody>")[1].split("</tbody>")[0].count("<tr")


def _all_rows(html: str) -> int:
    """Every row the table pages through: the JSON its script draws, or the cells."""
    block = re.search(r"<script type='application/json'>(.*?)</script>", html, re.S)
    return len(json.loads(block.group(1))) if block else _body_rows(html)


def test_every_distinct_row_ten_a_page(islands):
    html = (islands >> distinct())._repr_html_()
    assert _body_rows(html) == 10 and _all_rows(html) == 29
    assert "island 28" in html
    assert "data-page-size='10'" in html and "data-total='29'" in html
    assert ">TidyFrame (preview)<" in html
    assert html.count("<script>") == 1  # pages in place; the first page without scripts


def test_preview_max_caps_the_rows(islands, opts):
    html = islands._repr_html_()
    assert _all_rows(html) == 87  # under the default 100
    opts(preview_max=20, preview_rows=5)
    html = islands._repr_html_()
    assert _body_rows(html) == 5 and _all_rows(html) == 20
    assert "data-page-size='5'" in html
    assert "data-total='87'" in html  # 20 previewed of 87: table3 writes the label
    assert "shape: (87, 1)" in html


def test_preview_max_below_a_page_still_fills_the_page(islands, opts):
    opts(preview_max=3, preview_rows=10)
    assert _all_rows(islands._repr_html_()) == 10


def test_preview_digits(opts):
    tf = tidy(x=[0.000123456789, 1.9015544041450778])
    assert ">0.000123<" in tf._repr_html_()
    opts(preview_digits=6)
    html = tf._repr_html_()
    assert ">0.000123457<" in html and ">1.90155<" in html
    opts(preview_digits="full")
    assert ">0.000123456789<" in tf._repr_html_()
    opts(preview_digits=None)
    assert "0.000123" in repr(tf)  # the text preview is polars' own


def test_groups_and_uncounted_rows(opts):
    tf = tidy(g=["a", "b"], x=[1, 2]) >> group_by("g")
    assert ">TidyFrame (preview groups=[&#x27;g&#x27;])<" in tf._repr_html_()
    opts(preview_count=False, preview_max=10)
    html = (tidy(x=list(range(50))) >> filter(col("x") >= 0))._repr_html_()
    assert "shape: (?, 1)" in html and "data-total=''" in html  # not counted


def test_table3_hook(islands):
    rows, total, caption = (islands >> distinct()).__table3__(5)
    assert isinstance(rows, pl.DataFrame) and rows.height == 5
    assert total == 29 and caption == "TidyFrame (preview)"


def test_long_strings_and_html_escaped():
    html = tidy(s=["<b>" + "y" * 100])._repr_html_()
    assert "<b>yyy" not in html
    assert re.search(r"&lt;b&gt;y{36}…</span>", html)


def test_bad_option_shows_as_a_failed_preview(opts):
    opts(preview_digits=0)
    html = tidy(x=[1.5])._repr_html_()
    assert "digits" in html and "<details" in html
