"""Partial-run: select tidy >> filter → filtered intermediate."""

from __future__ import annotations

import polars as pl
import pytest

from tidy3 import TidyFrame, normalize_pipe_source, partial_run, tidy


@pytest.fixture
def cars_ns():
    cars = pl.DataFrame(
        {
            "mpg": [21.0, 22.8, 21.4, 18.7, 18.1, 14.3],
            "cyl": [6, 4, 6, 8, 6, 8],
            "hp": [110, 93, 110, 175, 105, 245],
        }
    )
    return {"cars": cars}


def test_normalize_wraps_multiline():
    src = """
    tidy(cars)
    >> filter(col("mpg") > 20)
    """
    code = normalize_pipe_source(src)
    assert code.strip().startswith("(")
    assert ">>" in code


def test_normalize_already_expression():
    src = 'tidy(cars) >> filter(col("mpg") > 20)'
    code = normalize_pipe_source(src)
    # may or may not re-wrap; must be valid eval expression
    compile(code, "<t>", "eval")


def test_partial_run_filter_only(cars_ns):
    """Selecting tidy >> filter shows filtered data — not later verbs."""
    source = """
    tidy(cars)
    >> filter(col("mpg") > 20)
    """
    result = partial_run(source, namespace=cars_ns)
    assert isinstance(result, TidyFrame)
    out = result.collect()
    assert out.height == 3
    assert (out["mpg"] > 20).all()
    # mutate not applied
    assert "km" not in out.columns


def test_partial_run_filter_then_mutate(cars_ns):
    source = """
    tidy(cars)
    >> filter(col("mpg") > 20)
    >> mutate(km=col("mpg") * 1.609)
    """
    result = partial_run(source, namespace=cars_ns)
    out = result.collect()
    assert out.height == 3
    assert "km" in out.columns


def test_partial_run_longer_prefix_differs(cars_ns):
    a = partial_run(
        """
        tidy(cars)
        >> filter(col("mpg") > 20)
        """,
        namespace=cars_ns,
    ).collect()
    b = partial_run(
        """
        tidy(cars)
        >> filter(col("mpg") > 20)
        >> mutate(km=col("mpg") * 1.609)
        """,
        namespace=cars_ns,
    ).collect()
    assert "km" not in a.columns
    assert "km" in b.columns
    assert a.height == b.height == 3


def test_partial_run_drops_trailing_pipe(cars_ns):
    source = """
    tidy(cars)
    >> filter(col("mpg") > 20)
    >>
    """
    result = partial_run(source, namespace=cars_ns)
    assert result.collect().height == 3


def test_preview_does_not_need_full_collect(cars_ns):
    """_repr_ / preview uses head only."""
    tf = partial_run(
        """
        tidy(cars)
        >> filter(col("mpg") > 20)
        """,
        namespace=cars_ns,
    )
    prev = tf.preview(2)
    assert prev.height == 2
    # HTML repr should work
    html = tf._repr_html_()
    assert "TidyFrame" in html
    assert "mpg" in html


def test_partial_run_rejects_leading_pipe(cars_ns):
    with pytest.raises(ValueError, match="start of the pipe"):
        partial_run(
            """
            >> filter(col("mpg") > 20)
            """,
            namespace=cars_ns,
        )


def test_partial_run_rejects_leading_dot(cars_ns):
    with pytest.raises(ValueError, match="leading"):
        partial_run(
            """
            .filter(col("mpg") > 20)
            """,
            namespace=cars_ns,
        )


def test_partial_run_named_frame(cars_ns):
    cars_ns["tf"] = tidy(cars_ns["cars"])
    result = partial_run(
        """
        tf
        >> filter(col("mpg") > 20)
        """,
        namespace=cars_ns,
    )
    assert result.collect().height == 3
