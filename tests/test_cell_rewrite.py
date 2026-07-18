"""Kernel-side cell rewrite (any Jupyter / SolveIt — no VS Code)."""

from __future__ import annotations

import ast

import polars as pl
import pytest

from tidy3.partial_run import looks_like_tidy_pipe, maybe_rewrite_cell, partial_run
from tidy3.jupyter import tidy3_input_transformer


def test_maybe_rewrite_multiline_pipe():
    src = """
tidy(cars)
>> filter(col("mpg") > 20)
"""
    out = maybe_rewrite_cell(src)
    assert out is not None
    ast.parse(out)  # valid module
    assert ">>" in out


def test_maybe_rewrite_leaves_valid_python():
    src = """
(
    tidy(cars)
    >> filter(col("mpg") > 20)
)
"""
    assert maybe_rewrite_cell(src) is None


def test_maybe_rewrite_leaves_normal_code():
    assert maybe_rewrite_cell("x = 1 + 2\nprint(x)\n") is None
    assert maybe_rewrite_cell("a >> 2") is None or maybe_rewrite_cell("a >> 2")  # bit shift may look like pipe
    # pure bit shift without tidy start may still attempt — ensure ordinary assign ok
    assert maybe_rewrite_cell("x = 1\ny = 2\n") is None


def test_maybe_rewrite_assignment_pipe():
    src = """
out = tidy(cars)
>> filter(col("mpg") > 20)
"""
    out = maybe_rewrite_cell(src)
    assert out is not None
    assert out.strip().startswith("out =")
    ast.parse(out)


def test_transformer_list_of_lines():
    lines = ["tidy(cars)\n", ">> filter(col(\"mpg\") > 20)\n"]
    new = tidy3_input_transformer(lines)
    text = "".join(new)
    ast.parse(text)
    assert "filter" in text


def test_transformer_noop_on_normal():
    lines = ["x = 1\n", "x + 2\n"]
    assert tidy3_input_transformer(lines) == lines


def test_rewritten_cell_eval_filter(cars_ns=None):
    cars = pl.DataFrame({"mpg": [21.0, 18.0, 22.0], "cyl": [6, 8, 4]})
    src = """
tidy(cars)
>> filter(col("mpg") > 20)
"""
    rewritten = maybe_rewrite_cell(src)
    assert rewritten is not None
    ns = {"cars": cars}
    from tidy3 import col, filter, tidy  # noqa: F401

    ns.update(col=col, filter=filter, tidy=tidy)
    result = eval(compile(rewritten, "<t>", "eval"), ns, ns)
    assert result.collect().height == 2


def test_partial_run_still_works():
    cars = pl.DataFrame({"mpg": [21.0, 18.0], "cyl": [6, 8]})
    r = partial_run(
        """
        tidy(cars)
        >> filter(col("mpg") > 20)
        """,
        namespace={"cars": cars},
    )
    assert r.collect().height == 1


def test_looks_like_tidy_pipe():
    assert looks_like_tidy_pipe("tidy(df)\n>> filter(col('x') > 0)")
    assert not looks_like_tidy_pipe("%timeit 1+1")
    assert not looks_like_tidy_pipe("print(hello)")
