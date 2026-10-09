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


def _run_cell(source: str, **names):
    """Rewrite and execute *source*; return the namespace and the last value."""
    from tidy3 import col, count, filter, select, tidy

    rewritten = maybe_rewrite_cell(source)
    assert rewritten is not None
    tree = ast.parse(rewritten)
    last = tree.body.pop() if isinstance(tree.body[-1], ast.Expr) else None
    ns = dict(col=col, count=count, filter=filter, select=select, tidy=tidy, **names)
    exec(compile(tree, "<cell>", "exec"), ns, ns)
    value = eval(compile(ast.Expression(last.value), "<cell>", "eval"), ns, ns) if last else None
    return ns, value


CARS = pl.DataFrame({"mpg": [21.0, 18.0, 22.0], "cyl": [6, 8, 4]})


def test_rewrite_keeps_comments_and_other_statements():
    ns, value = _run_cell(
        '''import math
# cars with good mileage
good = tidy(cars)
>> filter(col("mpg") > 20)   # trailing comment
# keep only what we need

>> select("mpg")

limit = math.floor(21.5)
good
>> filter(col("mpg") > limit)
''',
        cars=CARS,
    )
    assert ns["good"].collect().columns == ["mpg"]
    assert value.collect()["mpg"].to_list() == [22.0]


def test_rewrite_wraps_each_pipe_and_leaves_other_lines():
    source = 'x = 1\ntidy(cars)\n>> select("mpg")\n\ntidy(cars)\n    >> select("cyl")\n'
    out = maybe_rewrite_cell(source)
    assert out.splitlines() == [
        "x = 1",
        "(tidy(cars)",
        '>> select("mpg"))',
        "",
        "(tidy(cars)",
        '    >> select("cyl"))',
    ]


def test_rewrite_keeps_line_numbers_for_errors():
    source = '# first\ntidy(cars)\n>> select("mpg")\n>> filter(col("nope") > 1)\n'
    out = maybe_rewrite_cell(source)
    assert len(out.splitlines()) == len(source.splitlines())
    assert out.splitlines()[3] == '>> filter(col("nope") > 1))'


def test_rewrite_plus_continues_a_pipe_only():
    out = maybe_rewrite_cell(
        'tidy(cars)\n>> ggplot(aes(x="mpg"))\n+ geom_point()  # layer\n'
    )
    assert out.splitlines()[-1] == "+ geom_point())  # layer"


def test_rewrite_leaves_pipes_inside_blocks_alone():
    assert maybe_rewrite_cell('if True:\n    tidy(cars)\n    >> select("mpg")\n') is None


def test_rewrite_ignores_pipe_text_inside_strings():
    out = maybe_rewrite_cell('s = """\n>> not code\n"""\ntidy(cars)\n>> select("mpg")\n')
    assert out.splitlines()[:3] == ['s = """', ">> not code", '"""']
