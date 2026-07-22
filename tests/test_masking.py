"""R-style bare-name / backtick masking (source + AST)."""

from __future__ import annotations

import ast

import pytest

from tidy3.masking import (
    BT_NAME,
    COL_NAME,
    Tidy3MaskTransformer,
    apply_masking,
    default_known_names,
    rewrite_backticks,
    tidy3_backtick_transform,
)


def _norm(src: str) -> str:
    """Unparse-normalized form for stable compares."""
    return ast.unparse(ast.parse(apply_masking(src)))


def test_backtick_rewrite_to_sentinel():
    assert rewrite_backticks("select(`hp new`)") == f"select({BT_NAME}('hp new'))"
    assert rewrite_backticks("x = `a b` / cyl") == f"x = {BT_NAME}('a b') / cyl"


def test_backtick_transformer_lines():
    lines = ["mutate(x = `hp new` / cyl)\n"]
    out = tidy3_backtick_transform(lines)
    assert BT_NAME in "".join(out)
    assert "`" not in "".join(out)


def test_filter_bare_name_becomes_col():
    out = _norm("filter(mpg > 20)")
    assert COL_NAME in out
    assert 'mpg' in out
    # col("mpg") > 20
    assert f'{COL_NAME}("mpg")' in out or f"{COL_NAME}('mpg')" in out


def test_select_bare_name_becomes_string():
    out = _norm("select(mpg, cyl)")
    assert "mpg" in out and "cyl" in out
    assert COL_NAME not in out


def test_select_backtick_becomes_string():
    out = _norm("select(`hp new`, cyl)")
    assert "hp new" in out
    assert COL_NAME not in out


def test_mutate_expr_and_backtick():
    out = _norm('mutate(x = `hp new` / cyl, y = mpg * 2)')
    assert f'{COL_NAME}("hp new")' in out or f"{COL_NAME}('hp new')" in out
    assert f'{COL_NAME}("cyl")' in out or f"{COL_NAME}('cyl')" in out
    assert f'{COL_NAME}("mpg")' in out or f"{COL_NAME}('mpg')" in out


def test_if_else_args_masked_not_func_name():
    out = _norm("mutate(z = if_else(cyl > 4, 1, 0))")
    assert "if_else" in out
    assert f'{COL_NAME}("cyl")' in out or f"{COL_NAME}('cyl')" in out


def test_mean_string_arg_untouched():
    out = _norm('summarise(avg = mean("mpg"))')
    assert 'mean("mpg")' in out or "mean('mpg')" in out


def test_mean_bare_column_becomes_col():
    out = _norm("summarise(avg = mean(mpg))")
    assert f'{COL_NAME}("mpg")' in out or f"{COL_NAME}('mpg')" in out


def test_known_name_not_rewritten():
    known = default_known_names({"cars", "n", "mean"})
    out = apply_masking("filter(cars > 0)", known=known)
    # "cars" is known → left as Name
    tree = ast.parse(out)
    # still has a Name cars somewhere? filter(cars > 0) with known cars
    assert "cars" in out
    # should NOT be col("cars") if known
    assert f'{COL_NAME}("cars")' not in out and f"{COL_NAME}('cars')" not in out


def test_group_by_selector_args():
    out = _norm("group_by(cyl, gear)")
    assert COL_NAME not in out
    assert "cyl" in out and "gear" in out


def test_arrange_expr():
    out = _norm("arrange(desc(mpg))")
    assert "desc" in out
    assert f'{COL_NAME}("mpg")' in out or f"{COL_NAME}('mpg')" in out


def test_slice_max_order_by():
    out = _norm('slice_max(order_by=hp, n=1)')
    assert f'{COL_NAME}("hp")' in out or f"{COL_NAME}('hp')" in out


def test_pipe_with_filter_still_masks():
    src = "(tidy(cars) >> filter(mpg > 20) >> select(cyl, mpg))"
    out = _norm(src)
    assert f'{COL_NAME}("mpg")' in out or f"{COL_NAME}('mpg')" in out
    # select args strings
    assert COL_NAME in out  # from filter
    # select should use strings — check for Constant-like unparse
    assert "select(" in out


def test_transformer_is_idempotent_on_explicit_col():
    out = _norm('filter(col("mpg") > 20)')
    # col("mpg") stays a call to col / __tidy3_col__ not double-wrapped badly
    assert out.count("mpg") >= 1


def test_mutate_backtick_new_column_name():
    """R: mutate(`new hp` = hp * 1.1) → __tidy3_assign__('new hp', col('hp') * 1.1)."""
    from tidy3.masking import ASSIGN_NAME, rewrite_backtick_keyword_assigns

    raw = "mutate(`new hp` = hp * 1.1)"
    mid = rewrite_backtick_keyword_assigns(raw)
    assert ASSIGN_NAME in mid and "new hp" in mid
    out = _norm(raw)
    assert "new hp" in out
    assert f'{COL_NAME}("hp")' in out or f"{COL_NAME}('hp')" in out


def test_mutate_backtick_new_column_end_to_end():
    from tidy3 import mutate, tidy
    from tidy3.expr import col
    from tidy3.masking import ASSIGN_NAME, BT_NAME, COL_NAME, apply_masking, make_named_assign

    cars = tidy({"hp": [100, 200], "cyl": [4, 8]})
    src = apply_masking("(cars >> mutate(`new hp` = hp * 1.1))")
    ns = {
        "cars": cars,
        "mutate": mutate,
        "col": col,
        COL_NAME: col,
        BT_NAME: col,
        ASSIGN_NAME: make_named_assign,
    }
    out = eval(compile(src, "<t>", "eval"), ns).collect()
    assert "new hp" in out.columns
    vals = out["new hp"].to_list()
    assert abs(vals[0] - 110.0) < 1e-9 and abs(vals[1] - 220.0) < 1e-9
