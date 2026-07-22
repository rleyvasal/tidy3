"""Comprehensive matrix: every column-name context is masked correctly.

Covers bare names, backticks in expressions, and `` `new col` = expr ``
keyword assigns across verbs.
"""

from __future__ import annotations

import ast

import pytest

from tidy3 import (
    arrange,
    count,
    distinct,
    drop,
    filter,
    group_by,
    mutate,
    rename,
    select,
    slice_max,
    summarise,
    tidy,
    transmute,
)
from tidy3.expr import col
from tidy3.masking import (
    BT_NAME,
    COL_NAME,
    _ASSIGN_VERBS,
    _COUNT_VERBS,
    _EXPR_ARG_VERBS,
    _GROUP_VERBS,
    _RENAME_VERBS,
    _SELECTOR_ARG_VERBS,
    _SLICE_BY_VERBS,
    _SLICE_ORDER_VERBS,
    apply_masking,
    rewrite_backtick_keyword_assigns,
)


def _norm(src: str) -> str:
    return ast.unparse(ast.parse(apply_masking(src)))


def _ns(**extra):
    from tidy3.masking import ASSIGN_NAME, make_named_assign

    base = {
        "mutate": mutate,
        "transmute": transmute,
        "filter": filter,
        "select": select,
        "drop": drop,
        "rename": rename,
        "arrange": arrange,
        "group_by": group_by,
        "summarise": summarise,
        "distinct": distinct,
        "count": count,
        "slice_max": slice_max,
        "col": col,
        COL_NAME: col,
        BT_NAME: col,
        ASSIGN_NAME: make_named_assign,
    }
    base.update(extra)
    return base


def _eval_pipe(src: str, frame, **extra):
    code = apply_masking(src)
    ns = _ns(cars=frame, **extra)
    return eval(compile(code, "<matrix>", "eval"), ns)


# ── coverage inventory ───────────────────────────────────────────────────────


def test_all_public_column_verbs_are_classified():
    """No column-taking verb should be missing from the masking tables."""
    classified = (
        _EXPR_ARG_VERBS
        | _ASSIGN_VERBS
        | _SELECTOR_ARG_VERBS
        | _GROUP_VERBS
        | _COUNT_VERBS
        | _RENAME_VERBS
        | _SLICE_ORDER_VERBS
        | _SLICE_BY_VERBS
    )
    # Core dplyr-like verbs users write with bare names / backticks
    required = {
        "filter",
        "filter_out",
        "arrange",
        "mutate",
        "transmute",
        "summarise",
        "summarize",
        "reframe",
        "select",
        "drop",
        "relocate",
        "rename",
        "rename_with",
        "pull",
        "group_by",
        "ungroup",
        "count",
        "add_count",
        "tally",
        "add_tally",
        "distinct",
        "slice_min",
        "slice_max",
        "slice",
        "slice_head",
        "slice_tail",
        "rowwise",
        "drop_na",
        "fill",
        "nest",
        "pivot_longer",
        "pivot_wider",
    }
    missing = required - classified
    assert not missing, f"verbs missing from masking tables: {sorted(missing)}"


# ── source rewrite: `new col` = ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "src",
    [
        "mutate(`new hp` = hp * 1.1)",
        "transmute(`new hp` = hp)",
        "summarise(`avg mpg` = mean(mpg))",
        "reframe(`avg mpg` = mean(mpg))",
        "select(`new hp` = hp)",
        "rename(`new hp` = hp)",
        "group_by(`g col` = cyl > 4)",
        "distinct(cyl, `hi` = mpg > 20)",
    ],
)
def test_backtick_keyword_assign_rewrites_to_assign_sentinel(src: str):
    from tidy3.masking import ASSIGN_NAME

    mid = rewrite_backtick_keyword_assigns(src)
    assert ASSIGN_NAME in mid
    assert "`" not in mid


# ── AST masking by context ───────────────────────────────────────────────────


def test_filter_expr_and_by_selector():
    out = _norm("filter(mpg > 20, by=cyl)")
    assert f'{COL_NAME}("mpg")' in out or f"{COL_NAME}('mpg')" in out
    assert "cyl" in out


def test_mutate_rhs_and_new_name():
    out = _norm("mutate(`new hp` = hp * 1.1, z = mpg + cyl)")
    assert "new hp" in out
    assert f'{COL_NAME}("hp")' in out or f"{COL_NAME}('hp')" in out
    assert f'{COL_NAME}("mpg")' in out or f"{COL_NAME}('mpg')" in out


def test_select_bare_and_rename_backtick():
    out = _norm("select(mpg, cyl, `power` = hp)")
    assert "mpg" in out and "cyl" in out
    # rename value is selector string
    assert "power" in out
    assert f'{COL_NAME}("hp")' not in out  # selector mode → 'hp' not col


def test_rename_backtick_new_name():
    out = _norm("rename(`horse power` = hp)")
    assert "horse power" in out
    # old name as string selector
    assert "'hp'" in out or '"hp"' in out


def test_summarise_backtick_and_mean():
    out = _norm("summarise(`avg mpg` = mean(mpg), n = n())")
    assert "avg mpg" in out
    assert "mean" in out
    assert f'{COL_NAME}("mpg")' in out or f"{COL_NAME}('mpg')" in out


def test_group_by_selector_and_computed():
    out = _norm("group_by(cyl, big = mpg > 20)")
    assert "'cyl'" in out or '"cyl"' in out
    assert f'{COL_NAME}("mpg")' in out or f"{COL_NAME}('mpg')" in out


def test_count_and_tally_wt():
    out = _norm("count(cyl, wt=hp)")
    assert "'cyl'" in out or '"cyl"' in out
    assert f'{COL_NAME}("hp")' in out or f"{COL_NAME}('hp')" in out
    out2 = _norm("tally(wt=hp)")
    assert f'{COL_NAME}("hp")' in out2 or f"{COL_NAME}('hp')" in out2


def test_slice_max_order_by_and_by():
    out = _norm("slice_max(order_by=hp, n=1, by=cyl)")
    assert f'{COL_NAME}("hp")' in out or f"{COL_NAME}('hp')" in out
    assert "'cyl'" in out or '"cyl"' in out


def test_arrange_desc():
    out = _norm("arrange(desc(mpg), cyl)")
    assert "desc" in out
    assert f'{COL_NAME}("mpg")' in out or f"{COL_NAME}('mpg')" in out


def test_drop_and_pull_selectors():
    assert COL_NAME not in _norm("drop(hp, wt)")
    out = _norm("pull(mpg)")
    assert "'mpg'" in out or '"mpg"' in out


def test_distinct_computed_backtick():
    out = _norm("distinct(cyl, `is big` = mpg > 25)")
    assert "is big" in out
    assert f'{COL_NAME}("mpg")' in out or f"{COL_NAME}('mpg')" in out


# ── end-to-end evaluation ────────────────────────────────────────────────────


def test_e2e_mutate_spaced_column():
    cars = tidy({"hp": [100.0, 200.0], "mpg": [20.0, 30.0]})
    out = _eval_pipe("(cars >> mutate(`new hp` = hp * 1.1))", cars).collect()
    assert "new hp" in out.columns
    assert abs(out["new hp"][0] - 110.0) < 1e-9


def test_e2e_select_rename_spaced():
    cars = tidy({"hp": [1, 2], "mpg": [3, 4]})
    out = _eval_pipe("(cars >> select(`horse power` = hp, mpg))", cars).collect()
    assert "horse power" in out.columns
    assert "mpg" in out.columns
    assert "hp" not in out.columns


def test_e2e_rename_spaced():
    cars = tidy({"hp": [1, 2], "mpg": [3, 4]})
    out = _eval_pipe("(cars >> rename(`horse power` = hp))", cars).collect()
    assert "horse power" in out.columns
    assert "hp" not in out.columns


def test_e2e_filter_arrange_summarise_chain():
    cars = tidy(
        {
            "cyl": [4, 4, 6, 8],
            "mpg": [22.0, 24.0, 18.0, 15.0],
            "hp": [90.0, 95.0, 110.0, 200.0],
        }
    )
    src = """(
        cars
        >> filter(mpg > 16)
        >> mutate(`hp per cyl` = hp / cyl)
        >> group_by(cyl)
        >> summarise(n = n(), avg = mean(mpg))
        >> arrange(cyl)
    )"""
    from tidy3 import mean, n

    out = _eval_pipe(src, cars, mean=mean, n=n).collect()
    assert out.columns == ["cyl", "n", "avg"]
    assert out["cyl"].to_list() == [4, 6]


def test_e2e_filter_by_and_slice_max():
    cars = tidy(
        {
            "cyl": [4, 4, 6, 6],
            "hp": [80.0, 100.0, 120.0, 90.0],
            "mpg": [30.0, 25.0, 20.0, 22.0],
        }
    )
    out = _eval_pipe(
        "(cars >> slice_max(order_by=hp, n=1, by=cyl, with_ties=False))",
        cars,
    ).collect()
    assert out.height == 2
    # max hp per cyl: 100 for 4, 120 for 6
    assert set(out["hp"].to_list()) == {100.0, 120.0}
