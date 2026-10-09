"""dplyr 1.2 helpers: recode_values, replace_values, replace_when, when_any/all.

Results are compared with R in test_r_oracle_parity.py (dplyr_12_helpers);
these tests cover errors, argument checks, and notebook bare names.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tidy3 import col, mutate, recode_values, replace_values, tidy
from tidy3.masking import COL_NAME, apply_masking

BACKENDS = ["polars", "pandas"]
STATES = {"state": ["NC", "NYC", "CA", None, "Unknown"]}


@pytest.mark.parametrize("backend", BACKENDS)
def test_unmatched_error_raises_when_a_value_has_no_case(backend):
    frame = tidy(STATES, backend=backend)
    strict = recode_values("state", ("NC", 1), ("NYC", 2), ("CA", 3), unmatched="error")
    with pytest.raises(Exception, match="2 value\\(s\\) had no match"):
        (frame >> mutate(code=strict)).collect()


@pytest.mark.parametrize("backend", BACKENDS)
def test_unmatched_error_passes_when_every_value_matches(backend):
    frame = tidy(STATES, backend=backend)
    strict = recode_values(
        "state", ("NC", 1), (["NYC", "CA"], 2), (["Unknown", None], 0), unmatched="error"
    )
    out = (frame >> mutate(code=strict)).collect(as_="pandas")
    assert out["code"].tolist() == [1, 2, 2, 0, 0]


@pytest.mark.parametrize("backend", BACKENDS)
def test_lookup_table_recycles_a_single_target(backend):
    frame = tidy(STATES, backend=backend)
    out = (
        frame >> mutate(east=replace_values("state", from_=["NC", "NYC"], to="east"))
    ).collect(as_="pandas")
    assert out["east"].tolist()[:3] == ["east", "east", "CA"]
    assert pd.isna(out["east"].tolist()[3])


def test_argument_checks():
    with pytest.raises(TypeError, match="not both"):
        recode_values("state", ("NC", 1), from_=["NC"], to=[1])
    with pytest.raises(TypeError, match="both from_= and to="):
        recode_values("state", from_=["NC"])
    with pytest.raises(ValueError, match="length 1 or 2"):
        recode_values("state", from_=["NC", "CA"], to=[1, 2, 3])
    with pytest.raises(TypeError, match="default= can only be set"):
        recode_values("state", ("NC", 1), default=0, unmatched="error")
    with pytest.raises(ValueError, match="'default' or 'error'"):
        recode_values("state", ("NC", 1), unmatched="warn")


def test_bare_column_names_in_notebooks():
    out = ast_norm('mutate(code = recode_values(state, ("NC", 1)), any = when_any(a > 1, b))')
    assert f"recode_values({COL_NAME}('state'), ('NC', 1))" in out
    assert f"when_any({COL_NAME}('a') > 1, {COL_NAME}('b'))" in out


def ast_norm(source: str) -> str:
    import ast

    return ast.unparse(ast.parse(apply_masking(source)))


def test_case_values_and_none_match_missing():
    frame = tidy(STATES)
    out = (
        frame >> mutate(flag=recode_values(col("state"), (None, "missing"), default="ok"))
    ).collect(as_="pandas")
    assert out["flag"].tolist() == ["ok", "ok", "ok", "missing", "ok"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_case_match_none_matches_missing_values(backend):
    from tidy3 import case_match

    out = (
        tidy(STATES, backend=backend)
        >> mutate(flag=case_match("state", (None, "missing"), default="ok"))
    ).collect(as_="pandas")
    assert out["flag"].tolist() == ["ok", "ok", "ok", "missing", "ok"]
