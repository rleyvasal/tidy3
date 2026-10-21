"""Shape changes retain grouping and the original conversion diagnostics."""

import pandas as pd
import pytest

import tidy3 as t


OPERATIONS = [
    pytest.param(t.pack("packed", "x"), 1, id="pack"),
    pytest.param(t.unpack("record"), 1, id="unpack"),
    pytest.param(t.hoist("record", nested_value="nested_value"), 1, id="hoist"),
    pytest.param(t.separate_wider_delim("code", names=["left", "right"], delim="-"), 1, id="wider_delim"),
    pytest.param(t.separate_wider_regex("code", [("left", "[a-z]"), "-", ("right", "[a-z]")]), 1, id="wider_regex"),
    pytest.param(t.separate_wider_position("code", [("left", 1), 1, ("right", 1)]), 1, id="wider_position"),
    pytest.param(t.separate_longer_delim("code", "-"), 2, id="longer_delim"),
    pytest.param(t.separate_longer_position("code", 2), 2, id="longer_position"),
]


def grouped_frame(backend):
    source = pd.DataFrame({
        "g": pd.Categorical(["b", "a", "b"], categories=["a", "b", "unused"], ordered=True),
        "code": ["x-y", "p-q", "r-s"], "x": [2, 4, 8],
        "record": [{"nested_value": 10}, {"nested_value": 20}, {"nested_value": 30}],
        "raw": ["bad", "2", "3"],
    }, index=[4, 4, 0])
    return (t.tidy(source, backend=backend) >> t.mutate(parsed=t.parse_integer("raw"))
            >> t.group_by("g", drop=False))


@pytest.mark.parametrize("backend", ["polars", "pandas"])
@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("operation,multiplier", OPERATIONS)
def test_reshape_preserves_groups_levels_and_original_problems(backend, empty, operation, multiplier):
    frame = grouped_frame(backend)
    if empty:
        frame = frame >> t.head(0)
    expected_problems = t.problems(frame).collect().to_dicts()
    out = frame >> operation
    assert out._groups == ["g"]
    assert out._group_drop is False
    assert out._category_levels["g"] == ["a", "b", "unused"]
    collected = out.collect(as_="pandas")["g"]
    assert collected.cat.categories.tolist() == ["a", "b", "unused"]
    assert collected.cat.ordered
    assert t.problems(out).collect().to_dicts() == expected_problems
    assert expected_problems == [{"row": 1, "column": "raw", "original": "bad", "expected": "integer"}]
    with pytest.raises(t.ConversionError, match="bad"):
        t.stop_for_problems(out)
    summary = (out >> t.summarise(rows=t.n())).collect()
    assert summary["g"].to_list() == ["a", "b", "unused"]
    assert summary["rows"].to_list() == ([0, 0, 0] if empty else [multiplier, 2 * multiplier, 0])


@pytest.mark.parametrize("backend", ["polars", "pandas"])
def test_packing_a_group_column_keeps_only_surviving_groups(backend):
    frame = grouped_frame(backend) >> t.group_by("g", "code", drop=False)
    out = frame >> t.pack("packed", "code")
    assert out._groups == ["g"]
    assert out._group_drop is False
    assert t.problems(out).collect().height == 1


@pytest.mark.parametrize("backend", ["polars", "pandas"])
def test_separating_a_group_column_keeps_only_surviving_groups(backend):
    frame = grouped_frame(backend) >> t.group_by("g", "code", drop=False)
    out = frame >> t.separate_wider_delim("code", names=["left", "right"], delim="-")
    assert out._groups == ["g"]
    assert out._group_drop is False
    assert t.problems(out).collect().height == 1
