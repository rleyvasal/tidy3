"""Group metadata: call and pipe forms, 0-based rows, group_walk, nest_by.

Values are compared with dplyr in test_r_oracle_parity.py (group_info).
"""

from __future__ import annotations

import pytest

from tidy3 import (
    group_by,
    group_data,
    group_indices,
    group_keys,
    group_rows,
    group_size,
    group_trim,
    group_vars,
    group_walk,
    groups,
    n_groups,
    nest_by,
    tidy,
)

BACKENDS = ["polars", "pandas"]
DATA = {"g": ["b", "a", "b", "a", "b"], "x": [1, 2, 3, 4, 5]}


@pytest.mark.parametrize("backend", BACKENDS)
def test_functions_work_called_or_piped(backend):
    grouped = tidy(DATA, backend=backend) >> group_by("g")
    for fn in (group_rows, group_size, group_indices, groups, group_vars, n_groups):
        assert (grouped >> fn()) == fn(grouped)
    for fn in (group_data, group_keys):
        piped = (grouped >> fn()).collect(as_="pandas")
        called = fn(grouped).collect(as_="pandas")
        assert piped.astype(str).equals(called.astype(str))


@pytest.mark.parametrize("backend", BACKENDS)
def test_rows_are_zero_based_and_group_numbers_one_based(backend):
    grouped = tidy(DATA, backend=backend) >> group_by("g")
    assert group_rows(grouped) == [[1, 3], [0, 2, 4]]
    assert group_indices(grouped) == [2, 1, 2, 1, 2]
    assert group_size(grouped) == [2, 3]


@pytest.mark.parametrize("backend", BACKENDS)
def test_group_walk_visits_sorted_groups_and_returns_the_frame(backend):
    grouped = tidy(DATA, backend=backend) >> group_by("g")
    seen = []
    result = grouped >> group_walk(lambda part, key: seen.append((key["g"], part.collect().shape[0])))
    assert seen == [("a", 2), ("b", 3)]
    assert result is grouped


@pytest.mark.parametrize("backend", BACKENDS)
def test_nest_by_is_rowwise_by_its_keys(backend):
    nested = tidy(DATA, backend=backend) >> nest_by("g")
    assert nested._rowwise and nested._groups == ["g"]
    assert nested.collect(as_="pandas")["g"].tolist() == ["a", "b"]


def test_group_trim_without_groups_returns_the_frame():
    frame = tidy(DATA)
    assert group_trim(frame) is frame
    assert group_size(frame) == [5]
