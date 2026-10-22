"""Group metadata: call and pipe forms, 1-based rows, group_walk, nest_by.

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
    slice,
    ungroup,
    tidy,
)

DATA = {"g": ["b", "a", "b", "a", "b"], "x": [1, 2, 3, 4, 5]}


def test_functions_work_called_or_piped():
    grouped = tidy(DATA) >> group_by("g")
    for fn in (group_rows, group_size, group_indices, groups, group_vars, n_groups):
        assert (grouped >> fn()) == fn(grouped)
    for fn in (group_data, group_keys):
        piped = (grouped >> fn()).collect(as_="pandas")
        called = fn(grouped).collect(as_="pandas")
        assert piped.astype(str).equals(called.astype(str))


def test_rows_and_group_numbers_are_one_based_like_dplyr():
    grouped = tidy(DATA) >> group_by("g")
    assert group_rows(grouped) == [[2, 4], [1, 3, 5]]
    # Positions work with slice(), which is 1-based too.
    first = (grouped >> ungroup() >> slice(*group_rows(grouped)[0])).collect(as_="pandas")
    assert first["g"].tolist() == ["a", "a"]
    assert group_indices(grouped) == [2, 1, 2, 1, 2]
    assert group_size(grouped) == [2, 3]


def test_nan_and_missing_float_keys_share_one_group():
    # pandas stores both as NaN; Polars keeps them apart but neither may drop rows.
    grouped = tidy({"g": [1.0, float("nan"), None, 1.0], "x": [1, 2, 3, 4]}) >> group_by("g")
    assert group_rows(grouped) == [[1, 4], [2, 3]]
    assert group_size(grouped) == [2, 2]
    assert group_indices(grouped) == [1, 2, 2, 1]
    assert n_groups(grouped) == 2
    assert group_data(grouped).collect()[".rows"].to_list() == [[1, 4], [2, 3]]


def test_keys_and_sizes_match_row_positions():
    data = {"g": [i % 7 for i in range(50)], "h": ["a" if i % 3 else None for i in range(50)]}
    grouped = tidy(data) >> group_by("g", "h")
    rows = group_rows(grouped)
    assert group_size(grouped) == [len(r) for r in rows]
    assert n_groups(grouped) == len(rows)
    keys = group_keys(grouped).collect()
    assert [(data["g"][r[0] - 1], data["h"][r[0] - 1]) for r in rows] == keys.rows()
    numbers = group_indices(grouped)
    assert all(numbers[p - 1] == i for i, r in enumerate(rows, start=1) for p in r)


def test_group_walk_visits_sorted_groups_and_returns_the_frame():
    grouped = tidy(DATA) >> group_by("g")
    seen = []
    result = grouped >> group_walk(lambda part, key: seen.append((key["g"], part.collect().shape[0])))
    assert seen == [("a", 2), ("b", 3)]
    assert result is grouped


def test_nest_by_is_rowwise_by_its_keys():
    nested = tidy(DATA) >> nest_by("g")
    assert nested._rowwise and nested._groups == ["g"]
    assert nested.collect(as_="pandas")["g"].tolist() == ["a", "b"]


def test_group_trim_without_groups_returns_the_frame():
    frame = tidy(DATA)
    assert group_trim(frame) is frame
    assert group_size(frame) == [5]


def test_group_trim_drops_unused_levels_from_the_column_too():
    import polars as pl

    species = pl.Enum(["Adelie", "Chinstrap", "Gentoo", "King"])
    frame = tidy(pl.DataFrame({"id": [1, 2, 3], "species": pl.Series(["Gentoo", "Adelie", "Gentoo"], dtype=species)}))
    trimmed = frame >> group_by("species", drop=False) >> group_trim()
    # dplyr: levels(group_trim(...)$species) is c("Adelie", "Gentoo").
    assert trimmed.collect()["species"].dtype == pl.Enum(["Adelie", "Gentoo"])
    assert group_size(trimmed) == [1, 2]


def test_rowwise_without_keys_has_one_group_per_row():
    from tidy3 import rowwise

    frame = tidy({"x": [5, 6, 7]}) >> rowwise()
    # dplyr: group_data(rowwise(df))$.rows is list(1L, 2L, 3L).
    assert group_data(frame).collect()[".rows"].to_list() == [[1], [2], [3]]
    assert group_rows(frame) == [[1], [2], [3]]
    assert group_size(frame) == [1, 1, 1]
    assert n_groups(frame) == 3
    assert group_indices(frame) == [1, 2, 3]
