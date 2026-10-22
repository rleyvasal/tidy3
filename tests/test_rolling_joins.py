from datetime import date, timedelta

import polars as pl
import pytest

import tidy3 as t


@pytest.mark.parametrize("temporal", [False, True])
@pytest.mark.parametrize("op, expected", [
    (t.ge, [(0, 0), (0, 1), (1, None), (2, 2), (3, None)]),
    (t.gt, [(0, 0), (0, 1), (1, None), (2, 0), (2, 1), (3, None)]),
    (t.le, [(0, 2), (1, 0), (1, 1), (2, 2), (3, None)]),
    (t.lt, [(0, 2), (1, 0), (1, 1), (2, None), (3, None)]),
])
def test_closest_preserves_ties_nulls_and_row_order(op, expected, temporal):
    convert = (lambda x: date(2024, 1, 1) + timedelta(days=x)) if temporal else lambda x: x
    left = t.tidy({"lid": [0, 1, 2, 3], "x": [convert(3), convert(1), convert(4), None]})
    right = {"rid": [0, 1, 2, 3], "y": [convert(2), convert(2), convert(4), None]}
    out = (left >> t.left_join(right, by=t.join_by(t.closest(op("x", "y"))))).collect()
    assert out.select("lid", "rid").rows() == expected


@pytest.mark.parametrize("op, expected", [
    (t.ge, [(0, 2), (1, 1), (2, None), (3, 2)]),
    (t.gt, [(0, 2), (1, 1), (2, None), (3, 1)]),
    (t.le, [(0, 0), (0, 3), (1, 2), (2, 1), (3, 2)]),
    (t.lt, [(0, None), (1, 2), (2, 1), (3, 0), (3, 3)]),
])
def test_closest_preserves_polars_nan_and_infinity_rules(op, expected):
    left = t.tidy({"lid": [0, 1, 2, 3], "x": [float("nan"), 2., float("-inf"), float("inf")]})
    right = {"rid": [0, 1, 2, 3], "y": [float("nan"), 1., float("inf"), float("nan")]}
    out = (left >> t.left_join(right, by=t.join_by(t.closest(op("x", "y"))))).collect()
    assert out.select("lid", "rid").rows() == expected


def test_closest_can_match_nan_when_no_non_nan_candidate_exists():
    left = t.tidy({"x": [float("nan")]})
    out = (left >> t.left_join({"y": [float("nan")], "rid": [1]}, by=t.join_by(t.closest(t.ge("x", "y"))))).collect()
    assert out["rid"].to_list() == [1]


@pytest.mark.parametrize("left_size, right_size", [(0, 0), (0, 2), (2, 0)])
def test_closest_empty_inputs_preserve_schema(left_size, right_size):
    left = t.tidy(pl.DataFrame({"x": pl.Series(range(left_size), dtype=pl.Int64)}))
    right = pl.DataFrame({"y": pl.Series(range(right_size), dtype=pl.Int64)})
    out = (left >> t.left_join(right, by=t.join_by(t.closest(t.ge("x", "y"))))).collect()
    assert out.schema == {"x": pl.Int64, "y": pl.Int64}
    assert out.height == left_size
    assert out["y"].null_count() == left_size


@pytest.mark.parametrize("na_matches, expected", [("na", [1, 2, 3]), ("never", [None, 2, None])])
def test_closest_multiple_equality_keys_and_null_groups(na_matches, expected):
    left = t.tidy({"a": [None, "x", "x"], "b": [1, 1, None], "x": [3, 3, 3]}) >> t.group_by("a")
    right = {"c": [None, "x", "x"], "d": [1, 1, None], "y": [2, 2, 2], "rid": [1, 2, 3]}
    out = left >> t.left_join(right, by=t.join_by(("a", "c"), ("b", "d"), t.closest(t.ge("x", "y"))), na_matches=na_matches)
    assert out.collect()["rid"].to_list() == expected
    assert out._groups == ["a"]


def test_closest_applies_other_inequalities_before_choosing():
    left = t.tidy({"x": [5]})
    right = {"y": [4, 3], "upper": [2, 10]}
    out = (left >> t.left_join(right, by=t.join_by(t.closest(t.ge("x", "y")), t.le("x", "upper")))).collect()
    assert out["y"].to_list() == [3]


@pytest.mark.parametrize("kwargs, message", [
    ({"relationship": "many-to-one"}, "relationship.*violated"),
    ({"unmatched": "error"}, "unmatched='error'"),
])
def test_closest_keeps_relationship_and_unmatched_checks(kwargs, message):
    left = t.tidy({"x": [3]})
    right = {"y": [2, 2, 4]}
    with pytest.raises(ValueError, match=message):
        (left >> t.left_join(right, by=t.join_by(t.closest(t.ge("x", "y"))), **kwargs)).collect()


@pytest.mark.parametrize("grouped", [False, True])
def test_large_closest_join_builds_lazily_and_searches_only_keys(monkeypatch, grouped):
    rows = 200_000
    left = t.tidy(pl.DataFrame({"x": range(rows), "g": [None] * rows}).with_columns(
        *(pl.col("x").alias(f"payload_{i}") for i in range(20))
    ))
    right = pl.DataFrame({"y": range(0, rows, 10), "g": [None] * (rows // 10)})
    original = pl.LazyFrame.join_asof
    searches = []

    def search(self, other, **kwargs):
        searches.append((len(self.collect_schema()), len(other.collect_schema())))
        return original(self, other, **kwargs)

    def unexpected_collect(*args, **kwargs):
        raise AssertionError("building a rolling join must not collect its source")

    monkeypatch.setattr(pl.LazyFrame, "join_asof", search)
    with monkeypatch.context() as building:
        building.setattr(pl.LazyFrame, "collect", unexpected_collect)
        spec = t.join_by(*(["g"] if grouped else []), t.closest(t.ge("x", "y")))
        result = left >> t.left_join(right, by=spec)
    assert searches == ([(3, 2)] if grouped else [(2, 1)])
    out = result.collect()
    assert out.height == rows
    assert out["y"].to_list() == [i // 10 * 10 for i in range(rows)]
