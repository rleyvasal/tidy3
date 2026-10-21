"""Janitor-style duplicate records, including all members and key counts."""

import pandas as pd
import polars as pl
import pytest

import tidy3 as t


BACKENDS = ["polars", "pandas"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_get_dupes_returns_all_records_sorted_by_count_then_key(backend):
    frame = t.tidy({"row": list(range(1, 9)), "key": ["b", "a", "b", "c", "a", "a", None, None]}, backend=backend)
    out = (frame >> t.get_dupes("key")).collect()
    assert out.columns == ["key", "dupe_count", "row"]
    assert out["row"].to_list() == [2, 5, 6, 1, 3, 7, 8]
    assert out["dupe_count"].to_list() == [3, 3, 3, 2, 2, 2, 2]
    assert frame.columns == ["row", "key"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_get_dupes_default_keys_and_method_call(backend):
    frame = t.tidy({"a": ["b", "a", "b", "a", "x"], "b": [1, 2, 1, 2, 3]}, backend=backend)
    out = frame.get_dupes().collect()
    assert out.columns == ["a", "b", "dupe_count"]
    assert out["a"].to_list() == ["a", "a", "b", "b"]
    assert out["dupe_count"].to_list() == [2] * 4


@pytest.mark.parametrize("backend", BACKENDS)
def test_get_dupes_supports_composite_keys_and_tidyselect(backend):
    frame = t.tidy({"key_id": ["a", "a", "a", "b"], "key_sub": [1, 1, 2, 2], "value": [9, 8, 7, 6]}, backend=backend)
    for spec in [("key_id", "key_sub"), (t.starts_with("key"),), (["key_id", "key_sub"],), (t.everything() - "value",)]:
        out = (frame >> t.get_dupes(*spec)).collect()
        assert out.columns == ["key_id", "key_sub", "dupe_count", "value"]
        assert out["value"].to_list() == [9, 8]
    # An explicit empty selection uses one key combination for the table;
    # omitting arguments instead uses all columns, as janitor does.
    empty_keys = (frame >> t.get_dupes(t.any_of([]))).collect()
    assert empty_keys.columns == ["dupe_count", "key_id", "key_sub", "value"]
    assert empty_keys["dupe_count"].to_list() == [4] * 4


@pytest.mark.parametrize("backend", BACKENDS)
def test_get_dupes_is_global_and_preserves_group_metadata(backend):
    source = pd.DataFrame({"g": pd.Categorical(["b", "a", "c"], categories=["a", "b", "c", "unused"]), "id": [1, 1, 2]})
    frame = t.tidy(source, backend=backend) >> t.group_by("g", drop=False)
    out = frame >> t.get_dupes("id")
    assert out._groups == ["g"]
    assert out._group_drop is False
    assert out._category_levels == frame._category_levels
    assert out.collect()["dupe_count"].to_list() == [2, 2]
    assert out.collect().columns == ["id", "dupe_count", "g"]
    assert (out >> t.count()).collect()["n"].to_list() == [1, 1, 0, 0]


@pytest.mark.parametrize("backend", BACKENDS)
def test_get_dupes_respects_factor_order_and_unused_levels(backend):
    source = pd.DataFrame({"key": pd.Categorical(["a", "b", "a", "b", None, None], categories=["b", "a", "unused"], ordered=True), "row": range(6)})
    out = t.tidy(source, backend=backend) >> t.get_dupes("key")
    assert out.collect()["row"].to_list() == [1, 3, 0, 2, 4, 5]
    assert out._category_levels["key"] == ["b", "a", "unused"]
    assert (out >> t.count("key", drop=False)).collect()["n"].to_list() == [2, 2, 0, 2]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("values", [[], ["a"], ["a", "b"]])
def test_get_dupes_no_matches_retains_schema(backend, values):
    frame = t.tidy(pd.DataFrame({"key": pd.Series(values, dtype="string"), "row": pd.Series(range(len(values)), dtype="int64")}), backend=backend)
    out = (frame >> t.get_dupes("key")).collect()
    assert out.columns == ["key", "dupe_count", "row"]
    assert out.height == 0
    assert out["dupe_count"].dtype.is_integer()


@pytest.mark.parametrize("backend", BACKENDS)
def test_get_dupes_duplicate_pandas_indexes_and_diagnostics(backend):
    source = pd.DataFrame({"id": [2, 1, 2, 1], "raw": ["bad", "1", "2", "3"]}, index=[2, 2, 0, 0])
    frame = t.tidy(source, backend=backend) >> t.mutate(parsed=t.parse_integer("raw"))
    out = frame >> t.get_dupes("id")
    assert out.collect()["raw"].to_list() == ["1", "3", "bad", "2"]
    assert t.problems(out).collect()["original"].to_list() == ["bad"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_get_dupes_does_not_overwrite_count_column_or_ignore_missing_keys(backend):
    frame = t.tidy({"key": [1, 1], "dupe_count": [99, 100]}, backend=backend)
    with pytest.raises(ValueError, match="rename it first"):
        frame >> t.get_dupes("key")
    with pytest.raises(t.ColumnNotFoundError):
        t.tidy({"key": [1, 1]}, backend=backend) >> t.get_dupes("missing")


def test_get_dupes_stays_lazy(monkeypatch):
    frame = t.tidy({"key": [1, 1]})
    def no_collect(*args, **kwargs):
        raise AssertionError("get_dupes must not collect data when building the plan")
    monkeypatch.setattr(pl.LazyFrame, "collect", no_collect)
    out = frame >> t.get_dupes("key")
    assert out.columns == ["key", "dupe_count"]


def test_get_dupes_notebook_bare_columns_and_selectors():
    namespace = {name: getattr(t, name) for name in t.__all__}
    namespace["frame"] = t.tidy({"key": [1, 1, 2], "value": ["a", "b", "c"]})
    for call in ["get_dupes(key)", "get_dupes([key])", "get_dupes(~starts_with('value'))"]:
        exec("out = " + t.transform_source("frame >> " + call).strip(), namespace)
        assert namespace["out"].collect()["dupe_count"].to_list() == [2, 2]
