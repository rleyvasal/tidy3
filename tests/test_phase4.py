from __future__ import annotations

import pandas as pd
import polars as pl
import pytest

from tidy3 import (
    anti_join,
    between,
    closest,
    full_join,
    ge,
    group_by,
    inner_join,
    join_by,
    left_join,
    overlaps,
    right_join,
    rows_append,
    rows_delete,
    rows_insert,
    rows_patch,
    rows_update,
    rows_upsert,
    semi_join,
    tidy,
    within,
)


BACKENDS = ["polars", "pandas"]


def base(backend: str):
    return tidy(
        {
            "id": [1, 2, 3],
            "value": ["a", "b", None],
            "score": [10.0, 20.0, 30.0],
        },
        backend=backend,
    )


def collect(frame):
    return frame.collect(as_="pandas")


@pytest.mark.parametrize("backend", BACKENDS)
def test_rows_insert_and_append(backend):
    inserted = base(backend) >> rows_insert({"id": [4], "value": ["z"]})
    out = collect(inserted)
    assert out["id"].tolist() == [1, 2, 3, 4]
    assert pd.isna(out.loc[3, "score"])

    appended = base(backend) >> rows_append({"id": [3], "value": ["again"]})
    assert collect(appended)["id"].tolist() == [1, 2, 3, 3]


@pytest.mark.parametrize("backend", BACKENDS)
def test_rows_insert_conflict_policies(backend):
    with pytest.raises(ValueError, match="already exist"):
        collect(base(backend) >> rows_insert({"id": [3], "value": ["z"]}))
    ignored = base(backend) >> rows_insert(
        {"id": [3, 4], "value": ["bad", "good"]}, conflict="ignore"
    )
    assert collect(ignored)["id"].tolist() == [1, 2, 3, 4]


@pytest.mark.parametrize("backend", BACKENDS)
def test_rows_update_and_patch(backend):
    updated = base(backend) >> rows_update(
        {"id": [2, 3], "value": ["B", "C"], "score": [None, 35.0]}
    )
    out = collect(updated)
    assert out["value"].tolist() == ["a", "B", "C"]
    assert pd.isna(out.loc[1, "score"])
    assert out.loc[2, "score"] == 35.0

    patched = base(backend) >> rows_patch(
        {"id": [2, 3], "value": ["B", "C"], "score": [25.0, 35.0]}
    )
    out = collect(patched)
    assert out["value"].tolist() == ["a", "b", "C"]
    assert out["score"].tolist() == [10.0, 20.0, 30.0]


@pytest.mark.parametrize("backend", BACKENDS)
def test_rows_upsert_preserves_x_order_then_appends(backend):
    result = base(backend) >> rows_upsert(
        {"id": [2, 4], "value": ["B", "D"]}, by="id"
    )
    out = collect(result)
    assert out["id"].tolist() == [1, 2, 3, 4]
    assert out.loc[0, "value"] == "a"
    assert out.loc[1, "value"] == "B"
    assert pd.isna(out.loc[2, "value"])
    assert out.loc[3, "value"] == "D"


@pytest.mark.parametrize("backend", BACKENDS)
def test_rows_delete_and_unmatched_policies(backend):
    deleted = base(backend) >> rows_delete({"id": [2]})
    assert collect(deleted)["id"].tolist() == [1, 3]

    with pytest.raises(ValueError, match="absent"):
        collect(base(backend) >> rows_delete({"id": [9]}))
    ignored = base(backend) >> rows_delete({"id": [9]}, unmatched="ignore")
    assert collect(ignored)["id"].tolist() == [1, 2, 3]


@pytest.mark.parametrize("backend", BACKENDS)
def test_rows_update_unmatched_and_duplicate_y_keys(backend):
    with pytest.raises(ValueError, match="absent"):
        collect(base(backend) >> rows_update({"id": [9], "value": ["x"]}))
    ignored = base(backend) >> rows_update(
        {"id": [2, 9], "value": ["B", "x"]}, unmatched="ignore"
    )
    out = collect(ignored)
    assert out.loc[0, "value"] == "a"
    assert out.loc[1, "value"] == "B"
    assert pd.isna(out.loc[2, "value"])

    with pytest.raises(ValueError, match="unique"):
        collect(
            base(backend)
            >> rows_update({"id": [2, 2], "value": ["B", "again"]})
        )


@pytest.mark.parametrize("backend", BACKENDS)
def test_row_operations_preserve_groups(backend):
    grouped = base(backend) >> group_by("id")
    result = grouped >> rows_upsert({"id": [4], "value": ["d"]})
    assert result._groups == ["id"]


def test_polars_row_operations_remain_lazy_before_collection():
    result = base("polars") >> rows_upsert({"id": [4], "value": ["d"]})
    assert isinstance(result._lf, pl.LazyFrame)


@pytest.mark.parametrize("backend", BACKENDS)
def test_join_by_equality_can_use_different_key_names(backend):
    left = tidy({"id": [1, 2], "left": ["a", "b"]}, backend=backend)
    right = {"key": [2, 3], "right": ["B", "C"]}
    result = left >> inner_join(right, by=join_by(("id", "key")))
    out = collect(result)
    assert list(out.columns) == ["id", "left", "right"]
    assert out.to_dict(orient="records") == [{"id": 2, "left": "b", "right": "B"}]


@pytest.mark.parametrize("backend", BACKENDS)
def test_join_by_equality_matches_null_keys(backend):
    left = tidy({"key": [None, 1], "left": ["missing", "one"]}, backend=backend)
    right = {"key": [None], "right": ["matched"]}
    out = collect(left >> inner_join(right, by=join_by("key")))
    assert len(out) == 1
    assert pd.isna(out.loc[0, "key"])
    assert out.loc[0, "right"] == "matched"


def temporal_frames(backend):
    left = tidy(
        {"id": [1, 1, 2, 3], "sale": [2, 5, 4, 1]}, backend=backend
    )
    right = {
        "id": [1, 1, 2, 4],
        "promo": [1, 5, 2, 1],
        "label": ["early", "exact", "two", "unused"],
    }
    return left, right


@pytest.mark.parametrize("backend", BACKENDS)
def test_join_by_inequality_and_left_unmatched_rows(backend):
    left, right = temporal_frames(backend)
    result = left >> left_join(right, by=join_by("id", ge("sale", "promo")))
    out = collect(result)
    assert list(out[["id", "sale", "promo"]].iloc[:4].itertuples(index=False, name=None)) == [
        (1, 2, 1),
        (1, 5, 1),
        (1, 5, 5),
        (2, 4, 2),
    ]
    assert out.loc[4, "id"] == 3
    assert pd.isna(out.loc[4, "promo"])


@pytest.mark.parametrize("backend", BACKENDS)
def test_join_by_closest_is_a_rolling_join(backend):
    left, right = temporal_frames(backend)
    result = left >> left_join(
        right, by=join_by("id", closest(ge("sale", "promo")))
    )
    out = collect(result)
    assert out["promo"].iloc[:3].tolist() == [1, 5, 2]
    assert pd.isna(out["promo"].iloc[3])


@pytest.mark.parametrize("backend", BACKENDS)
def test_join_by_between_and_filtering_joins(backend):
    points = tidy({"point": [1, 4, 7]}, backend=backend)
    ranges = {"lower": [0, 3], "upper": [2, 5], "name": ["a", "b"]}
    specification = join_by(between("point", "lower", "upper"))
    matched = points >> inner_join(ranges, by=specification)
    assert collect(matched)["name"].tolist() == ["a", "b"]
    assert collect(points >> semi_join(ranges, by=specification))["point"].tolist() == [1, 4]
    assert collect(points >> anti_join(ranges, by=specification))["point"].tolist() == [7]


@pytest.mark.parametrize("backend", BACKENDS)
def test_join_by_within_and_overlaps(backend):
    left = tidy({"lo": [1, 4, 8], "hi": [2, 6, 9]}, backend=backend)
    right = {"start": [0, 5], "end": [3, 10], "name": ["first", "second"]}
    inside = left >> inner_join(
        right, by=join_by(within("lo", "hi", "start", "end"))
    )
    assert collect(inside)["name"].tolist() == ["first", "second"]
    touching = left >> inner_join(
        right, by=join_by(overlaps("lo", "hi", "start", "end"))
    )
    assert collect(touching)["name"].tolist() == ["first", "second", "second"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_join_by_right_and_full_include_unmatched_right_rows(backend):
    left, right = temporal_frames(backend)
    specification = join_by("id", ge("sale", "promo"))
    right_result = collect(left >> right_join(right, by=specification))
    assert "unused" in right_result["label"].tolist()
    full_result = collect(left >> full_join(right, by=specification))
    assert set(full_result["label"].dropna()) == {"early", "exact", "two", "unused"}
    assert 3 in full_result["id"].tolist()


def test_join_by_polars_plan_remains_lazy():
    left = tidy({"point": [1, 4, 7]})
    right = {"lower": [0, 3], "upper": [2, 5]}
    result = left >> inner_join(
        right, by=join_by(between("point", "lower", "upper"))
    )
    assert isinstance(result._lf, pl.LazyFrame)
    assert "IEJOIN JOIN" in result.explain()


@pytest.mark.parametrize("backend", BACKENDS)
def test_phase4_method_api_matches_pipe_api(backend):
    data = base(backend)
    piped = data >> rows_upsert({"id": [4], "value": ["d"]})
    method = data.rows_upsert({"id": [4], "value": ["d"]})
    pd.testing.assert_frame_equal(
        collect(piped), collect(method), check_dtype=False
    )

    left, right = temporal_frames(backend)
    specification = join_by("id", ge("sale", "promo"))
    piped_join = left >> left_join(right, by=specification)
    method_join = left.left_join(right, by=specification)
    pd.testing.assert_frame_equal(
        collect(piped_join), collect(method_join), check_dtype=False
    )
