from __future__ import annotations

import pytest

from tidy3 import (
    all as tidy_all,
    any as tidy_any,
    between,
    col,
    consecutive_id,
    distinct,
    filter,
    first,
    group_by,
    mean,
    mutate,
    na_if,
    near,
    nth,
    nest_join,
    lag,
    lead,
    last,
    sd,
    summarise,
    tidy,
    ungroup,
    var,
)


BACKENDS = ["polars", "pandas"]


def as_pandas(frame):
    return frame.collect(as_="pandas").reset_index(drop=True)


@pytest.mark.parametrize("backend", BACKENDS)
def test_mutate_assignments_can_reference_columns_created_earlier(backend):
    out = as_pandas(
        tidy({"x": [1, 2, 3]}, backend=backend)
        >> mutate(
            doubled=col("x") * 2,
            squared=col("doubled") ** 2,
            total=col("doubled") + col("squared"),
        )
    )

    assert out["doubled"].tolist() == [2, 4, 6]
    assert out["squared"].tolist() == [4, 16, 36]
    assert out["total"].tolist() == [6, 20, 42]


@pytest.mark.parametrize("backend", BACKENDS)
def test_grouped_dependent_mutate_reuses_new_columns(backend):
    out = as_pandas(
        tidy(
            {"g": ["a", "a", "b"], "x": [1.0, 3.0, 10.0]},
            backend=backend,
        )
        >> mutate(centered=col("x") - mean("x"), scaled=col("centered") * 2, by="g")
    )

    assert out["centered"].tolist() == [-1.0, 1.0, 0.0]
    assert out["scaled"].tolist() == [-2.0, 2.0, 0.0]


@pytest.mark.parametrize("backend", BACKENDS)
def test_mutate_keep_and_placement_controls(backend):
    frame = tidy(
        {"id": [1, 2], "x": [3, 4], "unused": [8, 9]},
        backend=backend,
    )

    used = as_pandas(
        frame >> mutate(doubled=col("x") * 2, keep="used", before="x")
    )
    none = as_pandas(frame.mutate(doubled=col("x") * 2, keep="none"))

    assert list(used.columns) == ["doubled", "x"]
    assert list(none.columns) == ["doubled"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_distinct_projects_keys_unless_keep_all(backend):
    frame = tidy(
        {"x": [1, 1, 2], "label": ["first", "second", "third"]},
        backend=backend,
    )

    projected = as_pandas(frame >> distinct("x"))
    retained = as_pandas(frame >> distinct("x", keep_all=True))

    assert projected.to_dict(orient="list") == {"x": [1, 2]}
    assert retained.to_dict(orient="list") == {
        "x": [1, 2],
        "label": ["first", "third"],
    }


@pytest.mark.parametrize("backend", BACKENDS)
def test_distinct_includes_and_preserves_grouping_columns(backend):
    frame = tidy(
        {"g": ["a", "a", "b"], "x": [1, 1, 1], "y": [2, 3, 4]},
        backend=backend,
    ) >> group_by("g")

    result = frame >> distinct("x")

    assert as_pandas(result).to_dict(orient="list") == {
        "g": ["a", "b"],
        "x": [1, 1],
    }
    assert result._groups == ["g"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_summarise_grouping_policy_defaults_to_drop_last(backend):
    frame = tidy(
        {"g": ["a", "a", "b"], "h": [1, 2, 1], "x": [1, 3, 5]},
        backend=backend,
    ) >> group_by("g", "h")

    default = frame >> summarise(avg=mean("x"))
    kept = frame >> summarise(avg=mean("x"), groups="keep")
    dropped = frame >> summarise(avg=mean("x"), groups="drop")
    rowwise = frame >> summarise(avg=mean("x"), groups="rowwise")

    assert default._groups == ["g"]
    assert kept._groups == ["g", "h"]
    assert dropped._groups is None
    assert rowwise._groups == ["g", "h"]
    assert rowwise._rowwise


@pytest.mark.parametrize("backend", BACKENDS)
def test_computed_additive_grouping_and_partial_ungroup(backend):
    frame = tidy(
        {"region": ["a", "a", "b"], "x": [1, 11, 21]},
        backend=backend,
    )
    grouped = (
        frame
        >> group_by("region")
        >> group_by(add=True, decade=col("x") // 10)
    )
    partially = grouped >> ungroup("decade")

    assert grouped._groups == ["region", "decade"]
    assert as_pandas(grouped)["decade"].tolist() == [0, 1, 2]
    assert partially._groups == ["region"]


def test_new_grouping_arguments_are_validated():
    with pytest.raises(ValueError, match="summarise.*groups"):
        summarise(n=1, groups="sometimes")
    with pytest.raises(TypeError, match="group_by.*add"):
        group_by("x", add="yes")


@pytest.mark.parametrize("backend", BACKENDS)
def test_value_predicates_and_consecutive_ids(backend):
    frame = tidy(
        {"x": [1.0, 1.000000001, 3.0, 3.0], "flag": [True, True, False, True]},
        backend=backend,
    )
    out = as_pandas(
        frame
        >> mutate(
            close=near(col("x"), 1.0),
            cleaned=na_if(col("x"), 3.0),
            run=consecutive_id(col("x")),
        )
        >> filter(between("x", 1.0, 3.0, bounds="[)"))
    )

    assert out["close"].tolist() == [True, True]
    assert out["cleaned"].tolist() == [1.0, 1.000000001]
    assert out["run"].tolist() == [1, 2]


@pytest.mark.parametrize("backend", BACKENDS)
def test_nth_boolean_and_variance_aggregates(backend):
    frame = tidy(
        {
            "g": ["a", "a", "b", "b"],
            "x": [3.0, 1.0, 8.0, 4.0],
            "order": [2, 1, 2, 1],
            "flag": [True, False, True, True],
        },
        backend=backend,
    )
    out = as_pandas(
        frame
        >> summarise(
            first_ordered=nth("x", 1, order_by="order"),
            any_flag=tidy_any("flag"),
            all_flag=tidy_all("flag"),
            variance=var("x"),
            deviation=sd("x"),
            by="g",
        )
    )

    assert out["first_ordered"].tolist() == [1.0, 4.0]
    assert out["any_flag"].tolist() == [True, True]
    assert out["all_flag"].tolist() == [False, True]
    assert out["variance"].tolist() == pytest.approx([2.0, 8.0])
    assert out["deviation"].tolist() == pytest.approx([2**0.5, 8**0.5])


@pytest.mark.parametrize("backend", BACKENDS)
def test_aggregates_offer_explicit_missing_value_control(backend):
    frame = tidy(
        {
            "g": ["a", "a", "b", "b"],
            "x": [1.0, None, 2.0, 4.0],
            "flag": [True, None, False, None],
        },
        backend=backend,
    )
    out = as_pandas(
        frame
        >> summarise(
            removed=mean("x", na_rm=True),
            propagated=mean("x", na_rm=False),
            any_value=tidy_any("flag", na_rm=False),
            all_value=tidy_all("flag", na_rm=False),
            by="g",
        )
    )

    assert out["removed"].tolist() == [1.0, 3.0]
    assert out["propagated"].isna().tolist() == [True, False]
    assert out["any_value"].tolist()[0] is True
    assert out["any_value"].isna().tolist()[1]
    assert out["all_value"].isna().tolist()[0]
    assert out["all_value"].tolist()[1] is False


@pytest.mark.parametrize("backend", BACKENDS)
def test_nest_join_preserves_left_rows_and_collects_matches(backend):
    left = tidy(
        {"id": [1, 2, 3], "label": ["a", "b", "c"]},
        backend=backend,
    )
    right = {"id": [1, 1, 2], "value": [10, 11, 20]}

    result = left >> nest_join(right, by="id", name="matches")
    out = as_pandas(result)

    assert [[row["value"] for row in rows] for rows in out["matches"]] == [
        [10, 11],
        [20],
        [],
    ]


@pytest.mark.parametrize("backend", BACKENDS)
def test_ordered_lead_lag_first_and_last(backend):
    frame = tidy(
        {
            "g": ["a", "a", "a", "b", "b"],
            "x": [30, 10, 20, 5, 4],
            "order": [3, 1, 2, 2, 1],
        },
        backend=backend,
    )
    mutated = as_pandas(
        frame
        >> mutate(
            previous=lag("x", order_by="order"),
            following=lead("x", order_by="order"),
            by="g",
        )
    )
    summary = as_pandas(
        frame
        >> summarise(
            beginning=first("x", order_by="order"),
            ending=last("x", order_by="order"),
            by="g",
        )
    )

    assert mutated.loc[[0, 2], "previous"].tolist() == [20, 10]
    assert mutated.loc[[1, 2], "following"].tolist() == [20, 30]
    assert mutated.loc[1, "previous"] != mutated.loc[1, "previous"]
    assert mutated.loc[0, "following"] != mutated.loc[0, "following"]
    assert summary["beginning"].tolist() == [10, 4]
    assert summary["ending"].tolist() == [30, 5]
