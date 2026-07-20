from __future__ import annotations

import pandas as pd
import pytest

from tidy3 import (
    arrange,
    bind_cols,
    col,
    consecutive_id,
    count,
    cummean,
    desc,
    distinct,
    first,
    group_by,
    mean,
    min_rank,
    mutate,
    n,
    n_distinct,
    nth,
    summarise,
    sum as tidy_sum,
    tidy,
)


BACKENDS = ["pandas", "polars"]


def as_pandas(frame):
    return frame.collect(as_="pandas")


@pytest.mark.parametrize("backend", BACKENDS)
def test_aggregate_defaults_propagate_missing_values_like_dplyr(backend):
    frame = tidy(
        {"g": ["a", "a", "b", "b"], "x": [1.0, None, 2.0, 4.0]},
        backend=backend,
    )
    out = as_pandas(
        frame
        >> summarise(
            default_mean=mean("x"),
            removed_mean=mean("x", na_rm=True),
            by="g",
        )
    )

    assert out["g"].tolist() == ["a", "b"]
    assert out["default_mean"].isna().tolist() == [True, False]
    assert out["removed_mean"].tolist() == [1.0, 3.0]


@pytest.mark.parametrize("backend", BACKENDS)
def test_numeric_cumulative_functions_propagate_missing_values(backend):
    frame = tidy(
        {
            "g": ["a", "b", "a", "b", "a"],
            "x": [1.0, 10.0, None, 20.0, 3.0],
        },
        backend=backend,
    )
    out = as_pandas(
        frame
        >> mutate(
            total=col("x").cum_sum(),
            average=cummean("x"),
            by="g",
        )
    )

    assert out["total"].tolist()[:2] == [1.0, 10.0]
    assert out["total"].isna().tolist() == [False, False, True, False, True]
    assert out["total"].tolist()[3] == 30.0
    assert out["average"].isna().tolist() == [False, False, True, False, True]
    assert out["average"].tolist()[3] == 15.0


@pytest.mark.parametrize("backend", BACKENDS)
def test_consecutive_id_treats_adjacent_missing_values_as_one_run(backend):
    frame = tidy(
        {"x": [None, None, 1.0, None, None, 2.0]}, backend=backend
    )
    out = as_pandas(frame >> mutate(run=consecutive_id("x")))

    assert out["run"].tolist() == [1, 1, 2, 3, 3, 4]


@pytest.mark.parametrize("backend", BACKENDS)
def test_consecutive_id_restarts_and_compares_within_groups(backend):
    frame = tidy(
        {
            "g": ["a", "b", "a", "b", "a"],
            "x": [1, 1, 1, 2, 2],
        },
        backend=backend,
    )
    out = as_pandas(
        frame >> mutate(run=consecutive_id("x"), by="g")
    )

    assert out["run"].tolist() == [1, 1, 1, 2, 2]


@pytest.mark.parametrize("backend", BACKENDS)
def test_arrange_is_stable_and_always_places_missing_values_last(backend):
    frame = tidy(
        pd.DataFrame(
            {"id": [1, 2, 3, 4], "x": [2.0, None, 2.0, 1.0]}
        ),
        backend=backend,
    )

    ascending = as_pandas(frame >> arrange("x"))
    descending = as_pandas(frame >> arrange(desc("x")))

    assert ascending["id"].tolist() == [4, 1, 3, 2]
    assert descending["id"].tolist() == [1, 3, 4, 2]


@pytest.mark.parametrize("backend", BACKENDS)
def test_persistent_groups_sort_but_transient_by_preserves_encounter_order(backend):
    frame = tidy(
        {"g": ["b", "a", "c", "a"], "x": [1, 2, 3, 4]},
        backend=backend,
    )

    persistent = as_pandas(
        frame >> group_by("g") >> summarise(rows=n(), groups="drop")
    )
    transient = as_pandas(frame >> summarise(rows=n(), by="g"))
    counted = as_pandas(frame >> count("g"))

    assert persistent["g"].tolist() == ["a", "b", "c"]
    assert transient["g"].tolist() == ["b", "a", "c"]
    assert counted["g"].tolist() == ["a", "b", "c"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_nth_default_only_applies_when_position_is_absent(backend):
    frame = tidy({"x": [None, 1.0]}, backend=backend)
    out = as_pandas(
        frame
        >> summarise(
            actual_missing=nth("x", 1, default=99),
            removed_missing=first("x", default=99, na_rm=True),
            absent=nth("x", 10, default=99),
            zero=nth("x", 0, default=99),
        )
    )

    assert pd.isna(out.loc[0, "actual_missing"])
    assert out.loc[0, "removed_missing"] == 1.0
    assert out.loc[0, "absent"] == 99
    assert out.loc[0, "zero"] == 99


@pytest.mark.parametrize("backend", BACKENDS)
def test_completed_core_parity_contracts(backend):
    frame = tidy(
        {"g": ["b", "a", "b"], "x": [3.0, 1.0, 2.0]},
        backend=backend,
    )
    arranged = as_pandas(
        frame >> group_by("g") >> arrange("x", by_group=True)
    )
    assert arranged[["g", "x"]].values.tolist() == [
        ["a", 1.0], ["b", 2.0], ["b", 3.0]
    ]

    ranked = as_pandas(frame >> mutate(rank=min_rank(desc("x"))))
    assert ranked["rank"].tolist() == [1, 3, 2]

    summary = as_pandas(
        frame >> summarise(a=tidy_sum("x"), b=tidy_sum(col("a")))
    )
    assert summary.to_dict(orient="records") == [{"a": 6.0, "b": 6.0}]

    unique = as_pandas(
        frame
        >> distinct(parity=col("x") % 2)
        >> summarise(n=n_distinct("parity", "parity"))
    )
    assert unique.loc[0, "n"] == 2


@pytest.mark.parametrize("backend", BACKENDS)
def test_mutate_deletes_left_to_right(backend):
    out = as_pandas(
        tidy({"x": [1, 2], "y": [3, 4]}, backend=backend)
        >> mutate(x=None)
    )
    assert out.to_dict(orient="list") == {"y": [3, 4]}

    with pytest.raises(KeyError, match="deleted earlier"):
        tidy({"x": [1]}, backend=backend) >> mutate(
            x=None, y=col("x")
        )


@pytest.mark.parametrize("backend", BACKENDS)
def test_bind_cols_repairs_names_and_recycles_size_one(backend):
    out = as_pandas(
        tidy({"x": [1, 2, 3]}, backend=backend)
        >> bind_cols(pd.DataFrame({"x": [9]}))
    )
    assert out.columns.tolist() == ["x...1", "x...2"]
    assert out["x...2"].tolist() == [9, 9, 9]


@pytest.mark.parametrize("backend", BACKENDS)
def test_group_by_drop_false_retains_unused_factor_levels(backend):
    frame = tidy(
        pd.DataFrame(
            {"g": pd.Categorical(["a"], categories=["a", "b"])}
        ),
        backend=backend,
    )
    out = as_pandas(
        frame
        >> group_by("g", drop=False)
        >> summarise(n=n(), groups="drop")
    )
    assert out["g"].tolist() == ["a", "b"]
    assert out["n"].tolist() == [1, 0]

    counted = as_pandas(frame >> count("g", drop=False))
    assert counted["g"].tolist() == ["a", "b"]
    assert counted["n"].tolist() == [1, 0]
