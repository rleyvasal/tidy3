from __future__ import annotations

import pandas as pd
import polars as pl
import pytest

from tidy3 import (
    complete,
    drop_na,
    expand,
    fill,
    group_by,
    nest,
    nesting,
    pivot_longer,
    pivot_wider,
    replace_na,
    separate,
    starts_with,
    tidy,
    unite,
    unnest,
    unnest_longer,
    unnest_wider,
)


BACKENDS = ["polars", "pandas"]


def as_pandas(frame):
    return frame.collect(as_="pandas").reset_index(drop=True)


@pytest.mark.parametrize("backend", BACKENDS)
def test_drop_na_selected_columns_and_replace_na(backend):
    frame = tidy(
        {
            "id": [1, 2, 3, 4],
            "x": [1.0, None, float("nan"), 4.0],
            "y": [None, 2.0, 3.0, 4.0],
        },
        backend=backend,
    )

    dropped = as_pandas(frame >> drop_na("x"))
    replaced = as_pandas(frame >> replace_na({"x": 0.0, "y": -1.0}))

    assert dropped["id"].tolist() == [1, 4]
    assert replaced["x"].tolist() == [1.0, 0.0, 0.0, 4.0]
    assert replaced["y"].tolist() == [-1.0, 2.0, 3.0, 4.0]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    ("direction", "expected"),
    [
        ("down", [None, 2.0, 2.0, None, 4.0, 4.0]),
        ("up", [2.0, 2.0, None, 4.0, 4.0, None]),
        ("downup", [2.0, 2.0, 2.0, 4.0, 4.0, 4.0]),
        ("updown", [2.0, 2.0, 2.0, 4.0, 4.0, 4.0]),
    ],
)
def test_fill_directions_respect_transient_groups(backend, direction, expected):
    frame = tidy(
        {
            "group": ["a", "a", "a", "b", "b", "b"],
            "x": [None, 2.0, None, None, 4.0, None],
        },
        backend=backend,
    )
    out = as_pandas(frame >> fill("x", direction=direction, by="group"))

    actual = out["x"].tolist()
    for value, wanted in zip(actual, expected):
        if wanted is None:
            assert pd.isna(value)
        else:
            assert value == wanted


@pytest.mark.parametrize("backend", BACKENDS)
def test_fill_uses_persistent_groups_and_preserves_grouping(backend):
    frame = tidy(
        {"group": ["a", "a", "b", "b"], "x": [1.0, None, None, 2.0]},
        backend=backend,
    ) >> group_by("group")
    result = frame.fill("x", direction="downup")

    assert as_pandas(result)["x"].tolist() == [1.0, 1.0, 2.0, 2.0]
    assert result._groups == ["group"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_pivot_longer_default_order_prefix_and_drop_na(backend):
    frame = tidy(
        {
            "id": [1, 2],
            "wk1": [10.0, 20.0],
            "wk2": [None, 40.0],
        },
        backend=backend,
    )
    result = frame >> pivot_longer(
        starts_with("wk"),
        names_to="week",
        names_prefix="wk",
        values_to="rank",
        values_drop_na=True,
    )
    out = as_pandas(result)

    assert out.to_dict(orient="list") == {
        "id": [1, 2, 2],
        "week": ["1", "1", "2"],
        "rank": [10.0, 20.0, 40.0],
    }
    if backend == "polars":
        assert isinstance(result.lazy(), pl.LazyFrame)


@pytest.mark.parametrize("backend", BACKENDS)
def test_pivot_longer_splits_names_into_multiple_columns(backend):
    frame = tidy(
        {"id": [1], "temp_day": [20], "rain_day": [3]},
        backend=backend,
    )
    out = as_pandas(
        frame
        >> pivot_longer(
            starts_with(("temp", "rain")),
            names_to=["measure", "period"],
            names_sep="_",
            values_to="reading",
        )
    )

    assert out.to_dict(orient="list") == {
        "id": [1, 1],
        "measure": ["temp", "rain"],
        "period": ["day", "day"],
        "reading": [20, 3],
    }


@pytest.mark.parametrize("backend", BACKENDS)
def test_pivot_longer_cols_vary_slowest(backend):
    out = as_pandas(
        tidy({"id": [1, 2], "a": [10, 20], "b": [30, 40]}, backend=backend)
        >> pivot_longer(["a", "b"], cols_vary="slowest")
    )
    assert list(out.itertuples(index=False, name=None)) == [
        (1, "a", 10),
        (2, "a", 20),
        (1, "b", 30),
        (2, "b", 40),
    ]


@pytest.mark.parametrize("backend", BACKENDS)
def test_pivot_wider_roundtrip_prefix_fill_and_method_form(backend):
    frame = tidy(
        {
            "id": [1, 1, 2],
            "metric": ["x", "y", "x"],
            "reading": [10, 30, 20],
        },
        backend=backend,
    )
    result = frame.pivot_wider(
        names_from="metric",
        values_from="reading",
        names_prefix="value_",
        values_fill=0,
    )
    out = as_pandas(result)

    assert out.to_dict(orient="list") == {
        "id": [1, 2],
        "value_x": [10, 20],
        "value_y": [30, 0],
    }


@pytest.mark.parametrize("backend", BACKENDS)
def test_pivot_wider_aggregates_duplicate_cells(backend):
    frame = tidy(
        {"id": [1, 1, 1], "name": ["x", "x", "y"], "value": [2, 3, 4]},
        backend=backend,
    )
    out = as_pandas(frame >> pivot_wider(values_fn="sum"))
    assert out.to_dict(orient="list") == {"id": [1], "x": [5], "y": [4]}


@pytest.mark.parametrize("backend", BACKENDS)
def test_separate_and_unite_roundtrip(backend):
    frame = tidy(
        {"id": [1, 2], "code": ["north-10", "south-20"]},
        backend=backend,
    )
    separated = frame >> separate("code", ["region", "number"], sep="-")
    united = separated >> unite("code", "region", "number", sep="-")

    assert as_pandas(separated).to_dict(orient="list") == {
        "id": [1, 2],
        "region": ["north", "south"],
        "number": ["10", "20"],
    }
    assert as_pandas(united).to_dict(orient="list") == {
        "id": [1, 2],
        "code": ["north-10", "south-20"],
    }


@pytest.mark.parametrize("backend", BACKENDS)
def test_separate_merge_and_left_fill(backend):
    frame = tidy({"x": ["a", "a:b:c"]}, backend=backend)
    merged = as_pandas(
        frame >> separate("x", ["key", "value"], sep=":", extra="merge")
    )
    left = as_pandas(
        frame >> separate("x", ["key", "value"], sep=":", fill="left")
    )

    assert pd.isna(merged.loc[0, "value"])
    assert merged.loc[1, "value"] == "b:c"
    assert pd.isna(left.loc[0, "key"])
    assert left.loc[0, "value"] == "a"
    assert left.loc[1, "key"] == "a"
    assert left.loc[1, "value"] == "b"


@pytest.mark.parametrize("backend", BACKENDS)
def test_unite_missing_value_controls(backend):
    frame = tidy({"a": ["x", None], "b": [None, "y"]}, backend=backend)
    kept = as_pandas(frame >> unite("z", "a", "b", na_rm=False))
    removed = as_pandas(frame >> unite("z", "a", "b", na_rm=True))

    assert kept["z"].tolist() == ["x_NA", "NA_y"]
    assert removed["z"].tolist() == ["x", "y"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_expand_crosses_columns_and_nesting_preserves_observed_pairs(backend):
    frame = tidy(
        {"a": [1, 1, 2], "b": ["x", "y", "y"]},
        backend=backend,
    )

    crossed = as_pandas(frame >> expand("a", "b"))
    observed = as_pandas(frame >> expand(nesting("a", "b")))

    assert list(crossed.itertuples(index=False, name=None)) == [
        (1, "x"),
        (1, "y"),
        (2, "x"),
        (2, "y"),
    ]
    assert list(observed.itertuples(index=False, name=None)) == [
        (1, "x"),
        (1, "y"),
        (2, "y"),
    ]


@pytest.mark.parametrize("backend", BACKENDS)
def test_grouped_expand_operates_within_each_group(backend):
    frame = tidy(
        {
            "g": ["a", "a", "b", "b"],
            "x": [1, 2, 1, 1],
            "y": ["u", "v", "u", "v"],
        },
        backend=backend,
    ) >> group_by("g")

    result = frame >> expand("x", "y")
    out = as_pandas(result)

    assert len(out[out["g"] == "a"]) == 4
    assert len(out[out["g"] == "b"]) == 2
    assert result._groups == ["g"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_complete_fills_only_implicit_values_when_requested(backend):
    frame = tidy(
        {"id": [1, 2], "key": ["a", "b"], "value": [None, 5.0]},
        backend=backend,
    )

    result = frame >> complete(
        "id", "key", fill={"value": 0.0}, explicit=False
    )
    out = as_pandas(result)

    assert len(out) == 4
    existing_missing = out[(out["id"] == 1) & (out["key"] == "a")]
    assert existing_missing["value"].isna().all()
    implicit = out[(out["id"] == 1) & (out["key"] == "b")]
    assert implicit["value"].tolist() == [0.0]


@pytest.mark.parametrize("backend", BACKENDS)
def test_nest_and_unnest_nested_records_roundtrip(backend):
    frame = tidy(
        {
            "g": ["a", "a", "b"],
            "x": [1, 2, 3],
            "y": ["u", "v", "w"],
        },
        backend=backend,
    )

    nested = frame >> nest("data", cols=["x", "y"])
    nested_out = as_pandas(nested)
    restored = as_pandas(nested >> unnest("data"))

    assert len(nested_out) == 2
    # pandas nests DataFrames; polars handoff is list/array of row records
    if backend == "pandas":
        assert list(nested_out["data"].iloc[0]["x"]) == [1, 2]
    else:
        first = list(nested_out["data"].iloc[0])
        assert [row["x"] for row in first] == [1, 2]
    assert restored.to_dict(orient="list") == {
        "g": ["a", "a", "b"],
        "x": [1, 2, 3],
        "y": ["u", "v", "w"],
    }


@pytest.mark.parametrize("backend", BACKENDS)
def test_unnest_longer_values_indices_and_empty_rows(backend):
    frame = tidy(
        {"id": [1, 2, 3], "items": [[10, 11], [], [30]]},
        backend=backend,
    )
    dropped = as_pandas(
        frame
        >> unnest_longer(
            "items", values_to="value", indices_to="position"
        )
    )
    kept = as_pandas(frame >> unnest("items", keep_empty=True))

    assert dropped.to_dict(orient="list") == {
        "id": [1, 1, 3],
        "position": [1, 2, 1],
        "value": [10, 11, 30],
    }
    assert kept["id"].tolist() == [1, 1, 2, 3]
    assert pd.isna(kept.loc[2, "items"])


@pytest.mark.parametrize("backend", BACKENDS)
def test_unnest_wider_dict_and_list_columns(backend):
    structs = tidy(
        {"id": [1, 2], "info": [{"x": 10, "y": 11}, {"x": 20, "y": 21}]},
        backend=backend,
    )
    lists = tidy(
        {"id": [1, 2], "point": [[10, 11], [20, 21]]},
        backend=backend,
    )

    struct_out = as_pandas(structs >> unnest_wider("info"))
    list_out = as_pandas(lists >> unnest_wider("point", names_sep="_"))

    assert struct_out.to_dict(orient="list") == {
        "id": [1, 2], "x": [10, 20], "y": [11, 21]
    }
    assert list_out.to_dict(orient="list") == {
        "id": [1, 2], "point_1": [10, 20], "point_2": [11, 21]
    }


def test_separate_convert_infers_polars_types():
    out = as_pandas(
        tidy({"x": ["a-1", "b-2"]})
        >> separate("x", ["letter", "number"], sep="-", convert=True)
    )
    assert out.to_dict(orient="list") == {
        "letter": ["a", "b"],
        "number": [1, 2],
    }
    assert pd.api.types.is_integer_dtype(out["number"])


@pytest.mark.parametrize("backend", BACKENDS)
def test_pivot_longer_value_sentinel_and_multi_value_wider(backend):
    long = as_pandas(
        tidy(
            {"id": [1, 2], "x_a": [1, 2], "y_a": [3, 4]},
            backend=backend,
        )
        >> pivot_longer(
            ["x_a", "y_a"],
            names_to=[".value", "set"],
            names_sep="_",
        )
    )
    assert long.to_dict(orient="list") == {
        "id": [1, 2],
        "set": ["a", "a"],
        "x": [1, 2],
        "y": [3, 4],
    }

    wide = as_pandas(
        tidy(
            {
                "id": [1, 1],
                "name": ["a", "b"],
                "v1": [10, 20],
                "v2": [30, 40],
            },
            backend=backend,
        )
        >> pivot_wider(
            names_from="name", values_from=["v1", "v2"]
        )
    )
    assert wide.to_dict(orient="list") == {
        "id": [1],
        "v1_a": [10],
        "v1_b": [20],
        "v2_a": [30],
        "v2_b": [40],
    }


def test_reshape_argument_validation():
    with pytest.raises(ValueError, match="cols_vary"):
        pivot_longer("x", cols_vary="random")
    with pytest.raises(ValueError, match="one of names_sep"):
        pivot_longer(
            "x", names_to=["a", "b"], names_sep="_", names_pattern="(.)"
        )
    with pytest.raises(ValueError, match="direction"):
        fill("x", direction="sideways")
