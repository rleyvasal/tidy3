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


def as_pandas(frame):
    return frame.collect(as_="pandas").reset_index(drop=True)


def test_drop_na_selected_columns_and_replace_na():
    frame = tidy(
        {
            "id": [1, 2, 3, 4],
            "x": [1.0, None, float("nan"), 4.0],
            "y": [None, 2.0, 3.0, 4.0],
        },
    )

    dropped = as_pandas(frame >> drop_na("x"))
    replaced = as_pandas(frame >> replace_na({"x": 0.0, "y": -1.0}))

    assert dropped["id"].tolist() == [1, 4]
    assert replaced["x"].tolist() == [1.0, 0.0, 0.0, 4.0]
    assert replaced["y"].tolist() == [-1.0, 2.0, 3.0, 4.0]


@pytest.mark.parametrize(
    ("direction", "expected"),
    [
        ("down", [None, 2.0, 2.0, None, 4.0, 4.0]),
        ("up", [2.0, 2.0, None, 4.0, 4.0, None]),
        ("downup", [2.0, 2.0, 2.0, 4.0, 4.0, 4.0]),
        ("updown", [2.0, 2.0, 2.0, 4.0, 4.0, 4.0]),
    ],
)
def test_fill_directions_respect_transient_groups(direction, expected):
    frame = tidy(
        {
            "group": ["a", "a", "a", "b", "b", "b"],
            "x": [None, 2.0, None, None, 4.0, None],
        },
    )
    out = as_pandas(frame >> fill("x", direction=direction, by="group"))

    actual = out["x"].tolist()
    for value, wanted in zip(actual, expected):
        if wanted is None:
            assert pd.isna(value)
        else:
            assert value == wanted


def test_fill_uses_persistent_groups_and_preserves_grouping():
    frame = tidy(
        {"group": ["a", "a", "b", "b"], "x": [1.0, None, None, 2.0]},
    ) >> group_by("group")
    result = frame.fill("x", direction="downup")

    assert as_pandas(result)["x"].tolist() == [1.0, 1.0, 2.0, 2.0]
    assert result._groups == ["group"]


@pytest.mark.parametrize("direction", ["down", "up", "downup", "updown"])
def test_fill_leaves_the_input_frame_unchanged(direction):
    frame = tidy(pd.DataFrame({"x": [None, 2.0, None]}))

    frame >> fill("x", direction=direction)

    assert pd.isna(as_pandas(frame)["x"]).tolist() == [True, False, True]


def test_pivot_longer_default_order_prefix_and_drop_na():
    frame = tidy(
        {
            "id": [1, 2],
            "wk1": [10.0, 20.0],
            "wk2": [None, 40.0],
        },
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
    assert isinstance(result.lazy(), pl.LazyFrame)


def test_pivot_longer_splits_names_into_multiple_columns():
    frame = tidy(
        {"id": [1], "temp_day": [20], "rain_day": [3]},
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


def test_pivot_longer_cols_vary_slowest():
    out = as_pandas(
        tidy({"id": [1, 2], "a": [10, 20], "b": [30, 40]})
        >> pivot_longer(["a", "b"], cols_vary="slowest")
    )
    assert list(out.itertuples(index=False, name=None)) == [
        (1, "a", 10),
        (2, "a", 20),
        (1, "b", 30),
        (2, "b", 40),
    ]


def test_pivot_wider_roundtrip_prefix_fill_and_method_form():
    frame = tidy(
        {
            "id": [1, 1, 2],
            "metric": ["x", "y", "x"],
            "reading": [10, 30, 20],
        },
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


def test_pivot_wider_aggregates_duplicate_cells():
    frame = tidy(
        {"id": [1, 1, 1], "name": ["x", "x", "y"], "value": [2, 3, 4]},
    )
    out = as_pandas(frame >> pivot_wider(values_fn="sum"))
    assert out.to_dict(orient="list") == {"id": [1], "x": [5], "y": [4]}


def test_separate_and_unite_roundtrip():
    frame = tidy(
        {"id": [1, 2], "code": ["north-10", "south-20"]},
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


def test_separate_merge_and_left_fill():
    frame = tidy({"x": ["a", "a:b:c"]})
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


def test_unite_missing_value_controls():
    frame = tidy({"a": ["x", None], "b": [None, "y"]})
    kept = as_pandas(frame >> unite("z", "a", "b", na_rm=False))
    removed = as_pandas(frame >> unite("z", "a", "b", na_rm=True))

    assert kept["z"].tolist() == ["x_NA", "NA_y"]
    assert removed["z"].tolist() == ["x", "y"]


def test_expand_crosses_columns_and_nesting_preserves_observed_pairs():
    frame = tidy(
        {"a": [1, 1, 2], "b": ["x", "y", "y"]},
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


def test_grouped_expand_operates_within_each_group():
    frame = tidy(
        {
            "g": ["a", "a", "b", "b"],
            "x": [1, 2, 1, 1],
            "y": ["u", "v", "u", "v"],
        },
    ) >> group_by("g")

    result = frame >> expand("x", "y")
    out = as_pandas(result)

    assert len(out[out["g"] == "a"]) == 4
    assert len(out[out["g"] == "b"]) == 2
    assert result._groups == ["g"]


def test_complete_fills_only_implicit_values_when_requested():
    frame = tidy(
        {"id": [1, 2], "key": ["a", "b"], "value": [None, 5.0]},
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


def test_nest_and_unnest_nested_records_roundtrip():
    frame = tidy(
        {
            "g": ["a", "a", "b"],
            "x": [1, 2, 3],
            "y": ["u", "v", "w"],
        },
    )

    nested = frame >> nest("data", cols=["x", "y"])
    nested_out = as_pandas(nested)
    restored = as_pandas(nested >> unnest("data"))

    assert len(nested_out) == 2
    # pandas nests DataFrames; polars handoff is list/array of row records
    first = list(nested_out["data"].iloc[0])
    assert [row["x"] for row in first] == [1, 2]
    assert restored.to_dict(orient="list") == {
        "g": ["a", "a", "b"],
        "x": [1, 2, 3],
        "y": ["u", "v", "w"],
    }


def test_unnest_longer_values_indices_and_empty_rows():
    frame = tidy(
        {"id": [1, 2, 3], "items": [[10, 11], [], [30]]},
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


def test_unnest_wider_dict_and_list_columns():
    structs = tidy(
        {"id": [1, 2], "info": [{"x": 10, "y": 11}, {"x": 20, "y": 21}]},
    )
    lists = tidy(
        {"id": [1, 2], "point": [[10, 11], [20, 21]]},
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


def test_pivot_longer_value_sentinel_and_multi_value_wider():
    long = as_pandas(
        tidy(
            {"id": [1, 2], "x_a": [1, 2], "y_a": [3, 4]},
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


def test_pivot_wider_duplicates_become_list_columns_with_tidyr_warning():
    import warnings

    data = {"id": [1, 1, 2], "key": ["a", "a", "b"], "val": [1.0, 2.0, 3.0]}
    with pytest.warns(UserWarning, match='values_fn="list"') as caught:
        out = as_pandas(
            tidy(data)
            >> pivot_wider(names_from="key", values_from="val")
        )
    assert "summarise(n=n(), by=['id', 'key'])" in str(caught[0].message)
    assert [list(cell) for cell in out["a"][:1]] == [[1.0, 2.0]]
    assert out["a"][1] is None and out["b"][0] is None

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        listed = as_pandas(
            tidy(data)
            >> pivot_wider(names_from="key", values_from="val", values_fn="list")
        )
        unique = as_pandas(
            tidy({"id": [1, 2], "key": ["a", "b"], "val": [1.0, 3.0]})
            >> pivot_wider(names_from="key", values_from="val")
        )
    assert list(listed["a"][0]) == [1.0, 2.0]
    assert unique["a"].tolist()[0] == 1.0


def test_pivot_wider_renaming_options_stay_lazy_on_polars(monkeypatch):
    """names_sep / names_glue / names_vary never collect the whole frame."""
    from tidy3.frame import TidyFrame

    frame = tidy(
        {"id": [1, 1, 2, 2], "key": ["a", "b", "a", "b"], "v": [1, 2, 3, 4], "w": [5, 6, 7, 8]}
    )

    def refuse(*_args, **_kwargs):
        raise AssertionError("pivot_wider collected the whole frame")

    monkeypatch.setattr(TidyFrame, "collect", refuse)
    plain = frame >> pivot_wider(names_from="key", values_from="v")
    styled = frame >> pivot_wider(
        names_from="key", values_from=["v", "w"], names_glue="{key}_{.value}", names_vary="slowest"
    )
    monkeypatch.undo()
    assert as_pandas(plain).columns.tolist() == ["id", "w", "a", "b"]
    assert as_pandas(styled).columns.tolist() == ["id", "a_v", "a_w", "b_v", "b_w"]


def test_check_pivot_spec_errors_like_tidyr():
    from tidy3 import check_pivot_spec

    with pytest.raises(ValueError, match="must have `.name` and `.value`"):
        check_pivot_spec({"x": [1]})
    with pytest.raises(ValueError, match="must be unique"):
        check_pivot_spec({".name": ["a", "a"], ".value": ["v", "v"]})
    assert as_pandas(check_pivot_spec({".name": ["a"], ".value": ["v"]})).shape == (1, 2)


def test_pivot_options_and_specs_never_convert_polars_frames_to_pandas(monkeypatch):
    """Every pivot option and spec function stays in Polars on that backend."""
    import polars as pl

    from tidy3 import build_longer_spec, build_wider_spec, pivot_longer_spec, pivot_wider_spec
    from tidy3.frame import TidyFrame

    long = tidy(
        {"id": ["i1", "i1", "i2", "i2"], "key": ["a", "b", "a", "b"],
         "v": [1.0, 2.0, 3.0, 4.0], "note": ["p", "q", "r", "s"]}
    )
    wide = tidy({"id": [1, 2], "mean_1": [5.0, 6.0], "sd_1": [0.1, 0.2]})

    def refuse(*_args, **_kwargs):
        raise AssertionError("converted a Polars frame to pandas")

    monkeypatch.setattr(TidyFrame, "collect", refuse)
    monkeypatch.setattr(pl.DataFrame, "to_pandas", refuse)
    results = [
        long >> pivot_wider(names_from="key", values_from="v", id_cols="id",
                            names_expand=True, id_expand=True,
                            unused_fn={"note": lambda s: "+".join(s)}),
        long >> pivot_wider(names_from="key", values_from="v", id_cols="id", unused_fn={"note": "first"}),
        long >> pivot_wider_spec(build_wider_spec(long, names_from="key", values_from="v"), id_cols="id"),
        wide >> pivot_longer_spec(
            build_longer_spec(wide, ["mean_1", "sd_1"], names_to=[".value", "visit"], names_sep="_")
        ),
    ]
    monkeypatch.undo()
    assert as_pandas(results[0])["note"].tolist() == ["p+q", "r+s"]
    assert as_pandas(results[1]).columns.tolist() == ["id", "a", "b", "note"]
    assert as_pandas(results[2]).columns.tolist() == ["id", "a", "b"]
    assert as_pandas(results[3]).columns.tolist() == ["id", "visit", "mean", "sd"]


def test_wider_spec_with_no_rows_is_typed_and_pivots_to_ids():
    import tidy3 as t

    empty = pl.DataFrame({
        "id": pl.Series([], dtype=pl.Int64),
        "k": pl.Series([], dtype=pl.String),
        "v": pl.Series([], dtype=pl.Float64),
    })
    spec = t.build_wider_spec(tidy(empty), names_from="k", values_from="v")
    assert spec.collect().schema == pl.Schema({".name": pl.String, ".value": pl.String, "k": pl.String})
    out = (tidy(empty) >> t.pivot_wider_spec(spec, id_cols="id")).collect()
    assert out.schema == pl.Schema({"id": pl.Int64}) and out.height == 0


def test_pivot_longer_ptypes_cast_without_losing_values():
    weeks = tidy({"id": [1, 2], "week_1": [5, 6], "week_2": [7, 8]})
    out = weeks >> pivot_longer(
        starts_with("week_"), names_to="week", names_prefix="week_",
        names_transform={"week": int}, names_ptypes={"week": float}, values_ptypes=float,
    )
    schema = out.collect().schema
    assert schema["week"] == pl.Float64 and schema["value"] == pl.Float64
    levels = weeks >> pivot_longer(
        starts_with("week_"), names_to="week", names_prefix="week_",
        names_ptypes={"week": pd.CategoricalDtype(["2", "1"])},
    )
    assert list(as_pandas(levels)["week"].cat.categories) == ["2", "1"]
    whole = tidy({"id": [1], "a": [1.0], "b": [2.0]}) >> pivot_longer(["a", "b"], values_ptypes=int)
    assert whole.collect().schema["value"] == pl.Int64

    with pytest.raises(TypeError, match="can't convert `week` <String> to <Int64>"):
        weeks >> pivot_longer(starts_with("week_"), names_to="week", names_ptypes={"week": int})
    with pytest.raises(ValueError, match="without losing values"):
        tidy({"id": [1], "a": [1.5], "b": [2.0]}) >> pivot_longer(["a", "b"], values_ptypes=int)
    with pytest.raises(ValueError, match="without losing values"):
        weeks >> pivot_longer(
            starts_with("week_"), names_to="week", names_prefix="week_",
            names_ptypes={"week": pl.Enum(["1"])},
        )
    with pytest.raises(KeyError, match="unknown columns"):
        weeks >> pivot_longer(starts_with("week_"), names_ptypes={"week": str})
    with pytest.raises(TypeError, match="Polars type"):
        weeks >> pivot_longer(starts_with("week_"), names_ptypes=list)


def test_pivot_longer_names_repair_and_groups():
    clash = tidy({"name": [1, 2], "g": ["a", "b"], "x": [1, 2], "y": [3, 4]}) >> group_by("g")
    with pytest.raises(ValueError, match="names must be unique.*'name'"):
        clash >> pivot_longer(["x", "y"])
    out = clash >> pivot_longer(["x", "y"], names_repair="unique")
    assert out.collect().columns == ["name_1", "g", "name_3", "value"]
    assert out._groups == ["g"]
    spaced = clash >> pivot_longer(["x", "y"], names_to="the key", values_to="1st", names_repair="universal")
    assert spaced.collect().columns == ["name", "g", "the_key", "_1st"]
    upper = clash >> pivot_longer(["x", "y"], names_to="key", names_repair=lambda n: [v.upper() for v in n])
    assert upper.collect().columns == ["NAME", "G", "KEY", "VALUE"]
    # A function that leaves duplicates cannot become a Polars frame.
    with pytest.raises(ValueError, match="Polars"):
        clash >> pivot_longer(["x", "y"], names_repair="minimal")
    with pytest.raises(ValueError, match="names_repair must be one of"):
        pivot_longer(["x"], names_repair="nope")
    with pytest.raises(ValueError, match="repeat"):
        pivot_longer(["x"], names_to=["a", "a"], names_sep="_")


def test_pivot_wider_names_repair():
    keyed = tidy({"a": [1, 2], "k": ["a", "b"], "v": [1, 2]})
    with pytest.raises(ValueError, match="names must be unique"):
        keyed >> pivot_wider(names_from="k", values_from="v")
    assert as_pandas(keyed >> pivot_wider(names_from="k", values_from="v", names_repair="unique")).columns.tolist() == ["a_1", "a_2", "b"]
    styled = keyed >> pivot_wider(names_from="k", values_from="v", names_glue="x {k}", names_repair="universal")
    assert as_pandas(styled).columns.tolist() == ["a", "x_a", "x_b"]
    odd = tidy({"id": [1, 2], "k": ["if", "1x"], "v": [1, 2]})
    assert as_pandas(odd >> pivot_wider(names_from="k", values_from="v", names_repair="universal")).columns.tolist() == ["id", "_if", "_1x"]


def test_pivot_wider_values_fn_functions_and_dicts():
    import warnings

    dup = tidy({"id": [1, 1, 2], "key": ["a", "a", "b"], "val": [1.0, 3.0, 5.0], "w": [10.0, 20.0, 30.0]})
    spread = as_pandas(
        dup >> pivot_wider(names_from="key", values_from="val", id_cols="id", values_fn=lambda s: s.max() - s.min())
    )
    assert spread["a"].tolist()[0] == 2.0 and pd.isna(spread["a"].tolist()[1])
    with pytest.warns(UserWarning, match="Values from `w` are not uniquely identified"):
        mixed = as_pandas(dup >> pivot_wider(names_from="key", values_from=["val", "w"], values_fn={"val": "sum"}))
    assert mixed["val_a"].tolist()[0] == 4.0
    assert list(mixed["w_a"][0]) == [10.0, 20.0]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        listed = as_pandas(dup >> pivot_wider(names_from="key", values_from=["val", "w"], values_fn={"val": "mean", "w": list}))
    assert listed["val_a"].tolist()[0] == 2.0 and list(listed["w_a"][0]) == [10.0, 20.0]
    # Absent cells stay missing even for sums and counts (Polars would fill 0).
    counted = as_pandas(dup >> pivot_wider(names_from="key", values_from="val", id_cols="id", values_fn="len"))
    assert counted["a"].tolist()[0] == 2 and pd.isna(counted["a"].tolist()[1])
    filled = as_pandas(dup >> pivot_wider(names_from="key", values_from=["val", "w"], values_fn="sum", values_fill={"w": 0}))
    assert pd.isna(filled["val_a"].tolist()[1]) and filled["w_a"].tolist()[1] == 0
    grouped = dup >> group_by("id") >> pivot_wider(names_from="key", values_from="val", id_cols="id", values_fn="n_unique")
    assert grouped._groups == ["id"]
    with pytest.raises(KeyError, match="not values_from"):
        dup >> pivot_wider(names_from="key", values_from="val", values_fn={"w": "sum"})
    with pytest.raises(ValueError, match="values_fn must be"):
        pivot_wider(values_fn="average")


def test_pivot_wider_spec_names_repair_and_values_fn():
    from tidy3 import build_wider_spec, pivot_wider_spec

    keyed = tidy({"a": [1, 1, 2], "k": ["a", "a", "b"], "v": [1, 2, 3]})
    spec = build_wider_spec(keyed, names_from="k", values_from="v")
    with pytest.raises(ValueError, match="names must be unique"):
        keyed >> pivot_wider_spec(spec, values_fn="sum")
    out = as_pandas(keyed >> pivot_wider_spec(spec, values_fn="sum", names_repair="unique"))
    assert out.columns.tolist() == ["a_1", "a_2", "b"]
    assert out["a_2"].tolist()[0] == 3


def test_pivot_longer_spec_ptypes_and_names_repair():
    from tidy3 import build_longer_spec, pivot_longer_spec

    weeks = tidy({"id": [1, 2], "week_1": [5.0, 6.0], "week_2": [7.0, 8.0]})
    spec = build_longer_spec(
        weeks, starts_with("week_"), names_to="week", names_prefix="week_",
        names_transform={"week": int}, names_ptypes={"week": float},
    )
    assert spec.collect().schema["week"] == pl.Float64
    out = weeks >> pivot_longer_spec(spec, values_ptypes=int, values_drop_na=True)
    assert out.collect().schema["value"] == pl.Int64
    clash = tidy({"name": [1, 2], "x": [1, 2], "y": [3, 4]})
    plain = build_longer_spec(clash, ["x", "y"])
    with pytest.raises(ValueError, match="names must be unique"):
        clash >> pivot_longer_spec(plain)
    assert as_pandas(clash >> pivot_longer_spec(plain, names_repair="unique")).columns.tolist() == ["name_1", "name_2", "value"]


def test_pivot_wider_values_fn_takes_tidy3_summaries():
    import tidy3

    dup = tidy({"id": [1, 1, 2], "key": ["a", "a", "b"], "val": [1.0, 3.0, 5.0]})
    for fn, first in ((tidy3.mean, 2.0), (tidy3.sum, 4.0), (tidy3.n_distinct, 2), (len, 2)):
        out = as_pandas(dup >> pivot_wider(names_from="key", values_from="val", values_fn=fn))
        assert out["a"].tolist()[0] == first
