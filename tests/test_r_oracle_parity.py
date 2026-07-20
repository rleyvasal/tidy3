from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
from functools import lru_cache
from typing import Any, Callable

import numpy as np
import pandas as pd
import polars as pl
import pytest

import tidy3
from tidy3 import (
    add_count,
    add_tally,
    anti_join,
    arrange,
    bind_cols,
    bind_rows,
    c_across,
    col,
    complete,
    case_match,
    consecutive_id,
    count,
    cross_join,
    distinct,
    drop,
    drop_na,
    expand,
    fill,
    filter,
    filter_out,
    full_join,
    group_by,
    head,
    inner_join,
    intersect,
    left_join,
    max as tidy_max,
    mean,
    min_rank,
    mutate,
    n,
    n_distinct,
    nest,
    nest_join,
    pivot_longer,
    pivot_wider,
    pull,
    reframe,
    relocate,
    rename,
    rename_with,
    replace_na,
    recode,
    right_join,
    rowwise,
    rows_append,
    rows_delete,
    rows_insert,
    rows_patch,
    rows_update,
    rows_upsert,
    select,
    semi_join,
    separate,
    separate_longer_delim,
    separate_wider_delim,
    setdiff,
    setequal,
    slice,
    slice_head,
    slice_max,
    slice_min,
    slice_tail,
    summarise,
    sum as tidy_sum,
    symdiff,
    tally,
    tidy,
    transmute,
    unite,
    ungroup,
    union as tidy_union,
    union_all,
    unnest,
    unnest_longer,
    unnest_wider,
)


ROOT = Path(__file__).resolve().parents[1]
R_CASES = ROOT / "tests" / "r_oracle_cases.R"
BACKENDS = ("pandas", "polars")


def _r_command() -> list[str] | None:
    manifest = os.environ.get("TIDY3_R_ORACLE_MANIFEST")
    pixi = shutil.which("pixi")
    if manifest and pixi:
        return [pixi, "run", "--manifest-path", manifest, "Rscript"]
    rscript = shutil.which("Rscript")
    return [rscript] if rscript else None


R_COMMAND = _r_command()


def _type_kind(series: pd.Series) -> str:
    dtype = series.dtype
    if isinstance(dtype, pd.CategoricalDtype):
        return "factor"
    if isinstance(dtype, pd.ArrowDtype) and str(dtype).startswith("dictionary<"):
        return "factor"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "datetime"
    if pd.api.types.is_bool_dtype(dtype):
        return "logical"
    if pd.api.types.is_integer_dtype(dtype):
        return "integer"
    if pd.api.types.is_float_dtype(dtype):
        return "double"
    nonmissing = series.dropna()
    if len(nonmissing) and all(
        isinstance(value, (bool, np.bool_)) for value in nonmissing
    ):
        return "logical"
    if len(nonmissing) and all(
        isinstance(value, (int, np.integer))
        and not isinstance(value, (bool, np.bool_))
        for value in nonmissing
    ):
        return "integer"
    if len(nonmissing) and all(
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_))
        for value in nonmissing
    ):
        return "double"
    if len(nonmissing) and all(
        isinstance(value, (list, tuple, dict)) for value in nonmissing
    ):
        return "list"
    return "character"


def _value(value: Any) -> Any:
    if value is pd.NA or value is None:
        return None
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return None
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).strftime("%Y-%m-%dT%H:%M:%S")
    if isinstance(value, dict):
        return {key: _normalise(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalise(item) for item in value]
    return value


def _frame_payload(frame: pd.DataFrame) -> dict[str, Any]:
    columns = list(frame.columns)
    types = [_type_kind(frame.iloc[:, index]) for index in range(len(columns))]
    data = [
        [_value(value) for value in frame.iloc[:, index].tolist()]
        for index in range(len(columns))
    ]
    return {
        "kind": "dataframe",
        "columns": columns[0] if len(columns) == 1 else columns,
        "types": types[0] if len(types) == 1 else types,
        "data": data,
    }


def _normalise(value: Any) -> Any:
    if isinstance(value, tidy3.TidyFrame):
        return _frame_payload(
            value.collect(
                as_="pandas", arrow_backed=value.backend == "polars"
            )
        )
    if isinstance(value, pl.DataFrame):
        return _frame_payload(value.to_pandas())
    if isinstance(value, pl.Series):
        return {"kind": "vector", "data": [_value(item) for item in value]}
    if isinstance(value, pd.DataFrame):
        return _frame_payload(value)
    if isinstance(value, (pd.Series, pd.Index, np.ndarray)):
        return {"kind": "vector", "data": [_value(item) for item in list(value)]}
    if isinstance(value, dict):
        return {key: _normalise(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalise(item) for item in value]
    return _value(value)


@lru_cache(maxsize=None)
def _oracle(case: str) -> Any:
    if R_COMMAND is None:
        pytest.skip(
            "R oracle unavailable; install R/dplyr/tidyr/jsonlite or set "
            "TIDY3_R_ORACLE_MANIFEST"
        )
    completed = subprocess.run(
        [*R_COMMAND, str(R_CASES), case],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def _base(backend: str):
    return tidy(
        pd.DataFrame(
            {
                "id": pd.Series([1, 2, 3, 4], dtype="int64"),
                "g": ["b", "a", "b", "a"],
                "x": [2.0, None, 4.0, 1.0],
                "y": [20.0, 10.0, 40.0, 30.0],
            }
        ),
        backend=backend,
    )


def _filter_missing(backend: str):
    return _base(backend) >> filter(col("x") > 1)


def _filter_out_missing(backend: str):
    return _base(backend) >> filter_out(col("x") > 1)


def _mutate_sequential(backend: str):
    return _base(backend) >> mutate(
        a=col("y") * 2, b=col("a") + 1, before="x"
    )


def _transmute(backend: str):
    return _base(backend) >> transmute(
        g=col("g"), a=col("y") * 2, b=col("a") + 1
    )


def _select_drop(backend: str):
    frame = _base(backend)
    return {"selected": frame >> select("g", "x"), "dropped": frame >> drop("y")}


def _rename(backend: str):
    return _base(backend) >> rename(score="x")


def _rename_with(backend: str):
    return _base(backend) >> rename_with(str.upper, "x", "y")


def _relocate(backend: str):
    return _base(backend) >> relocate("y", before="x")


def _pull(backend: str):
    return _base(backend) >> pull("y")


def _arrange(backend: str):
    frame = _base(backend)
    return {
        "ascending": frame >> arrange("x"),
        "descending": frame >> arrange(tidy3.desc("x")),
    }


def _distinct(backend: str):
    frame = tidy(
        {"g": ["b", "a", "b", "a"], "x": [1, 2, 1, 3]},
        backend=backend,
    )
    return frame >> distinct("g")


def _slices(backend: str):
    frame = _base(backend)
    return {
        "positions": frame >> slice(3, 1, 1),
        "first_rows": frame >> head(2),
        "head": frame >> slice_head(n=2),
        "tail": frame >> slice_tail(n=2),
        "minimum": frame >> slice_min("x", n=2),
        "maximum": frame >> slice_max("x", n=2),
    }


def _grouping(backend: str):
    frame = _base(backend)
    return {
        "persistent": frame
        >> group_by("g")
        >> summarise(n=n(), avg=mean("y"), groups="drop"),
        "transient": frame >> summarise(n=n(), avg=mean("y"), by="g"),
        "ungrouped": frame >> group_by("g") >> ungroup() >> summarise(n=n()),
    }


def _rowwise(backend: str):
    frame = tidy(
        {"id": [1, 2], "x": [1.0, 2.0], "y": [10.0, 20.0]},
        backend=backend,
    )
    return frame >> rowwise("id") >> mutate(
        total=tidy_sum(c_across("x", "y"))
    ) >> ungroup()


def _reframe(backend: str):
    frame = tidy(
        {"g": ["b", "b", "a"], "x": [2.0, 4.0, 10.0]},
        backend=backend,
    )
    return frame >> group_by("g") >> reframe(value=col("x"), avg=mean("x"))


def _counts(backend: str):
    frame = _base(backend)
    return {
        "counted": frame >> count("g"),
        "tallied": frame >> group_by("g") >> tally(),
        "added": frame >> add_count("g"),
        "add_tallied": frame >> group_by("g") >> add_tally() >> ungroup(),
    }


def _join_frames(backend: str):
    left = tidy(
        pd.DataFrame({"k": [1.0, 2.0, None], "x": ["a", "b", "c"]}),
        backend=backend,
    )
    right = pd.DataFrame({"k": [2.0, None, 3.0], "y": [20.0, 99.0, 30.0]})
    return left, right


def _joins(backend: str):
    left, right = _join_frames(backend)
    return {
        "left": left >> left_join(right, by="k"),
        "right": left >> right_join(right, by="k"),
        "inner": left >> inner_join(right, by="k"),
        "full": left >> full_join(right, by="k"),
        "semi": left >> semi_join(right, by="k"),
        "anti": left >> anti_join(right, by="k"),
        "cross": tidy({"x": [1, 2]}, backend=backend)
        >> cross_join(pd.DataFrame({"y": ["a", "b"]})),
    }


def _nest_join(backend: str):
    left = tidy({"k": [1, 2], "x": ["a", "b"]}, backend=backend)
    right = pd.DataFrame({"k": [1, 1, 2], "y": [10, 11, 20]})
    return left >> nest_join(right, by="k", name="matches") >> unnest("matches")


def _binds(backend: str):
    rows = tidy({"x": [1], "y": ["a"]}, backend=backend) >> bind_rows(
        pd.DataFrame({"x": [2], "z": [3.5]})
    )
    cols = tidy({"x": [1, 2]}, backend=backend) >> bind_cols(
        pd.DataFrame({"y": ["a", "b"]})
    )
    return {"rows": rows, "cols": cols}


def _sets(backend: str):
    left = tidy({"a": [1.0, 2.0, 2.0]}, backend=backend)
    right = pd.DataFrame({"a": [2.0, 3.0]})
    return {
        "union": left >> tidy_union(right),
        "union_all": left >> union_all(right),
        "intersect": left >> intersect(right),
        "setdiff": left >> setdiff(right),
        "symdiff": left >> symdiff(right),
        "setequal": left >> setequal(pd.DataFrame({"a": [2.0, 1.0]})),
    }


def _rows(backend: str):
    base = tidy(
        pd.DataFrame(
            {
                "id": [1, 2, 3],
                "value": [10.0, None, 30.0],
                "old": ["a", "b", "c"],
            }
        ),
        backend=backend,
    )
    fourth = pd.DataFrame({"id": [4], "value": [40.0], "old": ["d"]})
    correction = pd.DataFrame({"id": [2], "value": [20.0], "old": ["B"]})
    upsert = pd.DataFrame(
        {"id": [2, 4], "value": [20.0, 40.0], "old": ["B", "d"]}
    )
    return {
        "inserted": base >> rows_insert(fourth, by="id"),
        "appended": base >> rows_append(fourth),
        "updated": base >> rows_update(correction, by="id"),
        "patched": base >> rows_patch(correction, by="id"),
        "upserted": base >> rows_upsert(upsert, by="id"),
        "deleted": base >> rows_delete(pd.DataFrame({"id": [2]}), by="id"),
    }


def _missing_data(backend: str):
    frame = tidy(
        pd.DataFrame(
            {
                "g": ["a", "a", "b"],
                "id": [1, 2, 1],
                "value": [None, 2.0, None],
            }
        ),
        backend=backend,
    )
    return {
        "dropped": frame >> drop_na("value"),
        "replaced": frame >> replace_na({"value": 0}),
        "filled": frame >> fill("value", direction="downup"),
        "expanded": frame >> select("g", "id") >> expand("g", "id"),
        "completed": frame >> complete("g", "id", fill={"value": 0}),
    }


def _pivots(backend: str):
    wide = tidy(
        {"id": [1, 2], "a": [10, 20], "b": [30, 40]}, backend=backend
    )
    long = wide >> pivot_longer(
        ["a", "b"], names_to="name", values_to="value"
    )
    return {
        "long": long,
        "wide": long >> pivot_wider(names_from="name", values_from="value"),
    }


def _separate_unite(backend: str):
    frame = tidy({"id": [1, 2], "code": ["a-10", "b-20"]}, backend=backend)
    separated = frame >> separate("code", ["group", "number"], sep="-")
    return {
        "separated": separated,
        "united": separated >> unite("code", "group", "number", sep="-"),
    }


def _nest_roundtrip(backend: str):
    frame = tidy({"g": ["a", "a", "b"], "value": [1, 2, 3]}, backend=backend)
    return frame >> nest("data", cols=["value"]) >> unnest("data")


def _unnest_longer(backend: str):
    frame = tidy(
        pd.DataFrame({"id": [1, 2], "values": [[10, 11], [20]]}),
        backend=backend,
    )
    return frame >> unnest_longer("values", indices_to="position")


def _unnest_wider(backend: str):
    frame = tidy(
        pd.DataFrame(
            {
                "id": [1, 2],
                "values": [{"a": 10, "b": 20}, {"a": 30, "b": 40}],
            }
        ),
        backend=backend,
    )
    return frame >> unnest_wider("values")


def _empty_inputs(backend: str):
    empty = tidy(
        pd.DataFrame({"x": pd.Series(dtype="float64")}), backend=backend
    )
    grouped = tidy(
        pd.DataFrame(
            {
                "g": pd.Series(dtype="object"),
                "x": pd.Series(dtype="float64"),
            }
        ),
        backend=backend,
    )
    left = tidy(
        pd.DataFrame(
            {"k": pd.Series(dtype="int64"), "x": pd.Series(dtype="object")}
        ),
        backend=backend,
    )
    right = pd.DataFrame(
        {"k": pd.Series(dtype="int64"), "y": pd.Series(dtype="float64")}
    )
    return {
        "filtered": empty >> filter(col("x") > 0),
        "summary": empty >> summarise(n=n(), avg=mean("x")),
        "grouped": grouped >> group_by("g") >> summarise(n=n(), groups="drop"),
        "joined": left >> left_join(right, by="k"),
    }


def _categorical(backend: str):
    frame = tidy(
        pd.DataFrame(
            {"g": pd.Categorical(["a"], categories=["a", "b"]), "x": [1]}
        ),
        backend=backend,
    )
    return (
        frame
        >> group_by("g", drop=False)
        >> summarise(n=n(), groups="drop")
    )


def _categorical_count(backend: str):
    frame = tidy(
        pd.DataFrame(
            {"g": pd.Categorical(["a"], categories=["a", "b"]), "x": [1]}
        ),
        backend=backend,
    )
    return frame >> count("g", drop=False)


def _duplicate_names(backend: str):
    return tidy({"x": [1]}, backend=backend) >> bind_cols(
        pd.DataFrame({"x": [2]})
    )


def _descending_rank(backend: str):
    frame = tidy({"x": [3.0, 1.0, 2.0, None]}, backend=backend)
    return frame >> mutate(rank=min_rank(tidy3.desc("x")))


def _sequential_summary(backend: str):
    frame = tidy({"x": [1, 2, 3]}, backend=backend)
    return frame >> summarise(a=mean("x"), b=col("a") * 2)


def _sequential_summary_nested(backend: str):
    frame = tidy({"x": [1, 2, 3]}, backend=backend)
    return frame >> summarise(
        a=tidy_sum("x"),
        b=tidy_sum(col("a")),
        c=tidy_sum(col("a") + col("x")),
    )


def _mutate_delete(backend: str):
    return tidy({"x": [1, 2], "y": [3, 4]}, backend=backend) >> mutate(x=None)


def _computed_distinct(backend: str):
    frame = tidy(
        {"x": [1, 3, 2], "y": [10, 11, 12]}, backend=backend
    )
    return frame >> distinct(parity=col("x") % 2)


def _n_distinct_multi(backend: str):
    frame = tidy(
        {"x": [1, 1, 2], "y": ["a", "b", "b"]}, backend=backend
    )
    return frame >> summarise(n=n_distinct("x", "y"))


def _bind_cols_recycle(backend: str):
    return tidy({"x": [1, 2, 3]}, backend=backend) >> bind_cols(
        pd.DataFrame({"z": [9]})
    )


def _pivot_value(backend: str):
    frame = tidy(
        {"id": [1, 2], "x_a": [1, 2], "y_a": [3, 4]}, backend=backend
    )
    return frame >> pivot_longer(
        ["x_a", "y_a"], names_to=[".value", "set"], names_sep="_"
    )


def _pivot_wider_multi(backend: str):
    frame = tidy(
        {
            "id": [1, 1],
            "name": ["a", "b"],
            "v1": [10, 20],
            "v2": [30, 40],
        },
        backend=backend,
    )
    return frame >> pivot_wider(
        names_from="name", values_from=["v1", "v2"]
    )


def _pivot_wider_multi_names(backend: str):
    frame = tidy(
        {
            "id": [1, 1],
            "axis": ["x", "x"],
            "period": ["q1", "q2"],
            "value": [10, 20],
        },
        backend=backend,
    )
    return frame >> pivot_wider(
        names_from=["axis", "period"], values_from="value"
    )


def _separate_convert(backend: str):
    frame = tidy({"code": ["a-10", "b-20"]}, backend=backend)
    return frame >> separate(
        "code", ["group", "number"], sep="-", convert=True
    )


def _separate_convert_types(backend: str):
    frame = tidy(
        {"code": ["TRUE-1.5", "FALSE-2.0"]}, backend=backend
    )
    return frame >> separate(
        "code", ["flag", "number"], sep="-", convert=True
    )


def _arrange_by_group(backend: str):
    frame = tidy(
        {"g": ["b", "a", "b"], "x": [2.0, 3.0, 1.0]}, backend=backend
    )
    return frame >> group_by("g") >> arrange("x", by_group=True) >> ungroup()


def _type_coercion(backend: str):
    rows = tidy({"x": [1]}, backend=backend) >> bind_rows(
        pd.DataFrame({"x": [2.5]})
    )
    mutated = tidy({"x": [1, 2]}, backend=backend) >> mutate(
        y=col("x") / 2,
        flag=col("x") > 1,
        label=tidy3.if_else(col("flag"), "yes", "no"),
    )
    return {"rows": rows, "mutated": mutated}


def _new_helpers(backend: str):
    x = tidy(
        pd.DataFrame(
            {
                "code": pd.Series([1, 2, 3, None], dtype="Int64"),
                "text": ["a|b", "c", "d|e", None],
            }
        ),
        backend=backend,
    )
    labels = x >> mutate(
        bucket=case_match(
            "code", ((1, 2), "low"), (3, "high"), default="other"
        ),
        recoded=recode(
            "code", {1: "one", 2: "two"}, default="other", missing="missing"
        ),
    )
    longer = tidy({"text": ["a|b", "c"]}, backend=backend) >> separate_longer_delim(
        "text", "|"
    )
    wider = tidy({"text": ["a|b", "c"]}, backend=backend) >> separate_wider_delim(
        "text", ["left", "right"], "|"
    )
    return {"labels": labels, "longer": longer, "wider": wider}


ORACLE_CASES: dict[str, Callable[[str], Any]] = {
    "filter_missing": _filter_missing,
    "filter_out_missing": _filter_out_missing,
    "mutate_sequential": _mutate_sequential,
    "transmute": _transmute,
    "select_drop": _select_drop,
    "rename": _rename,
    "rename_with": _rename_with,
    "relocate": _relocate,
    "pull": _pull,
    "arrange": _arrange,
    "distinct": _distinct,
    "slices": _slices,
    "grouping": _grouping,
    "rowwise": _rowwise,
    "reframe": _reframe,
    "counts": _counts,
    "joins": _joins,
    "nest_join": _nest_join,
    "binds": _binds,
    "sets": _sets,
    "rows": _rows,
    "missing_data": _missing_data,
    "pivots": _pivots,
    "separate_unite": _separate_unite,
    "nest_roundtrip": _nest_roundtrip,
    "unnest_longer": _unnest_longer,
    "unnest_wider": _unnest_wider,
    "empty_inputs": _empty_inputs,
    "type_coercion": _type_coercion,
    "categorical": _categorical,
    "categorical_count": _categorical_count,
    "duplicate_names": _duplicate_names,
    "descending_rank": _descending_rank,
    "sequential_summary": _sequential_summary,
    "sequential_summary_nested": _sequential_summary_nested,
    "mutate_delete": _mutate_delete,
    "computed_distinct": _computed_distinct,
    "n_distinct_multi": _n_distinct_multi,
    "bind_cols_recycle": _bind_cols_recycle,
    "pivot_value": _pivot_value,
    "pivot_wider_multi": _pivot_wider_multi,
    "pivot_wider_multi_names": _pivot_wider_multi_names,
    "separate_convert": _separate_convert,
    "separate_convert_types": _separate_convert_types,
    "arrange_by_group": _arrange_by_group,
    "new_helpers": _new_helpers,
}


CASE_VERBS = {
    "filter_missing": {"filter"},
    "filter_out_missing": {"filter_out"},
    "mutate_sequential": {"mutate"},
    "transmute": {"transmute"},
    "select_drop": {"select", "drop"},
    "rename": {"rename"},
    "rename_with": {"rename_with"},
    "relocate": {"relocate"},
    "pull": {"pull"},
    "arrange": {"arrange"},
    "distinct": {"distinct"},
    "slices": {"slice", "head", "slice_head", "slice_tail", "slice_min", "slice_max"},
    "grouping": {"group_by", "ungroup", "summarise"},
    "rowwise": {"rowwise"},
    "reframe": {"reframe"},
    "counts": {"count", "tally", "add_count", "add_tally"},
    "joins": {"left_join", "right_join", "inner_join", "full_join", "semi_join", "anti_join", "cross_join"},
    "nest_join": {"nest_join"},
    "binds": {"bind_rows", "bind_cols"},
    "sets": {"union", "union_all", "intersect", "setdiff", "symdiff", "setequal"},
    "rows": {"rows_insert", "rows_append", "rows_update", "rows_patch", "rows_upsert", "rows_delete"},
    "missing_data": {"drop_na", "replace_na", "fill", "expand", "complete"},
    "pivots": {"pivot_longer", "pivot_wider"},
    "separate_unite": {"separate", "unite"},
    "nest_roundtrip": {"nest", "unnest"},
    "unnest_longer": {"unnest_longer"},
    "unnest_wider": {"unnest_wider"},
}


INVARIANT_ONLY = {
    "sample_n",
    "sample_frac",
    "slice_sample",
    "collect",
    "to_numpy",
    "glimpse",
    "peek",
    "nesting",
    "summarize",
}


@pytest.mark.parametrize("case", sorted(ORACLE_CASES))
@pytest.mark.parametrize("backend", BACKENDS)
def test_r_oracle_parity(case, backend):
    assert _normalise(ORACLE_CASES[case](backend)) == _oracle(case)


def test_every_public_frame_verb_has_a_parity_classification():
    public = {
        name
        for name in tidy3.__all__
        if callable(getattr(tidy3, name, None))
        and getattr(getattr(tidy3, name), "__module__", "")
        in {"tidy3.verbs", "tidy3.tidyr"}
    }
    differential = set().union(*CASE_VERBS.values())
    expected_gap_verbs = {
        "group_by",
        "group_split",
        "group_map",
        "group_modify",
        "group_nest",
        "with_groups",
        "hoist",
        "pack",
        "unpack",
        "separate_longer_delim",
        "separate_wider_delim",
    }
    assert public == differential | expected_gap_verbs | INVARIANT_ONLY


@pytest.mark.parametrize("backend", BACKENDS)
def test_stochastic_verbs_satisfy_dplyr_shape_and_group_invariants(backend):
    frame = tidy(
        {"g": ["a"] * 4 + ["b"] * 4, "x": list(range(8))},
        backend=backend,
    )
    assert len((frame >> tidy3.sample_n(3, seed=7)).collect(as_="pandas")) == 3
    assert len((frame >> tidy3.sample_frac(0.5, seed=7)).collect(as_="pandas")) == 4
    sampled = (
        frame >> group_by("g") >> tidy3.slice_sample(n=2, seed=7)
    ).collect(as_="pandas")
    assert sampled.groupby("g", observed=True).size().to_dict() == {"a": 2, "b": 2}


@pytest.mark.parametrize("backend", BACKENDS)
def test_weighted_slice_sample_support(backend):
    frame = tidy(
        {"x": [1, 2, 3], "weight": [1.0, 2.0, 2.0]}, backend=backend
    )
    sampled = frame >> tidy3.slice_sample(
        n=2, weight_by="weight", seed=7
    )
    assert len(sampled.collect(as_="pandas")) == 2
