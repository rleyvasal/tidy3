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
    case_when,
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
    recode_values,
    replace_values,
    replace_when,
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
    when_all,
    when_any,
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
    if rscript is None:
        return None
    # R alone is not enough: skip unless the packages the cases use load.
    probe = subprocess.run(
        [rscript, "-e", "library(dplyr); library(tidyr); library(jsonlite)"],
        capture_output=True,
    )
    return [rscript] if probe.returncode == 0 else None


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
        isinstance(value, (list, tuple, dict, np.ndarray)) for value in nonmissing
    ):
        return "list"
    return "character"


def _value(value: Any) -> Any:
    if isinstance(value, np.ndarray):  # a Polars list cell seen through pandas
        return [_value(item) for item in value.tolist()]
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
    dup = tidy(
        {"id": [1, 1, 2], "key": ["a", "a", "b"], "val": [1.0, 2.0, 3.0]},
        backend=backend,
    )
    with pytest.warns(UserWarning, match="not uniquely identified"):
        duplicated = dup >> pivot_wider(names_from="key", values_from="val")
    return {
        "long": long,
        "wide": long >> pivot_wider(names_from="name", values_from="value"),
        "duplicated": duplicated,
        "as_list": dup >> pivot_wider(names_from="key", values_from="val", values_fn="list"),
        "averaged": dup >> pivot_wider(names_from="key", values_from="val", values_fn="mean"),
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
        "text", ["left", "right"], "|", too_few="align_start"
    )
    edges = tidy({"id": [1, 2, 3], "text": ["a|b|c", "d", None], "z": [1, 2, 3]}, backend=backend)
    merged = edges >> separate_wider_delim(
        "text", ["left", "right"], "|", too_few="align_end", too_many="merge"
    )
    dropped = edges >> separate_wider_delim(
        "text", ["left", "right"], "|", too_few="align_start", too_many="drop"
    )
    return {
        "labels": labels,
        "longer": longer,
        "wider": wider,
        "merged": merged,
        "dropped": dropped,
    }


def _dplyr_12_helpers(backend: str):
    x = tidy(
        {
            "state": ["NC", "NYC", "CA", None, "NYC", "Unknown"],
            "score": [1.0, 2.0, 3.0, 4.0, 5.0, None],
        },
        backend=backend,
    )
    b = tidy(
        {
            "x": [True, True, True, False, False, False, None, None, None],
            "y": [True, False, None, True, False, None, True, False, None],
        },
        backend=backend,
    )
    pets = tidy(
        {"type": ["dog", "dog", "cat", "dog", "cat"], "age": [1.0, 3.0, 5.0, 2.0, 4.0]},
        backend=backend,
    )
    return {
        "values": x >> mutate(
            full=recode_values(
                "state", ("NC", "North Carolina"), ("NYC", "New York"), ("CA", "California")
            ),
            grouped=recode_values(
                "state", ("NC", "North Carolina"), (["NYC", "CA"], "elsewhere"),
                default="<not recorded>",
            ),
            label=recode_values("score", from_=[1, 2, 3, 4, 5], to=["SD", "D", "N", "A", "SA"]),
            replaced=replace_values("state", ("NYC", "NY")),
            no_missing=replace_values("state", (None, "Unknown (NA)")),
            tidied=replace_values("state", ([None, "Unknown"], "<not recorded>")),
            capped=replace_when("score", (col("score") > 3, 0.0)),
            big=col("score") > 3,
        ),
        "logic": b >> mutate(
            any_p=when_any(col("x"), col("y")),
            any_r=when_any(col("x"), col("y"), na_rm=True),
            all_p=when_all(col("x"), col("y")),
            all_r=when_all(col("x"), col("y"), na_rm=True),
        ),
        "pets": pets >> mutate(
            type=replace_when(
                "type", ((col("type") == "dog") & (col("age") <= 2), "puppy")
            )
        ),
        "filtered": x >> filter(when_any(col("state") == "CA", col("score") >= 5)),
        "oldest": pets
        >> group_by("type")
        >> filter(col("age") == tidy_max("age"))
        >> summarise(age_next=col("age") + 1),
        "strict": x >> mutate(
            band=case_when(
                (col("score") < 3, "low"),
                (col("score") >= 3, "high"),
                (col("score").is_null(), None),
                unmatched="error",
            )
        ),
    }


def _group_info(backend: str):
    import tidy3 as t3

    d = tidy(
        {"g": ["b", "a", "b", "a", "b"], "h": [1.0, 1.0, 2.0, 1.0, 2.0], "x": [1, 2, 3, 4, 5]},
        backend=backend,
    )
    gd = d >> group_by("g", "h")
    f = tidy(
        pd.DataFrame(
            {
                "k": pd.Categorical(["x", "x", None], categories=["x", "y"]),
                "g": ["b", "a", "b"],
            }
        ),
        backend=backend,
    )
    fd = f >> group_by("g", "k", drop=False)
    ungrouped = t3.group_data(d).collect(as_="pandas")
    ungrouped["n"] = t3.n_groups(d)
    return {
        "data": t3.group_data(gd),
        "keys": gd >> t3.group_keys(),
        "sizes": pd.DataFrame({"size": t3.group_size(gd)}),
        "indices": pd.DataFrame({"index": t3.group_indices(gd)}),
        "counts": pd.DataFrame(
            {"n": [t3.n_groups(gd)], "vars": [",".join(t3.group_vars(gd))]}
        ),
        "ungrouped": ungrouped,
        "empty": t3.group_data(fd),
        "trimmed": pd.DataFrame({"size": t3.group_size(fd >> t3.group_trim())}),
        "split": pd.concat(
            [part.collect(as_="pandas").head(1) for part in gd >> t3.group_split()],
            ignore_index=True,
        ),
        "modified": gd >> t3.group_modify(lambda part, key: part.collect(as_="pandas").head(1)),
        "nested": pd.DataFrame(
            {
                "g": (n := (d >> t3.nest_by("g")).collect(as_="pandas"))["g"],
                "rows": [len(cell) for cell in n["data"]],
            }
        ),
        "group_nested": pd.DataFrame(
            {
                "g": (m := (gd >> t3.group_nest()).collect(as_="pandas"))["g"],
                "h": m["h"],
                "rows": [len(cell) for cell in m["data"]],
            }
        ),
    }


def _ordering(backend: str):
    import tidy3 as t3

    d = tidy(
        {
            "g": ["a", "a", "b", "b", "a", "b", "a"],
            "t": [3.0, 1.0, 2.0, None, 2.0, 1.0, 1.0],
            "x": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0],
        },
        backend=backend,
    )
    return {
        "ungrouped": d >> mutate(
            running=t3.order_by("t", col("x").cum_sum()),
            previous=t3.with_order("t", t3.lag, "x"),
            latest_first=t3.order_by(t3.desc("t"), col("x").cum_sum()),
        ),
        "grouped": d
        >> group_by("g")
        >> mutate(
            running=t3.order_by("t", col("x").cum_sum()),
            rank=t3.with_order(t3.desc("t"), t3.row_number, "x"),
        )
        >> ungroup(),
    }


def _grids(backend: str):
    import tidy3 as t3

    df = tidy({"a": [2, 1, 2], "b": ["v", "u", "v"]}, backend=backend)
    counts = tidy({"x": ["a", "b", "c"], "n": [1.0, 0.0, 2.0]}, backend=backend)
    obs = tidy(
        {"country": ["A", "A", "B"], "year": [1952, 1962, 1957], "pop": [1.0, 2.0, 3.0]},
        backend=backend,
    )
    return {
        "expand_grid": t3.expand_grid(x=[3.0, 1.0, 3.0], y=["b", "a"]),
        "expand_grid_frame": t3.expand_grid(df, z=[1, 2]),
        "crossing": t3.crossing(x=[3.0, 1.0, 3.0, None], y=["b", "a"]),
        "crossing_frame": t3.crossing(df, z=[2, 1]),
        "uncounted": counts >> t3.uncount("n"),
        "uncounted_id": counts >> t3.uncount("n", id="copy"),
        "uncounted_kept": counts >> t3.uncount("n", remove=False),
        "sequence": pd.DataFrame({"value": t3.full_seq([1.0, 2.0, 4.0, 5.0, 10.0], 1)}),
        "completed": obs >> complete("country", year=t3.full_seq("year", 5)),
        "expanded": obs >> expand("country", year=t3.full_seq("year", 5)),
    }


def _splitting(backend: str):
    import tidy3 as t3

    d = tidy({"id": [1, 2, 3], "code": ["ab-12", "cd-3", None]}, backend=backend)
    pattern = [("letters", "[a-z]+"), "-", ("number", "[0-9]+")]
    dates = tidy({"id": [1, 2, 3], "code": ["20240115", "2023", None]}, backend=backend)
    chopped = tidy(
        {"g": ["a", "a", "b"], "x": [1, 2, 3], "y": [10.0, 20.0, 30.0]}, backend=backend
    ) >> t3.chop("x", "y")
    lists = tidy({"g": ["a", "b", "c"], "x": [[1, 2], [], None]}, backend=backend)
    return {
        "regex": d >> t3.separate_wider_regex("code", pattern),
        "regex_partial": tidy({"code": ["ab-12", "ab"]}, backend=backend)
        >> t3.separate_wider_regex("code", pattern, too_few="align_start"),
        "regex_prefixed": d >> t3.separate_wider_regex("code", pattern, names_sep="_"),
        "position": dates
        >> t3.separate_wider_position(
            "code", [("year", 4), ("month", 2), ("day", 2)], too_few="align_start"
        ),
        "position_skip": tidy({"code": ["20240115"]}, backend=backend)
        >> t3.separate_wider_position("code", [("year", 4), 2, ("day", 2)]),
        "longer": tidy({"id": [1, 2, 3], "x": ["abcde", "fg", "h"]}, backend=backend)
        >> t3.separate_longer_position("x", 2),
        "chopped": chopped,
        "unchopped": chopped >> t3.unchop("x", "y"),
        "unchop_drop": lists >> t3.unchop("x"),
        "unchop_keep": lists >> t3.unchop("x", keep_empty=True),
        "auto_wider": tidy(
            {"id": [1, 2], "v": [{"a": 1.0, "b": 2.0}, {"a": 3.0, "b": 4.0}]}, backend=backend
        )
        >> t3.unnest_auto("v"),
        "auto_longer": tidy({"id": [1, 2], "v": [[1.0, 2.0], [3.0]]}, backend=backend)
        >> t3.unnest_auto("v"),
    }


def _empty_groups(backend: str):
    d = tidy(
        pd.DataFrame(
            {
                "k": pd.Categorical(["x", "x", None], categories=["x", "y"]),
                "g": ["b", "a", "b"],
                "v": [1.0, 2.0, 3.0],
            }
        ),
        backend=backend,
    )
    stats = dict(n=n(), s=tidy_sum("v"), m=mean("v"), groups="drop")
    return {
        "kg": d >> group_by("k", "g", drop=False) >> summarise(**stats),
        "gk": d >> group_by("g", "k", drop=False) >> summarise(**stats),
        "k": d >> group_by("k", drop=False) >> summarise(n=n(), groups="drop"),
        "counted": d >> count("k", "g", drop=False),
        "sorted": d >> count("k", drop=False, sort=True),
    }


def _join_names(backend: str):
    import tidy3 as t3

    a = tidy({"id": [1, 2, 3], "v": [1.0, 2.0, 3.0], "w": [5.0, 6.0, 7.0]}, backend=backend)
    b = tidy({"id": [1, 2, 4], "v": [10.0, 20.0, 40.0]}, backend=backend)
    clash = tidy({"id": [1, 2], "v": [1.0, 2.0], "v_x": [9.0, 9.0]}, backend=backend)
    return {
        "left": a >> left_join(b, by="id"),
        "inner": a >> inner_join(b, by="id"),
        "full": a >> full_join(b, by="id"),
        "right": a >> right_join(b, by="id"),
        "pair": a >> left_join(b, by="id", suffix=("_l", "_r")),
        "kept": a >> left_join(b, by="id", keep=True),
        "clash": clash >> left_join(b, by="id"),
        "empty_left": a >> left_join(b, by="id", suffix=("", "_right")),
        "crossed": (a >> t3.slice_head(n=2)) >> t3.cross_join(b >> t3.slice_head(n=2)),
    }


def _pivot_options(backend: str):
    d = tidy(
        {"id": [1.0, 1.0, 2.0, 2.0], "extra": ["p", "q", "r", "s"], "key": ["a", "b", "a", "b"],
         "sub": ["x", "y", "x", "x"], "v": [1.0, 2.0, 3.0, 4.0], "w": [5.0, 6.0, 7.0, 8.0]},
        backend=backend,
    )
    f = tidy(pd.DataFrame({"id": [1.0, 2.0], "key": pd.Categorical(["a", "a"], categories=["a", "b", "c"]), "v": [1.0, 2.0]}), backend=backend)
    g = tidy(pd.DataFrame({"id": pd.Categorical(["i1", "i1"], categories=["i1", "i2"]), "key": ["a", "b"], "v": [1.0, 2.0]}), backend=backend)
    l = tidy({"id": [1, 2], "week_1": [5, 6], "week_2": [7, 8]}, backend=backend)
    return {
        "sep": d >> pivot_wider(id_cols="id", names_from=["key", "sub"], values_from="v", names_sep="."),
        "glue": d >> pivot_wider(id_cols="id", names_from="key", values_from=["v", "w"], names_glue="{key}_{.value}"),
        "slowest": d >> pivot_wider(id_cols="id", names_from="key", values_from=["v", "w"], names_vary="slowest"),
        "names_expanded": f >> pivot_wider(names_from="key", values_from="v", names_expand=True),
        "ids_expanded": g >> pivot_wider(names_from="key", values_from="v", id_expand=True),
        "unused": d >> pivot_wider(id_cols="id", names_from="key", values_from="v", unused_fn={"w": "max"}),
        "transformed": l >> pivot_longer(
            ["week_1", "week_2"], names_to="week", names_prefix="week_",
            names_transform={"week": int}, values_transform={"value": str},
        ),
    }


def _list_options(backend: str):
    import tidy3 as t3

    d = tidy({"id": [1, 2], "code": ["a-1", "b-2"]}, backend=backend)
    h = tidy(
        {"id": [1, 2], "info": [{"name": "x", "n": "1", "k": 5.0}, {"name": "y", "n": "2", "k": 6.0}]},
        backend=backend,
    )
    n = tidy({"g": [1.0, 1.0, 2.0], "x_a": [1, 2, 3], "x_b": [4, 5, 6]}, backend=backend)
    return {
        "sep": d >> t3.separate_wider_delim("code", ["l", "n"], "-", names_sep="_"),
        "kept": d >> t3.separate_wider_delim("code", ["l", "n"], "-", cols_remove=False),
        "indexed": tidy({"id": [1, 2], "x": [[10.0, 11.0], [20.0]]}, backend=backend)
        >> t3.unnest_longer("x", indices_include=True),
        "longer_int": tidy({"id": [1, 2], "x": [["1", "2"], ["3"]]}, backend=backend)
        >> t3.unnest_longer("x", transform=int),
        "wider_int": tidy({"id": [1, 2], "x": [{"a": "1", "b": "2"}, {"a": "3", "b": "4"}]}, backend=backend)
        >> t3.unnest_wider("x", transform={"a": int}),
        "hoisted": h >> t3.hoist("info", nm="name", n="n", transform={"n": int}) >> t3.select(~t3.all_of(["info"])),
        "hoisted_all": h >> t3.hoist("info", name="name", n="n", k="k"),
        "nested": n >> t3.nest("x", cols=["x_a", "x_b"], names_sep="_") >> t3.unnest("x"),
    }


def _specs(backend: str):
    import tidy3 as t3

    wide = tidy(
        {"id": [1, 2], "mean_1": [5.0, 6.0], "mean_2": [7.0, 8.0], "sd_1": [0.1, 0.2], "sd_2": [0.3, 0.4]},
        backend=backend,
    )
    long = tidy({"id": [1.0, 1.0, 2.0, 2.0], "key": ["a", "b", "a", "b"], "v": [1, 2, 3, 4], "w": [5, 6, 7, 8]}, backend=backend)
    s1 = t3.build_longer_spec(wide, ~t3.all_of(["id"]), names_to=[".value", "visit"], names_sep="_")
    s2 = t3.build_longer_spec(wide, t3.starts_with("mean"), names_to="visit", names_prefix="mean_", values_to="score")
    s3 = t3.build_wider_spec(long, names_from="key", values_from=["v", "w"])
    s4 = t3.build_wider_spec(long, names_from="key", values_from="v").collect(as_="pandas")
    s4[".name"] = ["first", "second"]
    return {
        "longer_spec": s1, "longer": wide >> t3.pivot_longer_spec(s1),
        "longer_values": wide >> t3.pivot_longer_spec(s2),
        "wider_spec": s3, "wider": long >> t3.pivot_wider_spec(s3),
        "renamed": long >> t3.pivot_wider_spec(s4, id_cols="id"),
        "glued": t3.build_wider_spec(long, names_from="key", values_from="v", names_glue="col_{key}", names_sort=True),
    }


def _helpers(backend: str):
    import tidy3 as t3

    rounded = (
        tidy({"x": [0.5, 1.5, -1.5, 1.25]}, backend=backend)
        >> mutate(z=t3.round("x"))
    ).collect()["z"]
    rounded_1 = (
        tidy({"x": [1.25, 1.35]}, backend=backend)
        >> mutate(z=t3.round("x", 1))
    ).collect()["z"]
    rounded_tens = (
        tidy({"x": [123.0, -15.0]}, backend=backend)
        >> mutate(z=t3.round("x", -1))
    ).collect()["z"]
    pairs = tidy(
        {"a": [1.0, None, 5.0], "b": [2.0, 3.0, None], "c": [0.0, 4.0, 4.0]},
        backend=backend,
    )
    paired = (
        pairs
        >> mutate(
            pmin=t3.pmin("a", "b"),
            pmin_rm=t3.pmin("a", "b", na_rm=True),
            pmax=t3.pmax("a", "c"),
        )
    ).collect()
    rows = tidy(
        {"a": [1.0, None, None], "b": [None, None, 4.0], "c": [2.0, None, 6.0]},
        backend=backend,
    )
    reduced = (
        rows
        >> mutate(
            row_sums=t3.row_sums("a", "b", "c"),
            row_sums_rm=t3.row_sums("a", "b", "c", na_rm=True),
            row_means_rm=t3.row_means("a", "b", "c", na_rm=True),
        )
    ).collect()
    return {
        "rounded": rounded,
        "rounded_1": rounded_1,
        "rounded_tens": rounded_tens,
        "pmin": paired["pmin"],
        "pmin_rm": paired["pmin_rm"],
        "pmax": paired["pmax"],
        "row_sums": reduced["row_sums"],
        "row_sums_rm": reduced["row_sums_rm"],
        "row_means_rm": reduced["row_means_rm"],
    }


def _lubridate_dates(backend: str):
    import tidy3 as t3
    from datetime import date

    frame = tidy(
        {
            "d": [
                date(2010, 1, 31),
                date(2012, 2, 29),
                date(2010, 1, 25),
                date(2010, 2, 3),
                date(2010, 1, 1),
            ]
        },
        backend=backend,
    )
    out = (
        frame
        >> mutate(
            rollback=t3.add_with_rollback("d", t3.months(1)),
            year_roll=t3.add_with_rollback("d", t3.years(1)),
            floor_week=t3.floor_date("d", "week"),
            ceiling_week=t3.ceiling_date("d", "week"),
            ceiling_month=t3.ceiling_date("d", "month"),
            days=t3.days_in_month("d"),
        )
        >> mutate(
            rollback=t3.as_character("rollback"),
            year_roll=t3.as_character("year_roll"),
            floor_week=t3.as_character("floor_week"),
            ceiling_week=t3.as_character("ceiling_week"),
            ceiling_month=t3.as_character("ceiling_month"),
        )
    )
    return out >> select(
        "rollback", "year_roll", "floor_week", "ceiling_week", "ceiling_month", "days"
    )


def _readr_parse(backend: str):
    import tidy3 as t3

    number = (
        tidy({"x": ["$1,234.50", "abc"]}, backend=backend)
        >> mutate(number=t3.parse_number("x"))
    ).collect()["number"]
    plain = (
        tidy({"x": ["1.5", "x"]}, backend=backend)
        >> mutate(plain=t3.parse_double("x"))
    ).collect()["plain"]
    integer = (
        tidy({"x": ["1234", "2.5"]}, backend=backend)
        >> mutate(integer=t3.parse_integer("x"))
    ).collect()["integer"]
    logical = (
        tidy({"x": ["true", "1", "0", "yes"]}, backend=backend)
        >> mutate(logical=t3.parse_logical("x"))
    ).collect()["logical"]
    return {"number": number, "plain": plain, "integer": integer, "logical": logical}


def _column(backend: str, values: list[Any], expr: Any) -> pl.Series:
    """One expression over one column ``x``, as a vector."""
    return (tidy({"x": values}, backend=backend) >> mutate(out=expr)).collect()["out"]


def _helper_edges(backend: str):
    import tidy3 as t3

    v = [1.0, None, float("nan"), float("inf")]
    half = [-2.5, 2.5, None]
    return {
        "is_na": _column(backend, v, t3.is_na("x")),
        "is_finite": _column(backend, v, t3.is_finite("x")),
        "is_infinite": _column(backend, v, t3.is_infinite("x")),
        "abs": _column(backend, half, t3.abs("x")),
        "floor": _column(backend, half, t3.floor("x")),
        "ceiling": _column(backend, half, t3.ceiling("x")),
        "round_2": _column(backend, [2.675, 0.125, 1.005], t3.round("x", 2)),
        "round_1": _column(backend, [0.15, 0.25, 0.35], t3.round("x", 1)),
        "round_tens": _column(backend, [125.0, 135.0, -15.0], t3.round("x", -1)),
        "pmin_text": (
            tidy({"a": ["nan", "b"], "b": ["c", "a"]}, backend=backend)
            >> mutate(out=t3.pmin("a", "b"))
        ).collect()["out"],
        "pmin_nan": (
            tidy({"a": [1.0, float("nan")], "b": [2.0, 3.0]}, backend=backend)
            >> mutate(out=t3.pmin("a", "b"))
        ).collect()["out"],
    }


def _is_nan(backend: str):
    import tidy3 as t3

    if backend == "pandas":
        pytest.skip("pandas stores a missing float as NaN, so is_nan is true for NA there")
    return _column(backend, [1.0, None, float("nan"), float("inf")], t3.is_nan("x"))


def _conversions(backend: str):
    import tidy3 as t3
    from datetime import date

    return {
        "as_double": _column(
            backend,
            ["1.5", " 2 ", "1e5", "-.5", "NaN", "abc", "1,234"],
            t3.as_double("x", strict=False),
        ),
        "as_integer": _column(backend, ["2", " 7 ", "-3", "007"], t3.as_integer("x")),
        "as_integer_num": _column(backend, [2.0, -3.0, None], t3.as_integer("x")),
        "as_logical_text": _column(
            backend,
            ["TRUE", "true", "T", "True", "FALSE", "false", "F", "tRuE", "yes", "1"],
            t3.as_logical("x", strict=False),
        ),
        "as_logical_num": _column(
            backend, [0.0, 2.0, -1.0, float("nan"), None], t3.as_logical("x")
        ),
        "as_character": _column(
            backend, [1.0, 1.5, 123456.0, 0.25, -3.0, 100001.0], t3.as_character("x")
        ),
        "as_logical_exact": _column(
            backend,
            [" TRUE", "TRUE ", "yes", "True", "T", "false", "F"],
            t3.as_logical("x", strict=False),
        ),
        "as_character_logical": _column(backend, [True, False, None], t3.as_character("x")),
        "as_date": _column(
            backend,
            ["2010-01-31", None],
            t3.as_character(t3.as_date("x")),
        ),
    }


def _lubridate_more(backend: str):
    import tidy3 as t3
    from datetime import date, datetime

    def days(values, expr):
        return _column(backend, values, t3.as_character(expr))

    d = [date(2024, 2, 29), date(2024, 1, 31), date(2010, 3, 31)]
    t = [datetime(2026, 10, 21, 15, 30), datetime(2026, 10, 18, 9, 0), datetime(2026, 10, 18)]
    dated = tidy({"d": d, "k": [1, None, -1]}, backend=backend)
    moved = (
        dated
        >> mutate(
            both=t3.add_with_rollback("d", t3.years(1) + t3.months(1)),
            back=t3.add_with_rollback("d", -(t3.years(1) + t3.months(1))),
            month_column=col("d") + t3.months(col("k")),
            year_minus=col("d") - t3.years(1),
            floor_quarter=t3.floor_date("d", "quarter"),
            ceiling_day=t3.ceiling_date("d", "day"),
        )
        >> mutate(
            both=t3.as_character("both"),
            back=t3.as_character("back"),
            month_column=t3.as_character("month_column"),
            year_minus=t3.as_character("year_minus"),
            floor_quarter=t3.as_character("floor_quarter"),
            ceiling_day=t3.as_character("ceiling_day"),
            year=t3.year("d"),
            month=t3.month("d"),
            day=t3.day("d"),
            days=t3.days_in_month("d"),
        )
    ).collect()
    return {
        "ymd": days(
            ["2010-01-31", "20100131", "2010/1/31", "2010.01.31", "2010 01 31", "bad"],
            t3.ymd("x"),
        ),
        "ymd_with_time": days(["2010-01-31 10:00:00"], t3.ymd("x"))[0],
        "mdy": days(["01/31/2010", "1-31-2010", "Jan 31 2010"], t3.mdy("x")),
        "dmy": days(["31.01.2010", "31012010"], t3.dmy("x")),
        "ym": days(["2010-02", "201002"], t3.ym("x")),
        "parts": {
            "year": moved["year"],
            "month": moved["month"],
            "day": moved["day"],
            "days": moved["days"],
        },
        "both": moved["both"],
        "back": moved["back"],
        "month_column": moved["month_column"],
        "year_minus": moved["year_minus"],
        "floor_week": days(t, t3.floor_date("x", "week")),
        "ceiling_week": days(t, t3.ceiling_date("x", "week")),
        "floor_week_monday": days(t, t3.floor_date("x", "week", week_start=1)),
        "floor_month": days(t, t3.floor_date("x", "month")),
        "ceiling_hour": days(t, t3.ceiling_date("x", "hour")),
        "floor_quarter": moved["floor_quarter"],
        "ceiling_day": moved["ceiling_day"],
        "ceiling_month": days(
            [date(2010, 1, 31), date(2010, 2, 1)], t3.ceiling_date("x", "month")
        ),
    }


def _readr_more(backend: str):
    import tidy3 as t3

    euro = t3.locale(decimal_mark=",", grouping_mark=".")
    kept = (
        tidy({"x": ["1", "a", "2", "b"]}, backend=backend)
        >> filter(col("x") != "1")
        >> mutate(n=t3.parse_double("x"))
    )
    found = t3.problems(kept).collect()
    return {
        "number": _column(
            backend,
            ["-$1,234", "$-5", "USD -7.5", "Price - 12", "50%", "abc"],
            t3.parse_number("x"),
        ),
        "double": _column(backend, ["1.5", " 1.5 ", "1e3", "abc"], t3.parse_double("x")),
        "double_grouped": _column(backend, ["1,234.5"], t3.parse_double("x"))[0],
        "double_euro": _column(backend, ["1,5", "-0,25"], t3.parse_double("x", locale=euro)),
        "integer": _column(backend, ["1234", " 7 ", "2.5", "x"], t3.parse_integer("x")),
        "logical": _column(
            backend,
            ["TRUE", "T", "true", "t", "1", "FALSE", "f", "0", "yes"],
            t3.parse_logical("x"),
        ),
        "date": _column(
            backend,
            ["31/01/2010", "bad"],
            t3.as_character(t3.parse_date("x", "%d/%m/%Y")),
        ),
        # readr reads %y with a fixed cutoff of 68; tidy3 asks for it.
        "date_cutoff": _column(
            backend,
            ["68/01/02", "69/01/02"],
            t3.as_character(t3.parse_date("x", "%y/%m/%d", cutoff=68)),
        ),
        "problem_rows": found["row"],
        "problem_values": found["original"],
    }


def _readr_files(backend: str):
    import tempfile

    import tidy3 as t3

    if backend == "pandas":
        pytest.skip("read_csv reads with Polars; the pandas backend has no reader of its own")
    folder = Path(tempfile.mkdtemp())
    cars_path = folder / "cars.csv"
    cars_path.write_text(
        "model,mpg,cyl,am\nMazda RX4,21.0, 6,TRUE\nDatsun 710,NA,4 ,false\nValiant, NA , 6,\n"
    )
    bad_path = folder / "bad.csv"
    bad_path.write_text("model,hp\nMazda RX4,110\nValiant,abc\nDatsun 710,2.5\n")
    cars = t3.read_csv(cars_path).collect()
    bad = t3.read_csv(bad_path, col_types={"model": "character", "hp": "integer"})
    found = t3.problems(bad).collect()
    return {
        "mpg": cars["mpg"],
        # readr guesses whole numbers as double; compare the values.
        "cyl": cars["cyl"].cast(pl.Float64),
        "am": cars["am"],
        "hp": bad.collect()["hp"],
        "problem_rows": found["row"],
        "problem_values": found["original"],
    }


def _round_grid(backend: str):
    import tidy3 as t3

    x = (
        [k / 1000 for k in range(2001)]
        + [-k / 1000 * 3.7 for k in range(1, 501)]
        + [k / 8 for k in range(401)]
        + [2.675, 1.005, 5.015, 0.285, 2.0**900, 123456789.123456789]
    )
    frame = tidy({"x": x, "big": [value * 1000 for value in x]}, backend=backend)
    out = (
        frame
        >> mutate(
            d0=t3.round("x", 0),
            d1=t3.round("x", 1),
            d2=t3.round("x", 2),
            d3=t3.round("x", 3),
            tens=t3.round("big", -1),
            hundreds=t3.round("big", -2),
        )
    ).collect()
    exact = lambda column: pl.Series(["%.17g" % value for value in out[column]])
    return {name: exact(name) for name in ("d0", "d1", "d2", "d3", "tens", "hundreds")}


def _readr_grid(backend: str):
    import tidy3 as t3

    euro = t3.locale(decimal_mark=",", grouping_mark=".")
    result = {
        "number": _column(
            backend,
            ["-$1,234", "$-5", "a-5", "- 5", "--5", "+5", "$+5", "1-2", "1,234,", "(5)",
             ".5", "-.5", "12abc34", "abc", "1e3", "1E3", "50%", "1,23", "1,2345",
             "12,34,567", "$1,234.50", "USD -7.5", "-", "Price 12", "1.2.3", "1,234.5.6",
             " 7 ", "", "Price - 12", "3eggs"],
            t3.parse_number("x"),
        ),
        "number_euro": _column(
            backend, ["1.234,5", "€1.234,5", "1,5"], t3.parse_number("x", locale=euro)
        ),
        "double": _column(
            backend,
            ["1,234", "1234", " 1e3", "1e+3", "1E-2", ".5", "5.", "+5", "-.5", "Inf", "inf",
             "NaN", "1 234", "", "1.2.3"],
            t3.parse_double("x"),
        ),
        "double_euro": _column(
            backend, ["1,5", "1.234,5", "1e3", ",5"], t3.parse_double("x", locale=euro)
        ),
        "integer": _column(
            backend, ["1,234", "+5", "007", "1e3", "-0", " 12 ", "1.0", ""], t3.parse_integer("x")
        ),
        "logical": _column(
            backend,
            ["True", "TRUE", "yes", "Y", "N", "no", " true", "F", "0"],
            t3.parse_logical("x"),
        ),
    }
    return result


def _readr_rows(backend: str):
    import tempfile

    import tidy3 as t3

    if backend == "pandas":
        pytest.skip("read_csv reads with Polars; the pandas backend has no reader of its own")
    folder = Path(tempfile.mkdtemp())
    result = {}

    def rows(text, **kwargs):
        path = folder / "file.csv"
        path.write_text(text)
        frame = t3.read_csv(path, col_types={"model": "character", "hp": "integer"}, **kwargs)
        return t3.problems(frame).collect()["row"]

    result["rows_plain"] = rows("model,hp\nMazda RX4,110\nValiant,abc\nDatsun 710,2.5\n")
    result["rows_skip"] = rows("junk line\nmodel,hp\nMazda RX4,110\nValiant,abc\n", skip=1)[0]
    result["rows_comment"] = rows(
        "model,hp\n# note\nMazda RX4,110\n# note\nValiant,abc\n", comment="#"
    )[0]
    return result


def _lubridate_grid(backend: str):
    import tidy3 as t3
    from datetime import date

    def days(values, expr):
        return _column(backend, values, t3.as_character(expr))

    d = [date(2010, 1, 31), date(2010, 1, 30), date(2010, 1, 1), date(2010, 4, 1), date(2010, 2, 17)]
    return {
        "ymd": days(["2010-01-31T10:00:00", "2010-01-31 10:00", "2010 Jan 31", "2010-jan-31"], t3.ymd("x")),
        "mdy": days(["January 31, 2010", "jan 31 2010", "Sept 1 2010", "31 Jan 2010"], t3.mdy("x")),
        "dmy": days(["31 Jan 2010", "31-January-2010", "1st March 2010"], t3.dmy("x")),
        "ym": days(["2010 Feb", "2010-February"], t3.ym("x")),
        "ceiling_week": days(d, t3.ceiling_date("x", "week")),
        "ceiling_month": days(d, t3.ceiling_date("x", "month")),
        "ceiling_quarter": days(d, t3.ceiling_date("x", "quarter")),
        "ceiling_year": days(d, t3.ceiling_date("x", "year")),
        "floor_week": days(d, t3.floor_date("x", "week")),
    }


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
    "dplyr_12_helpers": _dplyr_12_helpers,
    "group_info": _group_info,
    "ordering": _ordering,
    "grids": _grids,
    "splitting": _splitting,
    "empty_groups": _empty_groups,
    "join_names": _join_names,
    "pivot_options": _pivot_options,
    "list_options": _list_options,
    "specs": _specs,
    "helpers": _helpers,
    "lubridate_dates": _lubridate_dates,
    "readr_parse": _readr_parse,
    "helper_edges": _helper_edges,
    "is_nan": _is_nan,
    "conversions": _conversions,
    "lubridate_more": _lubridate_more,
    "readr_more": _readr_more,
    "readr_files": _readr_files,
    "round_grid": _round_grid,
    "readr_grid": _readr_grid,
    "readr_rows": _readr_rows,
    "lubridate_grid": _lubridate_grid,
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
    "new_helpers": {"separate_longer_delim", "separate_wider_delim"},
    "dplyr_12_helpers": {"mutate", "filter"},
    "group_info": {"group_split", "group_modify", "group_nest"},
    "ordering": {"mutate"},
    "grids": {"expand_grid", "crossing", "uncount", "full_seq", "complete", "expand"},
    "empty_groups": {"summarise", "count"},
    "pivot_options": {"pivot_wider", "pivot_longer"},
    "specs": {
        "build_longer_spec", "pivot_longer_spec", "build_wider_spec",
        "pivot_wider_spec", "check_pivot_spec",
    },
    "list_options": {"separate_wider_delim", "unnest_longer", "unnest_wider", "hoist", "nest", "unnest"},
    "join_names": {"left_join", "inner_join", "full_join", "right_join", "cross_join"},
    "splitting": {
        "separate_wider_regex", "separate_wider_position", "separate_longer_position",
        "chop", "unchop", "unnest_auto",
    },
}


INVARIANT_ONLY = {
    "sample_n",
    "sample_frac",
    "slice_sample",
    "collect",
    "compute",
    "to_numpy",
    "glimpse",
    "peek",
    "nesting",
    "summarize",
}


@pytest.mark.parametrize("case", sorted(ORACLE_CASES))
@pytest.mark.parametrize("backend", BACKENDS)
def test_r_oracle_parity(case, backend):
    oracle = _oracle(case)
    if isinstance(oracle, dict) and oracle.get("skip") is True:
        pytest.skip(str(oracle.get("reason") or "oracle case skipped"))
    assert _normalise(ORACLE_CASES[case](backend)) == oracle


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
        "group_map",
        "with_groups",
        "pack",
        "unpack",
    }
    # Column-name helpers from janitor and rlang, not dplyr or tidyr.
    not_dplyr = {"clean_names", "make_clean_names", "set_names"}
    assert public == differential | expected_gap_verbs | INVARIANT_ONLY | not_dplyr


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
