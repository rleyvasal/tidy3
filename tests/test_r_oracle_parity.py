from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import warnings
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
                as_="pandas", arrow_backed=True
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


# R's answers depend only on the case script, R, and its packages. One R
# process answers every case (``--batch`` in r_oracle_cases.R), and the answers
# are kept on disk for the next run with the same script and R setup.
# TIDY3_R_ORACLE_CACHE names the folder (default ~/.cache/tidy3/r-oracle);
# "off" always asks R.
_R_PACKAGES = ("dplyr", "tidyr", "jsonlite", "lubridate", "readr", "stringr", "forcats", "janitor")


def _r_setup() -> str:
    versions = ", ".join(f'"{name}"' for name in _R_PACKAGES)
    probe = subprocess.run(
        [*R_COMMAND, "-e", f"cat(R.version.string, sapply(c({versions}), function(p) "
         "tryCatch(format(packageVersion(p)), error=function(e) 'none')))"],
        capture_output=True, text=True, check=True,
    )
    return " ".join(R_COMMAND) + "\n" + probe.stdout


def _cache_file() -> Path | None:
    setting = os.environ.get("TIDY3_R_ORACLE_CACHE", "")
    if setting.lower() == "off":
        return None
    folder = Path(setting).expanduser() if setting else Path.home() / ".cache" / "tidy3" / "r-oracle"
    key = hashlib.sha256()
    key.update(R_CASES.read_bytes())
    key.update(_r_setup().encode())
    key.update(" ".join(sorted(ORACLE_CASES)).encode())
    return folder / f"{key.hexdigest()[:20]}.json"


@lru_cache(maxsize=None)
def _oracle_answers() -> dict[str, Any]:
    if R_COMMAND is None:
        pytest.skip(
            "R oracle unavailable; install R/dplyr/tidyr/jsonlite or set "
            "TIDY3_R_ORACLE_MANIFEST"
        )
    path = _cache_file()
    if path is not None and path.exists():
        try:
            return json.loads(path.read_text())
        except ValueError:
            pass  # a damaged file is recomputed below
    completed = subprocess.run(
        [*R_COMMAND, str(R_CASES), "--batch", *sorted(ORACLE_CASES)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    answers = json.loads(completed.stdout)
    if path is not None:
        # Parallel test runs may finish together; each renames a whole file.
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(f".{os.getpid()}.tmp")
        partial.write_text(completed.stdout)
        os.replace(partial, path)
    return answers


def _oracle(case: str) -> Any:
    answer = _oracle_answers()[case]
    if isinstance(answer, dict) and set(answer) == {"oracle_error"}:
        raise RuntimeError(f"R oracle case {case!r} failed in R: {answer['oracle_error']}")
    return answer


def _base():
    return tidy(
        pd.DataFrame(
            {
                "id": pd.Series([1, 2, 3, 4], dtype="int64"),
                "g": ["b", "a", "b", "a"],
                "x": [2.0, None, 4.0, 1.0],
                "y": [20.0, 10.0, 40.0, 30.0],
            }
        ),
    )


def _filter_missing():
    return _base() >> filter(col("x") > 1)


def _filter_out_missing():
    return _base() >> filter_out(col("x") > 1)


def _mutate_sequential():
    return _base() >> mutate(
        a=col("y") * 2, b=col("a") + 1, before="x"
    )


def _transmute():
    return _base() >> transmute(
        g=col("g"), a=col("y") * 2, b=col("a") + 1
    )


def _select_drop():
    frame = _base()
    return {"selected": frame >> select("g", "x"), "dropped": frame >> drop("y")}


def _rename():
    return _base() >> rename(score="x")


def _rename_with():
    return _base() >> rename_with(str.upper, "x", "y")


def _relocate():
    return _base() >> relocate("y", before="x")


def _pull():
    return _base() >> pull("y")


def _arrange():
    frame = _base()
    return {
        "ascending": frame >> arrange("x"),
        "descending": frame >> arrange(tidy3.desc("x")),
    }


def _distinct():
    frame = tidy(
        {"g": ["b", "a", "b", "a"], "x": [1, 2, 1, 3]},
    )
    return frame >> distinct("g")


def _slices():
    frame = _base()
    return {
        "positions": frame >> slice(3, 1, 1),
        "first_rows": frame >> head(2),
        "head": frame >> slice_head(n=2),
        "tail": frame >> slice_tail(n=2),
        "minimum": frame >> slice_min("x", n=2),
        "maximum": frame >> slice_max("x", n=2),
    }


def _grouping():
    frame = _base()
    return {
        "persistent": frame
        >> group_by("g")
        >> summarise(n=n(), avg=mean("y"), groups="drop"),
        "transient": frame >> summarise(n=n(), avg=mean("y"), by="g"),
        "ungrouped": frame >> group_by("g") >> ungroup() >> summarise(n=n()),
        "by_col": frame
        >> group_by(col("g"), pl.col("x"))
        >> ungroup(col("x"))
        >> summarise(n=n(), groups="drop"),
    }


def _rowwise():
    frame = tidy(
        {"id": [1, 2], "x": [1.0, 2.0], "y": [10.0, 20.0]},
    )
    return frame >> rowwise("id") >> mutate(
        total=tidy_sum(c_across("x", "y"))
    ) >> ungroup()


def _reframe():
    frame = tidy(
        {"g": ["b", "b", "a"], "x": [2.0, 4.0, 10.0]},
    )
    return frame >> group_by("g") >> reframe(value=col("x"), avg=mean("x"))


def _counts():
    frame = _base()
    return {
        "counted": frame >> count("g"),
        "tallied": frame >> group_by("g") >> tally(),
        "added": frame >> add_count("g"),
        "add_tallied": frame >> group_by("g") >> add_tally() >> ungroup(),
        "counted_computed": frame >> count(big=col("y") > 15),
        "added_computed": frame >> add_count(col("g"), big=col("y") > 15, name="k"),
    }


def _join_frames():
    left = tidy(
        pd.DataFrame({"k": [1.0, 2.0, None], "x": ["a", "b", "c"]}),
    )
    right = pd.DataFrame({"k": [2.0, None, 3.0], "y": [20.0, 99.0, 30.0]})
    return left, right


def _joins():
    left, right = _join_frames()
    return {
        "left": left >> left_join(right, by="k"),
        "right": left >> right_join(right, by="k"),
        "inner": left >> inner_join(right, by="k"),
        "full": left >> full_join(right, by="k"),
        "semi": left >> semi_join(right, by="k"),
        "anti": left >> anti_join(right, by="k"),
        "cross": tidy({"x": [1, 2]})
        >> cross_join(pd.DataFrame({"y": ["a", "b"]})),
    }


def _nest_join():
    left = tidy({"k": [1, 2], "x": ["a", "b"]})
    right = pd.DataFrame({"k": [1, 1, 2], "y": [10, 11, 20]})
    return left >> nest_join(right, by="k", name="matches") >> unnest("matches")


def _binds():
    rows = tidy({"x": [1], "y": ["a"]}) >> bind_rows(
        pd.DataFrame({"x": [2], "z": [3.5]})
    )
    cols = tidy({"x": [1, 2]}) >> bind_cols(
        pd.DataFrame({"y": ["a", "b"]})
    )
    return {"rows": rows, "cols": cols}


def _sets():
    left = tidy({"a": [1.0, 2.0, 2.0]})
    right = pd.DataFrame({"a": [2.0, 3.0]})
    return {
        "union": left >> tidy_union(right),
        "union_all": left >> union_all(right),
        "intersect": left >> intersect(right),
        "setdiff": left >> setdiff(right),
        "symdiff": left >> symdiff(right),
        "setequal": left >> setequal(pd.DataFrame({"a": [2.0, 1.0]})),
    }


def _rows():
    base = tidy(
        pd.DataFrame(
            {
                "id": [1, 2, 3],
                "value": [10.0, None, 30.0],
                "old": ["a", "b", "c"],
            }
        ),
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


def _missing_data():
    frame = tidy(
        pd.DataFrame(
            {
                "g": ["a", "a", "b"],
                "id": [1, 2, 1],
                "value": [None, 2.0, None],
            }
        ),
    )
    return {
        "dropped": frame >> drop_na("value"),
        "replaced": frame >> replace_na({"value": 0}),
        "filled": frame >> fill("value", direction="downup"),
        "expanded": frame >> select("g", "id") >> expand("g", "id"),
        "completed": frame >> complete("g", "id", fill={"value": 0}),
    }


def _pivots():
    wide = tidy(
        {"id": [1, 2], "a": [10, 20], "b": [30, 40]}
    )
    long = wide >> pivot_longer(
        ["a", "b"], names_to="name", values_to="value"
    )
    dup = tidy(
        {"id": [1, 1, 2], "key": ["a", "a", "b"], "val": [1.0, 2.0, 3.0]},
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


def _separate_unite():
    frame = tidy({"id": [1, 2], "code": ["a-10", "b-20"]})
    separated = frame >> separate("code", ["group", "number"], sep="-")
    return {
        "separated": separated,
        "united": separated >> unite("code", "group", "number", sep="-"),
    }


def _nest_roundtrip():
    frame = tidy({"g": ["a", "a", "b"], "value": [1, 2, 3]})
    return frame >> nest("data", cols=["value"]) >> unnest("data")


def _unnest_longer():
    frame = tidy(
        pd.DataFrame({"id": [1, 2], "values": [[10, 11], [20]]}),
    )
    return frame >> unnest_longer("values", indices_to="position")


def _unnest_wider():
    frame = tidy(
        pd.DataFrame(
            {
                "id": [1, 2],
                "values": [{"a": 10, "b": 20}, {"a": 30, "b": 40}],
            }
        ),
    )
    return frame >> unnest_wider("values")


def _empty_inputs():
    empty = tidy(
        pd.DataFrame({"x": pd.Series(dtype="float64")})
    )
    grouped = tidy(
        pd.DataFrame(
            {
                "g": pd.Series(dtype="object"),
                "x": pd.Series(dtype="float64"),
            }
        ),
    )
    left = tidy(
        pd.DataFrame(
            {"k": pd.Series(dtype="int64"), "x": pd.Series(dtype="object")}
        ),
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


def _categorical():
    frame = tidy(
        pd.DataFrame(
            {"g": pd.Categorical(["a"], categories=["a", "b"]), "x": [1]}
        ),
    )
    return (
        frame
        >> group_by("g", drop=False)
        >> summarise(n=n(), groups="drop")
    )


def _categorical_count():
    frame = tidy(
        pd.DataFrame(
            {"g": pd.Categorical(["a"], categories=["a", "b"]), "x": [1]}
        ),
    )
    return frame >> count("g", drop=False)


def _duplicate_names():
    return tidy({"x": [1]}) >> bind_cols(
        pd.DataFrame({"x": [2]})
    )


def _descending_rank():
    frame = tidy({"x": [3.0, 1.0, 2.0, None]})
    return frame >> mutate(rank=min_rank(tidy3.desc("x")))


def _sequential_summary():
    frame = tidy({"x": [1, 2, 3]})
    return frame >> summarise(a=mean("x"), b=col("a") * 2)


def _sequential_summary_nested():
    frame = tidy({"x": [1, 2, 3]})
    return frame >> summarise(
        a=tidy_sum("x"),
        b=tidy_sum(col("a")),
        c=tidy_sum(col("a") + col("x")),
    )


def _mutate_delete():
    return tidy({"x": [1, 2], "y": [3, 4]}) >> mutate(x=None)


def _computed_distinct():
    frame = tidy(
        {"x": [1, 3, 2], "y": [10, 11, 12]}
    )
    return frame >> distinct(parity=col("x") % 2)


def _n_distinct_multi():
    frame = tidy(
        {"x": [1, 1, 2], "y": ["a", "b", "b"]}
    )
    return frame >> summarise(n=n_distinct("x", "y"))


def _bind_cols_recycle():
    return tidy({"x": [1, 2, 3]}) >> bind_cols(
        pd.DataFrame({"z": [9]})
    )


def _pivot_value():
    frame = tidy(
        {"id": [1, 2], "x_a": [1, 2], "y_a": [3, 4]}
    )
    return frame >> pivot_longer(
        ["x_a", "y_a"], names_to=[".value", "set"], names_sep="_"
    )


def _pivot_wider_multi():
    frame = tidy(
        {
            "id": [1, 1],
            "name": ["a", "b"],
            "v1": [10, 20],
            "v2": [30, 40],
        },
    )
    return frame >> pivot_wider(
        names_from="name", values_from=["v1", "v2"]
    )


def _pivot_wider_multi_names():
    frame = tidy(
        {
            "id": [1, 1],
            "axis": ["x", "x"],
            "period": ["q1", "q2"],
            "value": [10, 20],
        },
    )
    return frame >> pivot_wider(
        names_from=["axis", "period"], values_from="value"
    )


def _separate_convert():
    frame = tidy({"code": ["a-10", "b-20"]})
    return frame >> separate(
        "code", ["group", "number"], sep="-", convert=True
    )


def _separate_convert_types():
    frame = tidy(
        {"code": ["TRUE-1.5", "FALSE-2.0"]}
    )
    return frame >> separate(
        "code", ["flag", "number"], sep="-", convert=True
    )


def _arrange_by_group():
    frame = tidy(
        {"g": ["b", "a", "b"], "x": [2.0, 3.0, 1.0]}
    )
    return frame >> group_by("g") >> arrange("x", by_group=True) >> ungroup()


def _type_coercion():
    rows = tidy({"x": [1]}) >> bind_rows(
        pd.DataFrame({"x": [2.5]})
    )
    mutated = tidy({"x": [1, 2]}) >> mutate(
        y=col("x") / 2,
        flag=col("x") > 1,
        label=tidy3.if_else(col("flag"), "yes", "no"),
    )
    return {"rows": rows, "mutated": mutated}


def _new_helpers():
    x = tidy(
        pd.DataFrame(
            {
                "code": pd.Series([1, 2, 3, None], dtype="Int64"),
                "text": ["a|b", "c", "d|e", None],
            }
        ),
    )
    labels = x >> mutate(
        bucket=case_match(
            "code", ((1, 2), "low"), (3, "high"), default="other"
        ),
        recoded=recode(
            "code", {1: "one", 2: "two"}, default="other", missing="missing"
        ),
    )
    longer = tidy({"text": ["a|b", "c"]}) >> separate_longer_delim(
        "text", "|"
    )
    wider = tidy({"text": ["a|b", "c"]}) >> separate_wider_delim(
        "text", ["left", "right"], "|", too_few="align_start"
    )
    edges = tidy({"id": [1, 2, 3], "text": ["a|b|c", "d", None], "z": [1, 2, 3]})
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


def _dplyr_12_helpers():
    x = tidy(
        {
            "state": ["NC", "NYC", "CA", None, "NYC", "Unknown"],
            "score": [1.0, 2.0, 3.0, 4.0, 5.0, None],
        },
    )
    b = tidy(
        {
            "x": [True, True, True, False, False, False, None, None, None],
            "y": [True, False, None, True, False, None, True, False, None],
        },
    )
    pets = tidy(
        {"type": ["dog", "dog", "cat", "dog", "cat"], "age": [1.0, 3.0, 5.0, 2.0, 4.0]},
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


def _group_info():
    import tidy3 as t3

    d = tidy(
        {"g": ["b", "a", "b", "a", "b"], "h": [1.0, 1.0, 2.0, 1.0, 2.0], "x": [1, 2, 3, 4, 5]},
    )
    gd = d >> group_by("g", "h")
    f = tidy(
        pd.DataFrame(
            {
                "k": pd.Categorical(["x", "x", None], categories=["x", "y"]),
                "g": ["b", "a", "b"],
            }
        ),
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


def _ordering():
    import tidy3 as t3

    d = tidy(
        {
            "g": ["a", "a", "b", "b", "a", "b", "a"],
            "t": [3.0, 1.0, 2.0, None, 2.0, 1.0, 1.0],
            "x": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0],
        },
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


def _grids():
    import tidy3 as t3

    df = tidy({"a": [2, 1, 2], "b": ["v", "u", "v"]})
    counts = tidy({"x": ["a", "b", "c"], "n": [1.0, 0.0, 2.0]})
    obs = tidy(
        {"country": ["A", "A", "B"], "year": [1952, 1962, 1957], "pop": [1.0, 2.0, 3.0]},
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


def _splitting():
    import tidy3 as t3

    d = tidy({"id": [1, 2, 3], "code": ["ab-12", "cd-3", None]})
    pattern = [("letters", "[a-z]+"), "-", ("number", "[0-9]+")]
    dates = tidy({"id": [1, 2, 3], "code": ["20240115", "2023", None]})
    chopped = tidy(
        {"g": ["a", "a", "b"], "x": [1, 2, 3], "y": [10.0, 20.0, 30.0]}
    ) >> t3.chop("x", "y")
    lists = tidy({"g": ["a", "b", "c"], "x": [[1, 2], [], None]})
    return {
        "regex": d >> t3.separate_wider_regex("code", pattern),
        "regex_partial": tidy({"code": ["ab-12", "ab"]})
        >> t3.separate_wider_regex("code", pattern, too_few="align_start"),
        "regex_prefixed": d >> t3.separate_wider_regex("code", pattern, names_sep="_"),
        "position": dates
        >> t3.separate_wider_position(
            "code", [("year", 4), ("month", 2), ("day", 2)], too_few="align_start"
        ),
        "position_skip": tidy({"code": ["20240115"]})
        >> t3.separate_wider_position("code", [("year", 4), 2, ("day", 2)]),
        "longer": tidy({"id": [1, 2, 3], "x": ["abcde", "fg", "h"]})
        >> t3.separate_longer_position("x", 2),
        "chopped": chopped,
        "unchopped": chopped >> t3.unchop("x", "y"),
        "unchop_drop": lists >> t3.unchop("x"),
        "unchop_keep": lists >> t3.unchop("x", keep_empty=True),
        "auto_wider": tidy(
            {"id": [1, 2], "v": [{"a": 1.0, "b": 2.0}, {"a": 3.0, "b": 4.0}]}
        )
        >> t3.unnest_auto("v"),
        "auto_longer": tidy({"id": [1, 2], "v": [[1.0, 2.0], [3.0]]})
        >> t3.unnest_auto("v"),
    }


def _empty_groups():
    d = tidy(
        pd.DataFrame(
            {
                "k": pd.Categorical(["x", "x", None], categories=["x", "y"]),
                "g": ["b", "a", "b"],
                "v": [1.0, 2.0, 3.0],
            }
        ),
    )
    stats = dict(n=n(), s=tidy_sum("v"), m=mean("v"), groups="drop")
    return {
        "kg": d >> group_by("k", "g", drop=False) >> summarise(**stats),
        "gk": d >> group_by("g", "k", drop=False) >> summarise(**stats),
        "k": d >> group_by("k", drop=False) >> summarise(n=n(), groups="drop"),
        "counted": d >> count("k", "g", drop=False),
        "sorted": d >> count("k", drop=False, sort=True),
    }


def _join_names():
    import tidy3 as t3

    a = tidy({"id": [1, 2, 3], "v": [1.0, 2.0, 3.0], "w": [5.0, 6.0, 7.0]})
    b = tidy({"id": [1, 2, 4], "v": [10.0, 20.0, 40.0]})
    clash = tidy({"id": [1, 2], "v": [1.0, 2.0], "v_x": [9.0, 9.0]})
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


def _pivot_options():
    d = tidy(
        {"id": [1.0, 1.0, 2.0, 2.0], "extra": ["p", "q", "r", "s"], "key": ["a", "b", "a", "b"],
         "sub": ["x", "y", "x", "x"], "v": [1.0, 2.0, 3.0, 4.0], "w": [5.0, 6.0, 7.0, 8.0]},
    )
    f = tidy(pd.DataFrame({"id": [1.0, 2.0], "key": pd.Categorical(["a", "a"], categories=["a", "b", "c"]), "v": [1.0, 2.0]}))
    g = tidy(pd.DataFrame({"id": pd.Categorical(["i1", "i1"], categories=["i1", "i2"]), "key": ["a", "b"], "v": [1.0, 2.0]}))
    l = tidy({"id": [1, 2], "week_1": [5, 6], "week_2": [7, 8]})
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


def _list_options():
    import tidy3 as t3

    d = tidy({"id": [1, 2], "code": ["a-1", "b-2"]})
    h = tidy(
        {"id": [1, 2], "info": [{"name": "x", "n": "1", "k": 5.0}, {"name": "y", "n": "2", "k": 6.0}]},
    )
    n = tidy({"g": [1.0, 1.0, 2.0], "x_a": [1, 2, 3], "x_b": [4, 5, 6]})
    return {
        "sep": d >> t3.separate_wider_delim("code", ["l", "n"], "-", names_sep="_"),
        "kept": d >> t3.separate_wider_delim("code", ["l", "n"], "-", cols_remove=False),
        "indexed": tidy({"id": [1, 2], "x": [[10.0, 11.0], [20.0]]})
        >> t3.unnest_longer("x", indices_include=True),
        "longer_int": tidy({"id": [1, 2], "x": [["1", "2"], ["3"]]})
        >> t3.unnest_longer("x", transform=int),
        "wider_int": tidy({"id": [1, 2], "x": [{"a": "1", "b": "2"}, {"a": "3", "b": "4"}]})
        >> t3.unnest_wider("x", transform={"a": int}),
        "hoisted": h >> t3.hoist("info", nm="name", n="n", transform={"n": int}) >> t3.select(~t3.all_of(["info"])),
        "hoisted_all": h >> t3.hoist("info", name="name", n="n", k="k"),
        "nested": n >> t3.nest("x", cols=["x_a", "x_b"], names_sep="_") >> t3.unnest("x"),
    }


def _specs():
    import tidy3 as t3

    wide = tidy(
        {"id": [1, 2], "mean_1": [5.0, 6.0], "mean_2": [7.0, 8.0], "sd_1": [0.1, 0.2], "sd_2": [0.3, 0.4]},
    )
    long = tidy({"id": [1.0, 1.0, 2.0, 2.0], "key": ["a", "b", "a", "b"], "v": [1, 2, 3, 4], "w": [5, 6, 7, 8]})
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


def _helpers():
    import tidy3 as t3

    rounded = (
        tidy({"x": [0.5, 1.5, -1.5, 1.25]})
        >> mutate(z=t3.round("x"))
    ).collect()["z"]
    rounded_1 = (
        tidy({"x": [1.25, 1.35]})
        >> mutate(z=t3.round("x", 1))
    ).collect()["z"]
    rounded_tens = (
        tidy({"x": [123.0, -15.0]})
        >> mutate(z=t3.round("x", -1))
    ).collect()["z"]
    pairs = tidy(
        {"a": [1.0, None, 5.0], "b": [2.0, 3.0, None], "c": [0.0, 4.0, 4.0]},
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


def _lubridate_dates():
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


def _readr_parse():
    import tidy3 as t3

    number = (
        tidy({"x": ["$1,234.50", "abc"]})
        >> mutate(number=t3.parse_number("x"))
    ).collect()["number"]
    plain = (
        tidy({"x": ["1.5", "x"]})
        >> mutate(plain=t3.parse_double("x"))
    ).collect()["plain"]
    integer = (
        tidy({"x": ["1234", "2.5"]})
        >> mutate(integer=t3.parse_integer("x"))
    ).collect()["integer"]
    logical = (
        tidy({"x": ["true", "1", "0", "yes"]})
        >> mutate(logical=t3.parse_logical("x"))
    ).collect()["logical"]
    return {"number": number, "plain": plain, "integer": integer, "logical": logical}


def _column(values: list[Any], expr: Any) -> pl.Series:
    """One expression over one column ``x``, as a vector."""
    return (tidy({"x": values}) >> mutate(out=expr)).collect()["out"]


def _helper_edges():
    import tidy3 as t3

    v = [1.0, None, float("nan"), float("inf")]
    half = [-2.5, 2.5, None]
    return {
        "is_na": _column(v, t3.is_na("x")),
        "is_finite": _column(v, t3.is_finite("x")),
        "is_infinite": _column(v, t3.is_infinite("x")),
        "abs": _column(half, t3.abs("x")),
        "floor": _column(half, t3.floor("x")),
        "ceiling": _column(half, t3.ceiling("x")),
        "round_2": _column([2.675, 0.125, 1.005], t3.round("x", 2)),
        "round_1": _column([0.15, 0.25, 0.35], t3.round("x", 1)),
        "round_tens": _column([125.0, 135.0, -15.0], t3.round("x", -1)),
        "pmin_text": (
            tidy({"a": ["nan", "b"], "b": ["c", "a"]})
            >> mutate(out=t3.pmin("a", "b"))
        ).collect()["out"],
        "pmin_nan": (
            tidy({"a": [1.0, float("nan")], "b": [2.0, 3.0]})
            >> mutate(out=t3.pmin("a", "b"))
        ).collect()["out"],
    }


def _is_nan():
    import tidy3 as t3

    return _column([1.0, None, float("nan"), float("inf")], t3.is_nan("x"))


def _conversions():
    import tidy3 as t3
    from datetime import date

    return {
        "as_double": _column(
            ["1.5", " 2 ", "1e5", "-.5", "NaN", "abc", "1,234"],
            t3.as_double("x", strict=False),
        ),
        "as_integer": _column(["2", " 7 ", "-3", "007"], t3.as_integer("x")),
        "as_integer_num": _column([2.0, -3.0, None], t3.as_integer("x")),
        "as_logical_text": _column(
            ["TRUE", "true", "T", "True", "FALSE", "false", "F", "tRuE", "yes", "1"],
            t3.as_logical("x", strict=False),
        ),
        "as_logical_num": _column(
            [0.0, 2.0, -1.0, float("nan"), None], t3.as_logical("x")
        ),
        "as_character_int": _column([1, 123456, -3, 100001], t3.as_character("x")),
        "as_character": _column([1.5, 0.25, -3.75, 22.8], t3.as_character("x")),
        "as_logical_exact": _column(
            [" TRUE", "TRUE ", "yes", "True", "T", "false", "F"],
            t3.as_logical("x", strict=False),
        ),
        "as_character_logical": _column(
            [True, False, None], t3.as_character("x", true="TRUE", false="FALSE")
        ),
        "as_date": _column(
            ["2010-01-31", None],
            t3.as_character(t3.as_date("x")),
        ),
    }


def _lubridate_more():
    import tidy3 as t3
    from datetime import date, datetime

    def days(values, expr):
        return _column(values, t3.as_character(expr))

    d = [date(2024, 2, 29), date(2024, 1, 31), date(2010, 3, 31)]
    t = [datetime(2026, 10, 21, 15, 30), datetime(2026, 10, 18, 9, 0), datetime(2026, 10, 18)]
    dated = tidy({"d": d, "k": [1, None, -1]})
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


def _readr_more():
    import tidy3 as t3

    euro = t3.locale(decimal_mark=",", grouping_mark=".")
    kept = (
        tidy({"x": ["1", "a", "2", "b"]})
        >> filter(col("x") != "1")
        >> mutate(n=t3.parse_double("x"))
    )
    found = t3.problems(kept).collect()
    return {
        "number": _column(
            ["-$1,234", "$-5", "USD -7.5", "Price - 12", "50%", "abc"],
            t3.parse_number("x"),
        ),
        "double": _column(["1.5", " 1.5 ", "1e3", "abc"], t3.parse_double("x")),
        "double_grouped": _column(["1,234.5"], t3.parse_double("x"))[0],
        "double_euro": _column(["1,5", "-0,25"], t3.parse_double("x", locale=euro)),
        "integer": _column(["1234", " 7 ", "2.5", "x"], t3.parse_integer("x")),
        "logical": _column(
            ["TRUE", "T", "true", "t", "1", "FALSE", "f", "0", "yes"],
            t3.parse_logical("x"),
        ),
        "date": _column(
            ["31/01/2010", "bad"],
            t3.as_character(t3.parse_date("x", "%d/%m/%Y")),
        ),
        # readr reads %y with a fixed cutoff of 68; tidy3 asks for it.
        "date_cutoff": _column(
            ["68/01/02", "69/01/02"],
            t3.as_character(t3.parse_date("x", "%y/%m/%d", cutoff=68)),
        ),
        "problem_rows": found["row"],
        "problem_values": found["original"],
    }


def _readr_files():
    import tempfile

    import tidy3 as t3

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


def _reshape_metadata():
    source = pd.DataFrame({
        "g": pd.Categorical(["b", "a", "b"], categories=["a", "b", "unused"], ordered=True),
        "code": ["x-y", "p-q", "r-s"], "x": [2, 4, 8],
        "record": [{"nested_value": 10}, {"nested_value": 20}, {"nested_value": 30}],
    })
    frame = tidy(source) >> group_by("g", drop=False)
    operations = {
        "pack": tidy3.pack("packed", "x"), "unpack": tidy3.unpack("record"),
        "hoist": tidy3.hoist("record", nested_value="nested_value"),
        "wider_delim": tidy3.separate_wider_delim("code", names=["left", "right"], delim="-"),
        "wider_regex": tidy3.separate_wider_regex("code", [("left", "[a-z]"), "-", ("right", "[a-z]")]),
        "wider_position": tidy3.separate_wider_position("code", [("left", 1), 1, ("right", 1)]),
        "longer_delim": tidy3.separate_longer_delim("code", "-"),
        "longer_position": tidy3.separate_longer_position("code", 2),
    }
    result = {}
    for name, operation in operations.items():
        for empty in (False, True):
            out = (frame >> head(0) if empty else frame) >> operation
            result[f"{name}_{'empty' if empty else 'rows'}"] = {
                "groups": out._groups[0], "drop": out._group_drop,
                "levels": pl.Series(out._category_levels["g"]),
                "summary": (out >> summarise(rows=n(), groups="drop")
                            >> mutate(g=tidy3.as_character("g"))),
            }
    return result


def _quiet_same(*frames, **kwargs):
    """``compare_df_cols_same`` of one frame warns and returns False."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return tidy3.compare_df_cols_same(*frames, verbose=False, **kwargs)


def _excel_text(values, **kwargs):
    from datetime import datetime

    converted = tidy3.excel_numeric_to_date(list(values), **kwargs)
    text = []
    for value in converted:
        if value is None:
            text.append(None)
        elif isinstance(value, datetime):
            text.append(value.strftime("%Y-%m-%dT%H:%M:%S.%f"))
        else:
            text.append(value.isoformat())
    return tidy(
        pd.DataFrame({"value": pd.Series(text, dtype="string")})
    )


def _janitor_totals():
    import tidy3 as t3

    source = tidy({"label": ["a", "b"], "yes": [1, None], "no": [2, 3], "note": ["x", "y"]})
    result = {"row": source >> t3.adorn_totals(),
              "col": source >> t3.adorn_totals("col"),
              "both": source >> t3.adorn_totals("both"),
              "propagate": source >> t3.adorn_totals("both", na_rm=False),
              "custom": source >> t3.adorn_totals("both", cols="yes", name=("All", "Grand")),
              "typed_fill": source >> t3.adorn_totals(cols="yes", fill=None),
              "row_then_col": source >> t3.adorn_totals() >> t3.adorn_totals("col"),
              "col_then_row": source >> t3.adorn_totals("col") >> t3.adorn_totals()}
    factor = tidy(pd.DataFrame({
        "g": pd.Categorical(["b", "a"], categories=["b", "a", "unused"], ordered=True),
        "n": [1, 2],
    })) >> t3.adorn_totals()
    result["factor"] = factor
    result["factor_levels"] = pl.Series(factor.factor_levels()["g"])
    result["factor_ordered"] = factor.collect(as_="pandas")["g"].cat.ordered
    numeric = tidy({"g": [1, 2], "n": [3, 4]})
    result["numeric_label"] = numeric >> t3.adorn_totals()
    result["numeric_first_selected"] = numeric >> t3.adorn_totals("both", cols=t3.everything())
    empty = tidy(pl.DataFrame(schema={"label": pl.String, "n": pl.Int64}))
    result["empty"] = empty >> t3.adorn_totals("both")
    missing = tidy(pl.DataFrame({"label": ["a", "b"], "n": pl.Series([None, float("nan")], dtype=pl.Float64)}))
    result["missing"] = missing >> t3.adorn_totals("both")
    report = tidy({"g": ["a", "a", "b"], "answer": ["yes", "no", "yes"]}) >> t3.tabyl("g", "answer")
    result["tabyl"] = report >> t3.adorn_totals("both")
    return result


def _janitor_percentages():
    import tidy3 as t3

    source = tidy({"label": ["a", "b", "c"], "yes": [1, None, 0], "no": [2, 3, 0],
                   "w": [0.5, float("nan"), 1.0], "note": ["x", "y", "z"]})
    strict = tidy({"label": ["a", "b", "c"], "yes": [1, None, 0], "no": [2, 3, 0]})
    totals = strict >> t3.adorn_totals("both")
    result = {}
    for denominator in ("row", "col", "all"):
        percentages = lambda **kw: t3.adorn_percentages(denominator, **kw)
        result[denominator] = source >> percentages()
        result[f"{denominator}_keep_na"] = strict >> percentages(na_rm=False)
        result[f"{denominator}_totals"] = totals >> percentages()
        result[f"{denominator}_row_totals"] = strict >> t3.adorn_totals() >> percentages()
        result[f"{denominator}_col_totals"] = strict >> t3.adorn_totals("col") >> percentages()
        result[f"{denominator}_selected"] = totals >> percentages(cols="yes")
        result[f"{denominator}_selected_totals"] = totals >> percentages(cols=["no", "Total"])
    result["first_column"] = tidy({"g": [1, 2], "n": [3, 1]}) >> t3.adorn_percentages("col", cols=t3.everything())
    result["text_selected"] = source >> t3.adorn_percentages(cols=["yes", "note"])
    result["empty"] = tidy(pl.DataFrame(schema={"label": pl.String, "n": pl.Int64})) >> t3.adorn_percentages("col")
    report = tidy({"g": ["a", "a", "b", "b"], "answer": ["yes", "no", "yes", "yes"]}) >> t3.tabyl("g", "answer")
    result["tabyl"] = report >> t3.adorn_totals("both") >> t3.adorn_percentages("all")
    factor = tidy(pd.DataFrame({
        "g": pd.Categorical(["b", "a"], categories=["b", "a", "unused"], ordered=True),
        "n": [1, 3],
    }))
    result["factor"] = factor >> t3.adorn_totals() >> t3.adorn_percentages("col")
    # JSON keeps about 16 digits, so recurring shares are compared at 12.
    rounded = pl.col(pl.Float64).map_elements(lambda v: float(f"{v:.12g}"), return_dtype=pl.Float64)
    return {name: frame._with_lf(frame._lf.with_columns(rounded), totals=frame._totals)
            for name, frame in result.items()}


def _janitor_formatting():
    import tidy3 as t3

    shares = tidy({"label": ["a", "b", "c", "d"], "x": [0.125, None, 0.00125, -0.00004],
                   "y": [1 / 3, float("nan"), 2 / 3, float("inf")], "n": [1, 2, 3, 4],
                   "note": ["p", "q", "r", "s"]})
    result = {}
    for digits in (0, 1, 2):
        for rounding in ("half to even", "half up"):
            name = f"digits_{digits}_{rounding.replace(' ', '_')}"
            result[name] = shares >> t3.adorn_pct_formatting(digits, rounding)
    result["no_sign"] = shares >> t3.adorn_pct_formatting(3, affix_sign=False)
    # JSON has no Inf, so the unformatted y leaves out the fourth row.
    result["selected"] = shares >> t3.slice_head(n=3) >> t3.adorn_pct_formatting(cols=["x", "note"])
    result["first_column"] = tidy({"p": [0.5, 0.25], "q": [1., 0.]}) >> t3.adorn_pct_formatting(cols=t3.everything())
    one_way = tidy({"g": ["a", "b", None, "a"]}) >> t3.tabyl("g")
    result["one_way"] = one_way >> t3.adorn_pct_formatting()
    result["one_way_totals"] = one_way >> t3.adorn_totals() >> t3.adorn_pct_formatting(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result["one_way_ns"] = one_way >> t3.adorn_ns()
    counts = tidy({"g": ["b", "a", "b", "a", "b", "c", None],
                   "answer": ["yes", "no", "yes", "yes", "no", "no", "yes"],
                   "z": [2., 10., 2., None, 10., 2., 2.]})
    report = counts >> t3.tabyl("g", "answer")
    totals = report >> t3.adorn_totals("both")
    formatted = totals >> t3.adorn_percentages("col") >> t3.adorn_pct_formatting()
    result["report"] = formatted
    result["rear"] = formatted >> t3.adorn_ns()
    result["front"] = formatted >> t3.adorn_ns("front")
    result["counts_rear"] = report >> t3.adorn_ns()
    result["counts_front"] = totals >> t3.adorn_ns("front")
    result["numbers_rear"] = report >> t3.adorn_percentages() >> t3.adorn_ns()
    result["numbers_front"] = report >> t3.adorn_percentages() >> t3.adorn_ns("front")
    result["row_totals"] = (report >> t3.adorn_totals() >> t3.adorn_percentages()
                            >> t3.adorn_pct_formatting() >> t3.adorn_ns())
    named = report >> t3.adorn_totals("both", name=("All", "Sum"))
    result["named_totals"] = (named >> t3.adorn_percentages("all")
                              >> t3.adorn_pct_formatting(2, "half up") >> t3.adorn_ns("front"))
    result["selected_ns"] = formatted >> t3.adorn_ns(cols="yes")
    custom = tidy({"g": ["a", "b"], "n": [1 / 3, 12345678.]})
    result["custom"] = tidy({"g": ["a", "b"], "n": [1234., 5.]}) >> t3.adorn_ns(ns=custom)
    result["custom_front"] = tidy({"g": ["a", "b"], "n": ["x", "y"]}) >> t3.adorn_ns("front", ns=custom)
    big = tidy({"g": ["a"] * 1500 + ["b"] * 2, "h": ["x"] * 1502}) >> t3.tabyl("g", "h")
    result["big_mark"] = big >> t3.adorn_percentages("col") >> t3.adorn_pct_formatting() >> t3.adorn_ns()
    result["format_func"] = formatted >> t3.adorn_ns(format_func=lambda n: f"{n:g}n")
    result["three_way"] = counts >> t3.tabyl("g", "answer", "z")
    result["three_way_no_na"] = counts >> t3.tabyl("g", "answer", "z", show_na=False)
    result["three_way_present"] = counts >> t3.tabyl("g", "answer", "z", show_missing_levels=False)
    three = counts >> t3.tabyl("g", "answer", "z") >> t3.adorn_totals("both")
    result["three_way_report"] = three >> t3.adorn_percentages("row") >> t3.adorn_pct_formatting(2) >> t3.adorn_ns()
    result["three_way_front"] = (three >> t3.adorn_percentages("all") >> t3.adorn_pct_formatting()
                                 >> t3.adorn_ns("front"))
    levels = tidy(pd.DataFrame({
        "g": pd.Categorical(["b", "a", "b"], categories=["b", "a", "unused"]),
        "h": ["x", "y", "x"],
        "z": pd.Categorical(["hi", "lo", "lo"], categories=["lo", "hi"]),
        "w": ["", "k", ""],
    }))
    result["three_way_factor"] = levels >> t3.tabyl("g", "h", "z")
    text = levels >> t3.tabyl("h", "g", "w")
    result["three_way_text"] = {key or "<empty>": table for key, table in text.items()}
    return result


def _janitor_tabyl():
    import tidy3 as t3

    source = tidy({"x": ["b", "a", "a", None]})
    factor = tidy(pd.DataFrame({"x": pd.Categorical(
        ["b", None, "b"], categories=["c", "b", "a"], ordered=True,
    )}))
    result = {"ordinary": source >> t3.tabyl("x"),
              "without_na": source >> t3.tabyl("x", show_na=False)}
    for show_na in (False, True):
        for missing in (False, True):
            result[f"factor_{int(show_na)}_{int(missing)}"] = factor >> t3.tabyl(
                "x", show_na=show_na, show_missing_levels=missing,
            )
    result["empty"] = t3.tabyl(pd.Series([], dtype="string", name="x"))
    result["empty_factor"] = tidy(pd.DataFrame({
        "x": pd.Categorical([], categories=["b", "a"]),
    })) >> t3.tabyl("x")
    result["all_missing"] = tidy(pl.DataFrame({"x": pl.Series([None, None], dtype=pl.String)})) >> t3.tabyl("x")
    result["nan_null"] = tidy(pl.DataFrame({"x": [None, float("nan"), 1., float("nan")]})) >> t3.tabyl("x")
    two_way = tidy({"x": ["b", "a", "a", None, "b"], "y": ["", "z", None, "m", "z"]})
    result["two_way"] = two_way >> t3.tabyl("x", "y")
    result["two_way_no_na"] = two_way >> t3.tabyl("x", "y", show_na=False)
    factors = tidy(pd.DataFrame({
        "x": pd.Categorical(["b", None, "b"], categories=["c", "b", "a"], ordered=True),
        "y": pd.Categorical(["z", "m", None], categories=["q", "z", "m"]),
    }))
    for show_na in (False, True):
        for missing in (False, True):
            result[f"two_way_factor_{int(show_na)}_{int(missing)}"] = factors >> t3.tabyl(
                "x", "y", show_na=show_na, show_missing_levels=missing,
            )
    result["two_way_empty"] = tidy(pd.DataFrame({
        "x": pd.Categorical([], categories=["b", "a"]),
        "y": pd.Categorical([], categories=["z", "m"]),
    })) >> t3.tabyl("x", "y")
    result["two_way_empty_after_drop"] = tidy({"x": ["a", None], "y": [None, "b"]}) >> t3.tabyl("x", "y", show_na=False)
    for label, values in (("numeric", [10, 2, 1]), ("boolean", [True, False, True]),
                          ("nan", [1., float("nan"), None])):
        result[f"two_way_{label}"] = tidy({"x": ["b", "a", "b"], "y": values}) >> t3.tabyl("x", "y")
    return result


def _janitor_cleaning():
    cleaning = tidy(
        pd.DataFrame(
            {
                "a": pd.Series([1.0, None, 2.0, None], dtype="float64"),
                "b": pd.Series([None, None, None, None], dtype="float64"),
                "c": pd.Series(["x", None, "", "y"], dtype="string"),
                "d": pd.Series([None, None, 1.0, None], dtype="float64"),
            }
        ),
    )
    constant = tidy(
        pd.DataFrame(
            {
                "mix": pd.Series([1.0, 1.0, None], dtype="float64"),
                "vary": pd.Series([1.0, 2.0, 1.0], dtype="float64"),
                "const": pd.Series([5.0, 5.0, 5.0], dtype="float64"),
                "allna": pd.Series([None, None, None], dtype="float64"),
            }
        ),
    )
    zero = tidy(
        pd.DataFrame(
            {
                "a": pd.Series([], dtype="float64"),
                "b": pd.Series([], dtype="string"),
            }
        ),
    )
    old = tidy(
        pd.DataFrame(
            {
                "Z": pd.Series([1.5], dtype="float64"),
                "A": pd.Series([2.5], dtype="float64"),
                "B": pd.Series([1], dtype="int64"),
            }
        ),
    )
    new = tidy(
        pd.DataFrame(
            {
                "M": pd.Series([3.5], dtype="float64"),
                "A": pd.Series([4.5], dtype="float64"),
                "B": pd.Series([5.0], dtype="float64"),
                "C": pd.Series(["x"], dtype="string"),
            }
        ),
    )
    one = tidy(
        pd.DataFrame(
            {
                "Z": pd.Series([1.5], dtype="float64"),
                "A": pd.Series([1], dtype="int64"),
            }
        ),
    )
    fac = tidy(
        pd.DataFrame(
            {"f": pd.Categorical(["y", "x"], categories=["y", "x"], ordered=True)}
        ),
    )
    headers = tidy(
        pd.DataFrame(
            {
                "a": pd.Series(["id", "1"], dtype="string"),
                "b": pd.Series(["code", "x"], dtype="string"),
            }
        ),
    )
    numbers = tidy(
        pd.DataFrame({"a": pd.Series([1.0, 9.0], dtype="float64")})
    )
    flags = tidy(pd.DataFrame({"a": [True, False]}))
    sheet = tidy(
        pd.DataFrame(
            {
                "a": pd.Series([None, "h1", "1"], dtype="string"),
                "b": pd.Series(["a", "h2", "2"], dtype="string"),
            }
        ),
    )
    seek = tidy(
        pd.DataFrame(
            {
                "a": pd.Series(["skip", "header", "data"], dtype="string"),
                "b": pd.Series(["no", "yes", "row"], dtype="string"),
            }
        ),
    )
    return {
        "empty": cleaning >> tidy3.remove_empty(["rows", "cols"]),
        "rows_cutoff": cleaning >> tidy3.remove_empty("rows", cutoff=0.5),
        "cols_cutoff": cleaning >> tidy3.remove_empty("cols", cutoff=0.5),
        "constant": constant >> tidy3.remove_constant(),
        "constant_na": constant >> tidy3.remove_constant(na_rm=True),
        "zero_cols": zero >> tidy3.remove_empty("cols"),
        "zero_constant": zero >> tidy3.remove_constant(),
        "classes": tidy3.compare_df_cols(old=old, new=new),
        "matched": tidy3.compare_df_cols(old=old, new=new, return_="match"),
        "mismatched": tidy3.compare_df_cols(old=old, new=new, return_="mismatch"),
        "bound": tidy3.compare_df_cols(
            old=old, new=new, return_="mismatch", bind_method="rbind"
        ),
        "one_classes": tidy3.compare_df_cols(one=one),
        "strict": tidy3.compare_df_cols(fac=fac, strict_description=True),
        "loose": tidy3.compare_df_cols(fac=fac, strict_description=False),
        "same_bind": tidy3.compare_df_cols_same(
            tidy({"A": [1.0]}),
            tidy({"B": [2.0]}),
            verbose=False,
        ),
        "same_rbind": tidy3.compare_df_cols_same(
            tidy({"A": [1.0]}),
            tidy({"B": [2.0]}),
            bind_method="rbind",
            verbose=False,
        ),
        "same_one": _quiet_same(one=one),
        "headers": headers >> tidy3.row_to_names(1),
        "numbers": numbers >> tidy3.row_to_names(1),
        "flags": flags >> tidy3.row_to_names(1),
        "found": sheet >> tidy3.row_to_names("find_header"),
        "complete": tidy3.find_header(seek),
        "header_value": tidy3.find_header(seek, value="header"),
        "by_name": tidy3.find_header(seek, value="yes", column="b"),
        "by_pos": tidy3.find_header(seek, value="yes", column=2),
        "excel_modern": _excel_text(
            [-1, 0, 1, 59, 60, 61, 42370, 40000, 40000.5]
        ),
        "excel_mac": _excel_text(
            [0, 1, 60, 40908], date_system="mac pre-2011"
        ),
        "excel_time": _excel_text(
            [40000.5, 40000.521, 60.5, 1.999999999],
            include_time=True,
        ),
        "excel_noround": _excel_text(
            [40000.521], include_time=True, round_seconds=False
        ),
    }


def _janitor_dupes():
    x = tidy({"g": ["x", "x", "y", "z", "y", "x", "a", "b"],
              "key_id": ["b", "a", "b", "c", "a", "a", None, None],
              "key_sub": [1, 2, 1, 3, 2, 9, 4, 4], "row": list(range(1, 9))})
    full = tidy({"a": ["b", "a", "b", "a", "x"], "b": [1, 2, 1, 2, 3]})
    grouped = x >> group_by("g") >> tidy3.get_dupes("key_id")
    factors = tidy(pd.DataFrame({"key": pd.Categorical(["a", "b", "a", "b", None, None], categories=["b", "a", "unused"]), "row": range(6)}))
    return {
        "single": x >> tidy3.get_dupes("key_id"),
        "composite": x >> tidy3.get_dupes("key_id", "key_sub"),
        "selector": x >> tidy3.get_dupes(tidy3.starts_with("key")),
        "all": full >> tidy3.get_dupes(),
        "empty": x >> head(0) >> tidy3.get_dupes("key_id"),
        "none": x >> tidy3.get_dupes("row"),
        "empty_selection": x >> tidy3.get_dupes(tidy3.any_of([])),
        "grouped": grouped, "group_names": grouped._groups[0],
        "factors": factors >> tidy3.get_dupes("key"),
    }


def _timezone_payload(frame):
    out = frame.collect()["at"]
    return {
        "epochs": pl.Series([v.timestamp() if v is not None else None for v in out], dtype=pl.Float64),
        "clocks": out.dt.strftime("%Y-%m-%d %H:%M:%S %z"),
    }


def _timezone_conversion():
    from datetime import date

    frame = tidy({"text": ["2024-01-02 12:34:56.125", "2024-07-02 12:34:56.5", "2024-11-03 05:30:00", "2024-11-03 06:30:00", None]}) >> mutate(x=tidy3.ymd_hms("text"))
    zones = ["UTC", "America/New_York", "Europe/Dublin", "Australia/Sydney", "Asia/Kathmandu"]
    result = {}
    for source in zones:
        local = frame >> mutate(x=tidy3.with_tz("x", source))
        for target in zones:
            for name in ["with_tz", "force_tz"]:
                result[f"{source}_{target}_{name}"] = _timezone_payload(local >> mutate(at=getattr(tidy3, name)("x", target)))
    dates = tidy({"x": [date(2024, 1, 2), date(2024, 3, 10), date(2024, 11, 3), None]})
    for target in zones:
        for name in ["with_tz", "force_tz"]:
            result[f"dates_{target}_{name}"] = _timezone_payload(dates >> mutate(at=getattr(tidy3, name)("x", target)))
    return result


def _timezone_dst():
    inputs = {
        "America/New_York": ["2024-03-10 01:59:59.125", "2024-03-10 02:30:00.125", "2024-03-10 03:00:00", "2024-11-03 01:30:00.125", "2024-11-03 02:00:00", None],
        "Europe/Dublin": ["2024-03-31 01:30:00.125", "2024-10-27 01:30:00.125", "2024-01-01 12:34:56", None],
        "Australia/Lord_Howe": ["2024-04-07 01:45:00.125", "2024-10-06 02:15:00.125", "2024-01-01 12:34:56", None],
        "Pacific/Apia": ["2011-12-29 12:30:00.125", "2011-12-30 12:30:00.125", "2011-12-31 12:30:00.125", None],
    }
    policies = {"NA_both": "NA", "pre": "pre", "post": "post", "boundary": "boundary", "default": ("NA", "post"), "mixed": ("boundary", "pre")}
    result = {}
    for target, values in inputs.items():
        frame = tidy({"text": values}) >> mutate(x=tidy3.ymd_hms("text"))
        for key, policy in policies.items():
            source = frame >> mutate(x=tidy3.floor_date("x", "second")) if key == "mixed" else frame
            result[f"{target}_{key}"] = _timezone_payload(source >> mutate(at=tidy3.force_tz("x", target, roll_dst=policy)))
    return result


def _timezone_resolved_repeats():
    frame = tidy({"text": ["2024-11-03 04:30:00.125", "2024-11-03 05:30:00.125", "2024-11-03 06:30:00.125", "2024-10-27 00:30:00.125", "2024-10-27 01:30:00.125", None]}) >> mutate(x=tidy3.ymd_hms("text"))
    result = {}
    for source in ["UTC", "America/Toronto", "America/Halifax", "Europe/Dublin"]:
        local = frame >> mutate(x=tidy3.with_tz("x", source))
        for target in ["UTC", "America/New_York", "America/Toronto", "Europe/Dublin"]:
            for policy in ["NA", "pre", "post", "boundary"]:
                values = (frame >> mutate(x=tidy3.floor_date("x", "second"))
                          >> mutate(x=tidy3.with_tz("x", source))) if policy == "boundary" else local
                result[f"{source}_{target}_{policy}"] = _timezone_payload(values >> mutate(at=tidy3.force_tz("x", target, roll_dst=policy)))
    return result


def _formatted_offsets():
    values = ["2024-01-01 12:00:00+0200", "2024-01-01 12:00:00-0500",
              "2024-01-01 12:00:00+02:00", "2024-01-01 12:00:00Z",
              "2024-06-01 00:30:00-0530", "bad", None]
    out = (tidy({"x": values})
           >> mutate(at=tidy3.parse_datetime("x", "%Y-%m-%d %H:%M:%S%z")))
    return {"clocks": out.collect()["at"].dt.strftime("%Y-%m-%d %H:%M:%S")}


def _calendar_rounding():
    values = ["2024-03-11 12:00:00", "2024-03-10 01:30:00", "2024-11-03 01:30:00.125",
              "2024-11-03T01:30:00.125-04:00", "2024-11-03T01:00:00-04:00", None]
    frame = tidy({"x": values}) >> mutate(x=tidy3.ymd_hms("x", tz="America/New_York"))
    result = {}
    for unit in ("second", "minute", "day", "week", "month", "quarter", "year"):
        for name in ("floor_date", "ceiling_date"):
            result[f"{unit}_{name}"] = _timezone_payload(frame >> mutate(at=getattr(tidy3, name)("x", unit)))
    result["week_monday"] = _timezone_payload(frame >> mutate(at=tidy3.floor_date("x", "week", week_start=1)))
    return result


def _date_helpers():
    from datetime import date, timedelta

    zone = "America/New_York"

    def epoch_days(frame, expr):
        return (frame >> mutate(at=expr)).collect()["at"].cast(pl.Int32)

    d = tidy({"x": [date(2010, 2, 1), date(2010, 2, 14), date(2010, 2, 15), date(2010, 2, 16), date(2024, 2, 29),
                    date(2023, 12, 31), date(2024, 1, 3), date(2024, 8, 16), None]})
    x = tidy({"x": ["2024-01-03 12:00:00", "2024-02-15 12:00:00", "2024-07-02 00:00:00.5",
                    "2024-07-02 23:59:59.4", "2024-03-10 12:00:00", "2024-03-10 12:30:00",
                    "2024-11-03 12:00:00", "2024-11-03 11:30:00", "2024-03-10 01:30:00",
                    "2024-11-03T01:30:00-05:00", "2024-08-16 06:00:00", None]}) >> mutate(x=tidy3.ymd_hms("x", tz=zone))
    result = {}
    for unit in ("day", "week", "month", "quarter", "year"):
        result[f"date_{unit}"] = epoch_days(d, tidy3.round_date("x", unit))
    result["date_week_monday"] = epoch_days(d, tidy3.round_date("x", "week", week_start=1))
    for unit in ("second", "minute", "hour", "day", "week", "month", "quarter", "year"):
        result[f"time_{unit}"] = _timezone_payload(x >> mutate(at=tidy3.round_date("x", unit)))
    result["time_week_monday"] = _timezone_payload(x >> mutate(at=tidy3.round_date("x", "week", week_start=1)))
    parts = tidy(pl.DataFrame({
        "year": [2023, 2023, 2023, 2023, 2023, 2023.7, -5, 0, None, 2024, 10000, -2023.5, 2024],
        "month": [2, 13, 0, 1, 1, 1.9, 1, 1, 1, 2, 1, 1, None],
        "day": [29, 1, 1, 0, 32, 1.5, 1, 1, 1, 29, 1, 1, 1],
    }, strict=False))
    result["make_date"] = epoch_days(parts, tidy3.make_date("year", "month", "day"))
    result["make_date_defaults"] = epoch_days(tidy({"y": [2020, 2021]}), tidy3.make_date("y"))
    clock = tidy(pl.DataFrame({
        "year": [2023, 2023, 2023, 2023, 2023, 2023, 2023, 2023, 2023, 2023, 2023, None, 2024],
        "month": [2, 1, 1, 1, 13, 0, -1, 1, 1, 1, 1, 1, 2],
        "day": [31, 1, 1, 1, 1, 1, 1, -5, 1, 1, 1, 1, 29],
        "hour": [0, 25, 0, 23, 0, 0, 0, 0, 1000, 0, 0, 0, 12],
        "min": [0, 0, -61, 59, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "sec": [0, 0, 0, 60, 0, 0, 0, 0, 0, -0.5, 1.25, 0, 30],
    }, strict=False))
    result["make_datetime"] = _timezone_payload(clock >> mutate(at=tidy3.make_datetime("year", "month", "day", "hour", "min", "sec")))
    dst = tidy({"day": [10, 3, 10, 10, 9, 3, 3], "hour": [2, 1, 1, 26, 26, 0, 1],
                "min": [30, 30, 90, 0, 30, 90, 0], "month": [3, 11, 3, 3, 3, 11, 11]})
    result["make_datetime_new_york"] = _timezone_payload(dst >> mutate(at=tidy3.make_datetime(2024, "month", "day", "hour", "min", tz=zone)))
    kolkata = tidy({"day": [1, 2]}) >> mutate(at=tidy3.make_datetime(2023, 1, "day", 0, 0, 1, tz="Asia/Kolkata"))
    result["make_datetime_kolkata"] = _timezone_payload(kolkata)
    span = tidy({"x": [date(2019, 12, 25) + timedelta(days=n) for n in range(1850)] + [None]})
    result["isoyear"] = (span >> mutate(at=tidy3.isoyear("x"))).collect()["at"]
    result["isoyear_time"] = (x >> mutate(at=tidy3.isoyear("x"))).collect()["at"]
    years = tidy({"x": [1900.0, 2000, 2023, 2024, None, -4, 0, 2000.5, -1.5, 2100]})
    result["leap_number"] = (years >> mutate(at=tidy3.leap_year("x"))).collect()["at"]
    integers = tidy({"x": [1900, 2000, 2024, None]})
    result["leap_integer"] = (integers >> mutate(at=tidy3.leap_year("x"))).collect()["at"]
    result["leap_date"] = (span >> mutate(at=tidy3.leap_year("x"))).collect()["at"]
    result["leap_time"] = (x >> mutate(at=tidy3.leap_year("x"))).collect()["at"]
    return result


def _time_spans():
    from datetime import date

    ny, lh = "America/New_York", "Australia/Lord_Howe"

    def vec(values):
        return values[0] if len(values) == 1 else pl.Series(values)

    def r10(values):
        return vec([None if value is None else round(value, 10) for value in values])

    def values(frame, expr):
        return (frame >> mutate(at=expr)).collect()["at"]

    def clocks(out):
        return {
            "epochs": vec([value.timestamp() if value is not None else None for value in out]),
            "clocks": vec(out.dt.strftime("%Y-%m-%d %H:%M:%S %z").to_list()),
        }

    def zoned(texts, tz):
        return tidy({"x": texts}) >> mutate(x=tidy3.ymd_hms("x", tz=tz))

    def rows(pairs, tz):
        """One zoned clock per (text, period), each added on its own row."""
        return clocks(pl.concat([values(zoned([text], tz), col("x") + span) for text, span in pairs]))

    def day_text(out):
        return vec(out.dt.strftime("%Y-%m-%d").to_list())

    def utc_text(out):
        return vec(out.dt.strftime("%Y-%m-%d %H:%M:%S%.3f").to_list())

    def kind(out):
        return "Date" if out.dtype == pl.Date else "POSIXct"

    def lengths(frame, expr):
        return r10(values(frame, tidy3.time_length(expr)).to_list())

    n = tidy({"n": [1.0, 1.5, None]})
    leap = tidy({"x": [date(2012, 2, 29)]})
    ends = tidy({"x": [date(2010, 1, 31), date(2012, 2, 29), date(2010, 3, 31)],
                 "k": [1, None, -1], "m": [1, 12, -1], "h": [1, None, 30]})
    noon = zoned(["2024-03-09 12:00:00", "2024-11-02 12:00:00"], ny)
    later = tidy({"x": ["2024-11-03 06:30:00"]}) >> mutate(x=tidy3.with_tz(tidy3.ymd_hms("x"), ny))
    constructors = {
        name: lengths(n, getattr(tidy3, name)(col("n")))
        for name in ("dseconds", "dminutes", "dhours", "ddays", "dweeks", "dmonths", "dyears")
    }
    scalars = [tidy3.duration(90, "minutes"), tidy3.duration(1.5, "days"),
               tidy3.duration(2, "weeks"), tidy3.duration(1, "month")]
    as_duration = lengths(tidy({"s": [90, None]}), tidy3.as_duration("s")).to_list() + [
        tidy3.time_length(tidy3.as_duration(value))
        for value in (tidy3.months(1), tidy3.days(1) + tidy3.hours(1), tidy3.weeks(1))
    ]

    def summary(days_, verb):
        frame = tidy({"k": days_}) >> mutate(d=tidy3.ddays(col("k"))) >> summarise(s=verb("d"))
        return (frame >> mutate(s=tidy3.time_length("s"))).collect()["s"][0]

    span = tidy({"k": [1.5, None]}) >> mutate(d=tidy3.ddays(col("k")))
    gaps = tidy({"a": [date(2012, 1, 31)], "b": [date(2012, 3, 1)]})
    clock_gap = tidy({"a": ["2024-03-10 00:00:00"], "b": ["2024-03-11 00:00:00"]}) >> mutate(
        a=tidy3.ymd_hms("a", tz=ny), b=tidy3.ymd_hms("b", tz=ny))
    leap_out = (leap >> mutate(
        a=col("x") + tidy3.ddays(365), b=col("x") + tidy3.dyears(1),
        c=col("x") + tidy3.ddays(1.5), m=col("x") - tidy3.ddays(1.5),
    )).collect()
    date_out = (leap >> mutate(
        d=col("x") + tidy3.days(1), h=col("x") + tidy3.hours(1), w=col("x") + tidy3.weeks(2),
        mixed=col("x") + (tidy3.days(1) - tidy3.hours(2)),
    )).collect()
    first = zoned(["2024-01-31 02:30:00"], ny)
    skipped_month = zoned(["2024-02-10 02:30:00"], ny)
    return {
        "constructors": constructors,
        "duration_units": r10([tidy3.time_length(value) for value in scalars]),
        "as_duration": r10(as_duration),
        "leap_dates": {
            "days365": day_text(leap_out["a"]), "year": utc_text(leap_out["b"]),
            "day_and_half": utc_text(leap_out["c"]), "minus": utc_text(leap_out["m"]),
            "classes": vec([kind(leap_out[name]) for name in ("a", "b", "c")]),
        },
        "ddays_dst": clocks(values(zoned(["2024-03-09 12:00:00", "2024-11-02 12:00:00", "2024-03-09 02:30:00"], ny),
                                   col("x") + tidy3.ddays(1))),
        "summaries": r10([summary([1, 2, 3], tidy3.sum), summary([1, 2], tidy3.mean), summary([1, 2, 4], tidy3.median)]),
        "time_length": {
            unit: r10(values(span, tidy3.time_length("d", unit)).to_list())
            for unit in ("second", "minute", "hour", "day", "week", "month", "year")
        },
        "date_difference": r10([values(gaps, tidy3.time_length(col("b") - col("a")))[0],
                                values(clock_gap, tidy3.as_numeric(col("b") - col("a"), "second"))[0]]),
        "comparisons": vec([tidy3.ddays(1) == tidy3.days(1), tidy3.dhours(1) < tidy3.minutes(61),
                            tidy3.days(1) == tidy3.hours(24), tidy3.months(1) == tidy3.days(30)]),
        "compare_column": values(tidy({"n": [3600, 90000]}), tidy3.dseconds(col("n")) > tidy3.days(1)),
        "days_hours_dst": {"days": clocks(values(noon, col("x") + tidy3.days(1))),
                           "hours": clocks(values(noon, col("x") + tidy3.hours(24)))},
        "skipped": rows([("2024-03-09 02:30:00", tidy3.days(1)), ("2024-03-10 01:30:00", tidy3.hours(1)),
                         ("2024-03-10 01:30:00", tidy3.minutes(30)), ("2024-03-10 01:30:00", tidy3.minutes(90))], ny),
        "autumn_hours": rows([("2024-11-03 00:30:00", tidy3.hours(1)), ("2024-11-03 00:30:00", tidy3.hours(2))], ny),
        "repeat_from_before": rows([("2024-11-02 01:30:00", tidy3.days(1))], ny),
        "repeat_from_after": {
            "tidy3": rows([("2024-11-04 01:30:00", -tidy3.days(1))], ny),
            # lubridate takes the earlier repeat; tidy3 keeps EST.
            "lubridate": "01:30 EDT",
        },
        "repeat_identity": {
            "tidy3": clocks(pl.concat([values(later, col("x") + span) for span in (
                tidy3.seconds(0), tidy3.hours(1), -tidy3.hours(1), tidy3.minutes(1))])),
            # lubridate moves x + seconds(0) to the earlier 01:30.
            "lubridate": vec(["01:30 EDT", "01:31 EDT"]),
        },
        "month_skipped": {
            "plus": clocks(values(skipped_month, col("x") + tidy3.months(1))),
            "rollback": clocks(values(skipped_month, tidy3.add_with_rollback("x", tidy3.months(1)))),
            "first": clocks(values(first, tidy3.add_with_rollback("x", tidy3.months(1), roll_to_first=True))),
        },
        "month_ends": {
            "rollback": {
                "months": day_text(values(ends, col("x") + tidy3.months(1))),
                "years": day_text(values(ends, col("x") + tidy3.years(1))),
                "mixed": day_text(values(ends, col("x") + (tidy3.months(1) + tidy3.days(1)))),
            },
            # lubridate's + gives NA where tidy3 rolls back.
            "lubridate_plus": {
                "months": vec([None, "2012-03-29", None]),
                "years": vec(["2011-01-31", None, "2011-03-31"]),
                "mixed": vec([None, "2012-03-30", None]),
            },
        },
        "roll_to_first": {
            "months": day_text(values(ends, tidy3.add_with_rollback("x", tidy3.months(col("m")), roll_to_first=True))),
            "years": day_text(values(ends, tidy3.add_with_rollback("x", tidy3.years(1), roll_to_first=True))),
            "mixed": day_text(values(ends, tidy3.add_with_rollback(
                "x", tidy3.months(1) + tidy3.days(1), roll_to_first=True))),
        },
        "column_counts": {
            "days": day_text(values(ends, col("x") + tidy3.days(col("k")))),
            "months": day_text(values(ends, col("x") + tidy3.months(col("k")))),
            "hours": utc_text(values(ends, col("x") + tidy3.hours(col("h")))),
        },
        "date_classes": vec([kind(date_out[name]) for name in ("d", "h", "w")]),
        "date_clock": {
            "hours": utc_text(date_out["h"]), "weeks": day_text(date_out["w"]),
            "seconds": utc_text(values(tidy({"x": ["2024-01-01 00:00:00"]}) >> mutate(x=tidy3.ymd_hms("x")),
                                       col("x") + tidy3.seconds(1.5))),
            "period": clocks(values(noon, col("x") + tidy3.period(hours=25))),
            "mixed_sign": utc_text(date_out["mixed"]),
        },
        "lord_howe": rows([("2024-10-05 02:15:00", tidy3.days(1)), ("2024-04-06 01:45:00", tidy3.days(1)),
                           ("2024-10-06 01:45:00", tidy3.minutes(30))], lh),
    }


def _datetime_epochs(values: list, parser: Callable, **kwargs):
    out = (tidy({"x": values}) >> mutate(at=parser("x", **kwargs))).collect()["at"]
    return pl.Series([value.timestamp() if value is not None else None for value in out], dtype=pl.Float64)


def _datetime_parsers():
    result = {}
    for name, calendar, named, compact in [
        ("ymd_hms", "2024-02-29", "2024 February 29", "20240229"),
        ("ymd_hm", "2024-02-29", "2024 February 29", "20240229"),
        ("mdy_hms", "02/29/2024", "February 29th 2024", "02292024"),
        ("dmy_hms", "29/02/2024", "29th February 2024", "29022024"),
    ]:
        parser = getattr(tidy3, name)
        clock, digits = ("12:34", "1234") if name == "ymd_hm" else ("12:34:56", "123456")
        values = [f"{calendar} {clock}", f"{named} {clock}", compact + digits,
                  f"{calendar}T{clock}Z", f"{calendar} {clock}+05:30",
                  f"{calendar} {clock} PM", f"{calendar} {clock} AM", "bad", None]
        if name != "ymd_hm":
            values.append(f"{calendar} {clock}.125")
        for zone in ("UTC", "America/New_York", "Asia/Kolkata"):
            result[f"{name}_{zone}"] = _datetime_epochs(values, parser, tz=zone)
        result[f"{name}_numeric"] = _datetime_epochs([int(compact + digits)], parser)[0]
    return result


def _seconds_of_day(values) -> pl.Series:
    return pl.Series(
        [None if v is None else v.hour * 3600 + v.minute * 60 + v.second + v.microsecond / 1e6 for v in values],
        dtype=pl.Float64,
    )


def _date_time_orders():
    result = {}
    calendars = {"ymd": "2024-02-29", "mdy": "02/29/2024", "dmy": "29/02/2024"}
    named = {"ymd": "2024 February 29", "mdy": "February 29th 2024", "dmy": "29th February 2024"}
    compact = {"ymd": "20240229", "mdy": "02292024", "dmy": "29022024"}
    impossible = {"ymd": "2023-02-29", "mdy": "02/29/2023", "dmy": "29/02/2023"}
    bare = {"ymd": "1999-01-02", "mdy": "01/02/1999", "dmy": "02/01/1999"}
    for order, cal in calendars.items():
        for clock in ("h", "hm"):
            name = f"{order}_{clock}"
            parser = getattr(tidy3, name)
            time = (["13", "1 PM", "12 AM", "24", "13:05"] if clock == "h"
                    else ["13:05", "1:05 PM", "12:05 AM", "24:00", "13:05:09"])
            digits = "13" if clock == "h" else "1305"
            values = [f"{cal} {time[0]}", f"{named[order]} {time[0]}", compact[order] + digits,
                      f"{cal}T{time[0]}", f"{cal} {time[1]}", f"{cal} {time[2]}", f"{cal} {time[3]}",
                      f"{impossible[order]} {time[0]}", f"{cal} {time[4]}", bare[order], "bad", None]
            for zone in ("UTC", "America/New_York", "Asia/Kolkata"):
                result[f"{name}_{zone}"] = _datetime_epochs(values, parser, tz=zone)
            for truncated in range(1, len(clock) + 1):
                result[f"{name}_truncated_{truncated}"] = _datetime_epochs(values, parser, truncated=truncated)
    long_sparse = ["2024-01-15"] * 150
    for position in (1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73,
                     79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149):
        long_sparse[position - 1] = "garbage"
    long_majority = ["05/06/2024" if i % 10 == 0 else "06/14/2024" if i % 10 == 5 else "13/06/2024"
                     for i in range(1, 301)]
    specs = {
        "single": (["2024-02-29 12:34:56", "2024/2/29 1:2:3", "20240229123456", "on 2024-02-29 12:34:56",
                    "2024-02-29 12:34:56.125", "Thursday 2024-02-29 12:34:56", "2024-02-29T12:34:56Z",
                    "2024-02-29 12:34:56+05:30", "x2024-02-29 12:34:56", "2024-02-30 12:00:00",
                    "2024-02-29 24:00:00", "2024-02-29 12:60:00", "2024-02-29 12:34", "2024_02_29 12.34.56",
                    "bad", "", None], "ymd HMS", {}),
        "single_ny": (["2024-02-29 12:34:56", "2024-07-04 08:00:00", "2024-02-29T12:34:56Z", None],
                      "Ymd HMS", {"tz": "America/New_York"}),
        "mixed": (["02/29/2024", "29/02/2024", "2024-02-29", "Feb 29, 2024", "29 February 2024", "2024 Feb 29",
                   "2024 Sept 29", "2024 FEB 29", "13/01/2024", "01/02/2024", "29/02/2024 10:00", "2024-02", None],
                  ["mdy", "dmy", "ymd", "dmy HM"], {}),
        "majority_dmy": (["01/02/2024", "13/01/2024", "14/01/2024"], ["mdy", "dmy"], {}),
        "majority_mdy": (["01/02/2024", "01/13/2024", "01/14/2024"], ["dmy", "mdy"], {}),
        "separators": (["01/02/2024", "13-01-2024", "14-01-2024", "02/13/2024"], ["mdy", "dmy"], {}),
        "truncated": (["2024-02-29 1:02:03", "2024-02-29 13:02", "2024-02-29 13", "2024-02-29", "2024-02"],
                      "ymd HMS", {"truncated": 3}),
        "ampm": (["2024-02-29 1:02:03 PM", "2024-02-29 01:02:03PM", "2024-02-29 12:02:03 AM",
                  "2024-02-29 1:02:03", "2024-02-29 01:02:03.5 pm", "2024-02-29 13:02:03 PM"], "ymd IMS p", {}),
        "ampm_24": (["2024-02-29 1:02:03 PM", "2024-02-29 13:02:03", "2024-02-29 13:02:03 PM",
                     "2024-02-29 12:00:00 AM"], "ymd HMS p", {}),
        "clocks": (["2024-02-29 1:02:03 PM", "2024-02-29 13:02:03", "2024-02-29 13:02"], ["ymd T", "ymd R"], {}),
        "offsets": (["2024-02-29 12:34:56Z", "2024-02-29 12:34:56+05:30", "2024-02-29 12:34:56+0530",
                     "2024-02-29 12:34:56-05", "2024-02-29 12:34:56"], "ymd HMS z", {"tz": "America/New_York"}),
        "dst": (["2024-03-10 02:30:00", "2024-11-03 01:30:00", "2024-03-10 03:30:00"], "ymd HMS",
                {"tz": "America/New_York"}),
        "names": (["Feb 29 2024", "February 29 2024", "02 29 2024"], "bdY", {}),
        "time_first": (["12:34:56 2024-02-29", "12:34 29/02/2024"], ["HMS ymd", "HM dmy"], {}),
        # lubridate reads two-digit years with a cutoff of 68.
        "two_digit": (["24-02-29", "02/29/24", "29.02.69", "2024-02-29"], ["ymd", "mdy", "dmy"], {"cutoff": 68}),
        "long_sparse": (long_sparse, "ymd", {}),
        "long_majority": (long_majority, ["mdy", "dmy"], {}),
        "exact": (["2024-02-29 12:34:56", "29/02/2024", "bad"], ["%Y-%m-%d %H:%M:%S", "%d/%m/%Y"], {"exact": True}),
    }
    for name, (values, orders, kwargs) in specs.items():
        result[f"parse_date_time_{name}"] = _datetime_epochs(
            values, lambda x, **kw: tidy3.parse_date_time(x, orders, **kw), **kwargs
        )
    result["parse_date_time_numeric"] = _datetime_epochs(
        [20240229, 2024022913], lambda x: tidy3.parse_date_time(x, ["ymd", "ymd H"])
    )
    default = ["12:34", "12:34:56", "1:02", "01:02:03.25", "12:34 PM", "12:34pm", "12:00 AM", "1:02:03 am",
               " 12:34 ", "23:60", "23:59:60", "12", "1234", "12.34", "abc", "", "NA", None, "12:34:56.125",
               "0:00 AM", "-1:00", "12:34 X", "T12:34", "12:34:56Z", "100:00", "12:3", "12:345", "12:034",
               "12:34:056", "12:34.5", "1:02:03PM", "12:34   PM", "12:34 a.m.", "24:00", "13:00 PM"]
    result["parse_time_default"] = _seconds_of_day(
        (tidy({"x": default}) >> mutate(t=tidy3.parse_time("x"))).collect()["t"]
    )
    formats = [
        ("%Hh%M", ["12h34", "07.05", "1205", "bad"]),
        ("%H.%M", ["12h34", "07.05", "1205"]),
        ("%H%M", ["12h34", "1205", "0705"]),
        ("%I:%M:%S %p", ["7:05:03 PM", "12:00:00 am", "13:00:00 PM", "7:05:03"]),
        ("%H:%M", ["7:05", "07:05:03", "7:5", "24:00", "23:59", "0:00"]),
        ("%H:%M:%OS", ["07:05:03.5", "7:5:3", "07:05:3.25", "07:05:03"]),
        ("%H:%M:%S", ["07:05:03.5", "07:05:003", "07:05:3"]),
        ("%T", ["07:05:03", "07:05"]),
        ("%R", ["07:05", "07:05:03"]),
        ("%I:%M %p", ["7:05 pm", "07:05PM", "0:05 AM", "7:05", "12:05 am"]),
        ("%H:%M %p", ["7:05 pm", "7:05"]),
        ("%Y-%m-%dT%H:%M", ["2024-01-01T07:05", "07:05"]),
        ("%H:%M %Z", ["07:05 UTC", "07:05:03 Z"]),
        ("%H%.%M", ["07h05", "07-05", "07 05", "0705"]),
        ("%H%*%M", ["07h05", "07--05", "0705", "07abc05"]),
        ("%H%+%M", ["07h05", "07--05", "0705"]),
        ("%H %M", ["0705", " 0705", "07 05"]),
        ("%H:%M:%S%z", ["12:34:56+01", "12:34:56"]),
        ("%H:%M:%S %z", ["12:34:56 +0100", "12:34:56 Z"]),
        ("%b %H:%M", ["Jan 12:34", "January 12:34", "jan 12:34"]),
    ]
    result["parse_time_formats"] = [
        _seconds_of_day((tidy({"x": values}) >> mutate(t=tidy3.parse_time("x", fmt))).collect()["t"])
        for fmt, values in formats
    ]
    return result


def _datetime_truncation():
    result = {}
    for name, calendar in [("ymd_hms", "1999-01-02"), ("ymd_hm", "1999-01-02"), ("mdy_hms", "01/02/1999"), ("dmy_hms", "02/01/1999")]:
        values = [f"{calendar} 12:34:56", f"{calendar} 12:34", f"{calendar} 12", calendar, "bad", None]
        for truncated in range(3 if name == "ymd_hm" else 4):
            result[f"{name}_{truncated}"] = _datetime_epochs(values, getattr(tidy3, name), truncated=truncated)
    return result


def _datetime_dst():
    values = ["2024-03-10 01:59:59.5", "2024-03-10 02:30:00", "2024-03-10 03:30:00", "2024-11-03 01:30:00",
              "2024-11-03T01:30:00-04:00", "2024-11-03T01:30:00-05", "2024-01-01T12:00:00Z",
              "2024-02-29 24:00:00", "2024-02-29 23:59:60", "2023-02-29 12:00:00", None]
    return _datetime_epochs(values, tidy3.ymd_hms, tz="America/New_York")


def _date_components():
    from datetime import date, timedelta
    import tidy3 as t3

    days = [date(2019, 12, 25) + timedelta(days=i) for i in range(1850)] + [None]
    frame = tidy({"d": days})
    out = (frame >> mutate(
        weekday=t3.wday("d"), monday=t3.wday("d", week_start=1),
        yearday=t3.yday("d"), week=t3.week("d"), iso=t3.isoweek("d"),
        q=t3.quarter("d"), hour=t3.hour("d"), minute=t3.minute("d"), second=t3.second("d"),
    )).collect()
    return {name: out[name] for name in out.columns if name != "d"}


def _weekday_labels():
    from datetime import date, timedelta
    import tidy3 as t3

    frame = tidy({"d": [date(2024, 1, 1) + timedelta(days=i) for i in range(7)] + [None]})
    result = {}
    for start in range(1, 8):
        for abbr in (True, False):
            series = (frame >> mutate(label=t3.wday("d", label=True, abbr=abbr, week_start=start))).collect(as_="pandas")["label"]
            result[f"{start}_{'abbr' if abbr else 'full'}"] = {
                "values": pl.Series([None if pd.isna(value) else value for value in series], dtype=pl.String),
                "levels": pl.Series(series.cat.categories.tolist()),
                "ordered": series.cat.ordered,
            }
    return result


def _fiscal_quarters():
    from datetime import date
    import tidy3 as t3

    frame = tidy({"d": [date(year, month, 15) for year in (2023, 2024) for month in range(1, 13)] + [None]})
    result = {}
    for start in range(1, 13):
        for kind in ("quarter", "year.quarter", "date_first", "date_last", "year_start/end"):
            out = frame >> mutate(q=t3.quarter("d", type=kind, fiscal_start=start))
            if kind in {"date_first", "date_last"}:
                out = out >> mutate(q=t3.as_character("q"))
            result[f"{start}_{kind}"] = out.collect()["q"]
    return result


def _clock_components():
    from datetime import datetime
    from zoneinfo import ZoneInfo
    import tidy3 as t3

    zone = ZoneInfo("America/Los_Angeles")
    frame = tidy({"d": [
        datetime(2024, 1, 1, 23, 59, 59, 125000, tzinfo=zone),
        datetime(2024, 3, 10, 1, 59, 59, 500000, tzinfo=zone),
        datetime(2024, 3, 10, 3, 0, 0, 250000, tzinfo=zone),
        datetime(2024, 11, 3, 1, 30, 0, 0, tzinfo=zone), None,
    ]})
    out = (frame >> mutate(h=t3.hour("d"), m=t3.minute("d"), s=t3.second("d"), weekday=t3.wday("d"))).collect()
    return {name: out[name] for name in ("h", "m", "s", "weekday")}


def _factor_payload(frame):
    factor = frame.collect(as_="pandas")["species"]
    return {
        "values": pl.Series([None if pd.isna(value) else value for value in factor], dtype=pl.String),
        "levels": pl.Series(factor.cat.categories.tolist(), dtype=pl.String),
    }


def _factor_ordering():
    import tidy3 as t3

    frame = tidy(pd.DataFrame({
        "species": pd.Categorical(
            ["Gentoo", "Adelie", "Gentoo", None, "Chinstrap"],
            categories=["Chinstrap", "Adelie", "Gentoo", "unused"],
        ),
        "mass": [5000., 3500., 5200., 4000., 3700.],
    }))
    expressions = {
        "reverse": t3.fct_rev("species"),
        "frequency": t3.fct_infreq("species"),
        "weighted": t3.fct_infreq("species", "mass"),
        "relevel": t3.fct_relevel("species", "Gentoo", "Adelie"),
        "after": t3.fct_relevel("species", "Adelie", after=2),
        "last": t3.fct_relevel("species", "Adelie", after=float("inf")),
        "callable": t3.fct_relevel("species", sorted),
        "median": t3.fct_reorder("species", "mass"),
        "mean_desc": t3.fct_reorder("species", "mass", t3.mean, desc=True),
    }
    return {name: _factor_payload(frame >> mutate(species=expr)) for name, expr in expressions.items()}


def _factor_values():
    import tidy3 as t3

    frame = tidy(pd.DataFrame({
        "species": pd.Categorical(["b", "a", "a", "c", None],
                                  categories=["b", "a", "c", "unused"], ordered=True),
        "w": [1., 1., 1., 4., 10.],
    }))
    expressions = {
        "recode": t3.fct_recode("species", renamed="a", spare="unused"),
        "remove": t3.fct_recode("species", b="a", NULL="c"),
        "collapse": t3.fct_collapse("species", pair=["a", "b"]),
        "remainder": t3.fct_collapse("species", pair=["a", "b"], other_level="rest"),
        "existing": t3.fct_collapse("species", b="a", other_level="rest"),
        "overlap": t3.fct_collapse("species", first=["a", "b"], last="a"),
        "other_collision": t3.fct_lump_min("species", 2, other_level="b"),
    }
    for n in (0, 1, 2, -1, -2, 10):
        expressions[f"n_{n}"] = t3.fct_lump_n("species", n)
        expressions[f"weighted_n_{n}"] = t3.fct_lump_n("species", n, w="w")
    for label, prop in (("zero", 0), ("low", .2), ("high", .4),
                        ("negative_low", -.2), ("negative_high", -.4), ("all", 1), ("negative_all", -1)):
        expressions[f"prop_{label}"] = t3.fct_lump_prop("species", prop)
        expressions[f"weighted_prop_{label}"] = t3.fct_lump_prop("species", prop, w="w")
    for minimum in (0, 1, 2, 3):
        expressions[f"min_{minimum}"] = t3.fct_lump_min("species", minimum)
        expressions[f"weighted_min_{minimum}"] = t3.fct_lump_min("species", minimum, w="w")
    for method in ("min", "average", "first", "last", "max"):
        expressions[f"ties_{method}"] = t3.fct_lump_n("species", 2, ties_method=method)
    result = {}
    for name, expression in expressions.items():
        changed = frame >> mutate(species=expression)
        result[name] = {**_factor_payload(changed),
                        "ordered": changed.collect(as_="pandas")["species"].cat.ordered}
    grouped = tidy({"g": [2, 2, 2, 1, 1, 1], "species": ["a", "a", "b", "b", "b", "a"]})
    expr = t3.fct_lump_n("species", 1)
    result["persistent"] = _factor_payload(grouped >> group_by("g") >> mutate(species=expr))
    result["transient"] = _factor_payload(grouped >> mutate(species=expr, by="g"))
    return result


def _factor_levels():
    import tidy3 as t3

    frame = tidy(pd.DataFrame({
        "species": pd.Categorical(["b", "a", "a", "c", None],
                                  categories=["b", "a", "c", "unused"], ordered=True),
        "x": [1., 2., None, 3., 4.],
        "y": [5., 1., 2., 7., 3.],
        "ties": [2., 2., 2., 1., 1.],
        "ones": [1., 1., 1., 1., 1.],
        "text": ["z", "y", None, "z", "x"],
        "numbers": [3., 1.5, None, 10., 3.],
        "integers": pd.array([2, 1, None, 2, 7], dtype="Int64"),
        "logical": pd.array([True, None, True, True, True], dtype="boolean"),
    }))
    expressions = {
        "drop": t3.fct_drop("species"),
        "drop_only": t3.fct_drop("species", only=["unused", "a", "zz"]),
        "expand": t3.fct_expand("species", "z", "a", "y"),
        "expand_after": t3.fct_expand("species", ["z", "y"], after=1),
        "expand_front": t3.fct_expand("species", "z", after=0),
        "inorder": t3.fct_inorder("species"),
        "inorder_unordered": t3.fct_inorder("species", ordered=False),
        "inorder_text": t3.fct_inorder("text"),
        "value_to_level": t3.fct_na_value_to_level("species", "(Missing)"),
        "value_to_existing": t3.fct_na_value_to_level("species", "c"),
        "level_to_value": t3.fct_na_level_to_value("species", ["a", "zz"]),
        "reorder2": t3.fct_reorder2("species", "x", "y", na_rm=True),
        "reorder2_first": t3.fct_reorder2("species", "x", "y", t3.first2, na_rm=True, desc=False),
        "reorder2_ties": t3.fct_reorder2("species", "ties", "y"),
        "reorder2_tied_scores": t3.fct_reorder2("species", "ties", "ones"),
        "as_factor": t3.as_factor("species"),
        "as_factor_text": t3.as_factor("text"),
        "as_factor_numbers": t3.as_factor("numbers"),
        "as_factor_integers": t3.as_factor("integers"),
        "as_factor_logical": t3.as_factor("logical"),
    }
    result = {}
    for name, expression in expressions.items():
        changed = frame >> mutate(species=expression)
        result[name] = {**_factor_payload(changed),
                        "ordered": changed.collect(as_="pandas")["species"].cat.ordered}
    grouped = tidy({"g": [2, 2, 1, 1, 1], "s": ["q", "p", "r", "p", None],
                    "x": [1., 2., 3., 1., 2.], "y": [1., 2., 3., 4., 5.]})
    calls = {"inorder": t3.fct_inorder("s"), "reorder2": t3.fct_reorder2("s", "x", "y", na_rm=True)}
    for name, expr in calls.items():
        persistent = grouped >> group_by("g") >> mutate(species=expr)
        result[f"persistent_{name}"] = _factor_payload(persistent)
        result[f"transient_{name}"] = _factor_payload(grouped >> mutate(species=expr, by="g"))
    return result


def _factor_grouped():
    import tidy3 as t3

    frame = tidy({"island": ["b", "a", "b", "a"], "species": ["Gentoo", "Adelie", "Gentoo", "Adelie"]})
    expr = t3.fct_rev(t3.fct_infreq("species"))
    return {
        "persistent": _factor_payload(frame >> group_by("island") >> mutate(species=expr)),
        "transient": _factor_payload(frame >> mutate(species=expr, by="island")),
    }


def _factor_reorder_text():
    import tidy3 as t3

    inf = float("inf")
    frame = tidy({"g": [2, 2, 2, 2, 2, 1, 1, 1, 1, 1], "s": ["a", "b", "a", "c", "b", "b", "c", "c", "a", "a"],
                  "x": [3., None, 1., inf, 4., 2., -inf, inf, None, 5.]})
    calls = {
        "mean_remove": t3.fct_reorder("s", "x", t3.mean, na_rm=True),
        "mean_desc": t3.fct_reorder("s", "x", t3.mean, na_rm=True, desc=True),
        "keep": t3.fct_reorder("s", "x", t3.mean, na_rm=False),
        "max_desc": t3.fct_reorder("s", "x", t3.max, na_rm=True, desc=True),
        "nested": t3.fct_rev(t3.fct_reorder("s", "x", t3.median, na_rm=True)),
        "computed": t3.fct_reorder("s", col("x") * 2 + 1, t3.sum, na_rm=True, desc=True),
    }
    result = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, expr in calls.items():
            result[f"persistent_{name}"] = _factor_payload(frame >> group_by("g") >> mutate(species=expr))
            result[f"transient_{name}"] = _factor_payload(frame >> mutate(species=expr, by="g"))
            result[f"ungrouped_{name}"] = _factor_payload(frame >> mutate(species=expr))
    return result


def _factor_missing():
    import tidy3 as t3

    frame = tidy(pd.DataFrame({
        "species": pd.Categorical(["Gentoo", "Adelie", "Gentoo", None], categories=["Gentoo", "Adelie", "unused"]),
        "mass": [None, 3500., 5000., None],
    }))
    return {
        "remove": _factor_payload(frame >> mutate(species=t3.fct_reorder("species", "mass", na_rm=True))),
        "keep": _factor_payload(frame >> mutate(species=t3.fct_reorder("species", "mass", na_rm=False, desc=True))),
        "default": _factor_payload(frame >> mutate(species=t3.fct_reorder("species", "mass", na_rm=True, default=-float("inf")))),
        "empty": _factor_payload(tidy(pd.DataFrame({"species": pd.Categorical([], categories=["Adelie", "Gentoo"])})) >> mutate(species=t3.fct_rev("species"))),
    }


def _round_extremes():
    import tidy3 as t3

    x = [1.25, -1.25, 0.0, 1.25e-310, -1.25e-310, 1.75e-310, 5e-324]
    frame = tidy({"x": x})
    result = {}
    for digits in (309, 310, 323, 324, -309):
        out = (frame >> mutate(r=t3.round("x", digits))).collect()
        result[str(digits)] = pl.Series(["%.17g" % value for value in out["r"]])
    return result


def _string_summaries():
    import tidy3 as t3

    cars = tidy({"cyl": [4, 4, 6], "model": ["Datsun", "Toyota", "Mazda"]})
    summaries = {
        "length": t3.str_length(t3.first("model")),
        "upper": t3.str_to_upper(t3.first("model")),
        "label": t3.str_c(t3.first("model"), "!"),
    }
    return {
        "whole": cars >> summarise(**summaries),
        "grouped": cars >> group_by("cyl") >> summarise(**summaries),
    }


def _inferred_csv_failures():
    import tempfile

    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "cars.csv"
        path.write_text("hp,mpg,am\n" + "110,21.5,TRUE\n" * 100 + "oops,oops,oops\n")
        frame = tidy3.read_csv(path)
    return {
        "hp": frame.collect()["hp"],
        "mpg": frame.collect()["mpg"],
        "am": frame.collect()["am"],
        "problems": tidy3.problems(frame),
    }


def _round_grid():
    import tidy3 as t3

    x = (
        [k / 1000 for k in range(2001)]
        + [-k / 1000 * 3.7 for k in range(1, 501)]
        + [k / 8 for k in range(401)]
        + [2.675, 1.005, 5.015, 0.285, 2.0**900, 123456789.123456789]
    )
    frame = tidy({"x": x, "big": [value * 1000 for value in x]})
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


def _readr_grid():
    import tidy3 as t3

    euro = t3.locale(decimal_mark=",", grouping_mark=".")
    result = {
        "number": _column(
            ["-$1,234", "$-5", "a-5", "- 5", "--5", "+5", "$+5", "1-2", "1,234,", "(5)",
             ".5", "-.5", "12abc34", "abc", "1e3", "1E3", "50%", "1,23", "1,2345",
             "12,34,567", "$1,234.50", "USD -7.5", "-", "Price 12", "1.2.3", "1,234.5.6",
             " 7 ", "", "Price - 12", "3eggs"],
            t3.parse_number("x"),
        ),
        "number_euro": _column(
            ["1.234,5", "€1.234,5", "1,5"], t3.parse_number("x", locale=euro)
        ),
        "double": _column(
            ["1,234", "1234", " 1e3", "1e+3", "1E-2", ".5", "5.", "+5", "-.5", "Inf", "inf",
             "NaN", "1 234", "", "1.2.3"],
            t3.parse_double("x"),
        ),
        "double_euro": _column(
            ["1,5", "1.234,5", "1e3", ",5"], t3.parse_double("x", locale=euro)
        ),
        "integer": _column(
            ["1,234", "+5", "007", "1e3", "-0", " 12 ", "1.0", ""], t3.parse_integer("x")
        ),
        "logical": _column(
            ["True", "TRUE", "yes", "Y", "N", "no", " true", "F", "0"],
            t3.parse_logical("x"),
        ),
    }
    return result


def _readr_rows():
    import tempfile

    import tidy3 as t3

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


def _factor_sort():
    ordered = tidy(pd.DataFrame({
        "f": pd.Categorical(
            ["a", "b", "a", None, "b"],
            categories=["b", "a", "unused"],
            ordered=True,
        ),
        "row": [1, 2, 3, 4, 5],
    }))
    unordered = tidy(pd.DataFrame({
        "f": pd.Categorical(
            ["a", "b", "a", None, "b"],
            categories=["b", "a", "unused"],
        ),
        "row": [1, 2, 3, 4, 5],
    }))
    empty = tidy(pd.DataFrame({
        "f": pd.Categorical([], categories=["b", "a", "unused"], ordered=True),
        "row": pd.Series([], dtype="int64"),
    }))
    missing = tidy(pd.DataFrame({
        "f": pd.Categorical([None, None], categories=["b", "a"]),
        "row": [1, 2],
    }))

    def payload(frame):
        collected = frame.collect(as_="pandas")
        factor = collected["f"]
        return {
            "values": pl.Series(
                [None if pd.isna(value) else value for value in factor], dtype=pl.String
            ),
            "rows": pl.Series(collected["row"].tolist(), dtype=pl.Int64),
            "levels": pl.Series(list(factor.cat.categories), dtype=pl.String),
            "ordered": bool(factor.cat.ordered),
        }

    return {
        "ascending": payload(ordered >> arrange("f")),
        "descending": payload(ordered >> arrange(tidy3.desc("f"))),
        "grouped": payload(
            ordered >> group_by("f") >> arrange("row", by_group=True) >> ungroup()
        ),
        "unordered": payload(unordered >> arrange("f")),
        "empty": payload(empty),
        "missing": payload(missing >> arrange("f")),
    }


def _string_case_rank():
    import tidy3 as t3

    x = ["Mazda RX4 Wag", "Hornet 4 Drive", "Merc 450SLC", "AMC Javelin", "HTTPServer",
         "camelCaseXML", "x1y2", "  __a-b.C  ", "don't stop", "\u00c9coleNormale", "\u65e5\u672cText", "a  b", "", None]
    ranked = tidy({"g": [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0],
                   "s": ["merc", "fiat", None, "x10", "x9", "honda", "x10"]})
    ranks = {"rank": t3.str_rank("s"), "natural": t3.str_rank("s", numeric=True)}
    return {
        "case": tidy({"x": x}) >> mutate(
            snake=t3.str_to_snake("x"),
            kebab=t3.str_to_kebab("x"),
            camel=t3.str_to_camel("x"),
            pascal=t3.str_to_camel("x", first_upper=True),
        ),
        "rank": ranked >> mutate(**ranks),
        "grouped": ranked >> group_by("g") >> mutate(**ranks) >> ungroup(),
        "vector": pl.Series(t3.str_rank(["b", "a", "c", "a"])),
    }


def _pivot_ptypes_repair():
    import warnings

    weeks = tidy({"id": [1, 2], "week_1": [5.0, 6.0], "week_2": [7.0, 8.0]})
    counts = tidy({"id": [1, 2], "n_1": [1, 2], "n_2": [3, 4], "sd_1": [0.5, 0.25], "sd_2": [1.0, 2.0]})
    clash = tidy({"name": [1, 2], "x": [1, 2], "y": [3, 4]})
    keyed = tidy({"a": [1, 2], "k": ["a", "b"], "v": [1, 2]})
    odd = tidy({"id": [1, 2], "k": ["a b", "1x"], "v": [1, 2]})
    dup = tidy({"id": [1.0, 1.0, 2.0], "key": ["a", "a", "b"], "val": [1.0, 3.0, 5.0], "w": [10.0, 20.0, 30.0]})
    week = {"names_to": "week", "names_prefix": "week_"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fn_dict = dup >> pivot_wider(names_from="key", values_from=["val", "w"], values_fn={"val": "sum"})
    return {
        "names_double": weeks >> pivot_longer(
            ["week_1", "week_2"], **week, names_transform={"week": int}, names_ptypes={"week": float}
        ),
        "names_factor": weeks >> pivot_longer(
            ["week_1", "week_2"], **week, names_ptypes={"week": pl.Enum(["2", "1"])}
        ),
        "values_double": counts >> pivot_longer(["n_1", "n_2"], values_ptypes={"value": float}),
        "values_all": counts >> pivot_longer(["n_1", "n_2"], values_ptypes=float),
        "values_dot": counts >> pivot_longer(
            ["n_1", "n_2", "sd_1", "sd_2"], names_to=[".value", "visit"], names_sep="_",
            values_ptypes={"n": float},
        ),
        "longer_unique": clash >> pivot_longer(["x", "y"], names_repair="unique"),
        "longer_universal": clash >> pivot_longer(
            ["x", "y"], names_to="the key", values_to="the value", names_repair="universal"
        ),
        "longer_function": clash >> pivot_longer(
            ["x", "y"], names_to="key", names_repair=lambda names: [n.upper() for n in names]
        ),
        "wider_unique": keyed >> pivot_wider(names_from="k", values_from="v", names_repair="unique"),
        "wider_universal": odd >> pivot_wider(names_from="k", values_from="v", names_repair="universal"),
        "wider_glue": keyed >> pivot_wider(
            names_from="k", values_from="v", names_glue="{k}", names_vary="slowest", names_repair="unique"
        ),
        "fn_dict": fn_dict,
        "fn_callable": dup >> pivot_wider(
            names_from="key", values_from="val", values_fn=lambda s: s.max() - s.min()
        ),
        "fn_length": dup >> pivot_wider(names_from="key", values_from=["val", "w"], values_fn=lambda s: s.len()),
        "fn_glue": dup >> pivot_wider(
            names_from="key", values_from="val", values_fn={"val": "mean"},
            names_glue="{key}_avg", values_fill={"val": 0},
        ),
        "fn_fill": dup >> pivot_wider(
            names_from="key", values_from=["val", "w"], values_fn="sum", values_fill={"w": 0}
        ),
        "spec_ptypes": weeks >> tidy3.pivot_longer_spec(
            tidy3.build_longer_spec(
                weeks, ["week_1", "week_2"], **week, names_transform={"week": int}, names_ptypes={"week": float}
            ),
            values_ptypes={"value": int},
        ),
        "spec_longer_unique": clash >> tidy3.pivot_longer_spec(
            tidy3.build_longer_spec(clash, ["x", "y"]), names_repair="unique"
        ),
        "spec_wider_unique": keyed >> tidy3.pivot_wider_spec(
            tidy3.build_wider_spec(keyed, names_from="k", values_from="v"), names_repair="unique"
        ),
    }


def _string_equal():
    frame = tidy(
        {
            "x": ["\u00e9", "\u00c9", "e", None, "a", "\u212b"],
            "y": ["e\u0301", "e\u0301", "e", None, "b", "\u00c5"],
        },
    )
    return frame >> mutate(
        equal=tidy3.str_equal("x", col("y")),
        ignore=tidy3.str_equal("x", col("y"), ignore_case=True),
    )


def _lubridate_grid():
    import tidy3 as t3
    from datetime import date

    def days(values, expr):
        return _column(values, t3.as_character(expr))

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


def _factor_combine():
    import tidy3 as t3

    def describe(frame: Any, name: str = "species") -> dict[str, Any]:
        column = frame.collect(as_="pandas")[name]
        factor = isinstance(column.dtype, pd.CategoricalDtype)
        return {
            "type": "factor" if factor else "character",
            "values": pl.Series(
                [None if pd.isna(value) else str(value) for value in column.tolist()], dtype=pl.String
            ),
            "levels": pl.Series(
                [str(level) for level in column.cat.categories] if factor else [], dtype=pl.String
            ),
        }

    def frame(data: dict[str, Any]) -> Any:
        return tidy(data)

    p = frame({"species": ["Adelie", "Gentoo", "Adelie", "Chinstrap", None],
               "mass": [3750.0, 5000.0, 3800.0, 3700.0, 4000.0]}) >> mutate(species=t3.fct_infreq("species"))
    text_lookup = frame({"species": ["Adelie", "Gentoo", "Emperor"], "w": [1.0, 2.0, 3.0]})
    other_levels = frame({"species": ["Gentoo", "Emperor"], "w": [2.0, 3.0]}) >> mutate(
        species=t3.fct_relevel("species", "Gentoo", "Emperor")
    )
    # Unused levels: relevel with every value, then filter one value out.
    same_levels = frame({"species": ["Gentoo", "Adelie", "Chinstrap"], "w": [2.0, 1.0, None]}) >> mutate(
        species=t3.fct_relevel("species", "Adelie", "Chinstrap", "Gentoo")
    ) >> t3.filter(~t3.is_na("w"))
    emperor = frame({"species": ["Emperor"], "mass": [1.0]}) >> mutate(species=t3.fct_rev("species"))
    years = frame({"species": ["Adelie", "Gentoo", "Chinstrap", None], "year": [2007.0, 2008.0, 2008.0, 2008.0],
                   "n": [1.0, 2.0, None, 3.0]}) >> mutate(
        species=t3.fct_relevel("species", "Gentoo", "Chinstrap", "Adelie")
    ) >> t3.filter(~t3.is_na("n"))
    completed = years >> t3.complete("species", "year")
    return {
        "left_text": describe(p >> t3.left_join(text_lookup, by="species")),
        "left_text_w": (p >> t3.left_join(text_lookup, by="species")).collect()["w"],
        "inner_text": describe(p >> t3.inner_join(text_lookup, by="species")),
        "full_text": describe(p >> t3.full_join(text_lookup, by="species")),
        "right_text": describe(text_lookup >> t3.right_join(p, by="species")),
        "semi_text": describe(p >> t3.semi_join(text_lookup, by="species")),
        "anti_text": describe(p >> t3.anti_join(text_lookup, by="species")),
        "left_other": describe(p >> t3.left_join(other_levels, by="species")),
        "full_other": describe(p >> t3.full_join(other_levels, by="species")),
        "left_same": describe(p >> t3.left_join(same_levels, by="species")),
        "bind_factors": describe(p >> t3.bind_rows(emperor)),
        "bind_factors_reversed": describe(emperor >> t3.bind_rows(p)),
        "bind_text": describe(p >> t3.bind_rows(frame({"species": ["Emperor"], "mass": [1.0]}))),
        "bind_missing_column": describe(p >> t3.bind_rows(frame({"mass": [1.0]}))),
        "complete_species_year": describe(completed),
        "complete_species_year_n": completed.collect()["n"],
        "complete_species": describe(years >> t3.complete("species")),
        "complete_nesting": describe(years >> t3.complete(t3.nesting("species"), "year")),
        "expand_species": describe(years >> t3.expand("species")),
        "complete_grouped": describe(years >> t3.group_by("year") >> t3.complete("species") >> t3.ungroup()),
    }


def _right_join_order():
    import tidy3 as t3

    x = tidy({"island": ["Torgersen", "Biscoe", "Dream", "Biscoe"], "from_x": ["x1", "x2", "x3", "x4"]})
    y = tidy({"island": ["Biscoe", "Anvers", "Torgersen", "Biscoe", None], "from_y": ["y1", "y2", "y3", "y4", "y5"]})
    return {
        "plain": x >> t3.right_join(y, by="island"),
        "keep": x >> t3.right_join(y, by="island", keep=True),
        "first": x >> t3.right_join(y, by="island", multiple="first"),
        "by_spec": x >> t3.right_join(y, by=t3.join_by("island")),
        "swapped": y >> t3.right_join(x, by="island"),
    }


def _many_groups():
    import tidy3 as t3

    ids = np.arange(1, 901)
    species = np.array(["Adelie", "Gentoo", "Chinstrap", "Emperor", "King"])[(ids * 3) % 5]
    mass = (((ids * 13) % 50) * 100).astype(float)
    p = tidy(
        pl.DataFrame({
            "id": ids, "g": (ids * 7) % 97, "species": species,
            "mass": pl.Series(mass).set(pl.Series(ids % 23 == 0), None), "bill": (ids * 17) % 41 + 0.5,
        }),
    )

    def levels_of(frame: Any) -> dict[str, Any]:
        column = frame.collect(as_="pandas")["s"]
        return {
            "values": pl.Series([str(v) for v in column.tolist()]),
            "levels": pl.Series([str(level) for level in column.cat.categories]),
        }

    grouped = p >> t3.group_by("g")
    return {
        "slice_max": grouped >> t3.slice_max("mass", n=1, na_rm=True) >> t3.ungroup() >> t3.arrange("id"),
        "slice_max_no_ties": grouped >> t3.slice_max("mass", n=2, with_ties=False, na_rm=True) >> t3.ungroup() >> t3.arrange("id"),
        "slice_min_prop": grouped >> t3.slice_min("bill", prop=0.3) >> t3.ungroup() >> t3.arrange("id"),
        "slice_head": grouped >> t3.slice_head(n=2) >> t3.ungroup() >> t3.arrange("id"),
        "slice_tail": grouped >> t3.slice_tail(n=1) >> t3.ungroup() >> t3.arrange("id"),
        "slice_positions": grouped >> t3.slice(1, 3) >> t3.ungroup() >> t3.arrange("id"),
        "reframe": p >> t3.reframe(q=t3.median("mass", na_rm=True), k=t3.n(), by="g") >> t3.arrange("g"),
        "nth": grouped >> t3.summarise(a=t3.nth("mass", 2), b=t3.nth("bill", -1), c=t3.nth("species", 2, order_by="bill")) >> t3.ungroup(),
        "filter": grouped >> t3.filter(t3.col("bill") > t3.mean("bill")) >> t3.ungroup(),
        "reorder": levels_of(grouped >> t3.mutate(s=t3.fct_reorder("species", "bill")) >> t3.ungroup()),
        "reorder_desc": levels_of(grouped >> t3.mutate(s=t3.fct_reorder("species", "bill", desc=True)) >> t3.ungroup()),
        "infreq": levels_of(grouped >> t3.mutate(s=t3.fct_infreq("species")) >> t3.ungroup()),
        "relevel": levels_of(grouped >> t3.mutate(s=t3.fct_relevel("species", "King", after=1)) >> t3.ungroup()),
        "rev": levels_of(grouped >> t3.mutate(s=t3.fct_rev("species")) >> t3.ungroup()),
    }


def _rolling_joins() -> dict[str, Any]:
    import tidy3 as t3

    left = t3.tidy({"g": ["b", "a", None, "a", "z"], "x": [4., 2., 2., 0., 9.], "lid": [1., 2., 3., 4., 5.]})
    right = {"h": ["a", "a", None, "b", "q"], "y": [1., 1., 1., 4., 8.], "rid": [1., 2., 3., 4., 5.]}
    result = {}
    for op in (t3.ge, t3.gt, t3.le, t3.lt):
        spec = t3.join_by(("g", "h"), t3.closest(op("x", "y")))
        for how in ("left", "inner", "right", "full", "semi", "anti"):
            result[f"{op.__name__}_{how}"] = left >> getattr(t3, f"{how}_join")(right, by=spec)
    spec = t3.join_by(("g", "h"), t3.closest(t3.ge("x", "y")))
    result["never"] = left >> t3.left_join(right, by=spec, na_matches="never")
    result["keep"] = left >> t3.left_join(right, by=spec, keep=True)
    return result


ORACLE_CASES: dict[str, Callable[[str], Any]] = {
    "rolling_joins": _rolling_joins,
    "reshape_metadata": _reshape_metadata,
    "janitor_dupes": _janitor_dupes,
    "janitor_cleaning": _janitor_cleaning,
    "janitor_tabyl": _janitor_tabyl,
    "janitor_totals": _janitor_totals,
    "janitor_percentages": _janitor_percentages,
    "janitor_formatting": _janitor_formatting,
    "timezone_conversion": _timezone_conversion,
    "timezone_dst": _timezone_dst,
    "timezone_resolved_repeats": _timezone_resolved_repeats,
    "formatted_offsets": _formatted_offsets,
    "calendar_rounding": _calendar_rounding,
    "date_helpers": _date_helpers,
    "time_spans": _time_spans,
    "datetime_parsers": _datetime_parsers,
    "date_time_orders": _date_time_orders,
    "datetime_truncation": _datetime_truncation,
    "datetime_dst": _datetime_dst,
    "date_components": _date_components,
    "weekday_labels": _weekday_labels,
    "fiscal_quarters": _fiscal_quarters,
    "clock_components": _clock_components,
    "factor_ordering": _factor_ordering,
    "factor_values": _factor_values,
    "factor_levels": _factor_levels,
    "factor_grouped": _factor_grouped,
    "factor_reorder_text": _factor_reorder_text,
    "factor_missing": _factor_missing,
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
    "factor_combine": _factor_combine,
    "right_join_order": _right_join_order,
    "many_groups": _many_groups,
    "round_extremes": _round_extremes,
    "string_summaries": _string_summaries,
    "inferred_csv_failures": _inferred_csv_failures,
    "readr_grid": _readr_grid,
    "readr_rows": _readr_rows,
    "lubridate_grid": _lubridate_grid,
    "factor_sort": _factor_sort,
    "string_equal": _string_equal,
    "string_case_rank": _string_case_rank,
    "pivot_ptypes_repair": _pivot_ptypes_repair,
}


CASE_VERBS = {
    "reshape_metadata": {
        "pack", "unpack", "hoist", "separate_wider_delim", "separate_wider_regex",
        "separate_wider_position", "separate_longer_delim", "separate_longer_position",
    },
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
    "pivot_ptypes_repair": {
        "pivot_wider", "pivot_longer", "build_longer_spec", "pivot_longer_spec",
        "build_wider_spec", "pivot_wider_spec",
    },
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


@pytest.mark.r_oracle
@pytest.mark.parametrize("case", sorted(ORACLE_CASES))
def test_r_oracle_parity(case):
    oracle = _oracle(case)
    if isinstance(oracle, dict) and oracle.get("skip") is True:
        pytest.skip(str(oracle.get("reason") or "oracle case skipped"))
    assert _normalise(ORACLE_CASES[case]()) == oracle


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
    }
    # Column-name helpers from janitor and rlang, not dplyr or tidyr.
    not_dplyr = {"clean_names", "make_clean_names", "set_names", "get_dupes"}
    assert public == differential | expected_gap_verbs | INVARIANT_ONLY | not_dplyr


def test_stochastic_verbs_satisfy_dplyr_shape_and_group_invariants():
    frame = tidy(
        {"g": ["a"] * 4 + ["b"] * 4, "x": list(range(8))},
    )
    assert len((frame >> tidy3.sample_n(3, seed=7)).collect(as_="pandas")) == 3
    assert len((frame >> tidy3.sample_frac(0.5, seed=7)).collect(as_="pandas")) == 4
    sampled = (
        frame >> group_by("g") >> tidy3.slice_sample(n=2, seed=7)
    ).collect(as_="pandas")
    assert sampled.groupby("g", observed=True).size().to_dict() == {"a": 2, "b": 2}


def test_weighted_slice_sample_support():
    frame = tidy(
        {"x": [1, 2, 3], "weight": [1.0, 2.0, 2.0]}
    )
    sampled = frame >> tidy3.slice_sample(
        n=2, weight_by="weight", seed=7
    )
    assert len(sampled.collect(as_="pandas")) == 2
