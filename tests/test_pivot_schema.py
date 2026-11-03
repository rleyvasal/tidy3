"""pivot_longer/pivot_wider hand their result's schema to the frame (Polars 2.1's
chained lazy pivots resolve it in time that doubles per pivot). The schema comes
from running the same lazy steps on an empty frame, so it must equal the real one."""

import warnings

import polars as pl
import pytest

import tidy3.tidyr as tidyr_module
from tidy3 import pivot_longer, pivot_wider, tidy

WIDE = pl.DataFrame(
    {
        "id": [1, 2, 3],
        "g": ["a", "b", "a"],
        "x_1": [1.5, None, 3.0],
        "x_2": [4.0, 5.0, None],
        "y_1": [1, 2, 3],
        "y_2": [4, 5, 6],
    }
)
LONG = pl.DataFrame(
    {
        "id": [1, 1, 2, 2, 3],
        "k": ["x", "y", "x", "y", "x"],
        "k2": ["u", "u", "v", "v", "u"],
        "n": [1, 2, 3, 4, 5],
        "f": [1.5, None, 2.5, 3.5, 4.5],
        "s": ["p", "q", "r", "s", "t"],
    }
)
DUPLICATED = pl.DataFrame(
    {"id": [1, 1, 1, 2], "k": ["x", "x", "y", "x"], "n": [1, 2, 3, 4], "f": [1.5, 2.5, 3.5, 4.5]}
)

LONGER = {
    "basic": (WIDE, ["x_1", "x_2"], {}),
    "names and values to": (WIDE, ["x_1", "x_2"], {"names_to": "n", "values_to": "val"}),
    "prefix": (WIDE, ["x_1", "x_2"], {"names_prefix": "x_", "names_to": "n"}),
    "two names, sep": (WIDE, ["x_1", "x_2", "y_1", "y_2"], {"names_to": ["var", "num"], "names_sep": "_"}),
    "one name, pattern": (WIDE, ["x_1", "x_2"], {"names_to": "num", "names_pattern": r"x_(.)"}),
    "two names, pattern": (WIDE, ["x_1", "x_2"], {"names_to": ["var", "num"], "names_pattern": r"(.)_(.)"}),
    ".value": (WIDE, ["x_1", "x_2", "y_1", "y_2"], {"names_to": [".value", "num"], "names_sep": "_"}),
    "drop missing": (WIDE, ["x_1", "x_2"], {"values_drop_na": True}),
    "slowest": (WIDE, ["x_1", "x_2", "y_1"], {"cols_vary": "slowest"}),
    "mixed types": (WIDE, ["x_1", "y_1"], {}),
    "value ptype": (WIDE, ["y_1", "y_2"], {"values_ptypes": pl.Float64}),
}

WIDER = {
    "basic": (LONG[["id", "k", "n"]], {"names_from": "k", "values_from": "n"}),
    "float values": (LONG[["id", "k", "f"]], {"names_from": "k", "values_from": "f"}),
    "string values": (LONG[["id", "k", "s"]], {"names_from": "k", "values_from": "s"}),
    "id_cols": (LONG, {"names_from": "k", "values_from": "n", "id_cols": "id"}),
    "no id columns": (LONG[["k", "n"]], {"names_from": "k", "values_from": "n"}),
    "prefix": (LONG[["id", "k", "n"]], {"names_from": "k", "values_from": "n", "names_prefix": "v_"}),
    "sorted names": (LONG[["id", "k", "n"]], {"names_from": "k", "values_from": "n", "names_sort": True}),
    "two names_from": (LONG[["id", "k", "k2", "n"]], {"names_from": ["k", "k2"], "values_from": "n"}),
    "two values_from": (LONG[["id", "k", "n", "f"]], {"names_from": "k", "values_from": ["n", "f"]}),
    "fill with an integer": (LONG[["id", "k", "n"]].head(4), {"names_from": "k", "values_from": "n", "values_fill": 0}),
    # Filling an integer column with 1.5 changes its type: the derived schema must too.
    "fill with a float": (LONG[["id", "k", "n"]].head(4), {"names_from": "k", "values_from": "n", "values_fill": 1.5}),
    "fill by column": (
        LONG[["id", "k", "n", "f"]].head(4),
        {"names_from": "k", "values_from": ["n", "f"], "values_fill": {"n": 0, "f": 0.5}},
    ),
    "repeated cells become lists": (DUPLICATED, {"names_from": "k", "values_from": "n"}),
    "list": (DUPLICATED, {"names_from": "k", "values_from": "n", "values_fn": "list"}),
    "sum": (DUPLICATED, {"names_from": "k", "values_from": "n", "values_fn": "sum"}),
    "mean": (DUPLICATED, {"names_from": "k", "values_from": "n", "values_fn": "mean"}),
    "first, two values": (DUPLICATED, {"names_from": "k", "values_from": ["n", "f"], "values_fn": "first"}),
    "different summary per value": (
        DUPLICATED,
        {"names_from": "k", "values_from": ["n", "f"], "values_fn": {"n": "max", "f": "min"}},
    ),
    "names given": (LONG[["id", "k", "n"]], {"names_from": "k", "values_from": "n", "names": ["x", "y", "z"]}),
    "repaired names": (
        LONG[["id", "k", "n"]].rename({"id": "x"}),
        {"names_from": "k", "values_from": "n", "names_repair": "unique"},
    ),
}


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


@pytest.mark.parametrize("name", LONGER)
def test_pivot_longer_derived_schema_is_the_real_one(monkeypatch, name):
    monkeypatch.setattr(tidyr_module, "_DERIVE_PIVOT_SCHEMAS", True)
    frame, columns, options = LONGER[name]
    out = pivot_longer(columns, **options)._fn(tidy(frame))  # the verb itself, before validation
    assert out._schema_cache is not None
    assert out._schema_cache == out._lf.collect_schema()


@pytest.mark.parametrize("name", WIDER)
def test_pivot_wider_derived_schema_is_the_real_one(monkeypatch, name):
    monkeypatch.setattr(tidyr_module, "_DERIVE_PIVOT_SCHEMAS", True)
    frame, options = WIDER[name]
    out = pivot_wider(**options)._fn(tidy(frame))
    assert out._schema_cache is not None
    assert out._schema_cache == out._lf.collect_schema()


def test_the_float_fill_changes_the_column_type():
    frame, options = WIDER["fill with a float"]
    out = (tidy(frame) >> pivot_wider(**options)).collect()
    assert set(out.schema.values()) == {pl.Int64, pl.Float64}


def test_functions_given_by_the_user_resolve_normally(monkeypatch):
    # Their result type is read off the data, which an empty frame does not have.
    monkeypatch.setattr(tidyr_module, "_DERIVE_PIVOT_SCHEMAS", True)
    out = pivot_wider(names_from="k", values_from="n", values_fn=lambda s: s.sum())._fn(tidy(DUPLICATED))
    assert out._schema_cache is None
    out = pivot_longer(["x_1", "x_2"], values_transform=float)._fn(tidy(WIDE))
    assert out._schema_cache is None


@pytest.mark.parametrize("name", ["basic", "two names, sep", ".value", "drop missing"])
def test_results_do_not_depend_on_deriving_the_schema(monkeypatch, name):
    frame, columns, options = LONGER[name]
    results = []
    for derive in (True, False):
        monkeypatch.setattr(tidyr_module, "_DERIVE_PIVOT_SCHEMAS", derive)
        results.append((tidy(frame) >> pivot_longer(columns, **options)).collect())
    assert results[0].equals(results[1])
    for name in ("basic", "fill with a float", "repeated cells become lists"):
        frame, options = WIDER[name]
        results = []
        for derive in (True, False):
            monkeypatch.setattr(tidyr_module, "_DERIVE_PIVOT_SCHEMAS", derive)
            results.append((tidy(frame) >> pivot_wider(**options)).collect())
        assert results[0].equals(results[1])
