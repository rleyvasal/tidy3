"""Category ordering, values, and factor metadata on both backends."""

import math

import pandas as pd
import polars as pl
import pytest

import tidy3 as t

BACKENDS = ["polars", "pandas"]


def levels(frame, name="species"):
    return frame.collect(as_="pandas")[name].cat.categories.tolist()


def penguins(backend):
    return t.tidy(pd.DataFrame({
        "species": pd.Categorical(
            ["Gentoo", "Adelie", "Gentoo", None, "Chinstrap"],
            categories=["Chinstrap", "Adelie", "Gentoo", "unused"],
        ),
        "mass": [5000., 3500., 5200., 4000., 3700.],
    }), backend=backend)


@pytest.mark.parametrize("backend", BACKENDS)
def test_relevel_preserves_values_unused_levels_and_metadata(backend):
    frame = penguins(backend) >> t.mutate(species=t.fct_relevel("species", "Gentoo", "Adelie"))
    assert levels(frame) == ["Gentoo", "Adelie", "Chinstrap", "unused"]
    assert frame.collect(as_="pandas")["species"].dropna().tolist() == ["Gentoo", "Adelie", "Gentoo", "Chinstrap"]
    assert frame._category_levels["species"] == levels(frame)
    ordered = (frame >> t.arrange("species")).collect(as_="pandas")["species"]
    assert ordered.dropna().tolist() == ["Gentoo", "Gentoo", "Adelie", "Chinstrap"]
    counted = frame >> t.count("species", drop=False)
    assert counted.collect(as_="pandas")["n"].tolist() == [2, 1, 1, 0, 1]


@pytest.mark.parametrize("backend", BACKENDS)
def test_relevel_after_callable_unknown_and_duplicate(backend):
    frame = penguins(backend)
    assert levels(frame >> t.mutate(species=t.fct_relevel("species", "Adelie", after=2))) == ["Chinstrap", "Gentoo", "Adelie", "unused"]
    assert levels(frame >> t.mutate(species=t.fct_relevel("species", "Adelie", after=math.inf))) == ["Chinstrap", "Gentoo", "unused", "Adelie"]
    assert levels(frame >> t.mutate(species=t.fct_relevel("species", sorted))) == ["Adelie", "Chinstrap", "Gentoo", "unused"]
    with pytest.warns(UserWarning, match="unknown"):
        assert levels(frame >> t.mutate(species=t.fct_relevel("species", "missing"))) == frame._category_levels["species"]
    with pytest.raises(ValueError, match="unique"):
        frame >> t.mutate(species=t.fct_relevel("species", "Adelie", "Adelie"))


@pytest.mark.parametrize("backend", BACKENDS)
def test_frequency_weights_ties_and_reverse(backend):
    frame = penguins(backend)
    assert levels(frame >> t.mutate(species=t.fct_infreq("species"))) == ["Gentoo", "Chinstrap", "Adelie", "unused"]
    assert levels(frame >> t.mutate(species=t.fct_infreq("species", w="mass"))) == ["Gentoo", "Chinstrap", "Adelie", "unused"]
    assert levels(frame >> t.mutate(species=t.fct_rev(t.fct_infreq("species")))) == ["unused", "Adelie", "Chinstrap", "Gentoo"]
    with pytest.raises(ValueError, match="non-negative"):
        frame >> t.mutate(species=t.fct_infreq("species", w=-t.col("mass")))
    weighted = t.tidy({"species": ["Gentoo", "Gentoo", "Adelie"], "weight": [1., 1., 10.]}, backend=backend)
    assert levels(weighted >> t.mutate(species=t.fct_infreq("species", "weight"))) == ["Adelie", "Gentoo"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_grouped_relevel_warns_once_per_distinct_missing_set(backend):
    frame = t.tidy({"g": [1, 1, 2, 3, 4], "species": ["a", "b", "a", "b", "a"]}, backend=backend)
    with pytest.warns(UserWarning) as recorded:
        out = frame >> t.group_by("g") >> t.mutate(species=t.fct_relevel("species", "b", "missing"))
    assert [str(w.message) for w in recorded] == [
        "fct_relevel(): unknown levels: ['missing']",
        "fct_relevel(): unknown levels: ['b', 'missing']",
    ]
    assert levels(out) == ["b", "a"]
    assert out.collect(as_="pandas")["species"].tolist() == ["a", "b", "a", "b", "a"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_grouped_string_factors_keep_merged_level_order(backend):
    frame = t.tidy({
        "g": [2, 2, 1, 1, 1],
        "species": ["b", "b", "a", "b", "a"],
        "mass": [1, 1, 10, 0, 10],
    }, backend=backend)
    grouped = frame >> t.group_by("g")
    assert levels(grouped >> t.mutate(species=t.fct_infreq("species"))) == ["a", "b"]
    assert levels(grouped >> t.mutate(species=t.fct_reorder("species", "mass"))) == ["b", "a"]
    assert levels(grouped >> t.mutate(species=t.fct_rev("species"))) == ["b", "a"]
    assert levels(grouped >> t.mutate(species=t.fct_relevel("species", "b"))) == ["b", "a"]
    with pytest.warns(UserWarning, match=r"unknown levels: \['a'\]"):
        moved = grouped >> t.mutate(species=t.fct_relevel("species", "a", after=math.inf))
    assert levels(moved) == ["b", "a"]
    plain = t.tidy({
        "species": ["b", "a", "b", None],
        "mass": [1, 5, 1, 3],
    }, backend=backend)
    assert levels(plain >> t.mutate(species=t.fct_infreq("species"))) == ["b", "a"]
    assert levels(plain >> t.mutate(species=t.fct_reorder("species", "mass"))) == ["b", "a"]
    assert levels(plain >> t.mutate(species=t.fct_rev("species"))) == ["b", "a"]


@pytest.mark.parametrize(
    ("fun", "expected"),
    [(t.median, ["a", "b", "c"]), (t.mean, ["a", "b", "c"]), (t.sum, ["a", "c", "b"])],
)
def test_reorder_skips_nan_values_on_both_backends(fun, expected):
    data = {"species": ["a", "a", "b", "b", "c"], "mass": [1.0, float("nan"), 2.0, 2.0, 3.0]}
    found = {}
    for backend in BACKENDS:
        with pytest.warns(UserWarning, match="removing missing x values"):
            frame = t.tidy(data, backend=backend) >> t.mutate(species=t.fct_reorder("species", "mass", fun))
            found[backend] = levels(frame)
    # The NaN is skipped, so "a" summarizes to 1.
    assert found["polars"] == found["pandas"] == expected


@pytest.mark.parametrize("backend", BACKENDS)
def test_reorder_summary_and_missing_values(backend):
    frame = penguins(backend)
    assert levels(frame >> t.mutate(species=t.fct_reorder("species", "mass"))) == ["Adelie", "Chinstrap", "Gentoo", "unused"]
    assert levels(frame >> t.mutate(species=t.fct_reorder("species", "mass", t.mean, desc=True))) == ["unused", "Gentoo", "Chinstrap", "Adelie"]
    assert levels(frame >> t.mutate(species=t.fct_reorder("species", "mass", lambda x: x.max(), default=-math.inf))) == ["unused", "Adelie", "Chinstrap", "Gentoo"]
    frame = t.tidy({"species": ["Adelie", "Gentoo", "Gentoo"], "mass": [3500., None, 5000.]}, backend=backend)
    with pytest.warns(UserWarning, match="missing"):
        assert levels(frame >> t.mutate(species=t.fct_reorder("species", "mass"))) == ["Adelie", "Gentoo"]
    assert levels(frame >> t.mutate(species=t.fct_reorder("species", "mass", na_rm=False, desc=True))) == ["Adelie", "Gentoo"]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("transient", [False, True])
def test_grouped_character_factors_reconcile_in_group_order(backend, transient):
    frame = t.tidy({"island": ["b", "a", "b", "a"], "species": ["Gentoo", "Adelie", "Gentoo", "Adelie"]}, backend=backend)
    if transient:
        out = frame >> t.mutate(species=t.fct_rev(t.fct_infreq("species")), by="island")
        expected = ["Gentoo", "Adelie"]
    else:
        out = frame >> t.group_by("island") >> t.mutate(species=t.fct_rev(t.fct_infreq("species")))
        expected = ["Adelie", "Gentoo"]
    assert levels(out) == expected
    assert t.group_vars(out) == ([] if transient else ["island"])


@pytest.mark.parametrize("backend", BACKENDS)
def test_across_sequential_assignments_and_type_selection(backend):
    frame = penguins(backend) >> t.mutate(t.across("species", t.fct_rev))
    assert levels(frame) == ["unused", "Gentoo", "Adelie", "Chinstrap"]
    out = frame >> t.mutate(ordered=t.fct_infreq("species"), reversed=t.fct_rev("ordered"))
    assert levels(out, "reversed") == list(reversed(levels(out, "ordered")))
    assert (out >> t.select(t.where(t.is_categorical))).columns == ["species", "ordered", "reversed"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_empty_and_all_missing_factors(backend):
    for values in [[], [None, None]]:
        frame = t.tidy(pd.DataFrame({"species": pd.Categorical(values, categories=["Adelie", "Gentoo"])}), backend=backend)
        assert levels(frame >> t.mutate(species=t.fct_rev("species"))) == ["Gentoo", "Adelie"]


def test_polars_factor_stage_keeps_source_snapshot_and_other_dtypes(tmp_path):
    path = tmp_path / "penguins.csv"
    path.write_text("species,mass\nAdelie,3500\nGentoo,5000\n")
    frame = t.scan_csv(path) >> t.mutate(species=t.fct_rev("species"))
    path.unlink()
    assert frame.collect()["mass"].to_list() == [3500, 5000]
    assert frame.collect().schema["mass"] == pl.Int64
    assert isinstance(frame._lf, pl.LazyFrame)


def test_vectors_and_invalid_inputs():
    result = t.fct_rev(["Gentoo", "Adelie", None])
    assert result.categories.tolist() == ["Gentoo", "Adelie"]
    with pytest.raises(TypeError, match="character"):
        t.fct_rev([1, 2])
    with pytest.raises(ValueError, match="after"):
        t.fct_relevel("species", "Gentoo", after=-1)


@pytest.mark.parametrize("backend", BACKENDS)
def test_factor_order_survives_plot_handoff(backend):
    from plot3 import aes, geom_point

    frame = penguins(backend) >> t.mutate(species=t.fct_relevel("species", "Gentoo"))
    plot = frame.ggplot(aes(x="species", y="mass")) + geom_point()
    assert plot.data.to_pandas()["species"].cat.categories.tolist() == ["Gentoo", "Chinstrap", "Adelie", "unused"]
    assert len(plot.html()) > 100


def test_factor_notebook_syntax():
    source = "penguins >> mutate(species=fct_reorder(species, mass, mean, na_rm=True))"
    code = t.transform_source(source)
    namespace = {name: getattr(t, name) for name in t.__all__}
    namespace["penguins"] = penguins("polars")
    exec("result = " + code.strip(), namespace)
    assert levels(namespace["result"]) == ["Adelie", "Chinstrap", "Gentoo", "unused"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_grouped_factor_values_follow_positions_with_duplicate_index(backend):
    pdf = pd.DataFrame({
        "island": ["b", "a", "b", "a"],
        "species": ["Gentoo", "Adelie", "Adelie", "Gentoo"],
    }, index=[0, 0, 1, 1])
    out = t.tidy(pdf, backend=backend) >> t.group_by("island") >> t.mutate(species=t.fct_rev("species"))
    assert out.collect(as_="pandas")["species"].tolist() == pdf["species"].tolist()
    assert levels(out) == ["Gentoo", "Adelie"]


def test_factor_snapshot_keeps_parse_problems(tmp_path):
    path = tmp_path / "penguins.csv"
    path.write_text("species,mass\nAdelie,3500\nGentoo,oops\n")
    frame = t.scan_csv(path) >> t.mutate(mass=t.parse_double("mass"))
    frame = frame >> t.mutate(species=t.fct_rev("species"))
    path.unlink()
    assert t.problems(frame).collect()["original"].to_list() == ["oops"]
    assert frame.collect()["mass"].to_list() == [3500., None]


def test_pandas_factors_keep_their_levels_when_leaving_pandas():
    frame = t.tidy({"species": ["Adelie", "Chinstrap", "Gentoo", None]}, backend="pandas")
    moved = frame >> t.mutate(species=t.fct_relevel("species", "Gentoo"))
    wanted = pl.Enum(["Gentoo", "Adelie", "Chinstrap"])
    assert moved.collect()["species"].dtype == wanted
    assert t.tidy(moved, backend="polars").collect()["species"].dtype == wanted
    assert moved.collect(as_="arrow").column("species").to_pylist() == ["Adelie", "Chinstrap", "Gentoo", None]
    # Imported pandas factors keep unused levels and the ordered flag.
    imported = t.tidy(pd.DataFrame({
        "species": pd.Categorical(["b", "a"], categories=["z", "b", "a"], ordered=True),
    }))
    assert imported.collect()["species"].dtype == pl.Enum(["z", "b", "a"])
    back = imported.collect(as_="pandas")["species"]
    assert back.cat.categories.tolist() == ["z", "b", "a"] and back.cat.ordered
    assert (imported >> t.arrange("species")).collect()["species"].to_list() == ["b", "a"]
