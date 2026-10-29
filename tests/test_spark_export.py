"""to_spark(): pipelines run on Spark and match tidy3, or are refused for a stated reason."""

from __future__ import annotations

import os
import shutil
import warnings

import polars as pl
import pytest
from polars.testing import assert_frame_equal

pytest.importorskip("pyspark")
if not (shutil.which("java") or os.environ.get("JAVA_HOME")):
    pytest.skip("Spark needs Java", allow_module_level=True)

import tidy3 as t
from tidy3 import *  # noqa: F403
from tidy3 import bench_api as api
from tidy3.frame import TidyFrame
from tidy3.spark_export import SparkExportError

PENGUINS = """species,island,bill_length_mm,body_mass_g,sex,year
Adelie,Torgersen,39.1,3750,male,2007
Adelie,Torgersen,39.5,3800,female,2007
Adelie,Torgersen,NA,NA,NA,2007
Adelie,Biscoe,37.8,3400,female,2008
Adelie,Dream,41.1,4050,male,2009
Adelie,Dream,36.0,3450,female,2009
Gentoo,Biscoe,46.1,4500,female,2007
Gentoo,Biscoe,50.0,5700,male,2007
Gentoo,Biscoe,48.7,5350,male,2008
Gentoo,Biscoe,47.5,NA,NA,2009
Gentoo,Biscoe,49.9,5400,male,2009
Gentoo,Biscoe,45.2,4750,female,2009
Chinstrap,Dream,46.5,3500,female,2007
Chinstrap,Dream,50.0,3900,male,2007
Chinstrap,Dream,51.3,3650,male,2008
Chinstrap,Dream,45.4,3525,female,2009
Chinstrap,Dream,52.7,3725,male,2009
Chinstrap,Dream,45.2,3700,female,2009
"""


@pytest.fixture(scope="module")
def spark():
    from pyspark.sql import SparkSession

    session = (
        SparkSession.builder.master("local[2]")
        .appName("tidy3-tests")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


@pytest.fixture(scope="module")
def penguins(tmp_path_factory):
    path = tmp_path_factory.mktemp("spark") / "penguins.csv"
    path.write_text(PENGUINS)
    return scan_csv(str(path), null_values="NA")


def _on_spark(frame, spark) -> pl.DataFrame:
    """Run *frame* on Spark and bring the rows back in tidy3's column types."""
    result = pl.from_arrow(frame.to_spark(spark).toArrow())
    schema = frame._lf.collect_schema()
    return result.cast({name: dtype for name, dtype in schema.items() if result.schema[name] != dtype})


def _same(frame, spark):
    """Spark gives tidy3's rows, in tidy3's order, to within float rounding."""
    expected = frame.collect(as_="polars")
    assert_frame_equal(_on_spark(frame, spark), expected, check_exact=False, rel_tol=1e-12, abs_tol=1e-12)


ISLANDS = tidy({"island": ["Biscoe", "Dream", "Torgersen"], "zone": ["south", "north", "north"]})

PIPELINES = {
    "readme_summary": lambda p: (
        p >> drop_na("sex") >> group_by("species", "sex")
        >> summarise(n=n(), bill_mm=mean("bill_length_mm").round(1),
                     mass_kg=(mean("body_mass_g") / 1000).round(2))
        >> arrange(desc("mass_kg"))
    ),
    "filter_select_mutate": lambda p: (
        p >> filter(col("species") == "Gentoo", col("body_mass_g") > 5000)
        >> select("species", "island", ends_with("_mm"))
        >> mutate(per_kg=col("bill_length_mm") / 10)
    ),
    "count_sort": lambda p: p >> count("species", "island", sort=True),
    "grouped_mutate": lambda p: (
        p >> group_by("species")
        >> mutate(mass_z=(col("body_mass_g") - mean("body_mass_g", na_rm=True)) / sd("body_mass_g", na_rm=True))
        >> ungroup() >> select("species", "body_mass_g", "mass_z")
    ),
    "na_rm_mean": lambda p: (
        p >> group_by("island")
        >> summarise(mass=mean("body_mass_g", na_rm=True), mass_na=mean("body_mass_g"),
                     heaviest=max("body_mass_g", na_rm=True))
    ),
    "case_when": lambda p: (
        p >> mutate(size=case_when((col("body_mass_g") >= 5000, "large"),
                                   (col("body_mass_g") >= 3600, "medium"), default="small"))
        >> count("size")
    ),
    "left_join": lambda p: p >> left_join(ISLANDS, by="island") >> count("zone", "species"),
    "distinct_slice": lambda p: p >> distinct("species", "island") >> arrange("species", "island") >> slice_head(n=4),
    "across_mean": lambda p: p >> group_by("species") >> summarise(across(ends_with("_mm"), lambda x: mean(x, na_rm=True))),
    "rename_relocate": lambda p: (
        p >> rename(mass="body_mass_g") >> relocate("mass") >> filter(col("year") == 2009)
        >> select("mass", "species", "year")
    ),
    "ungrouped_summary": lambda p: p >> summarise(n=n(), mass=mean("body_mass_g", na_rm=True)),
    "str_detect": lambda p: p >> filter(str_detect(col("species"), "^G")) >> count("species"),
    "slice_max": lambda p: (
        p >> group_by("species") >> slice_max("body_mass_g", n=2) >> ungroup() >> select("species", "body_mass_g")
    ),
    "row_number": lambda p: p >> group_by("species") >> mutate(i=row_number()) >> ungroup() >> select("species", "i"),
    "semi_join": lambda p: p >> semi_join(ISLANDS >> filter(col("zone") == "north"), by="island") >> count("island"),
}


@pytest.mark.parametrize("name", list(PIPELINES))
def test_penguin_pipelines_match_tidy3(name, penguins, spark):
    _same(PIPELINES[name](penguins), spark)


EDGES = tidy({
    "g": ["a", "a", "b", "b", None, "c"],
    "x": [2.5, 0.125, float("nan"), None, 1.0, -2.5],
    "k": [3, 1, 2, 1, 3, 2],
    "s": ["one", "two", "three", "four", "five", "six"],
})


@pytest.mark.parametrize(
    "frame",
    [
        # tidy3's round() is half to even: bround, not Spark's round.
        EDGES >> mutate(r0=col("x").round(0), r2=col("x").round(2)) >> select("s", "r0", "r2"),
        # arrange() puts NaN with the missing values, last.
        EDGES >> arrange(desc("x")) >> select("s", "x"),
        # Stable sort: ties keep their input order.
        EDGES >> arrange("k") >> select("k", "s"),
        EDGES >> distinct("k", keep_all=True),
        EDGES >> group_by("g") >> summarise(n=n(), x=mean("x", na_rm=True)),
        # Polars' remainder takes the divisor's sign; Spark's takes the dividend's.
        tidy({"a": [7, -7, 7, -7], "b": [3, 3, -3, -3]}) >> mutate(m=col("a") % col("b")),
        tidy({"u": pl.Series([1, 2, 3], dtype=pl.UInt32)}) >> mutate(v=col("u") * 2),
    ],
    ids=["round_ties", "arrange_nan_null", "stable_ties", "distinct_first", "null_group_key", "modulo", "unsigned"],
)
def test_edge_cases_match_tidy3(frame, spark):
    _same(frame, spark)


def test_a_multi_file_scan_keeps_file_order(tmp_path, spark):
    # Spark orders a scan's files by size; Polars reads them in name order.
    (tmp_path / "a.csv").write_text("x\n1\n2\n")
    (tmp_path / "b.csv").write_text("x\n" + "\n".join(str(i) for i in range(3, 400)) + "\n")
    _same(scan_csv(str(tmp_path / "*.csv")) >> mutate(y=col("x") * 2), spark)


def test_a_parquet_scan_keeps_polars_types_and_order(tmp_path, spark):
    path = tmp_path / "data.parquet"
    pl.DataFrame({"k": pl.Series(range(5000), dtype=pl.UInt32), "g": ["a", "b"] * 2500}).write_parquet(path, row_group_size=500)
    _same(scan_parquet(str(path)) >> group_by("g") >> mutate(c=col("k").cum_sum()) >> slice_head(n=20), spark)


def test_a_python_function_step_is_refused(spark):
    frame = tidy({"s": ["1,000", "2"]}) >> mutate(n=parse_number(col("s")))
    with pytest.raises(SparkExportError, match="Python function"):
        frame.to_spark(spark)


def test_refusals_are_export_errors(penguins, spark):
    frame = penguins >> count("species", "island") >> pivot_wider(names_from="island", values_from="n")
    with pytest.raises(t.ExportError, match="no Spark translation"):
        frame.to_spark(spark)


def test_spark_is_lazy_until_rows_are_asked_for(penguins, spark):
    from pyspark.sql import DataFrame

    frame = PIPELINES["readme_summary"](penguins)
    assert isinstance(frame.to_spark(spark), DataFrame)


# ── every benchmark case ────────────────────────────────────────────────

DATA = api.work(64)
CASES = api._cases()


def _final_frame(case, monkeypatch):
    """The TidyFrame a benchmark case collects, and what tidy3 returned."""
    seen: dict[str, object] = {}
    collect = TidyFrame.collect

    def capture(self, *args, **kwargs):
        out = collect(self, *args, **kwargs)
        if "frame" not in seen and isinstance(out, pl.DataFrame):
            seen["frame"], seen["out"] = self, out
        return out

    monkeypatch.setattr(TidyFrame, "collect", capture)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        case.tidy(DATA, case.prepare(DATA) if case.prepare else None)
    monkeypatch.setattr(TidyFrame, "collect", collect)
    return seen.get("frame"), seen.get("out")


@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
def test_every_benchmark_case_matches_on_spark_or_is_refused(case, monkeypatch, spark):
    frame, expected = _final_frame(case, monkeypatch)
    if frame is None:
        pytest.skip("not a frame query")
    try:
        got = _on_spark(frame, spark)
    except SparkExportError:
        return
    assert_frame_equal(got, expected, check_exact=False, rel_tol=1e-12, abs_tol=1e-12)


@pytest.mark.parametrize("grouped", [True, False], ids=["grouped", "ungrouped"])
def test_an_aggregation_of_an_aggregation(grouped, penguins, spark):
    # Polars broadcasts the inner min() within the group; Spark needs a window.
    frame = penguins >> group_by("species") if grouped else penguins
    _same(frame >> summarise(spread=sum(col("body_mass_g") - min("body_mass_g", na_rm=True), na_rm=True)), spark)
