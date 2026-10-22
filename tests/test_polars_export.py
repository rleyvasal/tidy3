"""to_polars_code(): exported pipelines run on Polars alone and match tidy3."""

from __future__ import annotations

import warnings

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import tidy3 as t
from tidy3 import bench_api as api
from tidy3.frame import TidyFrame
from tidy3.polars_export import _export

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


def _run(code: str, inputs: list[tuple[str, pl.DataFrame]]) -> pl.DataFrame:
    namespace: dict[str, object] = {}
    exec(code, namespace)
    return namespace["pipeline"](*[data for _name, data in inputs])


@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
def test_every_benchmark_case_exports_or_is_refused_for_a_stated_reason(case, monkeypatch):
    frame, expected = _final_frame(case, monkeypatch)
    if frame is None:
        pytest.skip("not a frame query")
    try:
        code, inputs = _export(frame)
    except t.ExportError as error:
        message = str(error)
        assert "has no exporter" not in message, message
        assert "Python function" in message or "values in the data" in message, message
        return
    assert "import tidy3" not in code and "tidy3." not in code
    assert_frame_equal(_run(code, inputs), expected)


def test_exported_code_is_a_plain_polars_function():
    frame = (
        t.tidy({"g": ["a", "b", "a"], "x": [1.0, 2.0, None]})
        >> t.filter(t.col("x") > 0)
        >> t.group_by("g")
        >> t.summarise(m=t.mean("x"), n=t.n())
        >> t.arrange(t.desc("m"))
    )
    code = frame.to_polars_code(name="summarise_x")
    assert code.startswith("import polars as pl\n")
    assert "def summarise_x(df: pl.DataFrame) -> pl.DataFrame:" in code
    assert "df: g (String), x (Float64)" in code
    namespace: dict[str, object] = {}
    exec(code, namespace)
    source = pl.DataFrame({"g": ["a", "b", "a"], "x": [1.0, 2.0, None]})
    assert_frame_equal(namespace["summarise_x"](source), frame.collect())
    assert t.to_polars_code(frame, name="summarise_x") == code
    with pytest.raises(ValueError, match="identifier"):
        frame.to_polars_code(name="not valid")


def test_joined_frames_become_arguments_and_lookup_tables_stay_inline():
    left = pl.DataFrame({"id": [1, 2, 3], "x": [10, 20, 30]})
    right = pl.DataFrame({"id": [1, 3], "w": ["a", "b"]})
    code, inputs = _export(t.tidy(left) >> t.left_join(right, by="id"))
    assert [name for name, _data in inputs] == ["df", "df2"]
    assert "def pipeline(df: pl.DataFrame, df2: pl.DataFrame)" in code
    sliced, inputs = _export(t.tidy(left) >> t.slice(1, 3))
    assert [name for name, _data in inputs] == ["df"]
    assert_frame_equal(_run(sliced, inputs), (t.tidy(left) >> t.slice(1, 3)).collect())


def test_python_function_columns_are_refused_with_the_column_and_function_named():
    frame = t.tidy({"s": ["2024-01-02"]}) >> t.mutate(d=t.ymd("s"))
    with pytest.raises(t.ExportError, match=r"column 'd' is computed by a Python function \(a text conversion"):
        frame.to_polars_code()
    rounded = t.tidy({"x": [2.675]}) >> t.mutate(r=t.round("x", 2))
    with pytest.raises(t.ExportError, match=r"column 'r' .*round\(\)"):
        rounded.to_polars_code()


def test_type_dependent_helpers_compile_to_plain_polars():
    import datetime as dt

    frame = t.tidy({
        "x": [1.0, None, float("nan")],
        "n": [1, None, 3],
        "at": [dt.datetime(2024, 1, 1, 5, 6, 7), None, dt.datetime(2024, 3, 1)],
        "day": [dt.date(2024, 1, 1), None, dt.date(2024, 2, 1)],
    }) >> t.mutate(
        missing=t.is_na("x"), finite=t.is_finite("n"), low=t.pmin("x", "n"), total=t.row_sums("x", "n", na_rm=True),
        hour=t.hour("at"), second=t.second("at"), up=t.ceiling_date("day", "month"), zoned=t.with_tz("at", "UTC"),
    )
    assert "python_udf" not in frame.explain(optimized=False)
    code, inputs = _export(frame)
    assert_frame_equal(_run(code, inputs), frame.collect())


def test_row_checks_are_skipped_in_exported_code():
    rows = pl.DataFrame({"id": [1, 2, 3], "x": [1.0, 2.0, 3.0]})
    frame = t.tidy(rows) >> t.rows_update(pl.DataFrame({"id": [2], "x": [9.0]}), by="id")
    code, inputs = _export(frame)
    assert_frame_equal(_run(code, inputs), frame.collect())


@pytest.mark.parametrize(
    "build, reason",
    [
        (lambda f: f >> t.mutate(g=t.fct_infreq("g")), "factor levels"),
        (lambda f: f >> t.distinct("g", keep_all=True) >> t.pivot_wider(names_from="g", values_from="x"), "pivot_wider"),
        (lambda f: f.compute() >> t.filter(t.col("x") > 0), "compute()"),
        (lambda f: f >> t.remove_constant(), "constant columns"),
    ],
    ids=["factor", "pivot_wider", "compute", "remove_constant"],
)
def test_steps_that_read_the_data_are_refused_with_the_step_named(build, reason):
    frame = t.tidy({"g": ["a", "b", "a"], "x": [1.0, 2.0, 3.0]})
    with pytest.raises(t.ExportError, match="values in the data") as error:
        build(frame).to_polars_code()
    assert reason in str(error.value)


def test_a_data_read_in_a_joined_frame_blocks_the_join():
    left = t.tidy({"g": ["a", "b"], "x": [1, 2]})
    right = t.tidy({"g": ["a", "b"], "f": ["p", "q"]}) >> t.mutate(f=t.fct_rev("f"))
    joined = left >> t.left_join(right, by="g") >> t.select("g", "f")
    with pytest.raises(t.ExportError, match="factor levels"):
        joined.to_polars_code()
    assert (left >> t.left_join(t.tidy({"g": ["a"], "y": [1]}), by="g")).to_polars_code()
