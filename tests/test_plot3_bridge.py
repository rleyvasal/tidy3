"""plot3 handoff (optional — skipped if plot3 not on path)."""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import pytest

PLOT3_ROOT = Path.home() / "plot3"
if PLOT3_ROOT.is_dir() and str(PLOT3_ROOT) not in sys.path:
    sys.path.insert(0, str(PLOT3_ROOT))

pytest.importorskip("plot3")

from plot3 import aes, geom_point, ggplot  # noqa: E402

from tidy3 import col, filter, select, tidy  # noqa: E402
from tidy3.partial_run import maybe_rewrite_cell  # noqa: E402


def test_ggplot_method_builds_html():
    df = pl.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 3.0],
            "y": [0.1, 0.2, 0.3, 0.4],
            "c": ["a", "a", "b", "b"],
        }
    )
    g = (
        tidy(df)
        >> filter(col("x") >= 0)
        >> select("x", "y", "c")
    ).ggplot(aes(x="x", y="y", colour="c")) + geom_point(size=4)
    html = g.html()
    assert "plot3" in html.lower() or "three" in html.lower() or len(html) > 100


def test_pipeable_ggplot_binds_tidyframe_after_layers():
    df = pl.DataFrame({"x": [0.0, 1.0], "y": [0.1, 0.2]})
    template = ggplot(aes(x="x", y="y")) + geom_point(size=4)

    g = tidy(df) >> template

    assert template.data is None
    assert g.data["x"].tolist() == [0.0, 1.0]
    assert len(g.layers) == 1
    assert len(g.html()) > 100


def test_pipeable_ggplot_operator_precedence_and_direct_frames():
    pdf = pl.DataFrame({"x": [0.0, 1.0], "y": [0.1, 0.2]}).to_pandas()

    from_pandas = pdf >> ggplot(aes(x="x", y="y")) + geom_point()
    from_polars = pl.from_pandas(pdf) >> ggplot(aes(x="x", y="y")) + geom_point()

    assert from_pandas.data.equals(pdf)
    assert from_polars.data.equals(pdf)


def test_unbound_ggplot_has_clear_error():
    with pytest.raises(ValueError, match="no data"):
        (ggplot(aes(x="x", y="y")) + geom_point()).html()


def test_multiline_pipe_into_ggplot_is_rewritten_and_evaluated():
    source = '''
tidy(df)
>> ggplot(aes(x="x", y="y"))
+ geom_point()
'''
    rewritten = maybe_rewrite_cell(source)
    assert rewritten is not None
    namespace = {
        "df": pl.DataFrame({"x": [1.0], "y": [2.0]}),
        "tidy": tidy,
        "ggplot": ggplot,
        "aes": aes,
        "geom_point": geom_point,
    }

    result = eval(compile(rewritten, "<plot-pipe>", "eval"), namespace)

    assert result.data.shape == (1, 2)
    assert len(result.layers) == 1
