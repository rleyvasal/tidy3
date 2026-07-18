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

from plot3 import aes, geom_point  # noqa: E402

from tidy3 import col, filter, select, tidy  # noqa: E402


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
