"""tidy3 imports and runs without pandas; pandas is only for pandas input and output."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"

SCRIPT = textwrap.dedent(
    """
    import builtins, sys, warnings
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "pandas" or name.startswith("pandas."):
            raise ImportError("pandas is not installed")
        return real_import(name, *args, **kwargs)

    builtins.__import__ = blocked
    sys.path.insert(0, sys.argv[1])
    import polars as pl
    import tidy3 as t

    frame = t.tidy({
        "species": ["Gentoo", "Adelie", "Adelie", None, "Chinstrap", "Gentoo", "Adelie"],
        "g": ["x", "x", "y", "y", "y", "x", "y"],
        "mass": [5.0, 3.0, 3.5, 4.0, 3.7, 5.5, float("nan")],
        "when": ["2024-02-29", "2024-03-10", None, "2023-12-31", "2024-01-15", "2024-07-04", "2024-11-03"],
    })
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nested = (frame >> t.mutate(f=t.fct_rev(t.fct_infreq("species")))).collect()
        grouped = (frame >> t.group_by("g") >> t.mutate(f=t.fct_lump_n(t.fct_infreq("species"), 1))).collect()
        reordered = (frame >> t.mutate(f=t.fct_reorder("species", "mass", lambda x: x.max()))).collect()
    assert nested.schema["f"] == pl.Enum(["Chinstrap", "Gentoo", "Adelie"]), nested.schema["f"]
    assert grouped.schema["f"] == pl.Enum(["Gentoo", "Other", "Adelie"]), grouped.schema["f"]
    assert reordered.schema["f"] == pl.Enum(["Adelie", "Chinstrap", "Gentoo"]), reordered.schema["f"]
    other = (frame >> t.mutate(
        s=t.str_c("species", "g", sep="-"), d=t.month(t.ymd("when")),
        i=t.as_integer(t.round("mass"), strict=False), c=t.as_character(t.fct_recode("species", Big="Gentoo")),
    )).collect()
    assert other.schema["c"] == pl.String and other["c"][0] == "Big"
    vector = t.fct_lump_n(["a", "a", "b", "c"], 1)
    assert isinstance(vector, pl.Series) and vector.dtype == pl.Enum(["a", "Other"])
    assert "pandas" not in sys.modules
    print("ok")
    """
)


def test_tidy3_runs_without_pandas():
    result = subprocess.run(
        [sys.executable, "-I", "-c", SCRIPT, str(SRC)], capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
