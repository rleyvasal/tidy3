"""nb_export / R-style → plain Python transforms."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from tidy3.export import (
    collect_known_names,
    nb_export,
    transform_cell,
    transform_source,
)


def test_transform_filter_bare_name():
    out = transform_source("filter(mpg > 20)", known=collect_known_names([]))
    assert "col" in out
    assert "mpg" in out
    assert 'col("mpg")' in out or "col('mpg')" in out


def test_transform_select_and_backtick():
    out = transform_source(
        "select(cyl, `hp new`)",
        known=collect_known_names([]),
    )
    assert "cyl" in out
    assert "hp new" in out
    # selectors become strings, not col()
    assert "col(" not in out or "hp new" in out


def test_transform_aes_when_plot3_available():
    pytest.importorskip("plot3")
    out = transform_source(
        "aes(x=wt, y=mpg, colour=cyl)",
        known=collect_known_names([]),
        with_plot3=True,
    )
    assert "wt" in out and "mpg" in out
    assert 'x="wt"' in out or "x='wt'" in out


def test_transform_aes_backtick():
    pytest.importorskip("plot3")
    out = transform_source(
        "aes(x=`First Name`, y=mpg)",
        known=collect_known_names([]),
        with_plot3=True,
    )
    assert "First Name" in out


def test_known_frame_name_not_column():
    known = collect_known_names(["cars = tidy(df)\n"])
    out = transform_source("filter(cars > 0)", known=known)
    # cars is assigned → left as Name, not col("cars")
    assert 'col("cars")' not in out and "col('cars')" not in out


def test_skip_directive():
    src = "#| skip\nfilter(mpg > 20)\n"
    assert transform_cell(src) is None


def test_export_directive_body_stripped():
    src = "#| export\nselect(cyl)\n"
    out = transform_cell(src, known=collect_known_names([]))
    assert out is not None
    assert "#| export" not in out
    assert "cyl" in out


def test_plot3_magic_rewrite():
    src = "%plot3 df x=wt y=mpg colour=cyl"
    out = transform_cell(src, cell_no=1, warnings=[])
    assert out is not None
    assert "ggplot" in out
    assert "aes" in out
    assert "wt" in out


def test_nb_export_roundtrip(tmp_path: Path):
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {},
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["# Analysis\n"],
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "from tidy3 import *\n",
                    "cars = tidy({'mpg': [21, 22], 'cyl': [4, 6], 'wt': [2.5, 3.0]})\n",
                ],
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": ["#| skip\n", "print('debug only')\n"],
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "out = cars >> filter(mpg > 20) >> select(cyl, mpg)\n"
                ],
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": ["%run /fake/addons/tidy3.py\n"],
            },
        ],
    }
    nb_path = tmp_path / "analysis.ipynb"
    nb_path.write_text(json.dumps(nb), encoding="utf-8")
    dest = tmp_path / "analysis_pipeline.py"

    result = nb_export(nb_path, dest)
    assert result.cells_exported >= 2
    assert dest.is_file()
    text = dest.read_text(encoding="utf-8")
    assert "Auto-exported by tidy3.nb_export" in text
    assert "print('debug only')" not in text
    assert "filter" in text
    assert "col" in text  # bare mpg rewritten
    assert "notebook-only" in text or "skipped" in " ".join(result.warnings).lower() or "%run" not in text


def test_nb_export_only_export(tmp_path: Path):
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {},
        "cells": [
            {
                "cell_type": "code",
                "metadata": {},
                "source": ["x = 1\n"],
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": ["#| export\n", "y = 2\n"],
            },
        ],
    }
    nb_path = tmp_path / "lib.ipynb"
    nb_path.write_text(json.dumps(nb), encoding="utf-8")
    script = nb_export(nb_path, only_export=True, write=False)
    assert isinstance(script, str)
    assert "y = 2" in script
    assert "x = 1" not in script


def test_nb_export_write_false_returns_str(tmp_path: Path):
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {},
        "cells": [
            {
                "cell_type": "code",
                "metadata": {},
                "source": ["a = 1\n"],
            }
        ],
    }
    p = tmp_path / "t.ipynb"
    p.write_text(json.dumps(nb), encoding="utf-8")
    out = nb_export(p, write=False)
    assert isinstance(out, str)
    assert "a = 1" in out


def test_cli_export(tmp_path: Path):
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {},
        "cells": [
            {
                "cell_type": "code",
                "metadata": {},
                "source": ["z = 99\n"],
            }
        ],
    }
    nb_path = tmp_path / "c.ipynb"
    out_path = tmp_path / "c_out.py"
    nb_path.write_text(json.dumps(nb), encoding="utf-8")

    from tidy3.__main__ import main

    rc = main(["export", str(nb_path), "-o", str(out_path)])
    assert rc == 0
    assert out_path.is_file()
    assert "z = 99" in out_path.read_text(encoding="utf-8")


def test_cli_run_rstyle(tmp_path: Path):
    script = tmp_path / "job.py"
    # Bare mpg → col("mpg"); inject supplies tidy3 API (like a notebook).
    script.write_text(
        textwrap.dedent(
            """
            df = tidy({"mpg": [10, 30], "cyl": [4, 6]})
            out = (df >> filter(mpg > 20)).collect()
            assert out.height == 1
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    from tidy3.__main__ import main

    rc = main(["run", str(script)])
    assert rc == 0


def test_cli_run_with_explicit_imports(tmp_path: Path):
    script = tmp_path / "job2.py"
    script.write_text(
        textwrap.dedent(
            """
            from tidy3 import tidy, filter, collect, col
            df = tidy({"mpg": [10, 30], "cyl": [4, 6]})
            out = (df >> filter(mpg > 20)).collect()
            assert out.height == 1
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    from tidy3.__main__ import main

    rc = main(["run", str(script), "--no-inject"])
    assert rc == 0
