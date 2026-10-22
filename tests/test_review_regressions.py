"""Regression cases from the October correctness review."""

from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import polars as pl
import pytest

import tidy3 as t


@pytest.mark.parametrize("kind,valid,dtype", [
    ("integer", "110", pl.Int64),
    ("double", "21.5", pl.Float64),
    ("logical", "TRUE", pl.Boolean),
])
def test_guessed_csv_types_record_late_failures(tmp_path, kind, valid, dtype):
    path = tmp_path / "cars.csv"
    path.write_text("hp\n" + f"{valid}\n" * 100 + "oops\n")
    frame = t.read_csv(path)
    path.unlink()  # problems must use the same snapshot as the data
    data = frame.collect()
    assert data["hp"].dtype == dtype
    assert data["hp"][-1] is None
    assert t.problems(frame).collect().to_dicts() == [
        {"row": 102, "column": "hp", "original": "oops", "expected": kind}
    ]


def test_guessed_csv_types_honor_strict(tmp_path):
    path = tmp_path / "cars.csv"
    path.write_text("hp\n" + "110\n" * 100 + "oops\n")
    with pytest.raises(t.ConversionError, match="oops"):
        t.read_csv(path, strict=True)


@pytest.mark.parametrize("grouped", [False, True])
def test_string_helpers_preserve_summary_size(grouped):
    cars = t.tidy({"cyl": [4, 4, 6], "model": ["Datsun", "Toyota", "Mazda"]})
    if grouped:
        cars = cars >> t.group_by("cyl")
    out = cars >> t.summarise(
        length=t.str_length(t.first("model")),
        upper=t.str_to_upper(t.first("model")),
        label=t.str_c(t.first("model"), "!"),
        equal=t.str_equal(t.first("model"), t.first("model")),
    )
    data = out.collect(as_="pandas")
    assert data["length"].tolist() == ([6, 5] if grouped else [6])
    assert data["upper"].tolist() == (["DATSUN", "MAZDA"] if grouped else ["DATSUN"])
    assert data["label"].tolist() == (["Datsun!", "Mazda!"] if grouped else ["Datsun!"])
    assert data["equal"].tolist() == ([True, True] if grouped else [True])


def test_round_preserves_float32_and_missing_values():
    frame = t.tidy(pl.DataFrame({"x": pl.Series([1.25, 2.75, None], dtype=pl.Float32)}))
    out = (frame >> t.mutate(r=t.round("x", 1))).collect()
    assert out["r"].dtype == pl.Float32
    assert out["r"].to_list() == pytest.approx([1.2, 2.8, None], nan_ok=True)


@pytest.mark.parametrize("digits", [309, -309, 324, -1000])
def test_round_extreme_digits(digits):
    frame = t.tidy({"x": [1.25, -1.25, 0.0, float("inf"), None]})
    out = (frame >> t.mutate(r=t.round("x", digits))).collect(as_="pandas")
    assert out["r"].tolist()[:4] == ([1.25, -1.25, 0.0, float("inf")] if digits > 0 else [0.0, 0.0, 0.0, float("inf")])
    assert pd.isna(out["r"].iloc[-1])


def test_round_subnormal_numbers():
    frame = t.tidy({"x": [1.25e-310, -1.25e-310, 1.75e-310]})
    out = (frame >> t.mutate(r=t.round("x", 310))).collect(as_="pandas")
    np.testing.assert_array_equal(out["r"].to_numpy(), [1e-310, -1e-310, 2e-310])


def test_import_from_repository_root_has_one_package_identity():
    result = subprocess.run(
        [sys.executable, "-c", "import tidy3; from tidy3.frame import TidyFrame; "
         "assert tidy3.TidyFrame is TidyFrame; "
         "assert tidy3.tidy({'x': [1]}).collect()['x'].to_list() == [1]; "
         "print(tidy3.__file__)"],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("src/tidy3/__init__.py")


def test_module_cli_from_repository_root():
    result = subprocess.run(
        [sys.executable, "-m", "tidy3", "--help"],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "export" in result.stdout and "run" in result.stdout
    assert "loader starting" not in result.stdout


@pytest.mark.parametrize("entrypoint", ["tidy3.py", "load.py"])
def test_notebook_loaders_still_enable_pipes(entrypoint):
    result = subprocess.run(
        [sys.executable, "-c", f"""
from IPython.core.interactiveshell import InteractiveShell
shell = InteractiveShell.instance()
shell.run_line_magic('run', {entrypoint!r})
result = shell.run_cell("cars = tidy({{'hp': [110, 93]}})\\ncars\\n>> filter(hp > 100)")
assert result.error_before_exec is None, result.error_before_exec
assert result.error_in_exec is None, result.error_in_exec
assert result.result.collect()['hp'].to_list() == [110]
"""],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
