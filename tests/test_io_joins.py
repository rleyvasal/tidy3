"""scan_* and joins."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from tidy3 import col, filter, inner_join, left_join, scan_csv, scan_parquet, tidy


@pytest.fixture
def tmp_parquet(tmp_path: Path) -> Path:
    path = tmp_path / "cars.parquet"
    pl.DataFrame(
        {
            "mpg": [21.0, 22.8, 14.3],
            "cyl": [6, 4, 8],
            "id": [1, 2, 3],
        }
    ).write_parquet(path)
    return path


@pytest.fixture
def tmp_csv(tmp_path: Path) -> Path:
    path = tmp_path / "cars.csv"
    pl.DataFrame(
        {
            "mpg": [21.0, 22.8, 14.3],
            "cyl": [6, 4, 8],
        }
    ).write_csv(path)
    return path


def test_scan_parquet_filter(tmp_parquet: Path):
    out = scan_parquet(tmp_parquet) >> filter(col("mpg") > 20)
    df = out.collect()
    assert df.height == 2
    assert (df["mpg"] > 20).all()


def test_scan_csv(tmp_csv: Path):
    df = (scan_csv(tmp_csv) >> filter(col("cyl") == 4)).collect()
    assert df.height == 1
    assert df["cyl"][0] == 4


def test_left_join():
    left = pl.DataFrame({"id": [1, 2, 3], "v": [10, 20, 30]})
    right = pl.DataFrame({"id": [1, 2], "w": ["a", "b"]})
    out = (tidy(left) >> left_join(right, on="id")).collect()
    assert out.height == 3
    assert out["w"].null_count() == 1


def test_inner_join():
    left = pl.DataFrame({"id": [1, 2, 3], "v": [10, 20, 30]})
    right = pl.DataFrame({"id": [1, 2], "w": ["a", "b"]})
    out = (tidy(left) >> inner_join(right, on="id")).collect()
    assert out.height == 2


def test_with_polars_escape():
    df = pl.DataFrame({"x": [1, 2, 3]})
    out = tidy(df).with_polars(lambda lf: lf.filter(pl.col("x") > 1)).collect()
    assert out.height == 2
