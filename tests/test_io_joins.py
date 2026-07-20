"""scan_* and joins."""

from __future__ import annotations

from pathlib import Path
import zipfile

import pandas as pd
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


def test_collect_streaming_engine_and_pipe_verb():
    from tidy3 import collect

    frame = tidy({"id": [1, 2, 3]}) >> filter(col("id") > 1)
    direct = frame.collect(engine="streaming")
    piped = frame >> collect(engine="streaming")
    assert direct.to_dict(as_series=False) == {"id": [2, 3]}
    assert piped.to_dict(as_series=False) == {"id": [2, 3]}


def test_polars_execution_modes_are_rejected_for_pandas_backend():
    frame = tidy({"id": [1, 2]}, backend="pandas")
    with pytest.raises(ValueError, match="Polars execution option"):
        frame.collect(engine="streaming")


def test_unknown_execution_engine_is_rejected():
    with pytest.raises(ValueError, match="engine must be one of"):
        tidy({"id": [1, 2]}).collect(engine="turbo")


@pytest.mark.parametrize("backend", ["polars", "pandas"])
def test_write_csv_and_scan_roundtrip(tmp_path: Path, backend: str):
    path = tmp_path / f"result-{backend}.csv"
    frame = tidy(
        {"id": [1, 2, 3], "value": [1.5, 2.5, 3.5]}, backend=backend
    ) >> filter(col("id") > 1)
    frame.write_csv(path)
    out = pl.read_csv(path)
    assert out.to_dict(as_series=False) == {
        "id": [2, 3],
        "value": [2.5, 3.5],
    }


@pytest.mark.parametrize("backend", ["polars", "pandas"])
def test_write_parquet_and_scan_roundtrip(tmp_path: Path, backend: str):
    path = tmp_path / f"result-{backend}.parquet"
    frame = tidy(
        {"id": [1, 2, 3], "value": [1.5, 2.5, 3.5]}, backend=backend
    ) >> filter(col("id") > 1)
    frame.write_parquet(path)
    out = pl.read_parquet(path)
    assert out.to_dict(as_series=False) == {
        "id": [2, 3],
        "value": [2.5, 3.5],
    }


def test_write_ipc_roundtrip(tmp_path: Path):
    path = tmp_path / "result.arrow"
    tidy({"id": [1, 2]}).write_ipc(path)
    assert pl.read_ipc(path)["id"].to_list() == [1, 2]


@pytest.mark.parametrize("format_", ["csv", "parquet", "ipc"])
def test_polars_writers_accept_streaming_engine(
    tmp_path: Path, format_: str
):
    path = tmp_path / f"streamed.{format_}"
    frame = tidy({"id": [1, 2, 3]}) >> filter(col("id") > 1)
    getattr(frame, f"write_{format_}")(path, engine="streaming")
    reader = {"csv": pl.read_csv, "parquet": pl.read_parquet, "ipc": pl.read_ipc}[
        format_
    ]
    assert reader(path)["id"].to_list() == [2, 3]


@pytest.mark.parametrize("format_", ["csv", "parquet", "ipc"])
def test_gpu_writers_materialize_before_serialization(
    tmp_path: Path, format_: str, monkeypatch
):
    path = tmp_path / f"gpu.{format_}"
    frame = tidy({"id": [1, 2, 3]})
    calls = []

    def fake_collect(self, as_="polars", **kwargs):
        calls.append((as_, kwargs))
        return pl.DataFrame({"id": [1, 2, 3]})

    monkeypatch.setattr(type(frame), "collect", fake_collect)
    getattr(frame, f"write_{format_}")(path, engine="gpu")

    assert calls == [("polars", {"engine": "gpu"})]
    reader = {"csv": pl.read_csv, "parquet": pl.read_parquet, "ipc": pl.read_ipc}[
        format_
    ]
    assert reader(path)["id"].to_list() == [1, 2, 3]


@pytest.mark.parametrize("backend", ["polars", "pandas"])
def test_write_excel_or_actionable_optional_dependency_error(
    tmp_path: Path, backend: str
):
    path = tmp_path / f"result-{backend}.xlsx"
    frame = tidy({"id": [1, 2], "value": [3.5, 4.5]}, backend=backend)
    try:
        import xlsxwriter  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match=r"tidy3\[excel\]"):
            frame.write_excel(path)
        return
    frame.write_excel(path, worksheet="Results", autofit=True)
    assert zipfile.is_zipfile(path)
    with zipfile.ZipFile(path) as workbook:
        assert "xl/worksheets/sheet1.xml" in workbook.namelist()


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


@pytest.mark.parametrize("backend", ["polars", "pandas"])
def test_natural_join_uses_common_columns(backend):
    left = pd.DataFrame({"id": [1, 2, 3], "v": [10, 20, 30]})
    right = pd.DataFrame({"id": [1, 2], "w": ["a", "b"]})

    out = (tidy(left, backend=backend) >> left_join(right)).collect(as_="pandas")

    assert out["id"].tolist() == [1, 2, 3]
    assert out["w"].iloc[:2].tolist() == ["a", "b"]
    assert pd.isna(out["w"].iloc[2])


def test_join_converts_tidyframe_from_the_other_backend():
    left_pd = pd.DataFrame({"id": [1, 2, 3], "v": [10, 20, 30]})
    right_pd = pd.DataFrame({"id": [1, 2], "w": ["a", "b"]})

    polars_out = (
        tidy(left_pd, backend="polars")
        >> left_join(tidy(right_pd, backend="pandas"), on="id")
    ).collect(as_="pandas")
    pandas_out = (
        tidy(left_pd, backend="pandas")
        >> left_join(tidy(right_pd, backend="polars"), on="id")
    ).collect(as_="pandas")

    pd.testing.assert_frame_equal(polars_out, pandas_out, check_dtype=False)


@pytest.mark.parametrize("backend", ["polars", "pandas"])
def test_natural_join_requires_at_least_one_common_column(backend):
    left = pd.DataFrame({"id": [1, 2]})
    right = pd.DataFrame({"other": [1, 2]})
    with pytest.raises(ValueError, match="no common columns"):
        tidy(left, backend=backend) >> inner_join(right)


def test_with_polars_escape():
    df = pl.DataFrame({"x": [1, 2, 3]})
    out = tidy(df).with_polars(lambda lf: lf.filter(pl.col("x") > 1)).collect()
    assert out.height == 2
