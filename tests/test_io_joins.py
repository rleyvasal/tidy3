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


def test_unknown_execution_engine_is_rejected():
    with pytest.raises(ValueError, match="engine must be one of"):
        tidy({"id": [1, 2]}).collect(engine="turbo")


def test_write_csv_and_scan_roundtrip(tmp_path: Path):
    path = tmp_path / "result.csv"
    frame = tidy(
        {"id": [1, 2, 3], "value": [1.5, 2.5, 3.5]}
    ) >> filter(col("id") > 1)
    frame.write_csv(path)
    out = pl.read_csv(path)
    assert out.to_dict(as_series=False) == {
        "id": [2, 3],
        "value": [2.5, 3.5],
    }


def test_write_parquet_and_scan_roundtrip(tmp_path: Path):
    path = tmp_path / "result.parquet"
    frame = tidy(
        {"id": [1, 2, 3], "value": [1.5, 2.5, 3.5]}
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


def test_write_excel_or_actionable_optional_dependency_error(
    tmp_path: Path
):
    path = tmp_path / "result.xlsx"
    frame = tidy({"id": [1, 2], "value": [3.5, 4.5]})
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
        names = workbook.read("xl/workbook.xml").decode()
        assert 'name="Results"' in names


def test_write_excel_sheets_and_leading_equals_stay_text(tmp_path: Path):
    path = tmp_path / "book.xlsx"
    notes = tidy({"note": ["=1+1", "ok"]})
    other = tidy({"b": [1]})
    with pytest.raises(TypeError, match="not both"):
        notes.write_excel(path, worksheet="Results", sheets={"Notes": notes})
    notes.write_excel(path, sheets={"Notes": notes, "Other": other})
    with zipfile.ZipFile(path) as workbook:
        listing = workbook.read("xl/workbook.xml").decode()
        assert 'name="Notes"' in listing
        assert 'name="Other"' in listing
        shared = workbook.read("xl/sharedStrings.xml").decode()
        assert "=1+1" in shared
        for name in workbook.namelist():
            if name.startswith("xl/worksheets/sheet"):
                assert "<f" not in workbook.read(name).decode()


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


def test_natural_join_uses_common_columns():
    left = pd.DataFrame({"id": [1, 2, 3], "v": [10, 20, 30]})
    right = pd.DataFrame({"id": [1, 2], "w": ["a", "b"]})

    out = (tidy(left) >> left_join(right)).collect(as_="pandas")

    assert out["id"].tolist() == [1, 2, 3]
    assert out["w"].iloc[:2].tolist() == ["a", "b"]
    assert pd.isna(out["w"].iloc[2])


def test_natural_join_requires_at_least_one_common_column():
    left = pd.DataFrame({"id": [1, 2]})
    right = pd.DataFrame({"other": [1, 2]})
    with pytest.raises(ValueError, match="no common columns"):
        tidy(left) >> inner_join(right)


def test_with_polars_escape():
    df = pl.DataFrame({"x": [1, 2, 3]})
    out = tidy(df).with_polars(lambda lf: lf.filter(pl.col("x") > 1)).collect()
    assert out.height == 2


@pytest.mark.parametrize("how", ["left", "inner", "right", "full", "semi", "anti"])
def test_small_inequality_join_matches_the_ie_join_path(how):
    import tidy3 as t

    left = pl.DataFrame({"k": [5, None, 2, 9, 2], "v": list("abcde")})
    right = pl.DataFrame({"lo": [1, 2, None, 6], "hi": [3, 9, 4, 6], "tag": list("wxyz")})
    spec = t.join_by(t.between("k", "lo", "hi"))
    verb = getattr(t, f"{how}_join")
    # An eager y of up to 64 rows runs as a nested loop; a lazy y as an IE join.
    small = (tidy(left) >> verb(right, by=spec)).collect()
    lazy = (tidy(left) >> verb(tidy(right.lazy()), by=spec)).collect()
    assert small.equals(lazy)
    big = pl.concat([right] * 20)
    assert (tidy(left) >> verb(big, by=spec)).collect().equals((tidy(left) >> verb(tidy(big.lazy()), by=spec)).collect())


@pytest.mark.parametrize("has_header", [True, False])
def test_read_csv_one_pass_parse_matches_the_scan(tmp_path, has_header):
    # read_csv() parses in memory once; scan_csv() keeps a lazy plan. Both
    # give the same values and the same problems(), numbered as readr does.
    import tidy3 as t3

    rows = "".join(f"{i}, {i}.5 ,true\n" for i in range(120)) + "oops,1.x,maybe\n 7 ,,T\n"
    path = tmp_path / "penguins.csv"
    path.write_text(("bill,mass,ok\n" if has_header else "") + rows)
    options = {"has_header": has_header, "na": ["", "NA"]}
    read = t3.read_csv(path, **options)
    scan = t3.scan_csv(path, trim_ws=True, **options)
    assert read.collect().equals(scan.collect())
    found = t3.problems(read).collect()
    assert found.equals(t3.problems(scan).collect())
    assert found["original"].to_list() == ["oops", "1.x", "maybe"]
    assert found["row"].to_list() == ([122, 122, 122] if has_header else [121, 121, 121])
