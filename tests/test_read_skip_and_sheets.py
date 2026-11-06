"""Rows above the header (skip= / skip_rows=) and read_excel() sheets."""

from __future__ import annotations

from pathlib import Path

import pytest

from tidy3 import read_csv, read_delim, read_excel, read_tsv, scan_csv

TITLE_LINES = "Motor Trend road tests\nsource: 1974\n"


def _csv(tmp_path: Path) -> Path:
    path = tmp_path / "cars.csv"
    path.write_text(TITLE_LINES + "model,mpg\nMazda RX4,21.0\nDatsun 710,22.8\n")
    return path


def test_skip_and_skip_rows_are_one_option(tmp_path: Path):
    path = _csv(tmp_path)
    for frame in (
        read_csv(path, skip=2),
        read_csv(path, skip_rows=2),
        scan_csv(path, skip=2),
        scan_csv(path, skip_rows=2),
    ):
        out = frame.collect()
        assert out.columns == ["model", "mpg"]
        assert out["mpg"].to_list() == [21.0, 22.8]


def test_skip_rows_in_tsv_and_delim(tmp_path: Path):
    tsv = tmp_path / "cars.tsv"
    tsv.write_text(TITLE_LINES + "model\tmpg\nValiant\t18.1\n")
    assert read_tsv(tsv, skip_rows=2).collect().columns == ["model", "mpg"]
    piped = tmp_path / "cars.txt"
    piped.write_text(TITLE_LINES + "model|mpg\nValiant|18.1\n")
    assert read_delim(piped, "|", skip_rows=2).collect().columns == ["model", "mpg"]


def test_skip_and_skip_rows_together_is_an_error(tmp_path: Path):
    path = _csv(tmp_path)
    with pytest.raises(TypeError, match="skip= or skip_rows="):
        read_csv(path, skip=2, skip_rows=2)
    with pytest.raises(TypeError, match="skip= or skip_rows="):
        read_csv(path, skip=0, skip_rows=2)
    with pytest.raises(ValueError, match="non-negative"):
        read_csv(path, skip_rows=-1)


def _workbook(tmp_path: Path) -> Path:
    import xlsxwriter

    path = tmp_path / "cars.xlsx"
    book = xlsxwriter.Workbook(path)
    first = book.add_worksheet("Road tests")  # first tab is not "Sheet1"
    for row, values in enumerate(
        [["Motor Trend road tests"], ["source: 1974"], ["model", "mpg"],
         ["Mazda RX4", 21.0], ["Datsun 710", 22.8]]
    ):
        first.write_row(row, 0, values)
    second = book.add_worksheet("Engines")
    for row, values in enumerate([["model", "cyl"], ["Valiant", 6]]):
        second.write_row(row, 0, values)
    book.close()
    return path


def test_read_excel_first_sheet_by_default(tmp_path: Path):
    pytest.importorskip("fastexcel")
    out = read_excel(_workbook(tmp_path)).collect()
    assert out.columns[0] == "Motor Trend road tests"


def test_read_excel_sheet_by_name_and_1_based_position(tmp_path: Path):
    pytest.importorskip("fastexcel")
    path = _workbook(tmp_path)
    assert read_excel(path, "Engines").collect().columns == ["model", "cyl"]
    assert read_excel(path, sheet=2).collect().columns == ["model", "cyl"]
    assert read_excel(path, sheet=1).collect().columns[0] == "Motor Trend road tests"
    for bad in (0, -1):
        with pytest.raises(ValueError, match="1-based"):
            read_excel(path, sheet=bad)
    with pytest.raises(TypeError, match="name or a position"):
        read_excel(path, sheet=True)
    with pytest.raises(TypeError, match="not both"):
        read_excel(path, sheet=1, sheet_name="Engines")


def test_read_excel_skip_reads_header_below_title_rows(tmp_path: Path):
    pytest.importorskip("fastexcel")
    path = _workbook(tmp_path)
    for out in (read_excel(path, skip=2), read_excel(path, skip_rows=2)):
        out = out.collect()
        assert out.columns == ["model", "mpg"]
        assert out["mpg"].to_list() == [21.0, 22.8]
    # read_options still reaches fastexcel, alongside skip
    out = read_excel(path, skip=2, read_options={"n_rows": 1}).collect()
    assert out["model"].to_list() == ["Mazda RX4"]
    with pytest.raises(TypeError, match="skip= or skip_rows="):
        read_excel(path, skip=2, skip_rows=2)
    with pytest.raises(TypeError, match="header_row"):
        read_excel(path, skip=2, read_options={"header_row": 1})
