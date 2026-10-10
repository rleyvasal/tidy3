"""compute(): run the plan once, keep piping from the in-memory result."""

from __future__ import annotations

import polars as pl
import pytest

from tidy3 import (
    c_across,
    col,
    compute,
    drop_na,
    filter,
    group_by,
    group_vars,
    mean,
    mutate,
    rowwise,
    scan_csv,
    summarise,
    sum,
    tidy,
)


def test_compute_reads_the_source_once(tmp_path):
    path = tmp_path / "cars.csv"
    path.write_text("mpg,cyl\n21,6\n22.8,4\nNA,8\n18.7,8\n")
    cars = scan_csv(str(path), null_values="NA") >> drop_na("mpg") >> compute()
    path.unlink()  # a lazy plan would now fail on every collect
    assert cars.collect().height == 3
    assert (cars >> filter(col("cyl") == 8)).collect()["mpg"].to_list() == [18.7]
    assert cars.to_numpy(columns=["mpg"]).shape == (3, 1)


def test_compute_keeps_groups_and_rowwise():
    frame = tidy({"g": ["a", "a", "b"], "x": [1.0, 3.0, 5.0], "y": [1, 1, 1]})
    grouped = frame >> group_by("g") >> compute()
    assert list(group_vars(grouped)) == ["g"]
    out = (grouped >> summarise(m=mean("x"))).collect().sort("g")
    assert out["m"].to_list() == [2.0, 5.0]
    by_row = frame >> rowwise() >> compute()
    totals = (by_row >> mutate(t=sum(c_across(["x", "y"])))).collect()["t"].to_list()
    assert totals == [2.0, 4.0, 6.0]


def test_compute_method_and_verb_agree():
    frame = tidy({"x": [3, 1, 2]}) >> mutate(y=col("x") * 2)
    assert frame.compute().collect().equals((frame >> compute()).collect())
    assert isinstance(frame.compute()._data, pl.LazyFrame)


def test_compute_on_the_pandas_backend_is_a_no_op():
    frame = tidy({"x": [1, 2]}, backend="pandas")
    assert frame.compute() is frame
    with pytest.raises(ValueError):
        frame.compute(engine="streaming")

