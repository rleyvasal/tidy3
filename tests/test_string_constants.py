"""A quoted string in mutate is text, as in dplyr, on both backends.

Columns are col("hp") in plain Python and the bare name hp in a notebook
pipe; nb_export writes the bare name as col("hp").
"""

from __future__ import annotations

import polars as pl
import pytest

import tidy3 as t3
from tidy3 import col, mutate, partial_run, tidy, transform_source



def mtcars():
    return tidy({"model": ["Mazda RX4", "Datsun 710"], "hp": [110, 93]})


def test_quoted_string_is_text():
    out = mtcars() >> mutate(source="mtcars", hp_copy=col("hp"))
    data = out.collect()
    assert data["source"].to_list() == ["mtcars", "mtcars"]
    assert data["hp_copy"].to_list() == [110, 93]
    # Even when the text matches a column name, it is still text.
    named = mtcars() >> mutate(label="hp")
    assert named.collect()["label"].to_list() == ["hp", "hp"]


def test_quoted_string_is_text_in_transmute_and_by():
    kept = mtcars() >> t3.transmute(source="mtcars", hp=col("hp"))
    assert kept.collect().to_dicts()[0] == {"source": "mtcars", "hp": 110}
    grouped = mtcars() >> mutate(source="mtcars", by="model")
    assert grouped.collect()["source"].to_list() == ["mtcars", "mtcars"]
    summary = mtcars() >> t3.summarise(source="mtcars", n=t3.n())
    assert summary.collect().to_dicts() == [{"source": "mtcars", "n": 2}]


def test_notebook_pipe_matches_r():
    namespace = {"mtcars": pl.DataFrame({"model": ["Mazda RX4", "Datsun 710"], "hp": [110, 93]})}
    out = partial_run(
        """
        tidy(mtcars)
        >> mutate(`data set` = "mtcars", source = "mtcars", hp_copy = hp,
                  fast = if_else(hp > 100, "Y", "N"))
        """,
        namespace=namespace,
    ).collect()
    assert out["data set"].to_list() == ["mtcars", "mtcars"]
    assert out["source"].to_list() == ["mtcars", "mtcars"]
    assert out["hp_copy"].to_list() == [110, 93]
    assert out["fast"].to_list() == ["Y", "N"]


def test_export_keeps_text_and_spells_columns_with_col():
    source = transform_source('mtcars >> mutate(source = "mtcars", hp_copy = hp)')
    assert source == "mtcars >> mutate(source='mtcars', hp_copy=col('hp'))"
