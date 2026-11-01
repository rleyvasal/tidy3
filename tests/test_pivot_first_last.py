"""pivot_wider(values_fn="first"/"last") lets Polars' pivot take the value
directly; it must give what summarising each cell first gives."""

import random
import warnings

import polars as pl
from polars.testing import assert_frame_equal
import pytest

from tidy3 import first, last, pivot_wider, tidy

SUMMARIES = {"first": first, "last": last}


def _frame(rng, rows):
    return pl.DataFrame(
        {
            "id": [rng.choice([1, 2, 3, None]) for _ in range(rows)],
            "k": [rng.choice(["a", "b", "c", None]) for _ in range(rows)],
            "v": [rng.choice([1.5, None, 2.5, float("nan")]) for _ in range(rows)],
            "w": [rng.choice(["p", None, "q"]) for _ in range(rows)],
        },
        schema={"id": pl.Int64, "k": pl.String, "v": pl.Float64, "w": pl.String},
    )


@pytest.mark.parametrize("name", SUMMARIES)
def test_first_and_last_match_summarising_each_cell(name):
    rng = random.Random(11)
    for rows in (1, 3, 8, 20, 60):
        data = _frame(rng, rows)
        for names in (None, ["a", "b", "z"]):
            for fill in (None, {"v": 0.0, "w": "-"}):
                options = {"names_from": "k", "values_from": ["v", "w"], "values_fill": fill}
                if names is not None:
                    options["names"] = names
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    direct = (tidy(data) >> pivot_wider(values_fn=name, **options)).collect()
                    summarised = (tidy(data) >> pivot_wider(values_fn=SUMMARIES[name], **options)).collect()
                assert_frame_equal(direct, summarised)
