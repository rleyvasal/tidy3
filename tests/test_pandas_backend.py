"""Pandas backend: verb correctness + parity with the polars backend."""

from __future__ import annotations

import pandas as pd
import polars as pl
import pytest

from tidy3 import (
    TidyFrame,
    arrange,
    col,
    collect,
    count,
    desc,
    distinct,
    drop,
    filter,
    group_by,
    head,
    left_join,
    mean,
    mutate,
    n,
    rename,
    sample_n,
    select,
    std,
    summarise,
    tidy,
)


@pytest.fixture
def cars() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "mpg": [21.0, 22.8, 21.4, 18.7, 18.1, 14.3],
            "cyl": [6, 4, 6, 8, 6, 8],
            "hp": [110, 93, 110, 175, 105, 245],
        }
    )


PIPE_SORT = ["cyl"]


def canonical(df) -> pd.DataFrame:
    """Sorted plain-pandas frame for cross-backend comparison."""
    pdf = df.to_pandas() if isinstance(df, pl.DataFrame) else df
    return pdf.sort_values(PIPE_SORT).reset_index(drop=True)


def run_pipe(tf: TidyFrame):
    return (
        tf
        >> filter(col("mpg") > 15)
        >> mutate(km=col("mpg") * 1.609)
        >> group_by("cyl")
        >> summarise(n=n(), avg=mean("km"), sd=std("mpg"))
        >> arrange("cyl")
        >> collect(as_="pandas")
    )


def test_backend_tag_and_types(cars):
    tp = tidy(cars, backend="pandas")
    assert tp.backend == "pandas"
    assert isinstance(tp._pdf, pd.DataFrame)
    assert tidy(cars).backend == "polars"


def test_pipeline_parity_polars_vs_pandas(cars):
    a = run_pipe(tidy(cars, backend="polars"))
    b = run_pipe(tidy(cars, backend="pandas"))
    pd.testing.assert_frame_equal(canonical(a), canonical(b), check_dtype=False)


def test_grouped_mutate_windowed_parity(cars):
    def pipe(tf):
        return (
            tf
            >> group_by("cyl")
            >> mutate(gmean=mean("mpg"), cs=col("hp").cum_sum())
            >> collect(as_="pandas")
        )

    a = pipe(tidy(cars))
    b = pipe(tidy(cars, backend="pandas"))
    key = ["cyl", "mpg", "hp"]
    pd.testing.assert_frame_equal(
        a.sort_values(key).reset_index(drop=True),
        b.sort_values(key).reset_index(drop=True),
        check_dtype=False,
    )


def test_grouped_filter_windowed(cars):
    out = (
        tidy(cars, backend="pandas")
        >> group_by("cyl")
        >> filter(col("mpg") > mean("mpg"))
        >> collect(as_="pandas")
    )
    exp = (
        tidy(cars)
        >> group_by("cyl")
        >> filter(col("mpg") > mean("mpg"))
        >> collect(as_="pandas")
    )
    assert sorted(out["mpg"]) == sorted(exp["mpg"])


def test_summarise_ungrouped(cars):
    out = (
        tidy(cars, backend="pandas")
        >> summarise(n=n(), avg=mean("mpg"))
        >> collect(as_="pandas")
    )
    assert out.shape == (1, 2)
    assert out["n"][0] == 6
    assert out["avg"][0] == pytest.approx(cars["mpg"].mean())


def test_select_drop_rename_group_hygiene(cars):
    tp = tidy(cars, backend="pandas") >> group_by("cyl")
    kept = (tp >> select("mpg")).collect(as_="pandas")
    assert list(kept.columns) == ["cyl", "mpg"]
    with pytest.raises(ValueError, match="grouping"):
        tp >> drop("cyl")
    ren = (tp >> rename(cylinders="cyl") >> summarise(n=n())).collect(as_="pandas")
    assert "cylinders" in ren.columns


def test_arrange_desc_and_head(cars):
    out = (
        tidy(cars, backend="pandas")
        >> arrange(desc("mpg"))
        >> head(2)
        >> collect(as_="pandas")
    )
    assert out["mpg"].tolist() == [22.8, 21.4]


def test_grouped_head_and_sample(cars):
    ngroups = cars["cyl"].nunique()
    h = (tidy(cars, backend="pandas") >> group_by("cyl") >> head(1)).collect(as_="pandas")
    assert len(h) == ngroups
    s = (
        tidy(cars, backend="pandas") >> group_by("cyl") >> sample_n(1, seed=1)
    ).collect(as_="pandas")
    assert len(s) == ngroups


def test_count_and_distinct(cars):
    c = (tidy(cars, backend="pandas") >> count("cyl")).collect(as_="pandas")
    assert c.set_index("cyl")["n"].to_dict() == {6: 3, 4: 1, 8: 2}
    d = (tidy(cars, backend="pandas") >> distinct("cyl")).collect(as_="pandas")
    assert len(d) == 3


def test_join_parity(cars):
    names = pd.DataFrame({"cyl": [4, 6, 8], "label": ["four", "six", "eight"]})

    def pipe(tf):
        return (tf >> left_join(names, on="cyl") >> collect(as_="pandas"))

    a = pipe(tidy(cars))
    b = pipe(tidy(cars, backend="pandas"))
    key = ["cyl", "mpg"]
    pd.testing.assert_frame_equal(
        a.sort_values(key).reset_index(drop=True)[sorted(a.columns)],
        b.sort_values(key).reset_index(drop=True)[sorted(b.columns)],
        check_dtype=False,
    )


def test_collect_conversions(cars):
    tp = tidy(cars, backend="pandas")
    assert isinstance(tp.collect(as_="pandas"), pd.DataFrame)
    assert isinstance(tp.collect(as_="polars"), pl.DataFrame)
    assert len(tp) == 6


def test_display_pandas_backend(cars):
    tp = tidy(cars, backend="pandas") >> filter(col("mpg") > 15)
    html = tp._repr_html_()
    assert "pandas" in html  # caption tags the backend
    assert "<style" not in html
    assert "TidyFrame" in repr(tp)


def test_unsupported_method_raises_clearly(cars):
    with pytest.raises(NotImplementedError, match="polars"):
        (
            tidy(cars, backend="pandas")
            >> mutate(x=col("mpg").rolling_mean(3))
            >> collect(as_="pandas")
        )


def test_summarise_derived_base_fast_path_parity(cars):
    # mean(col*2) → derived-series base, still one groupby.agg pass
    def pipe(tf):
        return (
            tf >> group_by("cyl") >> summarise(avg2=mean(col("mpg") * 2)) >> collect(as_="pandas")
        )

    a = pipe(tidy(cars))
    b = pipe(tidy(cars, backend="pandas"))
    pd.testing.assert_frame_equal(canonical(a), canonical(b), check_dtype=False)


def test_summarise_nested_agg_fallback_parity(cars):
    # mean(x - mean(x)): inner mean broadcasts per group → general evaluator
    def pipe(tf):
        return (
            tf
            >> group_by("cyl")
            >> summarise(dev=mean(col("mpg") - mean("mpg")))
            >> collect(as_="pandas")
        )

    a = pipe(tidy(cars))
    b = pipe(tidy(cars, backend="pandas"))
    pd.testing.assert_frame_equal(canonical(a), canonical(b), check_dtype=False)
    assert b["dev"].abs().max() == pytest.approx(0.0)  # mean deviation ≡ 0


def test_backend_option_default(cars):
    from tidy3 import options

    options(backend="pandas")
    try:
        assert tidy(cars).backend == "pandas"
    finally:
        options(backend="polars")
    assert tidy(cars).backend == "polars"
