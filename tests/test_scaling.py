from __future__ import annotations

import math

import pandas as pd
import pytest

from tidy3 import across, all_of, mutate, rescale, scale, tidy

BACKENDS = ["polars", "pandas"]


def collect(frame):
    return frame.collect(as_="pandas")


def _close(values, expected):
    assert len(values) == len(expected)
    for value, want in zip(values, expected):
        if want is None:
            assert pd.isna(value)
        else:
            assert math.isclose(value, want, rel_tol=1e-9, abs_tol=1e-12)


@pytest.fixture(params=BACKENDS)
def frame(request):
    return tidy(
        {
            "g": ["a", "a", "a", "b", "b", "b"],
            "x": [1.0, 2.0, 3.0, 10.0, None, 30.0],
            "k": [4.0, 4.0, 4.0, 4.0, None, 4.0],
        },
        backend=request.param,
    )


def test_scale_ignores_missing_values_like_r(frame):
    # R: scale(c(1, 2, 3, 10, NA, 30)), and its center=FALSE / scale=FALSE forms.
    out = collect(
        frame
        >> mutate(
            z=scale("x"),
            centred=scale("x", scale=False),
            rms=scale("x", center=False),
            same=scale("x", center=False, scale=False),
        )
    )
    mean, sd = 9.2, math.sqrt(590.8 / 4)
    rms = math.sqrt((1 + 4 + 9 + 100 + 900) / 4)
    xs = [1.0, 2.0, 3.0, 10.0, None, 30.0]
    _close(out["z"].tolist(), [None if v is None else (v - mean) / sd for v in xs])
    _close(out["centred"].tolist(), [None if v is None else v - mean for v in xs])
    _close(out["rms"].tolist(), [None if v is None else v / rms for v in xs])
    _close(out["same"].tolist(), xs)


def test_rescale_ignores_missing_values(frame):
    out = collect(frame >> mutate(unit=rescale("x"), wide=rescale("x", to=(-1, 1))))
    _close(out["unit"].tolist(), [0.0, 1 / 29, 2 / 29, 9 / 29, None, 1.0])
    _close(out["wide"].tolist(), [-1.0, -27 / 29, -25 / 29, -11 / 29, None, 1.0])


def test_scale_and_rescale_work_by_group_in_across(frame):
    out = collect(
        frame
        >> mutate(
            across(all_of(["x"]), scale, names="{col}_z"),
            across(all_of(["x"]), rescale, names="{col}_m"),
            by="g",
        )
    )
    half = math.sqrt(0.5)
    _close(out["x_z"].tolist(), [-1.0, 0.0, 1.0, -half, None, half])
    _close(out["x_m"].tolist(), [0.0, 0.5, 1.0, 0.0, None, 1.0])


def test_constant_column(frame):
    # R: scale() gives NaN (0 / 0); scales::rescale() gives the middle of `to`.
    out = collect(frame >> mutate(z=scale("k"), m=rescale("k"), m2=rescale("k", to=(2, 4))))
    assert out["z"].isna().all()
    _close(out["m"].tolist(), [0.5, 0.5, 0.5, 0.5, None, 0.5])
    _close(out["m2"].tolist(), [3.0, 3.0, 3.0, 3.0, None, 3.0])


def test_arguments_are_checked():
    with pytest.raises(TypeError, match="center must be a boolean"):
        scale("x", center=1)
    with pytest.raises(TypeError, match="scale must be a boolean"):
        scale("x", scale="yes")
    with pytest.raises(TypeError, match="pair of numbers"):
        rescale("x", to=(0, 1, 2))
