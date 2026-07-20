from __future__ import annotations

import numpy as np
import pytest

from tidy3 import col, collect, filter, starts_with, tidy, to_numpy


@pytest.mark.parametrize("backend", ["polars", "pandas"])
def test_to_numpy_projects_before_conversion(backend):
    frame = tidy(
        {
            "id": [1, 2, 3],
            "feature_a": [1.5, 2.5, 3.5],
            "feature_b": [4, 5, 6],
            "label": ["a", "b", "c"],
        },
        backend=backend,
    ) >> filter(col("id") > 1)

    matrix = frame.to_numpy(
        columns=starts_with("feature_"),
        dtype=np.float32,
        order="c",
        writable=True,
    )

    np.testing.assert_array_equal(
        matrix, np.array([[2.5, 5.0], [3.5, 6.0]], dtype=np.float32)
    )
    assert matrix.flags.c_contiguous
    assert matrix.flags.writeable


@pytest.mark.parametrize("backend", ["polars", "pandas"])
def test_collect_and_pipe_can_return_numpy(backend):
    frame = tidy(
        {"x": [1, 2], "y": [3, 4], "unused": ["a", "b"]},
        backend=backend,
    )

    collected = frame.collect(
        as_="numpy", columns=["x", "y"], dtype=np.float64, order="c"
    )
    piped_collect = frame >> collect(
        as_="numpy", columns=["x"], dtype=np.float32
    )
    piped_method = frame >> to_numpy(columns=["y"], dtype=np.int64)

    np.testing.assert_array_equal(collected, [[1.0, 3.0], [2.0, 4.0]])
    np.testing.assert_array_equal(piped_collect, [[1.0], [2.0]])
    np.testing.assert_array_equal(piped_method, [[3], [4]])
    assert collected.dtype == np.float64
    assert piped_collect.dtype == np.float32


@pytest.mark.parametrize("backend", ["polars", "pandas"])
def test_numpy_array_protocol_materializes_tidyframe(backend):
    frame = tidy({"x": [1, 2], "y": [3, 4]}, backend=backend)

    array = np.asarray(frame, dtype=np.float32)
    copied = np.array(frame, dtype=np.float64, copy=True)

    np.testing.assert_array_equal(array, [[1.0, 3.0], [2.0, 4.0]])
    np.testing.assert_array_equal(copied, array)
    assert array.dtype == np.float32
    assert copied.dtype == np.float64


def test_numpy_bridge_supports_polars_execution_engine():
    matrix = tidy({"x": [1, 2, 3]}).to_numpy(engine="streaming")
    np.testing.assert_array_equal(matrix, [[1], [2], [3]])


def test_numpy_conversion_options_require_numpy_output():
    with pytest.raises(ValueError, match="require as_='numpy'"):
        tidy({"x": [1]}).collect(as_="polars", dtype=np.float32)


def test_numpy_bridge_rejects_unknown_memory_order():
    with pytest.raises(ValueError, match="order must be"):
        tidy({"x": [1]}).to_numpy(order="rows")


def test_numpy_bridge_can_forbid_dtype_copy():
    frame = tidy({"x": [1, 2, 3]})
    with pytest.raises(RuntimeError, match="dtype conversion requires a copy"):
        frame.to_numpy(columns=["x"], dtype=np.float32, allow_copy=False)
