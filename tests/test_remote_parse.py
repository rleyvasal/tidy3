"""Remote payload parsing without a live CRAFT connection."""

from __future__ import annotations

import base64
import io

import polars as pl
import pytest

from tidy3.remote import _MARKER, _parse_preview


def test_parse_preview_ipc():
    df = pl.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
    buf = io.BytesIO()
    df.write_ipc(buf)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    out = f"noise\n{_MARKER}{b64}\ntidy3 remote: preview 2 rows\n"
    got = _parse_preview(out)
    assert got.shape == (2, 2)
    assert got["a"].to_list() == [1, 2]


def test_parse_preview_missing_marker():
    with pytest.raises(RuntimeError, match="no preview"):
        _parse_preview("some error traceback")
