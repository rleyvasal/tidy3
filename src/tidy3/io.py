"""Lazy file readers (scan) for large datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from tidy3.frame import TidyFrame


def scan_parquet(source: str | Path, **kwargs: Any) -> TidyFrame:
    """Lazily scan a Parquet file/glob — no full load until collect/preview."""
    return TidyFrame(pl.scan_parquet(source, **kwargs))


def scan_csv(source: str | Path, **kwargs: Any) -> TidyFrame:
    """Lazily scan a CSV file/glob."""
    return TidyFrame(pl.scan_csv(source, **kwargs))


def scan_ipc(source: str | Path, **kwargs: Any) -> TidyFrame:
    """Lazily scan Arrow IPC / Feather."""
    return TidyFrame(pl.scan_ipc(source, **kwargs))
