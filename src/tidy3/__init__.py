"""tidy3 — dplyr-style lazy data manipulation on Polars."""

from __future__ import annotations

from tidy3.expr import col, desc, first, last, max, mean, median, min, n, std
from tidy3.expr import sum as sum  # noqa: A001
from tidy3.frame import TidyFrame, options, tidy
from tidy3.io import scan_csv, scan_ipc, scan_parquet
from tidy3.partial_run import (
    looks_like_tidy_pipe,
    maybe_rewrite_cell,
    normalize_pipe_source,
    partial_run,
)
from tidy3.verbs import (
    arrange,
    collect,
    count,
    distinct,
    tally,
    drop,
    filter,
    group_by,
    head,
    inner_join,
    left_join,
    mutate,
    peek,
    rename,
    sample_frac,
    sample_n,
    select,
    slice_head,
    summarise,
    summarize,
    transmute,
    ungroup,
)

__version__ = "0.2.0"

__all__ = [
    "TidyFrame",
    "arrange",
    "col",
    "collect",
    "count",
    "desc",
    "distinct",
    "drop",
    "filter",
    "first",
    "group_by",
    "head",
    "inner_join",
    "last",
    "left_join",
    "looks_like_tidy_pipe",
    "max",
    "mean",
    "median",
    "min",
    "maybe_rewrite_cell",
    "mutate",
    "n",
    "normalize_pipe_source",
    "options",
    "partial_run",
    "peek",
    "rename",
    "sample_frac",
    "sample_n",
    "scan_csv",
    "scan_ipc",
    "scan_parquet",
    "select",
    "slice_head",
    "std",
    "sum",
    "summarise",
    "summarize",
    "tally",
    "tidy",
    "transmute",
    "ungroup",
    "__version__",
]
