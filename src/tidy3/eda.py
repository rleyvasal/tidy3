"""Lightweight EDA / inspection helpers (R-like free functions).

These return Python values (lists, ints, tuples), not :class:`TidyFrame`s —
same idea as base R ``names()`` / ``nrow()``, not dplyr verbs.

Also available as :class:`~tidy3.frame.TidyFrame` properties::

    cars.columns   # list[str]
    cars.names     # alias
    cars.shape     # (n_rows, n_cols)
    cars.dtypes    # {name: dtype}
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "names",
    "colnames",
    "nrow",
    "ncol",
    "dim",
    "dtypes",
]


def _as_tidy(data: Any):
    from tidy3.frame import TidyFrame, tidy

    if isinstance(data, TidyFrame):
        return data
    return tidy(data)


def names(data: Any) -> list[str]:
    """Column names (base R ``names()`` / ``colnames()``).

    Works on :class:`TidyFrame`, Polars, or pandas.
    """
    return list(_as_tidy(data).columns)


def colnames(data: Any) -> list[str]:
    """Alias of :func:`names` (base R ``colnames()``)."""
    return names(data)


def nrow(data: Any) -> int:
    """Number of rows (base R ``nrow()``). Materializes a count for lazy data."""
    return int(len(_as_tidy(data)))


def ncol(data: Any) -> int:
    """Number of columns (base R ``ncol()``). Schema-only for lazy data."""
    return int(_as_tidy(data).width)


def dim(data: Any) -> tuple[int, int]:
    """``(nrow, ncol)`` like base R ``dim()``."""
    tf = _as_tidy(data)
    return (int(len(tf)), int(tf.width))


def dtypes(data: Any) -> dict[str, Any]:
    """Mapping of column name → dtype (Polars or pandas dtype objects)."""
    return dict(_as_tidy(data).dtypes)
