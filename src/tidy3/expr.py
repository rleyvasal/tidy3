"""Column expressions and aggregation helpers (thin Polars wrappers)."""

from __future__ import annotations

from typing import Any

import polars as pl


def col(name: str) -> pl.Expr:
    """Column reference — maps to ``pl.col``."""
    return pl.col(name)


def desc(name: str | pl.Expr) -> pl.Expr:
    """Descending sort key for ``arrange``."""
    e = pl.col(name) if isinstance(name, str) else name
    return e.desc()


def n() -> pl.Expr:
    """Row count (within group if grouped)."""
    return pl.len()


def mean(x: str | pl.Expr) -> pl.Expr:
    e = pl.col(x) if isinstance(x, str) else x
    return e.mean()


def sum(x: str | pl.Expr) -> pl.Expr:  # noqa: A001
    e = pl.col(x) if isinstance(x, str) else x
    return e.sum()


def min(x: str | pl.Expr) -> pl.Expr:  # noqa: A001
    e = pl.col(x) if isinstance(x, str) else x
    return e.min()


def max(x: str | pl.Expr) -> pl.Expr:  # noqa: A001
    e = pl.col(x) if isinstance(x, str) else x
    return e.max()


def median(x: str | pl.Expr) -> pl.Expr:
    e = pl.col(x) if isinstance(x, str) else x
    return e.median()


def std(x: str | pl.Expr) -> pl.Expr:
    e = pl.col(x) if isinstance(x, str) else x
    return e.std()


def first(x: str | pl.Expr) -> pl.Expr:
    e = pl.col(x) if isinstance(x, str) else x
    return e.first()


def last(x: str | pl.Expr) -> pl.Expr:
    e = pl.col(x) if isinstance(x, str) else x
    return e.last()


def _as_expr(x: Any) -> pl.Expr:
    if isinstance(x, pl.Expr):
        return x
    if isinstance(x, str):
        return pl.col(x)
    raise TypeError(f"expected column name or polars Expr, got {type(x)!r}")
