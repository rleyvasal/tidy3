"""Lightweight EDA / inspection helpers (R-like free functions).

Column names
------------
* ``cars.columns`` / ``cars.names`` / ``names(cars)`` → plain ``list[str]``
  for programming (schema-only, no row scan).
* ``colnames(cars)`` → same names, but **displays** as copy/paste-ready
  selectors for ``select(...)``::

      mpg,
      cyl,
      `hp new`,
      wt,

  Odd identifiers are wrapped in backticks (built with ``chr(96)`` so this
  module stays safe under the Jupyter backtick preparser).
"""

from __future__ import annotations

import keyword
import re
from typing import Any, Iterable

__all__ = [
    "names",
    "colnames",
    "nrow",
    "ncol",
    "dim",
    "dtypes",
    "format_colnames",
    "ColnamesResult",
]

# Backtick character without a literal ` in this file (Jupyter R-style safety).
_BT = chr(96)
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _as_tidy(data: Any):
    from tidy3.frame import TidyFrame, tidy

    if isinstance(data, TidyFrame):
        return data
    return tidy(data)


def _column_names(data: Any) -> list[str]:
    """Reliable column-name accessor for tidy3 / Polars / pandas."""
    # Prefer native schema when already a TidyFrame
    from tidy3.frame import TidyFrame

    if isinstance(data, TidyFrame):
        return list(data.columns)
    # Polars
    if hasattr(data, "collect_schema"):
        try:
            return list(data.collect_schema().names())
        except Exception:
            pass
    if hasattr(data, "columns") and not callable(data.columns):
        cols = data.columns
        if isinstance(cols, list):
            return [str(c) for c in cols]
        # pandas Index
        try:
            return [str(c) for c in list(cols)]
        except Exception:
            pass
    return list(_as_tidy(data).columns)


def names(data: Any) -> list[str]:
    """Column names as a plain Python list (base R ``names()``).

    Use this (or ``cars.columns``) when you need to *program* with the names.
    For paste-into-``select`` display, use :func:`colnames`.
    """
    return _column_names(data)


def selector_token(name: str) -> str:
    """Format one column name for R-style bare / backtick select syntax."""
    name = str(name)
    if _IDENT.fullmatch(name) and not keyword.iskeyword(name):
        return name
    # Escape embedded backticks by doubling (R-style) if any.
    safe = name.replace(_BT, _BT + _BT)
    return f"{_BT}{safe}{_BT}"


def format_colnames(cols: Iterable[str], *, trailing_comma: bool = True) -> str:
    """Join column names as paste-ready ``select(...)`` lines."""
    lines: list[str] = []
    seq = list(cols)
    for i, col in enumerate(seq):
        token = selector_token(col)
        if trailing_comma or i < len(seq) - 1:
            lines.append(f"{token},")
        else:
            lines.append(token)
    return "\n".join(lines) + ("\n" if lines else "")


class ColnamesResult(list):
    """``list[str]`` of column names with a paste-ready ``repr``.

    Interactive display (notebook / REPL)::

        colnames(cars)
        # mpg,
        # cyl,
        # hp,

    Still a real list::

        colnames(cars)[0]          # 'mpg'
        list(colnames(cars))       # ['mpg', 'cyl', ...]
        'mpg' in colnames(cars)    # True
    """

    def __repr__(self) -> str:  # noqa: D105
        text = format_colnames(self)
        return text if text else "<empty colnames>"

    def __str__(self) -> str:  # noqa: D105
        return format_colnames(self)

    def _repr_pretty_(self, p, cycle) -> None:  # IPython
        p.text(repr(self) if not cycle else "...")

    def _repr_html_(self) -> str:
        # Plain <pre> so SolveIt/Jupyter show copyable text (no fancy table).
        from html import escape

        return f"<pre>{escape(str(self))}</pre>"


def colnames(data: Any) -> ColnamesResult:
    """Column names for **display / paste into** ``select(...)``.

    Returns a :class:`ColnamesResult` (list subclass). Printing it yields::

        mpg,
        cyl,
        `hp new`,
        wt,

    For a plain list (programming), use ``names(cars)`` or ``cars.columns``.
    """
    return ColnamesResult(_column_names(data))


def nrow(data: Any) -> int:
    """Number of rows (base R ``nrow()``). Materializes a count for lazy data."""
    return int(len(_as_tidy(data)))


def ncol(data: Any) -> int:
    """Number of columns (base R ``ncol()``). Schema-only for lazy data."""
    return int(len(_column_names(data)))


def dim(data: Any) -> tuple[int, int]:
    """``(nrow, ncol)`` like base R ``dim()``."""
    tf = _as_tidy(data)
    return (int(len(tf)), int(len(tf.columns)))


def dtypes(data: Any) -> dict[str, Any]:
    """Mapping of column name → dtype (Polars or pandas dtype objects)."""
    return dict(_as_tidy(data).dtypes)
