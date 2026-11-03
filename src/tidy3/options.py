"""Global tidy3 options (preview size, etc.)."""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass
class Options:
    preview_rows: int = 10
    preview: bool = True
    # Count all rows for the shape shown above a preview. Off: "?" and no
    # full pass over a lazy query (as dbplyr shows "??" for database tables).
    preview_count: bool = True
    # A notebook table pages through the first preview_max rows,
    # preview_rows a page, as R notebooks print a data frame. Every row is
    # saved with the notebook's output.
    preview_max: int = 100
    # Float digits in a notebook table: None as the text preview prints
    # them, k significant digits (as R's signif()), or "full".
    preview_digits: int | str | None = None


_OPTIONS = Options()


def get_options() -> Options:
    return _OPTIONS


def options(**kwargs) -> Options:
    """Update and return global options.

    Examples
    --------
    >>> options(preview_rows=20)
    """
    global _OPTIONS
    if "backend" in kwargs:
        from tidy3.frame import _NO_PANDAS_BACKEND

        raise TypeError(_NO_PANDAS_BACKEND)
    _OPTIONS = replace(_OPTIONS, **kwargs)
    return _OPTIONS
