"""Global tidy3 options (preview size, etc.)."""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass
class Options:
    preview_rows: int = 10
    preview: bool = True
    backend: str = "polars"  # default engine for tidy(): "polars" | "pandas"


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
    _OPTIONS = replace(_OPTIONS, **kwargs)
    return _OPTIONS
