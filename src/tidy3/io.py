"""File readers. ``scan_*`` stays lazy; ``read_*`` returns a computed frame.

``read_csv``, ``read_tsv``, and ``read_delim`` share options with
:func:`scan_csv`: ``col_types``, ``na``, ``skip``, ``comment``, ``trim_ws``,
and ``id``. ``trim_ws`` defaults to false for :func:`scan_csv`, so existing
scans do not change, and true for the readers. With ``col_types``, integer,
double, logical, date, and datetime columns are read as text and parsed by
the shared conversion engine, so ``"2.5"`` is a recorded problem rather than
a truncated integer and an id kept as text keeps its leading zeros.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import polars as pl

from tidy3.frame import TidyFrame

__all__ = [
    "scan_parquet",
    "scan_csv",
    "scan_ipc",
    "read_csv",
    "read_tsv",
    "read_delim",
    "read_excel",
    "excel_sheets",
]

_STRING_TYPES = frozenset({"character", "string", "str"})
_CAST_TYPES = frozenset(
    {"integer", "double", "numeric", "logical", "date", "datetime"}
)
_UTF8 = {
    "utf-8": "utf8",
    "utf8": "utf8",
    "utf-8-sig": "utf8",
    "utf8-lossy": "utf8-lossy",
    "utf-8-lossy": "utf8-lossy",
}


def scan_parquet(source: str | Path, **kwargs: Any) -> TidyFrame:
    """Lazily scan a Parquet file/glob — no full load until collect/preview."""
    return TidyFrame(pl.scan_parquet(source, **kwargs))


def scan_ipc(source: str | Path, **kwargs: Any) -> TidyFrame:
    """Lazily scan Arrow IPC / Feather."""
    return TidyFrame(pl.scan_ipc(source, **kwargs))


def _polars_encoding(encoding: str) -> str | None:
    if not isinstance(encoding, str):
        raise TypeError("encoding must be a string")
    return _UTF8.get(encoding.lower().replace("_", "-"))


def _is_glob(source: Any) -> bool:
    if isinstance(source, (list, tuple)):
        return True
    text = str(source)
    return any(mark in text for mark in "*?[")


def _type_key(kind: Any) -> str:
    if not isinstance(kind, str):
        raise TypeError(f"col_types values must be strings, got {kind!r}")
    return kind.strip().lower()


def _schema_overrides(col_types: dict[str, Any] | None) -> dict[str, pl.DataType]:
    if not col_types:
        return {}
    overrides: dict[str, pl.DataType] = {}
    for name, kind in col_types.items():
        key = _type_key(kind)
        if key in _STRING_TYPES or key in _CAST_TYPES:
            overrides[str(name)] = pl.String
            continue
        known = ", ".join(sorted(_STRING_TYPES | _CAST_TYPES))
        raise ValueError(f"col_types[{name!r}] must be one of: {known}")
    return overrides


def _cast_assignments(
    col_types: dict[str, Any] | None,
    *,
    strict: bool,
    marks: Any,
) -> dict[str, Any]:
    """String columns stay strings. Other named types go through the parsers.

    Numbers use ``parse_double`` / ``parse_integer`` so a locale's grouping
    mark is honoured. Logical values use readr's ``1``/``0``/``true``/``false``
    spellings. Dates are ISO ``YYYY-MM-DD``.
    """
    if not col_types:
        return {}
    from tidy3.convert import (
        as_date,
        as_datetime,
        parse_double,
        parse_integer,
        parse_logical,
    )
    from tidy3.expr import col

    assignments = {}
    for name, kind in col_types.items():
        key = _type_key(kind)
        column = col(str(name))
        if key in {"double", "numeric"}:
            assignments[str(name)] = parse_double(column, locale=marks, strict=strict)
        elif key == "integer":
            assignments[str(name)] = parse_integer(column, locale=marks, strict=strict)
        elif key == "logical":
            assignments[str(name)] = parse_logical(column, strict=strict)
        elif key == "date":
            assignments[str(name)] = as_date(column, strict=strict)
        elif key == "datetime":
            assignments[str(name)] = as_datetime(column, strict=strict)
    return assignments


def _trim_strings(lf: pl.LazyFrame, skip: set[str]) -> pl.LazyFrame:
    schema = lf.collect_schema()
    expressions = []
    for name, dtype in schema.items():
        if name in skip:
            continue
        if dtype == pl.String or dtype == pl.Utf8:
            expressions.append(pl.col(name).str.strip_chars().alias(name))
    if not expressions:
        return lf
    return lf.with_columns(expressions)


def _scan_table(
    source: Any,
    *,
    delim: str | None,
    col_types: dict[str, Any] | None,
    na: Any,
    skip: int,
    comment: str | None,
    trim_ws: bool,
    id: str | None,
    encoding: str,
    strict: bool,
    locale: Any,
    allow_decode: bool,
    kwargs: dict[str, Any],
) -> TidyFrame:
    if na is not None and "null_values" in kwargs:
        raise TypeError("pass na= or null_values=, not both")
    if delim is not None and "separator" in kwargs:
        raise TypeError("pass delim= or separator=, not both")
    if isinstance(skip, bool) or not isinstance(skip, int) or skip < 0:
        raise ValueError("skip must be a non-negative integer")
    if not isinstance(strict, bool):
        raise TypeError("strict must be a boolean")
    polars_encoding = _polars_encoding(encoding)
    if polars_encoding is None and not allow_decode:
        raise ValueError(
            "scan_csv() reads utf-8. Use read_csv(..., encoding=...) for other encodings."
        )
    if polars_encoding is None and _is_glob(source):
        raise ValueError(
            "an encoding other than utf-8 needs a single file path; globs stay utf-8"
        )
    options = dict(kwargs)
    if na is not None:
        options["null_values"] = na
    if skip:
        options["skip_rows"] = skip
    if comment is not None:
        options["comment_prefix"] = comment
    if delim is not None:
        options["separator"] = delim
    if id is not None:
        if not isinstance(id, str) or not id:
            raise TypeError("id= must be the name of the file-path column")
        options["include_file_paths"] = id
    overrides = _schema_overrides(col_types)
    if overrides:
        current = dict(options.get("schema_overrides") or {})
        current.update(overrides)
        options["schema_overrides"] = current
    if polars_encoding is None:
        text = Path(source).read_text(encoding=encoding)
        frame = pl.read_csv(io.BytesIO(text.encode("utf-8")), **options).lazy()
    else:
        if polars_encoding != "utf8":
            options["encoding"] = polars_encoding
        frame = pl.scan_csv(source, **options)
    if trim_ws:
        frame = _trim_strings(frame, {id} if id else set())
    result = TidyFrame(frame)
    assignments = _cast_assignments(col_types, strict=strict, marks=locale)
    if assignments:
        from tidy3.verbs import mutate

        result = result >> mutate(**assignments)
    return result


def scan_csv(
    source: str | Path,
    *,
    col_types: dict[str, Any] | None = None,
    na: Any = None,
    skip: int = 0,
    comment: str | None = None,
    trim_ws: bool = False,
    id: str | None = None,
    encoding: str = "utf-8",
    strict: bool = False,
    locale: Any = None,
    delim: str | None = None,
    **kwargs: Any,
) -> TidyFrame:
    """Lazily scan a CSV file or glob.

    ``trim_ws`` defaults to false so existing scans are unchanged; use
    :func:`read_csv` when you want readr's trimmed fields. ``na`` is
    Polars' ``null_values``. ``id`` records the source path of each row.
    Non-utf-8 files need :func:`read_csv`.
    """
    return _scan_table(
        source,
        delim=delim,
        col_types=col_types,
        na=na,
        skip=skip,
        comment=comment,
        trim_ws=trim_ws,
        id=id,
        encoding=encoding,
        strict=strict,
        locale=locale,
        allow_decode=False,
        kwargs=kwargs,
    )


def read_csv(
    source: str | Path,
    *,
    col_types: dict[str, Any] | None = None,
    na: Any = None,
    skip: int = 0,
    comment: str | None = None,
    trim_ws: bool = True,
    id: str | None = None,
    encoding: str = "utf-8",
    strict: bool = False,
    locale: Any = None,
    **kwargs: Any,
) -> TidyFrame:
    """Read a CSV into a :class:`~tidy3.frame.TidyFrame`.

    Same options as :func:`scan_csv`, with ``trim_ws=True``. ``col_types``
    keeps an id column as text (``"character"``) so leading zeros survive,
    and parses ``integer``, ``double``, ``logical``, ``date``, and
    ``datetime`` through the conversion engine. Date columns are ISO
    ``YYYY-MM-DD``. ``strict=False`` turns bad values into missing and
    records them for :func:`tidy3.convert.problems`.
    """
    delim = kwargs.pop("delim", None)
    if delim is None and "separator" not in kwargs:
        delim = ","
    frame = _scan_table(
        source,
        delim=delim,
        col_types=col_types,
        na=na,
        skip=skip,
        comment=comment,
        trim_ws=trim_ws,
        id=id,
        encoding=encoding,
        strict=strict,
        locale=locale,
        allow_decode=True,
        kwargs=kwargs,
    )
    return frame.compute()


def read_tsv(source: str | Path, **kwargs: Any) -> TidyFrame:
    """Read a tab-separated file. See :func:`read_csv`."""
    kwargs["delim"] = "\t"
    return read_csv(source, **kwargs)


def read_delim(source: str | Path, delim: str, **kwargs: Any) -> TidyFrame:
    """Read a delimited file. ``delim`` is required. See :func:`read_csv`."""
    if not isinstance(delim, str) or delim == "":
        raise TypeError("read_delim() requires delim=")
    kwargs["delim"] = delim
    return read_csv(source, **kwargs)


def _need_fastexcel() -> None:
    try:
        import fastexcel  # noqa: F401
    except ImportError as error:
        raise ImportError(
            "Reading Excel requires fastexcel. "
            "Install it with pip install 'tidy3[excel]'."
        ) from error


def excel_sheets(path: str | Path) -> list[str]:
    """Sheet names in an Excel workbook. Requires the ``excel`` extra."""
    _need_fastexcel()
    import fastexcel

    book = fastexcel.read_excel(path)
    return list(book.sheet_names)


def read_excel(
    path: str | Path,
    sheet: str | int | None = None,
    **kwargs: Any,
) -> TidyFrame:
    """Read one Excel sheet into a :class:`~tidy3.frame.TidyFrame`.

    Requires the optional ``fastexcel`` package (``pip install 'tidy3[excel]'``).
    ``sheet`` is a name or a 0-based index; the default is the first sheet.
    """
    _need_fastexcel()
    frame = pl.read_excel(
        path,
        sheet_name=0 if sheet is None else sheet,
        **kwargs,
    )
    from tidy3.frame import tidy

    return tidy(frame)
