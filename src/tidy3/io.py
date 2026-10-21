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
# readr's default na: an empty field and the text NA are missing.
_READR_NA = ("", "NA")
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


def _null_values(na: Any, name: str) -> list[str]:
    if isinstance(na, str):
        return [na]
    if isinstance(na, dict):
        return _null_values(na.get(name), name) if na.get(name) is not None else []
    if isinstance(na, (list, tuple)):
        return [str(value) for value in na]
    return []


def _null_after_trim(lf: pl.LazyFrame, na: Any, skip: set[str]) -> pl.LazyFrame:
    """Missing-value strings match the trimmed field, as in readr: " NA " is NA."""
    if na is None:
        return lf
    expressions = []
    for name, dtype in lf.collect_schema().items():
        values = _null_values(na, name)
        if name in skip or dtype != pl.String or not values:
            continue
        column = pl.col(name)
        expressions.append(
            pl.when(column.is_in(values)).then(None).otherwise(column).alias(name)
        )
    return lf.with_columns(expressions) if expressions else lf


def _guess_dtype(values: pl.Series) -> Any:
    """Integer, float, or logical when every sampled value is one; else None."""
    if values.str.contains(r"^[+-]?\d+$").all() and not values.cast(
        pl.Int64, strict=False
    ).null_count():
        return pl.Int64
    if not values.cast(pl.Float64, strict=False).null_count():
        return pl.Float64
    if values.str.to_lowercase().is_in(["true", "false"]).all():
        return pl.Boolean
    return None


def _guess_types(lf: pl.LazyFrame, skip: set[str], rows: int | None) -> pl.LazyFrame:
    """Guess column types from trimmed text, so a padded " 6" is still a number.

    Polars guesses before :func:`_trim_strings` would run, so the readers
    read every column as text, trim it, and guess here from the first
    *rows* rows (all rows when ``None``), as readr does.
    """
    names = [
        name
        for name, dtype in lf.collect_schema().items()
        if dtype == pl.String and name not in skip
    ]
    if not names:
        return lf
    sample = lf.select(names)
    sample = (sample if rows is None else sample.head(rows)).collect()
    expressions = []
    for name in names:
        values = sample[name].drop_nulls()
        if values.is_empty():
            continue
        dtype = _guess_dtype(values)
        if dtype == pl.Boolean:
            expressions.append(
                pl.col(name)
                .str.to_lowercase()
                .replace_strict({"true": True, "false": False}, return_dtype=pl.Boolean)
                .alias(name)
            )
        elif dtype is not None:
            expressions.append(pl.col(name).cast(dtype).alias(name))
    return lf.with_columns(expressions) if expressions else lf


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
    user_overrides = set(options.get("schema_overrides") or {})
    overrides = _schema_overrides(col_types)
    if overrides:
        current = dict(options.get("schema_overrides") or {})
        current.update(overrides)
        options["schema_overrides"] = current
    # Guess types after trimming, not before: read text, trim, then guess.
    guess = (
        trim_ws
        and not options.get("schema")
        and options.get("infer_schema", True) is not False
        and not options.get("try_parse_dates")
    )
    sample_rows = options.pop("infer_schema_length", 100) if guess else None
    if guess:
        options["infer_schema"] = False
    if polars_encoding is None:
        text = Path(source).read_text(encoding=encoding)
        frame = pl.read_csv(io.BytesIO(text.encode("utf-8")), **options).lazy()
    else:
        if polars_encoding != "utf8":
            options["encoding"] = polars_encoding
        frame = pl.scan_csv(source, **options)
    if trim_ws:
        frame = _trim_strings(frame, {id} if id else set())
        frame = _null_after_trim(frame, na, {id} if id else set())
    if guess:
        skip_guess = {*(col_types or {}), *user_overrides, *([id] if id else [])}
        frame = _guess_types(frame, {str(name) for name in skip_guess}, sample_rows)
    result = TidyFrame(frame)
    assignments = _cast_assignments(col_types, strict=strict, marks=locale)
    if assignments:
        from tidy3.verbs import mutate

        result = result >> mutate(**assignments)
        if options.get("has_header", True) and result._problems is not None:
            # readr numbers a file's problem rows counting the header line:
            # the first data row is row 2. Skipped and comment lines do not
            # count, which is what Polars' row position already gives.
            result = result._with_lf(
                result._lf,
                groups=None,
                problems=result._problems.with_columns(pl.col("row") + 1),
            )
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

    Same options as :func:`scan_csv`, with readr's defaults: ``trim_ws=True``
    and ``na=["", "NA"]``. Fields are trimmed before column types are
    guessed, so ``" 6"`` is the number 6, and ``" NA "`` is missing. Types
    are guessed from the first ``infer_schema_length`` rows (100). ``col_types``
    keeps an id column as text (``"character"``) so leading zeros survive,
    and parses ``integer``, ``double``, ``logical``, ``date``, and
    ``datetime`` through the conversion engine. Date columns are ISO
    ``YYYY-MM-DD``. ``strict=False`` turns bad values into missing and
    records them for :func:`tidy3.convert.problems`.
    """
    delim = kwargs.pop("delim", None)
    if delim is None and "separator" not in kwargs:
        delim = ","
    if na is None and "null_values" not in kwargs:
        na = list(_READR_NA)
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
