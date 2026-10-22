"""Janitor-style cleaning helpers.

``remove_empty`` with ``which="rows"`` stays lazy on Polars. Deciding which
columns are empty or constant, and reading a header row for ``row_to_names``,
looks at the current values, so those results discover their schema from the
data. Blank strings are kept: only missing values count as empty. Nothing
here prints janitor's status messages.
"""

from __future__ import annotations

import math
import warnings
from datetime import date, datetime
from typing import Any

import numpy as np
import polars as pl

from tidy3.verbs import Verb

_WHICH = frozenset({"rows", "cols"})
_RETURNS = frozenset({"all", "match", "mismatch"})
_BIND = frozenset({"bind_rows", "rbind"})


def _bool(name: str, value: Any) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")


def _missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    try:
        import pandas as pd

        if value is pd.NA or value is pd.NaT:
            return True
        flag = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(flag) if isinstance(flag, bool) else False


def _which(which: Any) -> tuple[str, ...]:
    if which is None or which == "both" or which == ():
        return ("rows", "cols")
    parts = (which,) if isinstance(which, str) else tuple(which)
    if not parts or any(not isinstance(part, str) or part not in _WHICH for part in parts):
        raise ValueError(
            'remove_empty() which must be "rows", "cols", "both", or both of '
            '"rows" and "cols"'
        )
    ordered: list[str] = []
    for part in parts:
        if part not in ordered:
            ordered.append(part)
    return tuple(ordered)


def _cutoff(cutoff: Any) -> float:
    if isinstance(cutoff, bool) or not isinstance(cutoff, (int, float)):
        raise TypeError("remove_empty() cutoff must be a number")
    value = float(cutoff)
    if not 0 < value <= 1:
        raise ValueError("remove_empty() cutoff must be > 0 and <= 1")
    return value


def _groups(tf: Any, names: list[str]) -> list[str] | None:
    kept = [group for group in (tf._groups or []) if group in names]
    return kept or None


def _levels(tf: Any, names: list[str]) -> dict[str, Any]:
    kept = set(names)
    return {name: levels for name, levels in tf._category_levels.items() if name in kept}


def _renamed(mapping: dict[str, str], names: list[str] | None) -> list[str] | None:
    if not names:
        return None
    return [mapping[name] for name in names]


def _finish_pandas(tf: Any, pdf: Any, names: list[str]) -> Any:
    return tf._with_pdf(pdf, _groups(tf, names), category_levels=_levels(tf, names))


def _finish_polars(tf: Any, lf: pl.LazyFrame, names: list[str]) -> Any:
    # Dropping every column also drops the rows: Polars cannot store a row
    # count without a column. Pandas keeps that row count.
    selected = lf.select(names) if names else lf.select([])
    return tf._with_lf(selected, _groups(tf, names), category_levels=_levels(tf, names))


def _float_dtype(dtype: Any) -> bool:
    is_float = getattr(dtype, "is_float", None)
    return bool(is_float()) if callable(is_float) else False


def _missing_expr(name: str, dtype: Any) -> pl.Expr:
    missing = pl.col(name).is_null()
    if _float_dtype(dtype):
        missing = missing | pl.col(name).is_nan()
    return missing


def _row_mask_polars(lf: pl.LazyFrame, cutoff: float) -> pl.LazyFrame:
    schema = lf.collect_schema()
    names = schema.names()
    width = len(names)
    if width == 0:
        if cutoff != 1:
            raise ValueError(
                "remove_empty() cutoff needs at least one column when removing rows"
            )
        return lf.filter(pl.lit(False))
    missing = pl.sum_horizontal(
        [_missing_expr(name, dtype).cast(pl.Int64) for name, dtype in schema.items()]
    )
    if cutoff == 1:
        return lf.filter(missing != width)
    return lf.filter(((width - missing) / width) > cutoff)


def _empty_columns(lf: pl.LazyFrame, cutoff: float) -> list[str]:
    """Columns that are empty enough to drop. This collects one count row."""
    schema = lf.collect_schema()
    names = schema.names()
    if not names:
        return []
    stats = lf.select(
        pl.len().alias("__n"),
        *[
            _missing_expr(name, dtype).cast(pl.Int64).sum().alias(f"__c{index}")
            for index, (name, dtype) in enumerate(schema.items())
        ],
    ).collect()
    height = int(stats["__n"][0])
    if cutoff != 1 and height == 0:
        raise ValueError(
            "remove_empty() cutoff needs at least one row when removing columns"
        )
    kept: list[str] = []
    for index, name in enumerate(names):
        missing = int(stats[f"__c{index}"][0])
        if cutoff == 1:
            keep = missing != height
        else:
            keep = ((height - missing) / height) > cutoff
        if keep:
            kept.append(name)
    return kept


def _pandas_missing(pdf: Any) -> Any:
    return pdf.isna()


def remove_empty(which: Any = "both", *, cutoff: float = 1) -> Verb:
    """Drop empty rows and columns (R **janitor::remove_empty**).

    A row or column is empty when every value is missing. ``""`` is a value,
    so convert blanks to missing first if those should go too. ``cutoff=1``
    (the default) keeps a row or column that has any non-missing value.
    A smaller cutoff keeps it only when the fraction of non-missing values
    is strictly greater than ``cutoff``. Rows are removed before columns.
    ``cutoff`` other than 1 cannot be combined with both rows and columns.

    ``remove_empty("rows")`` on Polars is a lazy filter. Dropping columns
    counts missing values first, so the set of surviving columns depends on
    the data. A zero-row frame has no non-missing values, so ``which="cols"``
    drops every column. Grouping, factor levels, and parsing diagnostics are
    kept for columns that remain. No removal message is printed.
    """
    targets = _which(which)
    limit = _cutoff(cutoff)
    if len(targets) > 1 and limit != 1:
        raise ValueError(
            "remove_empty() cutoff must be used with only one of "
            "which='rows' or which='cols', not both"
        )

    def _apply(tf: Any) -> Any:
        if tf._backend == "pandas":
            pdf = tf._pdf.copy()
            if "rows" in targets:
                pdf = _drop_empty_pandas_rows(pdf, limit)
            names = list(pdf.columns)
            if "cols" in targets:
                names = _empty_pandas_columns(pdf, limit)
                pdf = pdf.loc[:, names] if names else pdf.iloc[:, 0:0]
            return _finish_pandas(tf, pdf.reset_index(drop=True), [str(name) for name in names])
        lf = tf._lf
        if "rows" in targets:
            lf = _row_mask_polars(lf, limit)
        names = _empty_columns(lf, limit) if "cols" in targets else list(lf.collect_schema().names())
        return _finish_polars(tf, lf, names)

    return Verb(_apply, "remove_empty")


def _drop_empty_pandas_rows(pdf: Any, cutoff: float) -> Any:
    width = pdf.shape[1]
    if width == 0:
        if cutoff != 1:
            raise ValueError(
                "remove_empty() cutoff needs at least one column when removing rows"
            )
        return pdf.iloc[0:0]
    present = (~_pandas_missing(pdf)).sum(axis=1)
    if cutoff == 1:
        mask = present.to_numpy() != 0
    else:
        mask = (present / width).to_numpy() > cutoff
    return pdf.loc[mask]


def _empty_pandas_columns(pdf: Any, cutoff: float) -> list[str]:
    names = [str(name) for name in pdf.columns]
    height = len(pdf)
    if cutoff != 1 and height == 0:
        raise ValueError(
            "remove_empty() cutoff needs at least one row when removing columns"
        )
    if height == 0:
        return []
    missing = _pandas_missing(pdf).sum(axis=0)
    kept: list[str] = []
    for name in names:
        count = int(missing[name])
        if cutoff == 1:
            keep = count != height
        else:
            keep = ((height - count) / height) > cutoff
        if keep:
            kept.append(name)
    return kept


def _frozen(value: Any) -> Any:
    """A hashable stand-in for a list, array or dict cell, compared by content."""
    if isinstance(value, (list, tuple, np.ndarray)):
        return ("__list", tuple(_frozen(item) for item in value))
    if isinstance(value, dict):
        return ("__dict", tuple((key, _frozen(item)) for key, item in value.items()))
    return value.item() if isinstance(value, np.generic) else value


def _constant_pandas(series: Any, na_rm: bool) -> bool:
    if series.dtype == object and series.map(lambda v: isinstance(v, (list, tuple, np.ndarray, dict))).any():
        series = series.map(_frozen)
    if na_rm:
        return int(series.dropna().nunique(dropna=True)) <= 1
    return int(series.nunique(dropna=False)) <= 1


def _constant_columns(lf: pl.LazyFrame, na_rm: bool) -> list[str]:
    """Columns whose values are not constant. This collects one uniqueness row."""
    schema = lf.collect_schema()
    names = schema.names()
    if not names:
        return []
    counts = []
    for index, (name, dtype) in enumerate(schema.items()):
        value = pl.col(name)
        if na_rm and _float_dtype(dtype):
            value = pl.when(value.is_nan()).then(None).otherwise(value)
        if na_rm:
            value = value.drop_nulls()
        counts.append(value.n_unique().alias(f"__c{index}"))
    row = lf.select(counts).collect()
    return [name for index, name in enumerate(names) if int(row[f"__c{index}"][0]) > 1]


def remove_constant(*, na_rm: bool = False) -> Verb:
    """Drop columns with fewer than two distinct values (R **janitor::remove_constant**).

    Missing counts as its own value when ``na_rm=False``, so a column that
    mixes one value and missing is kept. ``na_rm=True`` ignores missing
    values first; a column that is entirely missing is still removed.
    On Polars, null and NaN are different values. Pandas stores both as NaN
    in a float column, so that column looks constant. An empty column has
    no values and is removed, including every column of a zero-row frame.
    Unused factor levels do not count. The check reads the current values.
    Grouping and factor levels are kept for columns that remain. No removal
    message is printed.
    """
    _bool("remove_constant() na_rm", na_rm)

    def _apply(tf: Any) -> Any:
        if tf._backend == "pandas":
            pdf = tf._pdf.copy()
            names = [
                str(name)
                for name in pdf.columns
                if not _constant_pandas(pdf[name], na_rm)
            ]
            kept = pdf.loc[:, names] if names else pdf.iloc[:, 0:0]
            return _finish_pandas(tf, kept.reset_index(drop=True), names)
        names = _constant_columns(tf._lf, na_rm)
        return _finish_polars(tf, tf._lf, names)

    return Verb(_apply, "remove_constant")


def _quote_levels(levels: Any) -> str:
    return ", ".join(f'"{level}"' for level in levels)


def _factor_class(levels: Any, ordered: bool, strict: bool) -> str:
    if not strict:
        return "factor"
    text = f"factor(levels=c({_quote_levels(levels)}))"
    if ordered:
        return f"ordered, {text}"
    return text


def _plain_polars(dtype: Any) -> str:
    if dtype == pl.Null or getattr(dtype, "base_type", lambda: None)() == pl.Null:
        return "logical"
    if (
        _float_dtype(dtype)
        or bool(getattr(dtype, "is_decimal", lambda: False)())
        or type(dtype).__name__.startswith("Decimal")
    ):
        return "numeric"
    is_integer = getattr(dtype, "is_integer", None)
    if callable(is_integer) and is_integer():
        return "integer"
    # Polars Boolean.is_boolean() is false; compare the dtype itself.
    if dtype == pl.Boolean or getattr(dtype, "base_type", lambda: None)() == pl.Boolean:
        return "logical"
    if dtype == pl.Date or getattr(dtype, "base_type", lambda: None)() == pl.Date:
        return "Date"
    if isinstance(dtype, pl.Datetime):
        return "POSIXct, POSIXt"
    if dtype in {pl.String, pl.Utf8, pl.Categorical} or str(getattr(dtype, "base_type", lambda: dtype)()) in {
        "String",
        "Utf8",
    }:
        return "character"
    return str(dtype)


def _pandas_class(series: Any, *, levels: Any, ordered: bool, strict: bool) -> str:
    import pandas as pd

    dtype = series.dtype
    if isinstance(dtype, pd.CategoricalDtype) or (
        isinstance(dtype, pd.ArrowDtype) and "dictionary" in str(dtype)
    ):
        if levels is None:
            levels = list(series.cat.categories) if hasattr(series.dtype, "categories") else []
            ordered = bool(getattr(series.dtype, "ordered", False))
        return _factor_class(levels, ordered, strict)
    if pd.api.types.is_bool_dtype(dtype):
        return "logical"
    if pd.api.types.is_integer_dtype(dtype):
        return "integer"
    if pd.api.types.is_float_dtype(dtype):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "POSIXct, POSIXt"
    if pd.api.types.is_string_dtype(dtype) and not pd.api.types.is_object_dtype(dtype):
        return "character"
    if pd.api.types.is_object_dtype(dtype):
        return _object_class(series)
    return str(dtype)


def _object_class(series: Any) -> str:
    seen: list[Any] = []
    for value in series.tolist():
        if _missing(value):
            continue
        seen.append(value)
        if len(seen) == 8:
            break
    if seen and all(type(value) is date for value in seen):
        return "Date"
    if seen and all(isinstance(value, datetime) for value in seen):
        return "POSIXct, POSIXt"
    return "character"


def _column_class(tf: Any, name: str, strict: bool) -> str:
    recorded = tf._category_levels.get(name)
    levels = list(recorded) if recorded is not None else None
    ordered = bool(getattr(recorded, "ordered", False)) if recorded is not None else False
    if tf._backend == "pandas":
        return _pandas_class(tf._pdf[name], levels=levels, ordered=ordered, strict=strict)
    dtype = tf._lf.collect_schema()[name]
    if isinstance(dtype, (pl.Enum, pl.Categorical)):
        if levels is None and isinstance(dtype, pl.Enum):
            levels = dtype.categories.to_list()
            ordered = True
        return _factor_class(levels or [], ordered, strict)
    return _plain_polars(dtype)


def describe_class(x: Any, *, strict_description: bool = True) -> str:
    """Describe a column the way janitor's ``describe_class`` does.

    Factors are ``"factor"`` unless ``strict_description=True``, which
    includes the levels and an ``ordered`` prefix. Other common classes use
    R's names: ``integer``, ``numeric``, ``logical``, ``character``,
    ``Date``, and ``POSIXct, POSIXt``. A Python ``int`` is ``integer``;
    pass a float for ``numeric``.
    """
    _bool("describe_class() strict_description", strict_description)
    from tidy3.frame import TidyFrame

    if isinstance(x, TidyFrame):
        raise TypeError(
            "describe_class() describes a column, not a frame; use compare_df_cols()"
        )
    if isinstance(x, pl.Series):
        frame = TidyFrame(pl.DataFrame({x.name or "value": x}).lazy())
        return _column_class(frame, frame.columns[0], strict_description)
    try:
        import pandas as pd

        if isinstance(x, pd.Series):
            frame = TidyFrame(x.to_frame(name=str(x.name or "value")))
            return _column_class(frame, frame.columns[0], strict_description)
    except ImportError:
        pass
    if isinstance(x, bool):
        return "logical"
    if isinstance(x, int):
        return "integer"
    if isinstance(x, float):
        return "numeric"
    if isinstance(x, str):
        return "character"
    if type(x) is date:
        return "Date"
    if isinstance(x, datetime):
        return "POSIXct, POSIXt"
    if x is None:
        return "logical"
    return type(x).__name__


def _as_frame(value: Any) -> Any:
    from tidy3.frame import TidyFrame, tidy

    if isinstance(value, TidyFrame):
        return value
    return tidy(value)


def _expand(name: str, value: Any) -> list[tuple[str, Any]]:
    if isinstance(value, dict):
        if not value:
            raise ValueError(f"compare_df_cols() mapping {name!r} is empty")
        items = []
        for index, (key, item) in enumerate(value.items(), start=1):
            label = str(key) if str(key) else f"{name}_{index}"
            items.append((label, _as_frame(item)))
        return items
    if isinstance(value, (list, tuple)) and not isinstance(value, (str, bytes)):
        if not value:
            raise ValueError(f"compare_df_cols() list {name!r} is empty")
        return [
            (f"{name}_{index}", _as_frame(item))
            for index, item in enumerate(value, start=1)
        ]
    return [(name, _as_frame(value))]


def _input_frames(args: tuple[Any, ...], named: dict[str, Any]) -> list[tuple[str, Any]]:
    items: list[tuple[str, Any]] = []
    for index, value in enumerate(args, start=1):
        items.extend(_expand(f"..{index}", value))
    for name, value in named.items():
        items.extend(_expand(str(name), value))
    if not items:
        raise TypeError("compare_df_cols() requires at least one data frame")
    labels = _unique_names([name for name, _frame in items])
    if any(label == "column_name" for label in labels):
        raise ValueError(
            "compare_df_cols() input names cannot be 'column_name'"
        )
    return list(zip(labels, [frame for _name, frame in items]))


def _unique_names(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    result: list[str] = []
    for name in names:
        if name not in seen:
            seen[name] = 1
            result.append(name)
            continue
        seen[name] += 1
        candidate = f"{name}_{seen[name]}"
        while candidate in seen:
            seen[name] += 1
            candidate = f"{name}_{seen[name]}"
        seen[candidate] = 1
        result.append(candidate)
    return result


def _bind_match(classes: list[str | None], method: str) -> bool:
    if method == "rbind":
        first = classes[0]
        return all(value == first for value in classes[1:])
    present = [value for value in classes if value is not None]
    anchor = present[0] if present else None
    return all(value is None or value == anchor for value in classes[1:])


def _class_frame(backend: str, labels: list[str], rows: list[tuple[str, list[str | None]]]) -> Any:
    from tidy3.frame import tidy

    columns: dict[str, list[Any]] = {"column_name": [name for name, _classes in rows]}
    for index, label in enumerate(labels):
        columns[label] = [classes[index] for _name, classes in rows]
    if backend == "pandas":
        import pandas as pd

        pdf = pd.DataFrame(
            {name: pd.Series(values, dtype="string") for name, values in columns.items()}
        )
        return tidy(pdf, backend="pandas")
    frame = pl.DataFrame(
        {name: pl.Series(name, values, dtype=pl.String) for name, values in columns.items()}
    )
    return tidy(frame, backend="polars")


def compare_df_cols(
    *frames: Any,
    return_: str = "all",
    bind_method: str = "bind_rows",
    strict_description: bool = False,
    **named: Any,
) -> Any:
    """Compare column names and classes before binding rows.

    Pass frames by keyword (``compare_df_cols(old=old, new=new)``) or in a
    mapping. Positional frames are named ``..1``, ``..2``, and so on. A list
    uses ``name_1``, ``name_2``. The result has ``column_name`` plus one
    column of class descriptions per frame. A column missing from a frame
    is missing in that description.

    ``return_`` is ``"all"``, ``"match"``, or ``"mismatch"``. With
    ``bind_method="bind_rows"`` (the default), a column missing from some
    frames still matches when every present class agrees. ``"rbind"``
    requires every frame to have the column and the same class. More than
    one frame sorts rows by column name, as R's ``merge`` does; one frame
    keeps its column order. ``strict_description=True`` treats different
    factor levels as different classes.

    Descriptions come from dtypes and recorded factor levels, not from a
    row scan, except that a pandas object column may be read far enough to
    tell a date from text. A frame with no columns is omitted. No input may
    be named ``column_name``.
    """
    if return_ not in _RETURNS:
        raise ValueError('compare_df_cols() return_ must be "all", "match", or "mismatch"')
    if bind_method not in _BIND:
        raise ValueError('compare_df_cols() bind_method must be "bind_rows" or "rbind"')
    _bool("compare_df_cols() strict_description", strict_description)
    inputs = _input_frames(frames, named)
    present: list[tuple[str, Any]] = []
    for label, frame in inputs:
        if frame.width == 0:
            warnings.warn(
                f"{label} has zero columns and will not appear in output.",
                stacklevel=2,
            )
            continue
        present.append((label, frame))
    backend = inputs[0][1].backend
    if not present:
        return _class_frame(backend, [], [])
    labels = [label for label, _frame in present]
    order: list[str] = []
    seen: set[str] = set()
    described: list[dict[str, str]] = []
    for _label, frame in present:
        classes = {
            name: _column_class(frame, name, strict_description) for name in frame.columns
        }
        described.append(classes)
        for name in frame.columns:
            if name not in seen:
                seen.add(name)
                order.append(name)
    if len(present) > 1:
        order = sorted(order)
    rows = [
        (
            name,
            [classes.get(name) for classes in described],
        )
        for name in order
    ]
    if return_ != "all" and len(present) == 1:
        warnings.warn(
            "Only one data frame was provided, so all its classes are returned.",
            stacklevel=2,
        )
    elif return_ != "all":
        want = return_ == "match"
        rows = [row for row in rows if _bind_match(row[1], bind_method) is want]
    return _class_frame(backend, labels, rows)


def compare_df_cols_same(
    *frames: Any,
    bind_method: str = "bind_rows",
    verbose: bool = True,
    **named: Any,
) -> bool:
    """Whether the frames can be row-bound under ``bind_method``.

    This is true when :func:`compare_df_cols` reports no mismatches. One
    frame is not the same as another frame, so a single frame with columns
    returns ``False``, as janitor does. ``verbose`` is accepted and ignored:
    the mismatch table is not printed. Use ``compare_df_cols(return_="mismatch")``.
    """
    _bool("compare_df_cols_same() verbose", verbose)
    mismatches = compare_df_cols(
        *frames,
        return_="mismatch",
        bind_method=bind_method,
        **named,
    )
    return mismatches.collect().height == 0


def _column_position(columns: list[str], column: Any) -> int:
    if isinstance(column, str):
        if column not in columns:
            raise KeyError(f"find_header() column not found: {column!r}")
        return columns.index(column)
    if isinstance(column, bool) or not isinstance(column, int) or column < 1:
        raise TypeError("find_header() column must be a name or a positive integer")
    if column > len(columns):
        raise IndexError(f"find_header() column position out of range: {column}")
    return column - 1


def _materialized(tf: Any) -> Any:
    if tf._backend == "pandas":
        return tf._pdf
    return tf._lf.collect()


def _cell(data: Any, row: int, column: int) -> Any:
    if isinstance(data, pl.DataFrame):
        return data.row(row)[column]
    return data.iat[row, column]


def _height(data: Any) -> int:
    if isinstance(data, pl.DataFrame):
        return data.height
    return len(data)


def _header_text(value: Any) -> str:
    if _missing(value):
        return "NA"
    # numpy.bool is not a Python bool, and str() of it is "True".
    if isinstance(value, bool) or type(value).__name__ in {"bool", "bool_"}:
        return "TRUE" if bool(value) else "FALSE"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, datetime):
        return value.isoformat(sep=" ")
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def find_header(
    data: Any = None,
    *,
    value: Any = None,
    column: Any = 1,
) -> Any:
    """Return the 1-based header row (R **janitor::find_header**).

    With no ``value``, this is the first row that has no missing values.
    With ``value``, this is the first row where ``column`` (a name or a
    1-based position, defaulting to the first column) equals that value.
    Several matches warn and use the first row. The search reads the
    current rows. ``find_header()`` without ``data`` is the pipe form.
    """
    if data is None:
        def _apply(tf: Any) -> int:
            return _find_header(tf, value=value, column=column)

        return Verb(_apply, "find_header")
    from tidy3.frame import TidyFrame, tidy

    frame = data if isinstance(data, TidyFrame) else tidy(data)
    return _find_header(frame, value=value, column=column)


def _find_header(tf: Any, *, value: Any, column: Any, data: Any = None) -> int:
    if data is None:
        data = _materialized(tf)
    columns = [str(name) for name in (data.columns if isinstance(data, pl.DataFrame) else data.columns)]
    height = _height(data)
    if value is None:
        if column != 1:
            raise TypeError(
                "find_header() column is only used together with value"
            )
        for row in range(height):
            if all(not _missing(_cell(data, row, index)) for index in range(len(columns))):
                return row + 1
        raise ValueError("No complete rows (rows with zero missing values) were found.")
    if not columns:
        raise ValueError(f"The value {value!r} was not found in column {column}")
    index = _column_position(columns, column)
    matches = [
        row + 1
        for row in range(height)
        if not _missing(_cell(data, row, index)) and _cell(data, row, index) == value
    ]
    if not matches:
        raise ValueError(f"The value {value!r} was not found in column {column}")
    if len(matches) > 1:
        warnings.warn(
            f"The value {value!r} was found {len(matches)} times in column {column}; "
            "using the first row.",
            stacklevel=2,
        )
    return matches[0]


def row_to_names(
    row_number: Any,
    *,
    remove_row: bool = True,
    remove_rows_above: bool = True,
    value: Any = None,
    column: Any = None,
) -> Verb:
    """Use one row as the column names (R **janitor::row_to_names**).

    ``row_number`` is 1-based, or ``"find_header"`` to call
    :func:`find_header`. ``value`` and ``column`` are only allowed with
    ``"find_header"``. The name row and the rows above it are removed by
    default. A missing name becomes ``"NA"``. Duplicate names are made
    unique (``id``, ``id_2``) because a frame cannot repeat column names;
    janitor 2.2.1 only warns. Reading the name row runs the plan up to
    this verb. Grouping and factor levels follow the new names.
    """
    _bool("row_to_names() remove_row", remove_row)
    _bool("row_to_names() remove_rows_above", remove_rows_above)
    if row_number != "find_header" and (value is not None or column is not None):
        raise TypeError(
            "row_to_names() value and column are only allowed when "
            "row_number='find_header'"
        )
    if row_number != "find_header" and (
        isinstance(row_number, bool)
        or not isinstance(row_number, int)
    ):
        raise TypeError("row_to_names() row_number must be a positive integer or 'find_header'")

    def _apply(tf: Any) -> Any:
        data = _materialized(tf)
        columns = [str(name) for name in data.columns]
        height = _height(data)
        if row_number == "find_header":
            chosen = _find_header(
                tf,
                value=value,
                column=1 if column is None else column,
                data=data,
            )
        else:
            chosen = row_number
        if isinstance(chosen, bool) or not isinstance(chosen, int) or chosen < 1 or chosen > height:
            raise ValueError(
                f"row_to_names() row_number {chosen} is outside 1:{height}"
            )
        raw = [_header_text(_cell(data, chosen - 1, index)) for index in range(len(columns))]
        names = _unique_names(raw)
        if names != raw:
            warnings.warn(
                f"Row {chosen} does not provide unique names. "
                "They were made unique; janitor only warns.",
                stacklevel=2,
            )
        removed: set[int] = set()
        if remove_row:
            removed.add(chosen)
        if remove_rows_above:
            removed.update(range(1, chosen))
        keep = [index for index in range(height) if (index + 1) not in removed]
        mapping = dict(zip(columns, names))
        groups = _renamed(mapping, tf._groups)
        levels = {
            mapping[name]: levels
            for name, levels in tf._category_levels.items()
            if name in mapping
        }
        if isinstance(data, pl.DataFrame):
            sliced = data[keep]
            sliced.columns = names
            return tf._with_lf(sliced.lazy(), groups, category_levels=levels)
        pdf = data.iloc[keep].copy()
        pdf.columns = names
        return tf._with_pdf(pdf.reset_index(drop=True), groups, category_levels=levels)

    return Verb(_apply, "row_to_names")
