"""Tidy reshaping and missing-data verbs for both tidy3 backends."""

from __future__ import annotations

import inspect
import re
import warnings
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import polars as pl

from tidy3.tidyselect import resolve_selection
from tidy3.verbs import Verb, _operation_groups


@dataclass(frozen=True)
class Nesting:
    columns: tuple[Any, ...]


def nesting(*cols: Any) -> Nesting:
    """Keep only combinations of columns that already occur together."""
    if not cols:
        raise TypeError("nesting() requires at least one column")
    return Nesting(tuple(cols))


@dataclass(frozen=True)
class FullSeq:
    """``full_seq("year", 1)``: filled in against a frame's column."""

    column: str
    period: float
    tol: float


def full_seq(x: Any, period: float, tol: float = 1e-6) -> Any:
    """Every value from ``min(x)`` to ``max(x)`` in steps of ``period``.

    Give a column name inside ``expand()``/``complete()``
    (``complete("country", year=full_seq("year", 1))``), or values for a
    plain list (``full_seq([1, 2, 5], 1)`` is ``[1, 2, 3, 4, 5]``). Raises
    if ``x`` has missing values or is not on a regular grid, like tidyr.
    """
    if not period or period <= 0:
        raise ValueError("full_seq() period must be a positive number")
    if isinstance(x, str):
        return FullSeq(x, float(period), float(tol))
    # A Series or array goes to NumPy directly; a missing value becomes NaN,
    # which the checks below reject as before.
    values = x.to_numpy() if hasattr(x, "to_numpy") else list(x)
    return _full_seq_values(values, period, tol)


def _full_seq_values(values: Any, period: float, tol: float) -> list[Any]:
    if len(values) == 0:
        return []
    fast = _full_seq_array(values, period, tol)
    if fast is not None:
        return fast
    import math

    if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in values):
        raise ValueError("full_seq() x must not contain missing values")
    numbers = [float(v) for v in values]
    low, high = min(numbers), max(numbers)
    for value in numbers:
        offset = (value - low) % period
        if offset > tol and period - offset > tol:
            raise ValueError("full_seq() x is not a regular sequence")
    count = int(round((high - low) / period))
    # Decimal numbers, like R's full_seq(); expand()/complete() then cast to
    # the column's own type, so a whole-number year column stays whole.
    return [low + index * period for index in range(count + 1)]


def _full_seq_array(values: list[Any], period: float, tol: float) -> list[float] | None:
    """The sequence when every value is a finite number. None asks for the slow path."""
    try:
        numbers = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if numbers.shape != (len(values),) or not np.isfinite(numbers).all():
        return None
    # Polars over the same memory: its remainder runs on several threads,
    # where NumPy's mod is the slowest step. The values are finite and at
    # least ``low``, so the remainder is in [0, period) as with np.mod.
    column = pl.Series(numbers)
    low = float(column.min())
    high = float(column.max())
    offset = (column - low) % period
    if ((offset > tol) & ((period - offset) > tol)).any():
        raise ValueError("full_seq() x is not a regular sequence")
    count = int(round((high - low) / period))
    return (low + np.arange(count + 1, dtype=np.float64) * period).tolist()


def _column_full_seq(lf: pl.LazyFrame, column: str, period: float, tol: float) -> list[float] | None:
    """full_seq of a numeric Polars column without copying the column into Python."""
    dtype = lf.collect_schema().get(column)
    if dtype is None or not (dtype.is_integer() or dtype.is_float()):
        return None
    # Column full_seq ignores nulls, matching the collect-then-drop path.
    # A float NaN is still a missing value and raises.
    number = pl.col(column).cast(pl.Float64)
    if dtype.is_float():
        nans = lf.select(number.is_nan().sum().alias("nans")).collect()
        if nans["nans"][0]:
            raise ValueError("full_seq() x must not contain missing values")
    stats = lf.select(number.min().alias("low"), number.max().alias("high")).collect()
    low = stats["low"][0]
    high = stats["high"][0]
    if low is None:
        return []
    low, high = float(low), float(high)
    offset = (number - low) % period
    off_grid = lf.select(
        (offset.is_not_null() & (offset > tol) & ((period - offset) > tol)).any().alias("bad")
    ).collect()
    if off_grid["bad"][0]:
        raise ValueError("full_seq() x is not a regular sequence")
    count = int(round((high - low) / period))
    return [low + index * period for index in range(count + 1)]


def _value_list(tf: Any, name: str, spec: Any) -> list[Any]:
    """Unique sorted values for a named ``expand()``/``complete()`` input."""
    if isinstance(spec, FullSeq):
        from tidy3.frame import note_data_read

        note_data_read("full_seq() reads the column's range from the data")
        fast = _column_full_seq(tf._lf, spec.column, spec.period, spec.tol)
        if fast is not None:
            return fast
        present = (
            tf._lf.select(pl.col(spec.column).drop_nulls()).collect().to_series().to_list()
        )
        return _full_seq_values(present, spec.period, spec.tol)
    values = list(spec.tolist() if hasattr(spec, "tolist") else spec)
    return _unique_sorted(values)


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and value != value)


def _unique_sorted(values: list[Any]) -> list[Any]:
    """tidyr's crossing order: distinct values sorted, missing last."""
    present = sorted({v for v in values if not _is_missing(v)})
    return [*present, None] if any(_is_missing(v) for v in values) else present


def _named_values_frame(tf: Any, name: str, values: list[Any], backend: str) -> Any:
    """One-column frame of *values*, cast to the column's type when present."""
    frame = pl.DataFrame({name: values}, strict=False).lazy()
    schema = tf._lf.collect_schema()
    if name in schema:
        frame = frame.with_columns(pl.col(name).cast(schema[name], strict=False))
    return frame


def _columns(tf: Any) -> list[str]:
    return tf._lf.collect_schema().names()


def _temp_name(columns: Iterable[str], base: str) -> str:
    occupied = set(columns)
    name = base
    index = 1
    while name in occupied:
        name = f"{base}_{index}"
        index += 1
    return name


def _result_groups(tf: Any, columns: Iterable[str]) -> list[str] | None:
    available = set(columns)
    groups = [name for name in (tf._groups or []) if name in available]
    return groups or None


def _is_float_dtype(dtype: Any) -> bool:
    try:
        return dtype.base_type() in {pl.Float32, pl.Float64}
    except AttributeError:
        return False


def drop_na(*cols: Any) -> Verb:
    """Drop rows missing a value in any selected column."""

    def _apply(tf):
        selected = resolve_selection(tf, cols) if cols else _columns(tf)
        if not selected:
            return tf
        schema = tf._lf.collect_schema()
        predicates = []
        for name in selected:
            valid = pl.col(name).is_not_null()
            if _is_float_dtype(schema[name]):
                valid = valid & pl.col(name).is_not_nan()
            predicates.append(valid)
        return tf._with_lf(
            tf._lf.filter(pl.all_horizontal(predicates)), groups=tf._groups
        )

    return Verb(_apply, "drop_na")


def replace_na(replace: dict[str, Any]) -> Verb:
    """Replace missing values using a ``{column: value}`` mapping."""
    if not isinstance(replace, dict) or not replace:
        raise TypeError("replace_na() requires a non-empty replacement mapping")

    def _apply(tf):
        columns = _columns(tf)
        missing = [name for name in replace if name not in columns]
        if missing:
            raise KeyError(f"replace_na() columns not found: {missing}")
        schema = tf._lf.collect_schema()
        expressions = []
        for name, value in replace.items():
            expression = pl.col(name)
            if _is_float_dtype(schema[name]):
                expression = expression.fill_nan(value)
            expressions.append(expression.fill_null(value).alias(name))
        return tf._with_lf(
            tf._lf.with_columns(expressions), groups=tf._groups
        )

    return Verb(_apply, "replace_na")


def fill(
    *cols: Any,
    direction: str = "down",
    by: Any = None,
) -> Verb:
    """Fill missing values down/up, respecting persistent or transient groups."""
    if not cols:
        raise TypeError("fill() requires at least one column selection")
    if direction not in {"down", "up", "downup", "updown"}:
        raise ValueError("direction must be down, up, downup, or updown")

    def _apply(tf):
        selected = resolve_selection(tf, cols)
        groups, transient = _operation_groups(tf, by, "fill")

        schema = tf._lf.collect_schema()
        strategies = {
            "down": ["forward"],
            "up": ["backward"],
            "downup": ["forward", "backward"],
            "updown": ["backward", "forward"],
        }[direction]
        expressions = []
        for name in selected:
            expression = pl.col(name)
            if _is_float_dtype(schema[name]):
                expression = expression.fill_nan(None)
            for strategy in strategies:
                expression = expression.fill_null(strategy=strategy)
            if groups:
                expression = expression.over(groups)
            expressions.append(expression.alias(name))
        return tf._with_lf(
            tf._lf.with_columns(expressions),
            groups=None if transient else tf._groups,
        )

    return Verb(_apply, "fill")


def _expansion_units(tf: Any, specs: tuple[Any, ...]) -> list[list[str]]:
    units: list[list[str]] = []
    for spec in specs:
        if isinstance(spec, Nesting):
            selected = resolve_selection(tf, spec.columns)
            if selected:
                units.append(selected)
        else:
            units.extend([[name] for name in resolve_selection(tf, [spec])])
    flattened = [name for unit in units for name in unit]
    if len(flattened) != len(set(flattened)):
        raise ValueError("expand() columns must be selected only once")
    return units


def _factor_units(tf: Any, specs: tuple[Any, ...]) -> dict[str, Any]:
    """Factor columns named outside ``nesting()``: they expand to every level.

    tidyr expands a bare factor to all its levels, including unused ones,
    while ``nesting()`` keeps only combinations that occur. Values are the
    Polars Enum dtype, or the pandas categorical dtype.
    """
    import pandas as pd

    bare = [
        name for spec in specs if not isinstance(spec, Nesting)
        for name in resolve_selection(tf, [spec])
    ]
    schema = tf._lf.collect_schema()
    return {name: schema[name] for name in bare if isinstance(schema[name], pl.Enum)}


def _pandas_levels(series: Any) -> Any:
    import pandas as pd

    values = list(series.cat.categories)
    if series.isna().any():
        values.append(None)
    return pd.DataFrame({series.name: pd.Categorical(values, dtype=series.dtype)})


def _polars_levels(tf: Any, name: str, dtype: Any) -> pl.LazyFrame:
    """Every level of an Enum column, then null when the column has one."""
    every = pl.LazyFrame({name: pl.Series(name, dtype.categories.to_list(), dtype=dtype)})
    missing = tf._lf.select(pl.col(name)).filter(pl.col(name).is_null()).head(1)
    return pl.concat([every, missing])


def expand(*cols: Any, **values: Any) -> Verb:
    """Generate observed-value combinations, within groups when grouped.

    Named inputs give the values to use instead of the observed ones, as in
    tidyr: ``expand("country", year=full_seq("year", 1))``.
    """
    if not cols and not values:
        raise TypeError("expand() requires at least one column")
    named_specs = dict(values)

    def _apply(tf):
        units = _expansion_units(tf, cols)
        levels = _factor_units(tf, cols)
        groups = list(tf._groups or [])
        named = {name: _value_list(tf, name, spec) for name, spec in named_specs.items()}
        selected = [name for unit in units for name in unit] + list(named)
        overlap = [name for name in selected if name in groups]
        if overlap:
            raise ValueError(
                f"expand() cannot expand grouping columns: {overlap}"
            )

        if groups:
            result = tf._lf.select(groups).unique(maintain_order=True)
            for unit in units:
                if unit[0] in levels:
                    name = unit[0]
                    every = pl.LazyFrame(
                        {name: pl.Series(name, levels[name].categories.to_list(), dtype=levels[name])}
                    )
                    values = pl.concat([
                        tf._lf.select(groups).unique(maintain_order=True).join(every, how="cross"),
                        tf._lf.select([*groups, name]).filter(pl.col(name).is_null()).unique(
                            maintain_order=True
                        ),
                    ])
                else:
                    values = tf._lf.select([*groups, *unit]).unique(
                        maintain_order=True
                    )
                result = result.join(
                    values, on=groups, how="inner", maintain_order="left_right"
                )
        else:
            result = None
            for unit in units:
                values = (
                    _polars_levels(tf, unit[0], levels[unit[0]]) if unit[0] in levels
                    else tf._lf.select(unit).unique(maintain_order=True)
                )
                result = (
                    values
                    if result is None
                    else result.join(values, how="cross")
                )
        for name, listed in named.items():
            frame = _named_values_frame(tf, name, listed, "polars")
            result = frame if result is None else result.join(frame, how="cross")
        if result is None:
            result = tf._lf.select([]).head(0)
        ordered = [*groups, *selected]
        return tf._with_lf(
            result.select(ordered).sort(ordered, nulls_last=True, maintain_order=True),
            groups=groups or None,
            rowwise=False,
        )

    return Verb(_apply, "expand")


def _grid_units(args: tuple[Any, ...], kwargs: dict[str, Any], *, crossing: bool) -> list[Any]:
    """A Polars LazyFrame for each expand_grid()/crossing() input."""
    from tidy3.frame import TidyFrame, tidy

    units = []
    for arg in args:
        if isinstance(arg, TidyFrame):
            unit = arg._lf
        elif isinstance(arg, pl.DataFrame):
            unit = arg.lazy()
        elif isinstance(arg, pl.LazyFrame):
            unit = arg
        else:
            unit = tidy(arg)._lf
        if crossing:
            names = unit.collect_schema().names()
            unit = unit.unique(maintain_order=True).sort(names, nulls_last=True, maintain_order=True)
        units.append(unit)
    for name, values in kwargs.items():
        listed = list(values.tolist() if hasattr(values, "tolist") else values)
        if crossing:
            listed = _unique_sorted(listed)
        units.append(pl.LazyFrame({name: listed}, strict=False))
    return units


def _cross(units: list[Any]) -> Any:
    from tidy3.frame import TidyFrame

    if not units:
        result = pl.LazyFrame()
    else:
        result = units[0]
        for unit in units[1:]:
            # The first input varies slowest, as in tidyr.
            result = result.join(unit, how="cross", maintain_order="left")
    return TidyFrame(result)


def expand_grid(*frames: Any, **values: Any) -> Any:
    """Every combination of the inputs, as given (tidyr ``expand_grid()``).

    Tables keep their rows together; named inputs are value lists. Nothing
    is de-duplicated or sorted, and the first input varies slowest::

        expand_grid(x=[1, 2], y=["a", "b"])
        expand_grid(stores, week=range(1, 53))
    """
    return _cross(_grid_units(frames, values, crossing=False))


def crossing(*frames: Any, **values: Any) -> Any:
    """Like :func:`expand_grid`, on each input's distinct values, sorted.

    Missing values sort last, as in tidyr ``crossing()``.
    """
    return _cross(_grid_units(frames, values, crossing=True))


def uncount(weights: Any, *, remove: bool = True, id: str | None = None) -> Verb:
    """Repeat each row ``weights`` times (tidyr ``uncount()``).

    ``weights`` is a column name or a number. The weights column is dropped
    unless ``remove=False``; ``id="copy"`` adds each copy's 1-based number.
    """

    def _apply(tf):
        columns = _columns(tf)
        name = weights if isinstance(weights, str) else None
        if name is not None and name not in columns:
            raise KeyError(f"uncount() weights column not found: {name!r}")
        count = pl.col(name) if name else pl.lit(weights)
        missing, negative, fractional = (
            tf._lf.select(
                count.is_null().any().alias("missing"),
                (count < 0).any().alias("negative"),
                (count.cast(pl.Float64) % 1 != 0).any().alias("fractional"),
            )
            .collect()
            .row(0)
        )
        _check_weights(missing, negative, fractional)
        copy = _temp_name(columns, "__tidy3_copy")
        # Drop zero weights first: explode() never sees an empty list, whose
        # handling differs between Polars 1 and 2.
        lf = (
            tf._lf.filter(count > 0)
            .with_columns(pl.int_ranges(0, count.cast(pl.Int64)).alias(copy))
            .explode(copy, **_EXPLODE_OPTIONS)
        )
        if id is not None:
            lf = lf.with_columns((pl.col(copy) + 1).alias(id))
        drop = [copy] + ([name] if remove and name is not None else [])
        lf = lf.drop(drop)
        ordered = lf.collect_schema().names()
        return tf._with_lf(lf, groups=_result_groups(tf, ordered))

    return Verb(_apply, "uncount")


# Polars 1.4x warns unless empty_as_null is explicit; older 1.x lacks it.
_EXPLODE_OPTIONS = (
    {"empty_as_null": True}
    if "empty_as_null" in inspect.signature(pl.LazyFrame.explode).parameters
    else {}
)


def _check_weights(missing: bool, negative: bool, fractional: bool) -> None:
    if missing:
        raise ValueError("uncount() weights must not be missing")
    if negative:
        raise ValueError("uncount() weights must be zero or more")
    if fractional:
        raise ValueError("uncount() weights must be whole numbers")


def complete(
    *cols: Any,
    fill: dict[str, Any] | None = None,
    explicit: bool = True,
    **values: Any,
) -> Verb:
    """Make implicit missing combinations explicit, optionally filling them.

    Named inputs give the values to complete over, as in tidyr:
    ``complete("country", year=full_seq("year", 1))``.
    """
    if not cols and not values:
        raise TypeError("complete() requires at least one column")
    if fill is not None and not isinstance(fill, dict):
        raise TypeError("complete() fill must be a mapping or None")

    def _apply(tf):
        expanded = expand(*cols, **values)._fn(tf)
        groups = list(tf._groups or [])
        units = _expansion_units(tf, cols)
        keys = [*groups, *(name for unit in units for name in unit), *values]
        marker = _temp_name(_columns(tf), "__tidy3_complete")
        replacements = fill or {}

        original = tf._lf.with_columns(pl.lit(True).alias(marker))
        output = expanded._lf.join(
            original,
            on=keys,
            how="left",
            maintain_order="left_right",
            nulls_equal=True,  # a missing key matches its own row, as in tidyr
        )
        schema = output.collect_schema()
        expressions = []
        for name, value in replacements.items():
            if name not in schema:
                raise KeyError(f"complete() fill column not found: {name!r}")
            missing = pl.col(name).is_null()
            if _is_float_dtype(schema[name]):
                missing = missing | pl.col(name).is_nan()
            if not explicit:
                missing = missing & pl.col(marker).is_null()
            expressions.append(
                pl.when(missing)
                .then(pl.lit(value))
                .otherwise(pl.col(name))
                .alias(name)
            )
        if expressions:
            output = output.with_columns(expressions)
        return tf._with_lf(
            output.drop(marker), groups=groups or None, rowwise=False
        )

    return Verb(_apply, "complete")


_NAME_REPAIRS = ("check_unique", "minimal", "unique", "universal")


def _check_names_repair(names_repair: Any) -> None:
    if not callable(names_repair) and names_repair not in _NAME_REPAIRS:
        choices = ", ".join(repr(choice) for choice in _NAME_REPAIRS)
        raise ValueError(f"names_repair must be one of {choices}, or a function")


def _python_name(name: str) -> str:
    """vctrs' "universal" name in Python spelling: a valid identifier.

    Characters an identifier cannot hold become ``_`` (vctrs: ``.``), and a
    name starting with a digit or a keyword gets a leading ``_`` (vctrs:
    ``..`` or ``.``).
    """
    import keyword

    text = "".join(char if ("a" + char).isidentifier() else "_" for char in name)
    if text and (text[0].isdigit() or keyword.iskeyword(text)):
        text = "_" + text
    return text


def _repair_names(names: list[str], names_repair: Any, verb: str) -> list[str]:
    """Output names after tidyr's ``names_repair``, in tidy3's spelling.

    ``"unique"`` numbers duplicated or empty names by position as
    ``bind_cols()`` does (``x_1``, where vctrs writes ``x...1``), and
    ``"universal"`` also makes each name a Python identifier. Polars columns
    need unique names, so a repair that leaves duplicates raises.
    """
    from tidy3.verbs import _repair_bind_names

    if callable(names_repair):
        repaired = [str(name) for name in names_repair(list(names))]
        if len(repaired) != len(names):
            raise ValueError(f"{verb}() names_repair must return {len(names)} names, not {len(repaired)}")
    elif names_repair == "unique":
        repaired = _repair_bind_names(list(names))
    elif names_repair == "universal":
        repaired = _repair_bind_names([_python_name(name) for name in names])
    else:
        repaired = list(names)
    duplicated = sorted({name for name in repaired if repaired.count(name) > 1})
    if names_repair == "check_unique" and (duplicated or "" in repaired):
        shown = ", ".join(repr(name) for name in duplicated) or "''"
        raise ValueError(
            f"{verb}(): names must be unique and not empty; these are not: {shown}. "
            'Use names_repair="unique" (or "universal") to repair them.'
        )
    if duplicated:
        shown = ", ".join(repr(name) for name in duplicated)
        raise ValueError(
            f"{verb}(): names_repair leaves duplicated names {shown}; Polars "
            'columns need unique names, so use names_repair="unique".'
        )
    return repaired


def _renamed(lf: pl.LazyFrame, internal: list[str], names: list[str]) -> pl.LazyFrame:
    """Select *internal* columns as *names*, which may swap or reuse names."""
    return lf.select(pl.col(old).alias(new) for old, new in zip(internal, names))


_PTYPES = {int: pl.Int64, float: pl.Float64, str: pl.String, bool: pl.Boolean}


def _ptype_dtype(ptype: Any, what: str) -> Any:
    """The Polars type for one ``names_ptypes``/``values_ptypes`` entry."""
    import datetime

    import pandas as pd

    if ptype in _PTYPES:
        return _PTYPES[ptype]
    if ptype is datetime.date:
        return pl.Date
    if ptype is datetime.datetime:
        return pl.Datetime("us")
    if isinstance(ptype, pd.CategoricalDtype):
        if ptype.categories is None:
            raise TypeError(f"{what}: a categorical prototype needs its categories")
        return pl.Enum([str(level) for level in ptype.categories])
    if isinstance(ptype, pl.DataType) or (isinstance(ptype, type) and issubclass(ptype, pl.DataType)):
        return ptype
    raise TypeError(
        f"{what} entries must be int, float, str, bool, datetime.date, "
        f"datetime.datetime, a pandas CategoricalDtype, or a Polars type, not {ptype!r}"
    )


def _cast_to(x: pl.Expr, source: Any, target: Any) -> tuple[pl.Expr, pl.Expr | None] | None:
    """vctrs' ``vec_cast()`` rules: the cast and its loss check, or None.

    Types convert only without losing information: whole doubles to
    integers, 0/1 to logicals, text or factors to a factor that has every
    value as a level. Text does not parse into numbers (use a transform).
    """
    if source == target:
        return x, None
    if source == pl.Null:
        return x.cast(target), None
    present = x.is_not_null()
    if target.is_float():
        if source.is_numeric() or source == pl.Boolean:
            return x.cast(target), None
        return None
    if target.is_integer():
        if source == pl.Boolean:
            return x.cast(target), None
        if source.is_integer():
            return x.cast(target, strict=False), present & x.cast(target, strict=False).is_null()
        if source.is_float():
            cast = x.cast(target, strict=False)
            lost = present & (cast.cast(pl.Float64) != x).fill_null(True)
            return cast, lost
        return None
    if target == pl.Boolean:
        if source.is_numeric():
            return x.cast(target), present & ~x.is_in([0, 1])
        return None
    if target == pl.String:
        if source in (pl.Categorical,) or isinstance(source, (pl.Enum, pl.Categorical)):
            return x.cast(target), None
        return None
    if isinstance(target, pl.Enum):
        if source == pl.String or isinstance(source, (pl.Enum, pl.Categorical)):
            cast = x.cast(pl.String).cast(target, strict=False)
            return cast, present & cast.is_null()
        return None
    if isinstance(target, pl.Datetime) or target == pl.Datetime:
        if source == pl.Date or isinstance(source, pl.Datetime):
            return x.cast(target), None
        return None
    return None


def _ptype_casts(lf: pl.LazyFrame, ptypes: Any, columns: dict[str, str], what: str) -> pl.LazyFrame:
    """Cast *columns* (requested name -> internal name) to their prototypes.

    One prototype applies to every column; a dict names the columns.
    Checking a lossy cast reads the data.
    """
    if ptypes is None:
        return lf
    if isinstance(ptypes, dict):
        unknown = [name for name in ptypes if name not in columns]
        if unknown:
            raise KeyError(f"{what} names unknown columns: {unknown}")
        chosen = dict(ptypes)
    else:
        chosen = {name: ptypes for name in columns}
    schema = lf.collect_schema()
    casts, checks = [], []
    for name, ptype in chosen.items():
        internal = columns[name]
        target = _ptype_dtype(ptype, what)
        source = schema[internal]
        found = _cast_to(pl.col(internal), source, target)
        if found is None:
            raise TypeError(f"{what}: can't convert `{name}` <{source}> to <{target}>")
        cast, lost = found
        casts.append(cast.alias(internal))
        if lost is not None:
            checks.append((name, source, target, lost.any().alias(f"__tidy3_lost_{len(checks)}")))
    if checks:
        from tidy3.frame import note_data_read

        note_data_read(f"{what} checks that the cast keeps every value")
        lost_any = lf.select([check for *_, check in checks]).collect().row(0)
        for (name, source, target, _), lost in zip(checks, lost_any):
            if lost:
                raise ValueError(
                    f"{what}: can't convert `{name}` <{source}> to <{target}> without losing values"
                )
    return lf.with_columns(casts) if casts else lf


def _transform_exprs(transform: Any, columns: dict[str, str], what: str) -> list[pl.Expr]:
    """``names_transform``/``values_transform`` on *columns* (name -> internal).

    *transform* is one function for every column or a dict by column name.
    Python types (``int``, ``float``, ``str``, ``bool``) cast in the engine;
    other functions run on each value, missing values passing through.
    """
    if transform is None:
        return []
    mapping = (
        dict(transform)
        if isinstance(transform, dict)
        else {name: transform for name in columns}
    )
    unknown = [name for name in mapping if name not in columns]
    if unknown:
        raise KeyError(f"{what} names unknown columns: {unknown}")
    expressions = []
    for name, fn in mapping.items():
        x = pl.col(columns[name])
        if fn in _PTYPES:
            expressions.append(x.cast(_PTYPES[fn]))
        else:
            expressions.append(x.map_elements(fn, skip_nulls=True, return_dtype=None))
    return expressions


def _transform_columns(tf: Any, transform: Any, columns: list[str], what: str) -> Any:
    """Apply ``names_transform``/``values_transform`` to *columns* of *tf*."""
    expressions = _transform_exprs(transform, {name: name for name in columns}, what)
    if not expressions:
        return tf
    return tf._with_lf(tf._lf.with_columns(expressions), groups=tf._groups)


def _any_present(lf: pl.LazyFrame, columns: list[str]) -> pl.Expr:
    """Rows where any of *columns* is present: not null, and not NaN (vctrs)."""
    schema = lf.collect_schema()
    return pl.any_horizontal(
        pl.col(name).is_not_null() & pl.col(name).is_not_nan()
        if _is_float_dtype(schema[name])
        else pl.col(name).is_not_null()
        for name in columns
    )


def pivot_longer(
    cols: Any,
    *,
    names_to: str | list[str] | tuple[str, ...] | None = "name",
    values_to: str = "value",
    names_prefix: str | None = None,
    names_sep: str | None = None,
    names_pattern: str | None = None,
    names_ptypes: Any = None,
    names_transform: Any = None,
    names_repair: Any = "check_unique",
    values_drop_na: bool = False,
    values_ptypes: Any = None,
    values_transform: Any = None,
    cols_vary: str = "fastest",
) -> Verb:
    """Lengthen selected columns, with optional name splitting.

    As in tidyr, ``names_transform`` / ``values_transform`` convert the new
    name and value columns: one function, or a dict by column name, e.g.
    ``names_transform={"week": int}`` turns ``"1"`` into ``1``. Then
    ``names_ptypes`` / ``values_ptypes`` check the types: a Python type, a
    Polars type, or a pandas ``CategoricalDtype``, cast only when no value
    changes (``pl.Enum(["b", "a"])`` makes a factor with those levels).
    ``names_repair`` handles output names that clash: ``"check_unique"``
    (the default) raises, ``"unique"`` numbers them (``name_1``),
    ``"universal"`` also makes them identifiers, or pass a function.
    """
    if cols_vary not in {"fastest", "slowest"}:
        raise ValueError("cols_vary must be 'fastest' or 'slowest'")
    if not isinstance(values_to, str) or not values_to:
        raise TypeError("values_to must be a non-empty string")
    if names_sep is not None and names_pattern is not None:
        raise ValueError("supply only one of names_sep or names_pattern")
    _check_names_repair(names_repair)
    name_columns = (
        []
        if names_to is None
        else [names_to]
        if isinstance(names_to, str)
        else list(names_to)
    )
    value_sentinel = ".value" in name_columns
    if len(name_columns) > 1 and names_sep is None and names_pattern is None:
        raise ValueError(
            "multiple names_to columns require names_sep or names_pattern"
        )
    if len(set(name_columns)) != len(name_columns):
        raise ValueError("names_to must not repeat a name")

    def _apply(tf):
        pivoted = resolve_selection(tf, [cols])
        if not pivoted:
            raise ValueError("pivot_longer() selected no columns")
        all_columns = _columns(tf)
        identifiers = [name for name in all_columns if name not in pivoted]
        # Build under private names and rename once at the end, so a new
        # column may share an identifier's name until names_repair decides.
        taken = list(all_columns)

        def private(base: str) -> str:
            name = _temp_name(taken, base)
            taken.append(name)
            return name

        variable = private("__tidy3_name")
        row = private("__tidy3_row")
        id_columns = [private("__tidy3_id") for _ in identifiers]
        key_columns = [
            ".value" if name == ".value" else private("__tidy3_key")
            for name in name_columns
        ]
        value_column = private("__tidy3_value")

        lf = (
            tf._lf.rename(dict(zip(identifiers, id_columns)))
            .with_row_index(row)
            .unpivot(
                on=pivoted,
                index=[*id_columns, row],
                variable_name=variable,
                value_name=value_column,
            )
        )
        # Sort explicitly: unpivot()'s row order changed in Polars 2.
        if cols_vary == "fastest":
            lf = lf.sort(row, maintain_order=True)
        else:
            position = pl.col(variable).replace_strict(
                {name: index for index, name in enumerate(pivoted)},
                return_dtype=pl.UInt32,
            )
            lf = lf.sort(position, row, maintain_order=True)
        name_expression = pl.col(variable)
        if names_prefix:
            name_expression = name_expression.str.replace(
                rf"^(?:{names_prefix})", ""
            )
        if len(name_columns) == 1:
            lf = lf.with_columns(name_expression.alias(key_columns[0]))
        elif len(name_columns) > 1:
            if names_pattern:
                pieces = [
                    name_expression.str.extract(names_pattern, index + 1).alias(name)
                    for index, name in enumerate(key_columns)
                ]
            else:
                split = name_expression.str.split_exact(
                    names_sep, len(name_columns) - 1
                )
                pieces = [
                    split.struct.field(f"field_{index}").alias(name)
                    for index, name in enumerate(key_columns)
                ]
            lf = lf.with_columns(pieces)
        lf = lf.drop(variable)
        keys = {
            name: key
            for name, key in zip(name_columns, key_columns)
            if name != ".value"
        }
        if value_sentinel:
            from tidy3.frame import note_data_read

            note_data_read("pivot_longer() takes .value column names from the data")
            value_names = (
                lf.select(".value")
                .unique(maintain_order=True)
                .collect()
                .get_column(".value")
                .to_list()
            )
            lf = lf.pivot(
                on=".value",
                on_columns=value_names,
                index=[*id_columns, row, *keys.values()],
                values=value_column,
                aggregate_function="first",
                maintain_order=True,
            )
            values = {name: name for name in value_names}
        else:
            values = {values_to: value_column}
        transforms = [
            *_transform_exprs(names_transform, keys, "names_transform"),
            *_transform_exprs(values_transform, values, "values_transform"),
        ]
        if transforms:
            lf = lf.with_columns(transforms)
        lf = _ptype_casts(lf, names_ptypes, keys, "names_ptypes")
        lf = _ptype_casts(lf, values_ptypes, values, "values_ptypes")
        if values_drop_na:
            lf = lf.filter(_any_present(lf, list(values.values())))
        names = _repair_names(
            [*identifiers, *keys, *values], names_repair, "pivot_longer"
        )
        lf = _renamed(lf, [*id_columns, *keys.values(), *values.values()], names)
        return tf._with_lf(
            lf,
            groups=_result_groups(tf, names),
            rowwise=False,
        )

    return Verb(_apply, "pivot_longer")


def _level_values(tf: Any, name: str) -> list[Any]:
    """Every value a column can take: factor levels, else sorted distinct.

    On Polars only that column is read, lazily.
    """
    levels = tf._category_levels.get(name)
    if levels is not None:
        return list(levels)
    dtype = tf._lf.collect_schema()[name]
    if isinstance(dtype, pl.Enum):
        return dtype.categories.to_list()
    from tidy3.frame import note_data_read

    note_data_read("expand()/complete() read the values present in the data")
    return (
        tf._lf.select(pl.col(name).drop_nulls().unique().sort())
        .collect()
        .to_series()
        .to_list()
    )


def pivot_wider(
    *,
    names_from: Any = "name",
    values_from: Any = "value",
    id_cols: Any = None,
    names_prefix: str = "",
    names_sep: str = "_",
    names_glue: str | None = None,
    names_sort: bool = False,
    names_vary: str = "fastest",
    names_expand: bool = False,
    names_repair: Any = "check_unique",
    id_expand: bool = False,
    values_fill: Any = None,
    values_fn: Any = None,
    unused_fn: Any = None,
    names: Iterable[Any] | None = None,
) -> Verb:
    """Widen a name/value pair; discovers output names when not supplied.

    Like tidyr: ``names_sep`` joins name parts, ``names_glue`` builds names
    from a template (``"{key}_{.value}"``), ``names_vary`` orders the columns
    (``"fastest"``: v_a, v_b, w_a, w_b; ``"slowest"``: v_a, w_a, v_b, w_b),
    ``names_expand`` / ``id_expand`` add columns / rows for every possible
    value (unused factor levels too), and ``unused_fn`` summarises columns
    that are not ids, names, or values instead of dropping them.
    ``names_repair`` handles output names that clash, as in
    :func:`pivot_longer`. ``values_fill`` is one value, or a dict by
    ``values_from`` column.

    Values that are not uniquely identified become list columns with a
    warning; ``values_fn="list"`` asks for them, ``values_fn="mean"`` (or
    ``sum``, ``first``, …) summarises duplicates instead. ``values_fn`` may
    also be a function of each cell's values (a Polars Series), such as
    ``lambda s: s.max() - s.min()``, or a dict of names and functions by
    ``values_from`` column.
    """
    if names_vary not in {"fastest", "slowest"}:
        raise ValueError("names_vary must be 'fastest' or 'slowest'")
    if not isinstance(names_sep, str):
        raise TypeError("names_sep must be a string")
    _check_names_repair(names_repair)
    _values_fns(values_fn)
    plain = (
        names_sep == "_" and names_glue is None and names_vary == "fastest"
        and not names_expand and not id_expand and unused_fn is None
    )
    if plain:
        return _pivot_wider_core(
            names_from=names_from,
            values_from=values_from,
            id_cols=id_cols,
            names_prefix=names_prefix,
            names_sort=names_sort,
            values_fill=values_fill,
            values_fn=values_fn,
            names=names,
            names_repair=names_repair,
        )

    def _apply(tf):
        name_columns = resolve_selection(tf, [names_from])
        value_columns = resolve_selection(tf, [values_from])
        all_columns = _columns(tf)
        identifiers = (
            [c for c in all_columns if c not in {*name_columns, *value_columns}]
            if id_cols is None
            else resolve_selection(tf, [id_cols])
        )
        unused = [
            c for c in all_columns
            if c not in {*identifiers, *name_columns, *value_columns}
        ]
        combos = _pivot_combinations(tf, name_columns, names, names_expand, names_sort)
        out = _pivot_wider_core(
            names_from=name_columns if len(name_columns) > 1 else name_columns[0],
            values_from=value_columns if len(value_columns) > 1 else value_columns[0],
            id_cols=identifiers if identifiers else None,
            names_prefix="",
            names_sort=False,
            values_fill=values_fill,
            values_fn=values_fn,
            names=[c if len(name_columns) > 1 else c[0] for c in combos],
            names_repair=None,
        )._fn(tf)
        produced = _columns(out)
        kept_ids = [c for c in produced if c in identifiers]
        value_outputs = [c for c in produced if c not in identifiers]
        pairs = [(value, combo) for value in value_columns for combo in combos]
        if len(value_outputs) != len(pairs):
            raise RuntimeError("pivot_wider(): unexpected output columns")
        labels = [
            _pivot_label(value, combo, name_columns, value_columns, names_prefix, names_sep, names_glue)
            for value, combo in pairs
        ]
        order = list(range(len(pairs)))
        if names_vary == "slowest":
            order.sort(key=lambda i: (combos.index(pairs[i][1]), value_columns.index(pairs[i][0])))
        out = out._with_lf(
            out._lf.select([*kept_ids, *[value_outputs[i] for i in order]]), groups=out._groups
        )
        if id_expand and kept_ids:
            if isinstance(values_fill, dict):
                fills = {value_outputs[i]: values_fill[pairs[i][0]] for i in order if pairs[i][0] in values_fill}
            else:
                fills = values_fill
            out = _expand_ids(tf, out, kept_ids, fills)
        extra: list[str] = []
        if unused_fn is not None and unused:
            before = _columns(out)
            out = _summarise_unused(tf, out, kept_ids, unused, unused_fn)
            extra = [c for c in _columns(out) if c not in before]
        current = [*kept_ids, *[value_outputs[i] for i in order], *extra]
        names_out = _repair_names(
            [*kept_ids, *[labels[i] for i in order], *extra], names_repair, "pivot_wider"
        )
        out = out._with_lf(_renamed(out._lf, current, names_out), groups=out._groups)
        return _with_frame_groups(out, _result_groups(tf, names_out), tf._category_levels)

    return Verb(_apply, "pivot_wider")


def _with_frame_groups(out: Any, groups: Any, levels: dict[str, list[Any]]) -> Any:
    return out._with_lf(out._lf, groups=groups, category_levels=levels)


def _pivot_combinations(tf, name_columns, names, names_expand, names_sort) -> list[tuple]:
    """The ``names_from`` combinations that become columns, in order.

    Reads only the name columns (lazily on Polars).
    """
    import itertools

    if names is not None:
        return [(v,) for v in names] if len(name_columns) == 1 else [tuple(v) for v in names]
    if names_expand:
        # Every combination of every possible value, sorted (as tidyr).
        return list(itertools.product(*[_level_values(tf, n) for n in name_columns]))
    from tidy3.frame import note_data_read

    note_data_read("pivot_wider() takes new column names from the data")
    seen = tf._lf.select(name_columns).unique(maintain_order=True).collect()
    if names_sort:
        seen = seen.sort(name_columns)
    return seen.rows()


def _pivot_label(value, combo, name_columns, value_columns, prefix, sep, glue) -> str:
    """Output column name for one value column and names_from combination."""
    import re

    if glue is not None:
        fields = {".value": value, **dict(zip(name_columns, map(str, combo)))}
        return re.sub(
            r"\{([^{}]+)\}",
            lambda m: fields.get(m.group(1).strip(), m.group(0)),
            glue,
        )
    parts = list(map(str, combo))
    if len(value_columns) > 1:
        parts = [value, *parts]
    return prefix + sep.join(parts)


def _expand_ids(tf: Any, out: Any, ids: list[str], values_fill: Any) -> Any:
    """``id_expand``: a row for every combination of id values (and levels)."""
    import itertools

    combos = list(itertools.product(*[_level_values(tf, n) for n in ids]))
    fill = [c for c in _columns(out) if c not in ids]
    schema = out._lf.collect_schema()
    grid = pl.LazyFrame(
        {name: [combo[i] for combo in combos] for i, name in enumerate(ids)},
        strict=False,
    ).with_columns(pl.col(name).cast(schema[name], strict=False) for name in ids)
    joined = grid.join(out._lf, on=ids, how="left", nulls_equal=True, maintain_order="left")
    if isinstance(values_fill, dict):
        joined = joined.with_columns(pl.col(c).fill_null(values_fill[c]) for c in fill if c in values_fill)
    elif values_fill is not None:
        joined = joined.with_columns(pl.col(c).fill_null(values_fill) for c in fill)
    return out._with_lf(joined, groups=out._groups)


_POLARS_SUMMARIES = {"first", "last", "min", "max", "sum", "mean", "median", "len", "n_unique", "std", "var"}


def _summary_expr(tf: Any, column: str, fn: Any, ids: list[str], what: str = "unused_fn") -> pl.Expr:
    """One ``unused_fn`` (or ``values_fn``) summary as a Polars expression.

    Named summaries ("max", "first", …) and tidy3's own summaries
    (``mean``, ``n_distinct``, which a notebook's bare ``mean`` names) run
    natively. Another Python function gets each group's values as a Polars
    Series; its result type comes from the first group, because Polars needs
    to know it up front.
    """
    if isinstance(fn, str):
        if fn not in _POLARS_SUMMARIES:
            raise ValueError(f"{what} summary not known: {fn!r}")
        return getattr(pl.col(column), fn)().alias(column)
    if not callable(fn):
        raise TypeError(f"{what} values must be functions or summary names")
    if getattr(fn, "__module__", "").startswith("tidy3."):
        from tidy3.expr import Expr, col, to_polars

        built = fn(col(column))
        if isinstance(built, Expr):
            return to_polars(built).alias(column)
    sample = tf._lf
    if ids:
        first = tf._lf.select(ids).head(1)
        sample = tf._lf.join(first, on=ids, how="semi", nulls_equal=True)
    from tidy3.frame import note_data_read

    note_data_read(f"{what} infers its result type from the data")
    values = sample.select(pl.col(column)).collect().to_series()
    dtype = pl.Series([fn(values)]).dtype
    return (
        pl.col(column)
        .map_batches(lambda series: pl.Series([fn(series)], dtype=dtype), returns_scalar=True, return_dtype=dtype)
        .alias(column)
    )


def _summarise_unused(tf: Any, out: Any, ids: list[str], unused: list[str], unused_fn: Any) -> Any:
    """``unused_fn``: summarise the leftover columns per id row."""
    fns = unused_fn if isinstance(unused_fn, dict) else {c: unused_fn for c in unused}
    stray = [c for c in fns if c not in unused]
    if stray:
        raise KeyError(f"unused_fn names columns that are not unused: {stray}")
    exprs = [_summary_expr(tf, column, fn, ids) for column, fn in fns.items()]
    if not ids:
        return out._with_lf(out._lf.join(tf._lf.select(exprs), how="cross"), groups=out._groups)
    summary = tf._lf.group_by(ids, maintain_order=True).agg(exprs)
    joined = out._lf.join(summary, on=ids, how="left", nulls_equal=True, maintain_order="left")
    return out._with_lf(joined, groups=out._groups)


# ── pivot specs (tidyr's build_*_spec / pivot_*_spec) ────────────────────


def _spec_frame(spec: Any) -> Any:
    """A spec as a Polars frame (it is small), checked like tidyr's."""
    from tidy3.frame import TidyFrame

    checked = getattr(spec, "_spec_table", None)
    if checked is not None:
        return checked
    if isinstance(spec, TidyFrame):
        frame = spec._lf.collect()
    elif isinstance(spec, pl.DataFrame):
        frame = spec
    elif hasattr(spec, "to_dict") and hasattr(spec, "columns"):
        frame = pl.from_pandas(spec)
    else:
        frame = pl.DataFrame(spec, strict=False)
    if ".name" not in frame.columns or ".value" not in frame.columns:
        raise ValueError("spec must have `.name` and `.value` columns")
    if frame[".name"].is_duplicated().any():
        raise ValueError("spec `.name` must be unique")
    return frame


def check_pivot_spec(spec: Any) -> Any:
    """Check a pivot spec: ``.name`` and ``.value`` columns, unique names."""
    from tidy3.frame import tidy

    return tidy(_spec_frame(spec))


def build_longer_spec(
    data: Any,
    cols: Any,
    *,
    names_to: str | list[str] | tuple[str, ...] | None = "name",
    values_to: str = "value",
    names_prefix: str | None = None,
    names_sep: str | None = None,
    names_pattern: str | None = None,
    names_ptypes: Any = None,
    names_transform: Any = None,
) -> Any:
    """The spec ``pivot_longer()`` would use, as a table you can edit (tidyr).

    One row per pivoted column: ``.name`` (the column), ``.value`` (where its
    values go), and one column per ``names_to`` entry. Pass it to
    :func:`pivot_longer_spec`. Only column names are read, not the data.
    ``names_transform`` and then ``names_ptypes`` apply as in
    :func:`pivot_longer`.
    """
    import re

    from tidy3.frame import TidyFrame, tidy

    tf = data if isinstance(data, TidyFrame) else tidy(data)
    pivoted = resolve_selection(tf, [cols])
    if not pivoted:
        raise ValueError("build_longer_spec() selected no columns")
    targets = (
        [] if names_to is None
        else [names_to] if isinstance(names_to, str)
        else list(names_to)
    )
    if len(targets) > 1 and names_sep is None and names_pattern is None:
        raise ValueError("multiple names_to columns require names_sep or names_pattern")
    rows = []
    for column in pivoted:
        name = re.sub(rf"^(?:{names_prefix})", "", column) if names_prefix else column
        if names_pattern:
            found = re.search(names_pattern, name)
            pieces = list(found.groups()) if found else [None] * len(targets)
        elif len(targets) > 1:
            pieces = name.split(names_sep, len(targets) - 1)
        else:
            pieces = [name]
        if targets and len(pieces) != len(targets):
            raise ValueError("name split did not produce the requested number of columns")
        rows.append(pieces)
    spec = {
        ".name": pivoted,
        ".value": [r[targets.index(".value")] for r in rows] if ".value" in targets else [values_to] * len(rows),
    }
    for index, target in enumerate(targets):
        if target != ".value":
            spec[target] = [r[index] for r in rows]
    out = tidy(pl.DataFrame(spec, schema={k: pl.String for k in spec}))
    extra = [t for t in targets if t != ".value"]
    out = _transform_columns(out, names_transform, extra, "names_transform")
    lf = _ptype_casts(out._lf, names_ptypes, {name: name for name in extra}, "names_ptypes")
    return out._with_lf(lf, groups=out._groups)


def _spec_parts(table: Any) -> tuple[list[str], list[str], list[tuple]]:
    keys = [c for c in table.columns if c not in {".name", ".value"}]
    values = list(dict.fromkeys(table[".value"].to_list()))
    combos = list(dict.fromkeys(table.select(keys).rows())) if keys else [()]
    return keys, values, combos


def pivot_longer_spec(
    spec: Any,
    *,
    cols_vary: str = "fastest",
    names_repair: Any = "check_unique",
    values_drop_na: bool = False,
    values_ptypes: Any = None,
    values_transform: Any = None,
) -> Verb:
    """Lengthen with a spec from :func:`build_longer_spec` (tidyr).

    Edit the spec first to rename or regroup: every ``.name`` column moves
    into rows, keyed by the spec's other columns, with values in ``.value``.
    ``names_repair``, ``values_ptypes``, and ``values_transform`` work as in
    :func:`pivot_longer`.
    """
    if cols_vary not in {"fastest", "slowest"}:
        raise ValueError("cols_vary must be 'fastest' or 'slowest'")
    _check_names_repair(names_repair)
    table = _spec_frame(spec)
    keys, values, combos = _spec_parts(table)
    lookup = {(r[".value"], tuple(r[k] for k in keys)): r[".name"] for r in table.iter_rows(named=True)}

    def _apply(tf):
        columns = _columns(tf)
        missing = [name for name in table[".name"].to_list() if name not in columns]
        if missing:
            raise KeyError(f"spec names columns the data does not have: {missing}")
        ids = [c for c in columns if c not in set(table[".name"].to_list())]
        # Private names until names_repair decides, as in pivot_longer().
        taken = list(columns)

        def private(base: str) -> str:
            name = _temp_name(taken, base)
            taken.append(name)
            return name

        row = private("__tidy3_row")
        order = private("__tidy3_combo")
        id_columns = [private("__tidy3_id") for _ in ids]
        key_columns = [private("__tidy3_key") for _ in keys]
        value_columns = {value: private("__tidy3_value") for value in values}
        base = tf._lf.with_row_index(row)
        pieces = []
        for position, combo in enumerate(combos):
            pieces.append(
                base.select(
                    pl.col(row),
                    *[pl.col(i).alias(p) for i, p in zip(ids, id_columns)],
                    *[pl.lit(v).alias(k) for k, v in zip(key_columns, combo)],
                    *[
                        (pl.col(lookup[(value, combo)]) if (value, combo) in lookup else pl.lit(None)).alias(internal)
                        for value, internal in value_columns.items()
                    ],
                    pl.lit(position).alias(order),
                )
            )
        lf = pl.concat(pieces, how="diagonal_relaxed").sort(
            [row, order] if cols_vary == "fastest" else [order, row], maintain_order=True
        )
        transforms = _transform_exprs(values_transform, value_columns, "values_transform")
        if transforms:
            lf = lf.with_columns(transforms)
        lf = _ptype_casts(lf, values_ptypes, value_columns, "values_ptypes")
        if values_drop_na:
            lf = lf.filter(_any_present(lf, list(value_columns.values())))
        names = _repair_names([*ids, *keys, *values], names_repair, "pivot_longer_spec")
        lf = _renamed(lf, [*id_columns, *key_columns, *value_columns.values()], names)
        return tf._with_lf(lf, groups=None)

    return Verb(_apply, "pivot_longer_spec")


def build_wider_spec(
    data: Any,
    *,
    names_from: Any = "name",
    values_from: Any = "value",
    names_prefix: str = "",
    names_sep: str = "_",
    names_glue: str | None = None,
    names_sort: bool = False,
    names_vary: str = "fastest",
    names_expand: bool = False,
) -> Any:
    """The spec ``pivot_wider()`` would use, as a table you can edit (tidyr).

    One row per output column: ``.name``, ``.value`` (the value column it
    comes from), and the ``names_from`` values. Pass it to
    :func:`pivot_wider_spec`. Only the name columns are read.
    """
    from tidy3.frame import TidyFrame, tidy

    if names_vary not in {"fastest", "slowest"}:
        raise ValueError("names_vary must be 'fastest' or 'slowest'")
    tf = data if isinstance(data, TidyFrame) else tidy(data)
    name_columns = resolve_selection(tf, [names_from])
    value_columns = resolve_selection(tf, [values_from])
    combos = _pivot_combinations(tf, name_columns, None, names_expand, names_sort)
    pairs = [(value, combo) for value in value_columns for combo in combos]
    if names_vary == "slowest":
        pairs.sort(key=lambda pair: (combos.index(pair[1]), value_columns.index(pair[0])))
    spec = pl.DataFrame(
        {
            ".name": [
                _pivot_label(v, c, name_columns, value_columns, names_prefix, names_sep, names_glue)
                for v, c in pairs
            ],
            ".value": [v for v, _ in pairs],
            **{name: [c[i] for _, c in pairs] for i, name in enumerate(name_columns)},
        },
        strict=False,
    )
    # Typed even with no rows, as in tidyr: text labels, name columns as in data.
    schema = (
        tf._lf.collect_schema()
    )
    table = spec.with_columns(
        pl.col(".name").cast(pl.String),
        pl.col(".value").cast(pl.String),
        *(pl.col(name).cast(schema[name]) for name in name_columns),
    )
    out = tidy(table)
    if not table[".name"].is_duplicated().any():
        # Checked here once; pivot_wider_spec() takes the table as it is.
        # Verbs build new frames without it, so an edited spec is checked again.
        out._spec_table = table
    return out


def pivot_wider_spec(
    spec: Any,
    *,
    names_repair: Any = "check_unique",
    id_cols: Any = None,
    id_expand: bool = False,
    values_fill: Any = None,
    values_fn: Any = None,
    unused_fn: Any = None,
) -> Verb:
    """Widen with a spec from :func:`build_wider_spec` (tidyr).

    Each spec row is one output column: ``.name`` takes the ``.value``
    column's values where the data's name columns equal the spec's.
    ``names_repair``, ``values_fill``, and ``values_fn`` work as in
    :func:`pivot_wider`.
    """
    _check_names_repair(names_repair)
    _values_fns(values_fn)
    table = _spec_frame(spec)
    keys, values, combos = _spec_parts(table)

    def _apply(tf):
        all_columns = _columns(tf)
        identifiers = (
            [c for c in all_columns if c not in {*keys, *values}]
            if id_cols is None
            else resolve_selection(tf, [id_cols])
        )
        unused = [c for c in all_columns if c not in {*identifiers, *keys, *values}]
        if table.height == 0:
            # No output columns: one row per distinct id, as tidyr returns.
            from tidy3.verbs import distinct

            return tf >> distinct(*identifiers)
        out = _pivot_wider_core(
            names_from=keys if len(keys) > 1 else keys[0],
            values_from=values if len(values) > 1 else values[0],
            id_cols=identifiers or None,
            values_fill=values_fill,
            values_fn=values_fn,
            names=[c if len(keys) > 1 else c[0] for c in combos],
            names_repair=None,
        )._fn(tf)
        produced = _columns(out)
        kept_ids = [c for c in produced if c in identifiers]
        outputs = [c for c in produced if c not in identifiers]
        pairs = [(v, c) for v in values for c in combos]
        by_pair = dict(zip(pairs, outputs))
        sources, labels = [], []
        for row in table.iter_rows(named=True):
            sources.append(by_pair[(row[".value"], tuple(row[k] for k in keys))])
            labels.append(row[".name"])
        out = out._with_lf(out._lf.select([*kept_ids, *sources]), groups=out._groups)
        if id_expand and kept_ids:
            if isinstance(values_fill, dict):
                source_values = dict(zip(outputs, (v for v, _ in pairs)))
                fills = {c: values_fill[source_values[c]] for c in sources if source_values[c] in values_fill}
            else:
                fills = values_fill
            out = _expand_ids(tf, out, kept_ids, fills)
        extra: list[str] = []
        if unused_fn is not None and unused:
            before = _columns(out)
            out = _summarise_unused(tf, out, kept_ids, unused, unused_fn)
            extra = [c for c in _columns(out) if c not in before]
        names_out = _repair_names([*kept_ids, *labels, *extra], names_repair, "pivot_wider_spec")
        out = out._with_lf(_renamed(out._lf, [*kept_ids, *sources, *extra], names_out), groups=out._groups)
        return _with_frame_groups(out, None, tf._category_levels)

    return Verb(_apply, "pivot_wider_spec")


def _warn_not_unique(values: list[str], by: list[str]) -> None:
    """tidyr's warning for values that are not uniquely identified."""
    shown = ", ".join(f"`{name}`" for name in values)
    warnings.warn(
        f"Values from {shown} are not uniquely identified; output will contain "
        "list-cols.\n"
        '* Use values_fn="list" to suppress this warning.\n'
        '* Use values_fn="mean" (or "sum", "first", ...) to summarise duplicates.\n'
        "* Use the following tidy3 code to identify duplicates:\n"
        f"  data >> summarise(n=n(), by={by!r}) >> filter(col(\"n\") > 1)",
        UserWarning,
        stacklevel=4,
    )


def _values_fns(values_fn: Any) -> Any:
    """Check ``values_fn``: None, a summary name, a function, or a dict of them."""
    def check(fn: Any) -> None:
        if fn is None or fn is list or callable(fn):
            return
        if fn != "list" and fn not in _POLARS_SUMMARIES:
            names = ", ".join(sorted(_POLARS_SUMMARIES | {"list"}))
            raise ValueError(f"values_fn must be a function or one of {names}, not {fn!r}")

    if isinstance(values_fn, dict):
        for fn in values_fn.values():
            check(fn)
    else:
        check(values_fn)
    return values_fn


def _pivot_wider_core(
    *,
    names_from: Any = "name",
    values_from: Any = "value",
    id_cols: Any = None,
    names_prefix: str = "",
    names_sort: bool = False,
    values_fill: Any = None,
    values_fn: Any = None,
    names: Iterable[Any] | None = None,
    names_repair: Any = "check_unique",
) -> Verb:
    """Widen a name/value pair; discovers output names when not supplied.

    Like tidyr, values that are not uniquely identified by the id and name
    columns become list columns with a warning; ``values_fn="list"`` asks for
    list columns without the warning, and ``values_fn="mean"`` (or ``sum``,
    ``first``, …) summarises duplicates instead. ``values_fn`` may also be a
    function of each cell's values (a Polars Series), or a dict of these by
    value column; columns it leaves out behave as with no ``values_fn``.

    ``names_repair=None`` is for the callers that rename the value columns
    themselves: those come back under private names, after the ids.
    """
    _values_fns(values_fn)

    def _apply(tf):
        name_columns = resolve_selection(tf, [names_from])
        value_columns = resolve_selection(tf, [values_from])
        if not name_columns or not value_columns:
            raise ValueError(
                "pivot_wider() must select names_from and values_from columns"
            )
        name_argument: Any = (
            name_columns[0] if len(name_columns) == 1 else name_columns
        )
        value_argument: Any = (
            value_columns[0] if len(value_columns) == 1 else value_columns
        )
        all_columns = _columns(tf)
        if id_cols is None:
            identifiers = [
                name for name in all_columns
                if name not in {*name_columns, *value_columns}
            ]
        else:
            identifiers = resolve_selection(tf, [id_cols])
        if isinstance(values_fn, dict):
            unknown = [name for name in values_fn if name not in value_columns]
            if unknown:
                raise KeyError(f"values_fn names columns that are not values_from: {unknown}")
            fns = {name: values_fn.get(name) for name in value_columns}
        else:
            fns = {name: values_fn for name in value_columns}
        # Polars' own pivot summaries fill absent cells (a sum of nothing is
        # 0); tidyr leaves them missing, so summaries run before the pivot.
        native = values_fn is None or values_fn == "list"
        unsummarised = [name for name in value_columns if fns[name] is None]

        # Identifiers move to private names while pivoting, so a new column
        # may share an identifier's name until names_repair decides.
        taken = list(all_columns)

        def private(base: str) -> str:
            name = _temp_name(taken, base)
            taken.append(name)
            return name

        temp_id = private("__tidy3_id") if not identifiers else None
        id_columns = [temp_id] if temp_id else [private("__tidy3_id") for _ in identifiers]
        lf = tf._lf.rename(dict(zip(identifiers, id_columns)))
        if temp_id:
            lf = lf.with_columns(pl.lit(0).alias(temp_id))
        # tidyr turns values that are not uniquely identified into lists, so
        # check for duplicates (with the name discovery when there is one).
        duplicates = None
        if unsummarised:
            count = private("__tidy3_n")
            duplicates = (
                lf.group_by([*id_columns, *name_columns])
                .agg(pl.len().alias(count))
                .select((pl.col(count).max() > 1).fill_null(False))
            )
        if names is None:
            from tidy3.frame import note_data_read

            note_data_read("pivot_wider() takes new column names from the data")
            discovery = lf.select(name_columns).unique(
                maintain_order=not names_sort
            )
            if duplicates is not None:
                discovered, duplicated = pl.collect_all([discovery, duplicates])
                duplicates = duplicated
            else:
                discovered = discovery.collect()
            if names_sort:
                discovered = discovered.sort(name_columns)
            combinations = discovered.rows()
            on_columns: Any = (
                discovered.get_column(name_columns[0])
                if len(name_columns) == 1
                else discovered
            )
        else:
            if len(name_columns) == 1:
                on_columns = list(names)
                combinations = [(value,) for value in names]
            else:
                combinations = [tuple(value) for value in names]
                on_columns = pl.DataFrame(
                    {
                        column: [row[index] for row in combinations]
                        for index, column in enumerate(name_columns)
                    }
                )
        if isinstance(duplicates, pl.LazyFrame):
            duplicates = duplicates.collect()
        duplicated = duplicates is not None and bool(duplicates.item())
        if duplicated:
            _warn_not_unique(unsummarised, [*identifiers, *name_columns])
        empty_lists = False
        if native:
            empty_lists = values_fn == "list" or duplicated
            aggregate: Any = pl.element() if empty_lists else None
        else:
            # Summarise each cell first, then pivot one value per cell.
            keys = [*identifiers, *name_columns]
            summaries = []
            for name in value_columns:
                fn = fns[name]
                if fn is None:
                    summaries.append(pl.col(name) if duplicated else pl.col(name).first())
                elif fn == "list" or fn is list:
                    summaries.append(pl.col(name))
                else:
                    summaries.append(_summary_expr(tf, name, fn, keys, "values_fn"))
            lf = lf.group_by([*id_columns, *name_columns], maintain_order=True).agg(summaries)
            aggregate = "first"
        out = lf.pivot(
            on=name_argument,
            on_columns=on_columns,
            index=id_columns,
            values=value_argument,
            aggregate_function=aggregate,
            maintain_order=True,
        )
        output_columns = out.collect_schema().names()
        actual_values = [
            name for name in output_columns if name not in id_columns
        ]
        if empty_lists:
            # Combinations with no rows come back as [] — make them missing.
            out = out.with_columns(
                pl.when(pl.col(name).list.len() > 0).then(pl.col(name)).alias(name)
                for name in actual_values
            )
        # Value columns come out value-major: every combination of the first
        # value column, then the next.
        sources = [value for value in value_columns for _ in combinations]
        if values_fill is not None:
            fills = (
                values_fill
                if isinstance(values_fill, dict)
                else {name: values_fill for name in value_columns}
            )
            expressions = [
                pl.col(name).fill_null(fills[source]).alias(name)
                for name, source in zip(actual_values, sources)
                if source in fills
            ]
            if expressions:
                out = out.with_columns(expressions)
        if temp_id:
            out = out.drop(temp_id)
            id_columns, identifiers = [], []
        if names_repair is None:
            values = [private("__tidy3_value") for _ in actual_values]
            names_out = [*identifiers, *values]
        else:
            generated = (
                [
                    f"{value}_{'_'.join(map(str, combination))}"
                    for value in value_columns
                    for combination in combinations
                ]
                if len(value_columns) > 1
                else [
                    "_".join(map(str, combination))
                    for combination in combinations
                ]
            )
            names_out = _repair_names(
                [*identifiers, *(f"{names_prefix}{name}" for name in generated)],
                names_repair,
                "pivot_wider",
            )
        out = _renamed(out, [*id_columns, *actual_values], names_out)
        return tf._with_lf(
            out,
            groups=_result_groups(tf, names_out),
            rowwise=False,
        )

    return Verb(_apply, "pivot_wider")


def separate(
    column: str,
    into: Iterable[str | None],
    *,
    sep: str = r"[^A-Za-z0-9]+",
    remove: bool = True,
    convert: bool = False,
    extra: str = "warn",
    fill: str = "warn",
) -> Verb:
    """Split one string column into several columns using a regex."""
    names = list(into)
    if not names:
        raise TypeError("separate() requires at least one output name")
    if extra not in {"warn", "drop", "merge"}:
        raise ValueError("extra must be warn, drop, or merge")
    if fill not in {"warn", "right", "left"}:
        raise ValueError("fill must be warn, right, or left")

    def _apply(tf):
        columns = _columns(tf)
        if column not in columns:
            raise KeyError(f"column not found: {column!r}")
        kept_names = [name for name in names if name is not None]
        collisions = [
            name for name in kept_names if name in columns and name != column
        ]
        if collisions:
            raise ValueError(f"separate() output columns already exist: {collisions}")

        sentinel = "\x1f"
        text = pl.col(column).cast(pl.String)
        # A separator with no regex syntax splits directly; a regex is first
        # rewritten to a sentinel character that Polars can split on.
        literal = bool(sep) and not re.search(r"[\\.^$|?*+()\[\]{}]", sep)
        source = text if literal else text.str.replace_all(sep, sentinel)
        piece_sep = sep if literal else sentinel
        if extra == "merge":
            split = source.str.splitn(piece_sep, len(names))
        elif literal and fill != "left":
            split = source.str.split_exact(piece_sep, len(names) - 1)
        else:
            split = source.str.split(piece_sep)
        schema = tf._lf.collect_schema()
        expressions: dict[str, pl.Expr] = {}
        for index, name in enumerate(names):
            if name is None:
                continue
            if extra == "merge":
                value = split.struct.field(f"field_{index}")
                if not literal:
                    value = value.str.replace_all(sentinel, sep, literal=True)
            elif literal and fill != "left":
                value = split.struct.field(f"field_{index}")
            else:
                if fill == "left":
                    value = (
                        pl.when(split.list.len() < len(names))
                        .then(
                            split.list.get(
                                index - len(names), null_on_oob=True
                            )
                        )
                        .otherwise(split.list.get(index, null_on_oob=True))
                    )
                else:
                    value = split.list.get(index, null_on_oob=True)
            expressions[name] = value.alias(name)
        output = []
        for name in schema.names():
            if name == column:
                if not remove:
                    output.append(pl.col(column))
                output.extend(expressions.values())
            else:
                output.append(pl.col(name))
        lf = tf._lf.select(output)
        if convert and kept_names:
            lf = lf.with_columns(
                pl.when(pl.col(name) == "NA")
                .then(None)
                .otherwise(pl.col(name))
                .alias(name)
                for name in kept_names
            )
            stats = []
            for index, name in enumerate(kept_names):
                value = pl.col(name)
                valid_int = value.is_null() | value.cast(
                    pl.Int64, strict=False
                ).is_not_null()
                valid_float = value.is_null() | value.cast(
                    pl.Float64, strict=False
                ).is_not_null()
                upper = value.str.to_uppercase()
                valid_bool = value.is_null() | upper.is_in(
                    ["TRUE", "FALSE", "T", "F"]
                )
                stats.extend(
                    [
                        value.is_not_null().any().alias(f"n_{index}"),
                        valid_int.all().alias(f"i_{index}"),
                        valid_float.all().alias(f"f_{index}"),
                        valid_bool.all().alias(f"b_{index}"),
                    ]
                )
            from tidy3.frame import note_data_read

            note_data_read("separate(convert=True) infers column types from the data")
            inferred = lf.select(stats).collect().row(0, named=True)
            conversions = []
            for index, name in enumerate(kept_names):
                if not inferred[f"n_{index}"]:
                    continue
                if inferred[f"i_{index}"]:
                    expression = pl.col(name).cast(pl.Int64)
                elif inferred[f"f_{index}"]:
                    expression = pl.col(name).cast(pl.Float64)
                elif inferred[f"b_{index}"]:
                    upper = pl.col(name).str.to_uppercase()
                    expression = (
                        pl.when(upper.is_in(["TRUE", "T"]))
                        .then(True)
                        .when(upper.is_in(["FALSE", "F"]))
                        .then(False)
                        .otherwise(None)
                    )
                else:
                    continue
                conversions.append(expression.alias(name))
            if conversions:
                lf = lf.with_columns(conversions)
        ordered = lf.collect_schema().names()
        return tf._with_lf(
            lf,
            groups=_result_groups(tf, ordered),
            rowwise=False,
        )

    return Verb(_apply, "separate")


def unite(
    column: str,
    *cols: Any,
    sep: str = "_",
    remove: bool = True,
    na_rm: bool = False,
) -> Verb:
    """Combine selected columns into one string column."""
    if not isinstance(column, str) or not column:
        raise TypeError("unite() output column must be a non-empty string")

    def _apply(tf):
        selected = resolve_selection(tf, cols) if cols else _columns(tf)
        if not selected:
            raise ValueError("unite() selected no columns")
        columns = _columns(tf)
        if column in columns and column not in selected:
            raise ValueError(f"unite() output column already exists: {column!r}")
        position = min(columns.index(name) for name in selected)

        values = [pl.col(name).cast(pl.String) for name in selected]
        if not na_rm:
            values = [value.fill_null("NA") for value in values]
        combined = pl.concat_str(
            values, separator=sep, ignore_nulls=na_rm
        ).alias(column)
        output = []
        inserted = False
        for index, name in enumerate(columns):
            if index == position:
                output.append(combined)
                inserted = True
            if name in selected and remove:
                continue
            if name == column and inserted:
                continue
            output.append(pl.col(name))
        lf = tf._lf.select(output)
        ordered = lf.collect_schema().names()
        return tf._with_lf(
            lf,
            groups=_result_groups(tf, ordered),
            rowwise=False,
        )

    return Verb(_apply, "unite")


def nest(
    column: str = "data",
    *,
    cols: Any = None,
    by: Any = None,
    names_sep: str | None = None,
) -> Verb:
    """Collapse selected columns into a nested list-column.

    Polars stores list-of-struct; pandas stores a DataFrame per group
    (same representation as ``group_nest``). ``names_sep="_"`` drops the
    ``{column}_`` prefix from the nested names (``x_a`` -> ``a``), as tidyr's
    ``.names_sep``.
    """
    if not isinstance(column, str) or not column:
        raise TypeError("nest() column must be a non-empty string")
    if cols is not None and by is not None:
        raise ValueError("nest() accepts only one of cols= or by=")

    def _apply(tf):
        columns = _columns(tf)
        if column in columns:
            raise ValueError(f"nest() output column already exists: {column!r}")
        if by is not None:
            identifiers = resolve_selection(tf, [by])
            nested = [name for name in columns if name not in identifiers]
        elif cols is not None:
            nested = resolve_selection(tf, [cols])
            identifiers = [name for name in columns if name not in nested]
        elif tf._groups:
            identifiers = list(tf._groups)
            nested = [name for name in columns if name not in identifiers]
        else:
            raise TypeError("nest() requires cols=, by=, or a grouped frame")
        if not nested:
            raise ValueError("nest() selected no columns to nest")
        if names_sep is not None:
            prefix = f"{column}{names_sep}"
            renames = {
                name: name[len(prefix):]
                for name in nested
                if name.startswith(prefix) and len(name) > len(prefix)
            }
            if renames:
                tf = (
                    tf._with_lf(tf._lf.rename(renames), groups=tf._groups)
                )
                nested = [renames.get(name, name) for name in nested]
                columns = _columns(tf)


        # Prefer exclude() when nesting "everything except keys" so the plan
        # tracks the schema without rebuilding a Python name list at collect.
        if identifiers and set(nested) == set(columns) - set(identifiers):
            if len(identifiers) == 1:
                records = pl.struct(pl.exclude(identifiers[0]))
            else:
                records = pl.struct(pl.exclude(identifiers))
        else:
            records = pl.struct(nested)
        if identifiers:
            output = tf._lf.group_by(
                identifiers, maintain_order=True
            ).agg(records.alias(column))
        else:
            output = tf._lf.select(records.implode().alias(column))
        groups = _result_groups(tf, identifiers)
        return tf._with_lf(output, groups=groups, rowwise=False)

    return Verb(_apply, "nest")


def unnest_longer(
    column: str,
    *,
    values_to: str | None = None,
    indices_to: str | None = None,
    indices_include: bool | None = None,
    keep_empty: bool = False,
    transform: Any = None,
) -> Verb:
    """Expand each element of a list-column into its own row.

    ``indices_include=True`` adds each element's 1-based position as
    ``{column}_id`` (or ``indices_to``); ``transform`` converts the values,
    e.g. ``transform=int``, as in tidyr.
    """
    if indices_include and indices_to is None:
        indices_to = f"{values_to or column}_id"
    core = _unnest_longer_core(
        column, values_to=values_to, indices_to=indices_to, keep_empty=keep_empty
    )
    if transform is None:
        return core

    def _apply(tf):
        out = core._fn(tf)
        return _transform_columns(out, transform, [values_to or column], "transform")

    return Verb(_apply, "unnest_longer")


def _unnest_longer_core(
    column: str,
    *,
    values_to: str | None = None,
    indices_to: str | None = None,
    keep_empty: bool = False,
) -> Verb:
    """Expand each element of a list-column into its own row."""

    def _apply(tf):
        columns = _columns(tf)
        if column not in columns:
            raise KeyError(f"column not found: {column!r}")
        value_name = values_to or column
        if value_name != column and value_name in columns:
            raise ValueError(f"unnest_longer() output column exists: {value_name!r}")
        if indices_to and indices_to in columns:
            raise ValueError(f"unnest_longer() index column exists: {indices_to!r}")

        lf = tf._lf
        explode_columns = [column]
        if indices_to:
            lf = lf.with_columns(
                pl.int_ranges(1, pl.col(column).list.len() + 1).alias(indices_to)
            )
            explode_columns.append(indices_to)
        lf = lf.explode(
            explode_columns,
            empty_as_null=keep_empty,
            keep_nulls=keep_empty,
        )
        if value_name != column:
            lf = lf.rename({column: value_name})
        ordered = lf.collect_schema().names()
        if indices_to:
            ordered.remove(indices_to)
            ordered.insert(ordered.index(value_name), indices_to)
            lf = lf.select(ordered)
        return tf._with_lf(
            lf,
            groups=_result_groups(tf, ordered),
            rowwise=False,
        )

    return Verb(_apply, "unnest_longer")


def unnest(
    column: str,
    *,
    keep_empty: bool = False,
    names_sep: str | None = None,
) -> Verb:
    """Expand a list-column, widening nested row records when present.

    On the pandas backend, nested cells may be list/tuple of dicts (legacy /
    polars-handoff style) or DataFrames (``nest`` / ``group_nest`` /
    ``nest_join``).
    """

    def _apply(tf):
        columns = _columns(tf)
        if column not in columns:
            raise KeyError(f"column not found: {column!r}")

        dtype = tf._lf.collect_schema()[column]
        if dtype.base_type() != pl.List:
            raise TypeError("unnest() requires a list column")
        inner = dtype.inner
        output = tf._lf.explode(
            column,
            empty_as_null=keep_empty,
            keep_nulls=keep_empty,
        )
        if inner.base_type() == pl.Struct:
            output = output.unnest(column, separator=names_sep)
        ordered = output.collect_schema().names()
        return tf._with_lf(
            output,
            groups=_result_groups(tf, ordered),
            rowwise=False,
        )

    return Verb(_apply, "unnest")


def unnest_wider(
    column: str, *, names_sep: str | None = None, transform: Any = None
) -> Verb:
    """Expand a struct/dict or fixed-width list column into columns.

    ``transform`` converts the new columns: one function, or a dict by name
    (``transform={"a": int}``), as in tidyr.
    """
    core = _unnest_wider_core(column, names_sep=names_sep)
    if transform is None:
        return core

    def _apply(tf):
        out = core._fn(tf)
        before = set(_columns(tf))
        new = [name for name in _columns(out) if name not in before]
        return _transform_columns(out, transform, new, "transform")

    return Verb(_apply, "unnest_wider")


def _unnest_wider_core(column: str, *, names_sep: str | None = None) -> Verb:
    """Expand a struct/dict or fixed-width list column into columns."""

    def _apply(tf):
        columns = _columns(tf)
        if column not in columns:
            raise KeyError(f"column not found: {column!r}")

        dtype = tf._lf.collect_schema()[column]
        if dtype.base_type() == pl.Struct:
            lf = tf._lf.unnest(column, separator=names_sep)
        elif dtype.base_type() == pl.List:
            if names_sep is None:
                raise ValueError("names_sep is required for unnamed list values")
            from tidy3.frame import note_data_read

            note_data_read("unnest_wider() reads the longest list from the data")
            width = (
                tf._lf.select(pl.col(column).list.len().max())
                .collect()
                .item()
                or 0
            )
            expressions = [
                pl.col(column)
                .list.get(index, null_on_oob=True)
                .alias(f"{column}{names_sep}{index + 1}")
                for index in range(width)
            ]
            output = []
            for name in columns:
                if name == column:
                    output.extend(expressions)
                else:
                    output.append(pl.col(name))
            lf = tf._lf.select(output)
        else:
            raise TypeError("unnest_wider() requires a struct or list column")
        ordered = lf.collect_schema().names()
        return tf._with_lf(
            lf,
            groups=_result_groups(tf, ordered),
            rowwise=False,
        )

    return Verb(_apply, "unnest_wider")


def separate_longer_delim(cols: Any, delim: str) -> Verb:
    """Split delimited values into multiple rows."""
    if not isinstance(delim, str) or not delim:
        raise ValueError("separate_longer_delim() delim must be non-empty")

    def _apply(tf):
        return _pl_separate_longer_delim(tf, cols, delim)

    return Verb(_apply, "separate_longer_delim")


def separate_wider_delim(
    column: str,
    names: list[str] | tuple[str, ...],
    delim: str,
    *,
    too_few: str = "error",
    too_many: str = "error",
    names_sep: str | None = None,
    cols_remove: bool = True,
) -> Verb:
    """Split one delimited column into named columns, like tidyr.

    ``names_sep="_"`` names the pieces ``{column}_{name}``;
    ``cols_remove=False`` keeps the original column after them.

    ``too_few`` is ``error`` (default), ``align_start``, or ``align_end``;
    ``too_many`` is ``error`` (default), ``drop``, or ``merge`` (the extra
    pieces stay joined in the last column). Missing values stay missing.
    """
    if not names:
        raise TypeError("separate_wider_delim() requires output names")
    if too_few not in {"align_start", "align_end", "error"}:
        raise ValueError("too_few must be align_start, align_end, or error")
    if too_many not in {"error", "drop", "merge"}:
        raise ValueError("too_many must be error, drop, or merge")

    def _apply(tf):
        return _pl_separate_wider_delim(tf, column, list(names), delim, too_few, too_many, names_sep, cols_remove)

    return Verb(_apply, "separate_wider_delim")


# ── Polars versions of the separate_* verbs (no pandas round trip) ──────


def _pl_place(tf: Any, source: str, new: list[pl.Expr], keep_source: bool) -> Any:
    """Select with *new* columns where *source* was (and *source* after them)."""
    columns = _columns(tf)
    out = []
    for name in columns:
        if name == source:
            out.extend(new)
            if keep_source:
                out.append(pl.col(source))
        else:
            out.append(pl.col(name))
    lf = tf._lf.select(out)
    return tf._with_lf(lf, groups=_result_groups(tf, lf.collect_schema().names()))


def _pl_count(tf: Any, condition: pl.Expr) -> int:
    """How many rows meet *condition* (one small query)."""
    return int(tf._lf.select(condition.sum()).collect().item() or 0)


def _pl_separate_longer_delim(tf: Any, cols: Any, delim: str) -> Any:
    selected = resolve_selection(tf, [cols])
    if len(selected) != 1:
        raise ValueError("separate_longer_delim() currently accepts one column")
    column = selected[0]
    lf = tf._lf.with_columns(pl.col(column).str.split(delim)).explode(column, **_EXPLODE_OPTIONS)
    return tf._with_lf(lf, groups=_result_groups(tf, lf.collect_schema().names()))


def _pl_separate_wider_delim(tf, column, names, delim, too_few, too_many, names_sep, cols_remove):
    source = _one_column(tf, column, "separate_wider_delim")
    width = len(names)
    parts = pl.col(source).str.split(delim)
    # Counting delimiters avoids building a list per row; both checks share one query.
    count = (
        (pl.col(source).str.count_matches(delim, literal=True) + 1) if delim else parts.list.len()
    ).cast(pl.Int64)  # signed: width - count may be negative
    present = pl.col(source).is_not_null()
    checks = {}
    if too_few == "error":
        checks["short"] = (present & (count < width)).sum()
    if too_many == "error":
        checks["long"] = (present & (count > width)).sum()
    found = tf._lf.select(**checks).collect().row(0, named=True) if checks else {}
    too_short, too_long = int(found.get("short") or 0), int(found.get("long") or 0)
    if too_short:
        raise ValueError(
            f"separate_wider_delim() expected {width} pieces in each value "
            f"of {source!r}; {too_short} value(s) had fewer. "
            'Use too_few="align_start" or "align_end" to fill with missing.'
        )
    if too_long:
        raise ValueError(
            f"separate_wider_delim() expected {width} pieces in each value "
            f"of {source!r}; {too_long} value(s) had more. "
            'Use too_many="drop" or "merge".'
        )
    if delim and too_few != "align_end":
        # Fields come straight from the split: missing ones are null, and
        # splitn keeps the rest of the text in the last field for "merge".
        fields = (
            pl.col(source).str.splitn(delim, width) if too_many == "merge"
            else pl.col(source).str.split_exact(delim, width - 1)
        )
        new = [
            fields.struct.field(f"field_{index}").alias(_output_name(source, name, names_sep))
            for index, name in enumerate(names)
        ]
        return _pl_place(tf, source, new, not cols_remove)
    shift = (pl.lit(width) - count).clip(lower_bound=0) if too_few == "align_end" else pl.lit(0)
    new = []
    for index, name in enumerate(names):
        position = pl.lit(index) - shift
        piece = pl.when(position >= 0).then(parts.list.get(position, null_on_oob=True))
        if too_many == "merge" and index == width - 1:
            rest = parts.list.slice(width - 1).list.join(delim)
            piece = pl.when(count > width).then(rest).otherwise(piece)
        new.append(piece.alias(_output_name(source, name, names_sep)))
    return _pl_place(tf, source, new, not cols_remove)


def _pl_separate_wider_regex(tf, column, pieces, too_few, names_sep, cols_remove):
    source = _one_column(tf, column, "separate_wider_regex")
    names = [name for name, _ in pieces if name is not None]

    def anchored(count: int) -> str:
        body = "".join(
            f"({pattern})" if name is not None else f"(?:{pattern})"
            for name, pattern in pieces[:count]
        )
        return f"^(?:{body})$"

    value = pl.col(source)
    full = anchored(len(pieces))
    failed = _pl_count(tf, value.is_not_null() & ~value.str.contains(full))
    if failed and too_few == "error":
        raise ValueError(
            f"separate_wider_regex(): each value of {source!r} must match the "
            f"whole pattern; {failed} value(s) did not. "
            'Use too_few="align_start" to keep partial matches.'
        )
    # Longest leading run of pieces that matches the whole value wins.
    attempts = [len(pieces)] + list(range(len(pieces) - 1, 0, -1))
    new = []
    for slot, name in enumerate(names):
        expr = None
        for count in attempts:
            kept = [n for n, _ in pieces[:count] if n is not None]
            if slot >= len(kept):
                continue
            pattern = anchored(count)
            piece = value.str.extract(pattern, slot + 1)
            matched = value.str.contains(pattern)
            expr = (
                pl.when(matched).then(piece)
                if expr is None
                else expr.when(matched).then(piece)
            )
        new.append(
            (expr.otherwise(None) if expr is not None else pl.lit(None, dtype=pl.String))
            .alias(_output_name(source, name, names_sep))
        )
    return _pl_place(tf, source, new, not cols_remove)


def _pl_separate_wider_position(tf, column, pieces, total, too_few, too_many, names_sep, cols_remove):
    source = _one_column(tf, column, "separate_wider_position")
    length = pl.col(source).str.len_chars()
    # Both counts in one pass over the column; a missing value's length is
    # missing, which a boolean sum leaves out.
    checks = {}
    if too_few == "error":
        checks["short"] = (length < total).sum()
    if too_many == "error":
        checks["long"] = (length > total).sum()
    counts = tf._lf.select(**checks).collect().row(0, named=True) if checks else {}
    short = int(counts.get("short") or 0)
    long = int(counts.get("long") or 0)
    if short:
        raise ValueError(
            f"separate_wider_position(): expected {total} characters in each "
            f'value of {source!r}; {short} value(s) were shorter. Use too_few="align_start".'
        )
    if long:
        raise ValueError(
            f"separate_wider_position(): expected {total} characters in each "
            f'value of {source!r}; {long} value(s) were longer. Use too_many="drop".'
        )
    new, start = [], 0
    for name, width in pieces:
        if name is not None:
            part = pl.col(source).str.slice(start, int(width))
            if too_few == "error":
                # Every present value has at least the total width, so no
                # piece is empty: the slice is the column.
                new.append(part.alias(_output_name(source, name, names_sep)))
            else:
                # An empty piece is missing; its byte length says so without
                # counting characters.
                new.append(
                    pl.when(part.str.len_bytes() > 0).then(part).alias(
                        _output_name(source, name, names_sep)
                    )
                )
        start += int(width)
    return _pl_place(tf, source, new, not cols_remove)


def _pl_separate_longer_position(tf, column, width, keep_empty):
    source = _one_column(tf, column, "separate_longer_position")
    chunks = pl.col(source).str.extract_all(rf"(?s).{{1,{width}}}")
    lf = tf._lf.with_columns(chunks.alias(source))
    if not keep_empty:
        # Empty strings give no pieces and no row; missing values keep one row.
        lf = lf.filter(pl.col(source).is_null() | (pl.col(source).list.len() > 0))
    lf = lf.explode(source, **_EXPLODE_OPTIONS)
    return tf._with_lf(lf, groups=_result_groups(tf, lf.collect_schema().names()))


def _one_column(tf: Any, column: Any, verb: str) -> str:
    selected = resolve_selection(tf, [column])
    if len(selected) != 1:
        raise ValueError(f"{verb}() requires one column")
    return selected[0]


def _output_name(source: str, name: str, names_sep: str | None) -> str:
    return f"{source}{names_sep}{name}" if names_sep is not None else name


def separate_wider_regex(
    column: Any,
    patterns: Any,
    *,
    names_sep: str | None = None,
    too_few: str = "error",
    cols_remove: bool = True,
) -> Verb:
    """Split a column into columns with regular expressions (tidyr).

    ``patterns`` is a list of ``(name, regex)`` pairs and bare regexes that
    are matched but not kept, or a dict of named pieces::

        separate_wider_regex("code", [("letters", "[a-z]+"), "-", ("number", "[0-9]+")])

    Each value must match the whole pattern; ``too_few="align_start"``
    fills the pieces that match from the start and leaves the rest missing.
    """
    import re

    pieces = list(patterns.items()) if isinstance(patterns, dict) else [
        piece if isinstance(piece, tuple) else (None, piece) for piece in patterns
    ]
    if not pieces:
        raise TypeError("separate_wider_regex() needs at least one pattern")
    if too_few not in {"error", "align_start"}:
        raise ValueError("too_few must be error or align_start")
    names = [name for name, _ in pieces if name is not None]

    def compiled(count: int):
        body = "".join(
            f"({pattern})" if name is not None else f"(?:{pattern})"
            for name, pattern in pieces[:count]
        )
        return re.compile(body)

    full = compiled(len(pieces))
    partial = [compiled(count) for count in range(len(pieces) - 1, 0, -1)]

    def split(value: Any) -> list[Any] | None:
        if value is None or (isinstance(value, float) and value != value):
            return [None] * len(names)
        found = full.fullmatch(str(value))
        if found:
            return list(found.groups())
        if too_few == "error":
            return None
        for pattern in partial:
            found = pattern.fullmatch(str(value))
            if found:
                groups = list(found.groups())
                return groups + [None] * (len(names) - len(groups))
        return [None] * len(names)

    def _apply(tf):
        return _pl_separate_wider_regex(tf, column, pieces, too_few, names_sep, cols_remove)

    return Verb(_apply, "separate_wider_regex")


def separate_wider_position(
    column: Any,
    widths: Any,
    *,
    names_sep: str | None = None,
    too_few: str = "error",
    too_many: str = "error",
    cols_remove: bool = True,
) -> Verb:
    """Split a column into columns of fixed widths (tidyr).

    ``widths`` is a list of ``(name, width)`` pairs and bare widths that are
    skipped, or a dict::

        separate_wider_position("date", [("year", 4), ("month", 2), ("day", 2)])

    Values shorter than the total width raise unless
    ``too_few="align_start"``; longer ones raise unless ``too_many="drop"``.
    """
    pieces = list(widths.items()) if isinstance(widths, dict) else [
        piece if isinstance(piece, tuple) else (None, piece) for piece in widths
    ]
    if not pieces or any(int(width) <= 0 for _, width in pieces):
        raise ValueError("separate_wider_position() widths must be positive")
    if too_few not in {"error", "align_start"}:
        raise ValueError("too_few must be error or align_start")
    if too_many not in {"error", "drop"}:
        raise ValueError("too_many must be error or drop")
    total = sum(int(width) for _, width in pieces)

    def _apply(tf):
        return _pl_separate_wider_position(tf, column, pieces, total, too_few, too_many, names_sep, cols_remove)

    return Verb(_apply, "separate_wider_position")


def separate_longer_position(column: Any, width: int, *, keep_empty: bool = False) -> Verb:
    """Split each value into ``width``-character pieces, one row each (tidyr).

    Missing values stay one missing row; empty strings are dropped unless
    ``keep_empty=True``.
    """
    if int(width) <= 0:
        raise ValueError("separate_longer_position() width must be positive")

    def _apply(tf):
        return _pl_separate_longer_position(tf, column, int(width), keep_empty)

    return Verb(_apply, "separate_longer_position")


def chop(*cols: Any) -> Verb:
    """Collect columns into lists, one row per combination of the others (tidyr)."""
    if not cols:
        raise TypeError("chop() requires at least one column")

    def _apply(tf):
        chopped = resolve_selection(tf, cols)
        columns = _columns(tf)
        keys = [name for name in columns if name not in chopped]
        lf = (
            tf._lf.group_by(keys, maintain_order=True).agg(chopped)
            if keys
            else tf._lf.select(pl.col(name).implode() for name in chopped)
        )
        return tf._with_lf(lf.select(columns), groups=_result_groups(tf, columns))

    return Verb(_apply, "chop")


def unchop(*cols: Any, keep_empty: bool = False) -> Verb:
    """Turn list columns back into rows, the inverse of :func:`chop` (tidyr).

    Rows whose lists are empty or missing are dropped unless
    ``keep_empty=True``, which keeps one row with missing values.
    """
    if not cols:
        raise TypeError("unchop() requires at least one column")

    def _apply(tf):
        listed = resolve_selection(tf, cols)
        columns = _columns(tf)
        lf = tf._lf
        if not keep_empty:
            lf = lf.filter(pl.col(listed[0]).list.len().fill_null(0) > 0)
        lf = lf.explode(listed, **_EXPLODE_OPTIONS)
        return tf._with_lf(lf, groups=_result_groups(tf, columns))

    return Verb(_apply, "unchop")


def unnest_auto(column: str) -> Verb:
    """``unnest_wider`` for named elements, ``unnest_longer`` otherwise (tidyr)."""

    def _apply(tf):
        named = isinstance(tf._lf.collect_schema()[column], pl.Struct)
        verb = unnest_wider(column) if named else unnest_longer(column)
        return tf >> verb

    return Verb(_apply, "unnest_auto")


def hoist(
    column: str,
    *paths: Any,
    remove: bool = True,
    transform: Any = None,
    **named_paths: Any,
) -> Verb:
    """Pull fields out of a dict/list column into columns, as tidyr's hoist().

    ``hoist("info", name="name", city=["address", "city"])`` adds ``name`` and
    ``city`` just before ``info``. Pulled fields are removed from ``info``
    (the column goes when nothing is left) unless ``remove=False``;
    ``transform`` converts the new columns (``transform={"n": int}``).
    """
    def _apply(tf):
        specs = list(named_paths.items()) or [(str(path), path) for path in paths]
        return _pl_hoist(tf, column, specs, remove, transform)
    return Verb(_apply, "hoist")


def _pl_path(base: pl.Expr, dtype: Any, keys: list[Any]) -> tuple[pl.Expr | None, Any]:
    """The expression (and dtype) for *keys* inside a struct/list value."""
    expr, current = base, dtype
    for key in keys:
        if isinstance(current, pl.Struct) and isinstance(key, str):
            fields = {field.name: field.dtype for field in current.fields}
            if key not in fields:
                return None, None
            expr = expr.struct.field(key)
            current = fields[key]
        elif isinstance(current, (pl.List, pl.Array)) and isinstance(key, int):
            expr = expr.list.get(key, null_on_oob=True)
            current = current.inner
        else:
            return None, None
    return expr, current


def _pl_without(value: pl.Expr, dtype: Any, paths: list[list[Any]]) -> pl.Expr | None:
    """*value* (a struct) rebuilt without the fields at *paths*; None if empty."""
    drop_here = {path[0] for path in paths if len(path) == 1}
    deeper: dict[str, list[list[Any]]] = {}
    for path in paths:
        if len(path) > 1 and isinstance(path[0], str):
            deeper.setdefault(path[0], []).append(path[1:])
    kept = []
    for field in dtype.fields:
        if field.name in drop_here:
            continue
        inner = value.struct.field(field.name)
        if field.name in deeper and isinstance(field.dtype, pl.Struct):
            inner = _pl_without(inner, field.dtype, deeper[field.name])
            if inner is None:
                continue
        kept.append(inner.alias(field.name))
    return pl.struct(kept) if kept else None


def _pl_hoist(tf: Any, column: str, specs: list[tuple[str, Any]], remove: bool, transform: Any) -> Any:
    """hoist() on a Polars struct column, without leaving Polars."""
    dtype = tf._lf.collect_schema()[column]
    if not isinstance(dtype, (pl.Struct, pl.List)):
        raise TypeError(f"hoist() needs a struct or list column; {column!r} is {dtype}")
    hoisted, paths = [], []
    for name, path in specs:
        keys = list(path) if isinstance(path, (list, tuple)) else [path]
        paths.append(keys)
        expr, _ = _pl_path(pl.col(column), dtype, keys)
        hoisted.append((expr if expr is not None else pl.lit(None)).alias(name))
    rest = _pl_without(pl.col(column), dtype, paths) if remove and isinstance(dtype, pl.Struct) else pl.col(column)
    columns = _columns(tf)
    out = []
    for name in columns:
        if name == column:
            out.extend(hoisted)
            if rest is not None:
                out.append(rest.alias(column))
        else:
            out.append(pl.col(name))
    lf = tf._lf.select(out)
    result = tf._with_lf(lf, groups=_result_groups(tf, lf.collect_schema().names()))
    return _transform_columns(result, transform, [name for name, _ in specs], "transform")


def pack(name: str, *cols: Any) -> Verb:
    """Pack selected columns into a structured column."""
    def _apply(tf):
        selected = resolve_selection(tf, cols or [c for c in _columns(tf) if c not in (tf._groups or [])])
        lf = tf._lf.with_columns(pl.struct(selected).alias(name)).drop(selected)
        return tf._with_lf(lf, groups=_result_groups(tf, lf.collect_schema().names()))
    return Verb(_apply, "pack")


def unpack(column: str) -> Verb:
    """Expand a structured dictionary column into ordinary columns."""
    def _apply(tf):
        import pandas as pd

        return tf >> unnest_wider(column)
    return Verb(_apply, "unpack")


__all__ = [
    "build_longer_spec",
    "build_wider_spec",
    "check_pivot_spec",
    "pivot_longer_spec",
    "pivot_wider_spec",
    "chop",
    "separate_longer_position",
    "separate_wider_position",
    "separate_wider_regex",
    "unchop",
    "unnest_auto",
    "crossing",
    "expand_grid",
    "full_seq",
    "uncount",
    "complete",
    "drop_na",
    "expand",
    "fill",
    "nest",
    "nesting",
    "pivot_longer",
    "pivot_wider",
    "replace_na",
    "separate",
    "unite",
    "unnest",
    "unnest_longer",
    "unnest_wider",
    "separate_longer_delim",
    "separate_wider_delim",
    "hoist",
    "pack",
    "unpack",
]
