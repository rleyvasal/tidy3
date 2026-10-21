"""Column helpers for missing values, math, and row-wise totals.

Each helper builds a backend-neutral expression, so it runs inside
``mutate``, ``filter``, ``summarise``, and ``across`` on Polars and on
pandas. ``abs`` and ``round`` share names with Python built-ins, as
``sum``, ``min``, ``max``, ``any``, and ``all`` already do.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from tidy3.expr import Expr, _func, col

__all__ = [
    "is_na",
    "is_nan",
    "is_finite",
    "is_infinite",
    "abs",
    "floor",
    "ceiling",
    "round",
    "pmin",
    "pmax",
    "row_sums",
    "row_means",
    "RowReduce",
]

_FLAGS = ("is_na", "is_nan", "is_finite", "is_infinite")
_MATH = {"abs", "floor", "ceiling", "round"}
_PAIRS = {"pmin", "pmax"}
_ROWS = {"row_sums", "row_means"}
FUNCTIONS = frozenset((*_FLAGS, *_MATH, *_PAIRS, *_ROWS))


def _bool(name: str, value: bool) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")


def _count(name: str, value: int | None) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _value(x: Any) -> Any:
    return col(x) if isinstance(x, str) else x


def _column(x: Any) -> str | None:
    if isinstance(x, str):
        return x
    if isinstance(x, Expr) and x.node[0] == "col":
        return x.node[1]
    return None


class RowReduce:
    """Row-wise sum or mean of a column selection (``pick`` or ``c_across``)."""

    __slots__ = ("selector", "operation", "na_rm", "min_count", "requires_rowwise")

    def __init__(
        self,
        selector: Any,
        operation: str,
        *,
        na_rm: bool,
        min_count: int | None,
        requires_rowwise: bool,
    ):
        self.selector = selector
        self.operation = operation
        self.na_rm = na_rm
        self.min_count = min_count
        self.requires_rowwise = requires_rowwise

    def expand(self, tf: Any) -> Expr:
        if self.requires_rowwise and not tf._rowwise:
            raise ValueError("c_across() must be used after rowwise()")
        from tidy3.tidyselect import resolve_selection

        columns = resolve_selection(tf, [self.selector])
        columns = [name for name in columns if name not in (tf._groups or [])]
        kwargs: dict[str, Any] = {"na_rm": self.na_rm}
        if self.operation == "sum":
            kwargs["min_count"] = self.min_count
        name = "row_sums" if self.operation == "sum" else "row_means"
        return _func(name, *[col(column) for column in columns], **kwargs)


def _selection(cols: tuple[Any, ...], operation: str, na_rm: bool, min_count: int | None):
    """A lone ``pick``, ``c_across``, or selector stands for its columns."""
    if len(cols) != 1:
        return None
    item = cols[0]
    kind = type(item).__name__
    if kind == "ColumnSet":
        return RowReduce(
            item.selector,
            operation,
            na_rm=na_rm,
            min_count=min_count,
            requires_rowwise=bool(item.requires_rowwise),
        )
    if kind == "Selector":
        return RowReduce(
            item, operation, na_rm=na_rm, min_count=min_count, requires_rowwise=False
        )
    return None


def _flat(cols: tuple[Any, ...]) -> tuple[Any, ...]:
    if len(cols) == 1 and isinstance(cols[0], (list, tuple)):
        return tuple(cols[0])
    return cols


def is_na(x: Any) -> Expr:
    """True where *x* is missing: Polars null or NaN.

    ``filter(is_na(body_mass_g))`` keeps both. Polars ``is_null`` does not
    see NaN, and ``is_nan`` does not see null; this is the check that sees
    both, as R's ``is.na`` does. The result is never itself missing, so
    ``filter`` can keep the row.
    """
    return _func("is_na", _value(x))


def is_nan(x: Any) -> Expr:
    """True only for NaN, not for null.

    On the pandas backend a missing float is stored as one value, so a
    missing float is both ``is_na`` and ``is_nan``. Polars keeps them apart:
    null is not NaN.
    """
    return _func("is_nan", _value(x))


def is_finite(x: Any) -> Expr:
    """True for finite numbers. Null, NaN, and infinity are false."""
    return _func("is_finite", _value(x))


def is_infinite(x: Any) -> Expr:
    """True for positive or negative infinity. Null and NaN are false."""
    return _func("is_infinite", _value(x))


def abs(x: Any) -> Expr:  # noqa: A001
    """Absolute value. Shadows Python's ``abs`` when imported from tidy3."""
    return _func("abs", _value(x))


def floor(x: Any) -> Expr:
    """Largest integer not above *x* (toward -infinity)."""
    return _func("floor", _value(x))


def ceiling(x: Any) -> Expr:
    """Smallest integer not below *x* (toward +infinity). R's ``ceiling``."""
    return _func("ceiling", _value(x))


def round(x: Any, digits: int = 0) -> Expr:  # noqa: A001
    """Round *x* as R's ``round`` does (R 4.0 and later).

    An exact tie goes to the even digit: ``round(0.5)`` is 0, ``round(1.5)``
    is 2, and ``round(0.125, 2)`` is 0.12. Otherwise the result is the
    candidate nearer to the number as stored, so ``round(2.675, 2)`` is
    2.67 because 2.675 is stored as 2.67499999.... Negative ``digits`` round
    to tens, hundreds, and so on (``round(123, -1)`` is 120). Integer
    columns stay integers. Shadows Python's ``round`` when imported from
    tidy3.
    """
    if isinstance(digits, bool) or not isinstance(digits, int):
        raise TypeError("round() digits must be an integer")
    return _func("round", _value(x), digits=digits)


def pmin(*args: Any, na_rm: bool = False) -> Expr:
    """Element-wise minimum, as R's ``pmin``.

    ``mutate(hp_capped = pmin(hp, 200))`` caps a column. With the default
    ``na_rm=False``, any missing input (null or NaN) makes that row missing.
    ``na_rm=True`` skips missing inputs; a row of only missing values stays
    missing.
    """
    _bool("pmin() na_rm", na_rm)
    if not args:
        raise TypeError("pmin() requires at least one value")
    return _func("pmin", *[_value(arg) for arg in args], na_rm=na_rm)


def pmax(*args: Any, na_rm: bool = False) -> Expr:
    """Element-wise maximum, as R's ``pmax``. See :func:`pmin`."""
    _bool("pmax() na_rm", na_rm)
    if not args:
        raise TypeError("pmax() requires at least one value")
    return _func("pmax", *[_value(arg) for arg in args], na_rm=na_rm)


def row_sums(*cols: Any, na_rm: bool = False, min_count: int | None = None) -> Any:
    """Sum across columns in each row.

    ``na_rm=False`` (the default, as in R's ``rowSums``) makes the row
    missing when any value is missing. ``na_rm=True`` skips missing values
    and, like R, sums an all-missing row to 0. ``min_count`` changes that:
    a row with fewer than ``min_count`` real values is missing instead of 0,
    so ``min_count=1`` makes an empty row missing.
    """
    _bool("row_sums() na_rm", na_rm)
    _count("row_sums() min_count", min_count)
    cols = _flat(cols)
    selected = _selection(cols, "sum", na_rm, min_count)
    if selected is not None:
        return selected
    if not cols:
        raise TypeError("row_sums() requires at least one column")
    return _func(
        "row_sums",
        *[_value(column) for column in cols],
        na_rm=na_rm,
        min_count=min_count,
    )


def row_means(*cols: Any, na_rm: bool = False) -> Any:
    """Mean across columns in each row.

    ``na_rm=False`` propagates a missing value. ``na_rm=True`` averages the
    values that are present. A row with no values is missing, not zero.
    """
    _bool("row_means() na_rm", na_rm)
    cols = _flat(cols)
    selected = _selection(cols, "mean", na_rm, None)
    if selected is not None:
        return selected
    if not cols:
        raise TypeError("row_means() requires at least one column")
    return _func("row_means", *[_value(column) for column in cols], na_rm=na_rm)


# ── flags ───────────────────────────────────────────────────────────────


def _polars_flags(series: pl.Series) -> dict[str, pl.Series]:
    size = series.len()
    false = pl.Series([False] * size)
    null = series.is_null()
    dtype = series.dtype
    if dtype.is_float():
        nan = series.is_nan().fill_null(False)
        infinite = series.is_infinite().fill_null(False)
        return {
            "is_na": null | nan,
            "is_nan": nan,
            "is_finite": series.is_finite().fill_null(False),
            "is_infinite": infinite,
        }
    if dtype.is_integer() or dtype.is_numeric():
        return {
            "is_na": null,
            "is_nan": false,
            "is_finite": ~null,
            "is_infinite": false,
        }
    return {"is_na": null, "is_nan": false, "is_finite": false, "is_infinite": false}


def _pandas_flags(series: pd.Series) -> dict[str, pd.Series]:
    index = series.index
    false = pd.Series(False, index=index)
    missing = series.isna()
    if pd.api.types.is_float_dtype(series):
        raw = series.tolist()
        nan = pd.Series(
            [isinstance(value, float) and math.isnan(value) for value in raw],
            index=index,
        )
        infinite = pd.Series(
            [isinstance(value, float) and math.isinf(value) for value in raw],
            index=index,
        )
        finite = pd.Series(
            [
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                for value in raw
            ],
            index=index,
        )
        # float64 stores null as NaN, so those rows are both is_na and is_nan.
        return {
            "is_na": missing | nan,
            "is_nan": nan,
            "is_finite": finite,
            "is_infinite": infinite,
        }
    if pd.api.types.is_bool_dtype(series):
        return {"is_na": missing, "is_nan": false, "is_finite": false, "is_infinite": false}
    if pd.api.types.is_integer_dtype(series) or pd.api.types.is_numeric_dtype(series):
        return {
            "is_na": missing,
            "is_nan": false,
            "is_finite": ~missing,
            "is_infinite": false,
        }
    return {"is_na": missing, "is_nan": false, "is_finite": false, "is_infinite": false}


def _flag_expr(name: str, value: pl.Expr) -> pl.Expr:
    def batch(series: pl.Series) -> pl.Series:
        return _polars_flags(series)[name]

    return value.map_batches(batch, return_dtype=pl.Boolean)


# ── math ────────────────────────────────────────────────────────────────


def _nullify_nan(expr: pl.Expr) -> pl.Expr:
    """Treat NaN as missing so pmin/row sums match R, where NaN is NA.

    Only float columns hold NaN; text such as ``"nan"`` is left alone, so
    ``pmin(c("nan", "b"), c("c", "a"))`` is ``c("c", "a")`` as in R.
    """

    def batch(series: pl.Series) -> pl.Series:
        return series.fill_nan(None) if series.dtype.is_float() else series

    return expr.map_batches(batch, return_dtype=pl.dtype_of(expr), is_elementwise=True)


def _pair(name: str, args: tuple[Any, ...], na_rm: bool) -> pl.Expr:
    values = [_nullify_nan(arg) for arg in args]
    result = values[0] if len(values) == 1 else (
        pl.min_horizontal(values) if name == "pmin" else pl.max_horizontal(values)
    )
    if na_rm:
        return result
    missing = pl.any_horizontal([value.is_null() for value in values])
    return pl.when(missing).then(None).otherwise(result)


# R's round() (R 4.0 and later): of the two candidates either side of x,
# take the one nearer to the double x actually holds; on an exact tie take
# the even one. 2.675 is stored as 2.67499999..., so it rounds to 2.67,
# while 0.125 is exact and rounds to 0.12. When x already has more than 15
# significant digits before the rounding place, R returns it unchanged.
_ROUND_KEEP_DIGITS = 15


def _round_expr(value: pl.Expr, digits: int) -> pl.Expr:
    def batch(series: pl.Series) -> pl.Series:
        if series.dtype.is_integer() and digits >= 0:
            return series
        # NumPy, not Polars arithmetic: Polars divides a column by a constant
        # through its reciprocal, so 3 / 10 would come out 0.30000000000000004.
        values = series.cast(pl.Float64).to_numpy()
        rounded = pl.Series(series.name, _round_numpy(values, digits))
        rounded = pl.select(
            pl.when(series.is_null()).then(None).otherwise(rounded)
        ).to_series()
        return rounded.cast(series.dtype)

    # Integers stay integers; floats stay floats.
    return value.map_batches(batch, return_dtype=pl.dtype_of(value), is_elementwise=True)


def _round_numpy(values: np.ndarray, digits: int) -> np.ndarray:
    """R's ``round`` on a float array; shared by both backends."""
    x = np.asarray(values, dtype=float)
    if digits > 323:
        return x.copy()
    if digits < -308:
        return np.where(np.isfinite(x) & (x != 0), 0.0, x)
    with np.errstate(all="ignore"):
        size = np.abs(x)
        if digits > 308:
            # Split the scale so tiny numbers can round past 308 places
            # without computing an overflowing power of ten.
            extra = 10.0 ** (digits - 308)
            scaled = (size * 1e308) * extra
            whole = np.floor(scaled)
            lower, upper = (whole / 1e308) / extra, (np.ceil(scaled) / 1e308) / extra
        elif digits >= 0:
            scale = 10.0**digits
            whole = np.floor(size * scale)
            lower, upper = whole / scale, np.ceil(size * scale) / scale
        else:
            scale = 10.0**-digits
            whole = np.floor(size / scale)
            lower, upper = whole * scale, np.ceil(size / scale) * scale
        down, up = size - lower, upper - size
        nearer = np.where((down < up) | ((down == up) & (np.fmod(whole, 2) == 0)), lower, upper)
        keep = ~np.isfinite(size) | (size == 0)
        if digits > 0:
            keep |= np.log10(size) + digits > _ROUND_KEEP_DIGITS
        return np.where(keep, x, np.copysign(nearer, x))


def _row_expr(
    operation: str,
    args: tuple[Any, ...],
    *,
    na_rm: bool,
    min_count: int | None,
) -> pl.Expr:
    if not args:
        if operation == "mean":
            return pl.lit(None, dtype=pl.Float64)
        if min_count:
            return pl.lit(None, dtype=pl.Float64)
        return pl.lit(0)
    values = [_nullify_nan(arg) for arg in args]
    present = pl.sum_horizontal(
        [value.is_not_null().cast(pl.Int64) for value in values]
    )
    total = pl.sum_horizontal(values)
    if operation == "mean":
        result = pl.when(present == 0).then(None).otherwise(total / present)
    else:
        result = total.cast(pl.Float64)
        if min_count:
            result = pl.when(present < min_count).then(None).otherwise(result)
    if not na_rm:
        missing = pl.any_horizontal([value.is_null() for value in values])
        result = pl.when(missing).then(None).otherwise(result)
    return result


def compile_polars(name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> pl.Expr:
    """Compile a helper to a Polars expression."""
    if name in _FLAGS:
        return _flag_expr(name, args[0])
    if name == "abs":
        return args[0].abs()
    if name == "floor":
        return args[0].floor()
    if name == "ceiling":
        return args[0].ceil()
    if name == "round":
        return _round_expr(args[0], int(kwargs["digits"]))
    if name in _PAIRS:
        return _pair(name, args, bool(kwargs["na_rm"]))
    if name in _ROWS:
        return _row_expr(
            "sum" if name == "row_sums" else "mean",
            args,
            na_rm=bool(kwargs["na_rm"]),
            min_count=kwargs.get("min_count"),
        )
    raise ValueError(f"unknown helper: {name}")


def _as_series(value: Any, index: pd.Index) -> pd.Series:
    if isinstance(value, pd.Series):
        return value.reindex(index)
    return pd.Series(value, index=index)


def _pandas_pair(name: str, args: tuple[Any, ...], na_rm: bool, index: pd.Index) -> pd.Series:
    frame = pd.concat([_as_series(arg, index) for arg in args], axis=1)
    # NaN counts as missing, as in R.
    frame = frame.mask(frame.apply(
        lambda column: column.map(lambda value: isinstance(value, float) and math.isnan(value))
    ))
    result = frame.min(axis=1, skipna=na_rm) if name == "pmin" else frame.max(axis=1, skipna=na_rm)
    return result


def _pandas_rows(
    operation: str,
    args: tuple[Any, ...],
    *,
    na_rm: bool,
    min_count: int | None,
    index: pd.Index,
) -> pd.Series:
    if not args:
        fill = np.nan if operation == "mean" or min_count else 0
        return pd.Series(fill, index=index)
    frame = pd.concat([_as_series(arg, index) for arg in args], axis=1)
    frame = frame.mask(frame.apply(
        lambda column: column.map(lambda value: isinstance(value, float) and math.isnan(value))
    ))
    present = frame.notna().sum(axis=1)
    if na_rm:
        total = frame.sum(axis=1, min_count=0)
    else:
        total = frame.sum(axis=1, min_count=frame.shape[1])
    if operation == "mean":
        result = total / present.replace(0, np.nan)
        result = result.mask(present == 0)
    else:
        result = total
        if min_count:
            result = result.mask(present < min_count)
    return result


def eval_pandas(
    name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    df: pd.DataFrame,
) -> Any:
    """Evaluate a helper against a pandas frame."""
    # Inputs keep their own index: inside a grouped summarise, mean(hp) has
    # one value per group, indexed by group, not one per row of *df*.
    index = next((arg.index for arg in args if isinstance(arg, pd.Series)), df.index)
    if name in _FLAGS:
        series = _as_series(args[0], index)
        return _pandas_flags(series)[name]
    series = _as_series(args[0], index) if args else None
    if name == "abs":
        return series.abs()
    if name == "floor":
        return np.floor(series)
    if name == "ceiling":
        return np.ceil(series)
    if name == "round":
        digits = int(kwargs["digits"])
        if pd.api.types.is_integer_dtype(series) and digits >= 0:
            return series
        values = series.to_numpy(dtype=float, na_value=np.nan)
        rounded = pd.Series(_round_numpy(values, digits), index=series.index)
        if pd.api.types.is_integer_dtype(series):
            return rounded.astype(series.dtype)
        return rounded
    if name in _PAIRS:
        return _pandas_pair(name, args, bool(kwargs["na_rm"]), index)
    if name in _ROWS:
        return _pandas_rows(
            "sum" if name == "row_sums" else "mean",
            args,
            na_rm=bool(kwargs["na_rm"]),
            min_count=kwargs.get("min_count"),
            index=index,
        )
    raise ValueError(f"unknown helper: {name}")
