"""Column helpers for missing values, math, and row-wise totals.

Each helper builds a backend-neutral expression, so it runs inside
``mutate``, ``filter``, ``summarise``, and ``across``. ``abs`` and
``round`` share names with Python built-ins, as
``sum``, ``min``, ``max``, ``any``, and ``all`` already do.
"""

from __future__ import annotations

import math
import sys
from typing import Any

import numpy as np
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
    "round_half_up",
    "signif_half_up",
    "round_to_fraction",
    "pmin",
    "pmax",
    "row_sums",
    "row_means",
    "RowReduce",
]

_FLAGS = ("is_na", "is_nan", "is_finite", "is_infinite")
_MATH = {"abs", "floor", "ceiling", "round", "round_half_up", "signif_half_up", "round_to_fraction"}
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

    pandas stores a missing float as NaN, so there a missing float is both
    ``is_na`` and ``is_nan``. Polars keeps them apart: null is not NaN.
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


def round_half_up(x: Any, digits: int = 0) -> Expr:
    """Round *x* with ties away from zero, as janitor's ``round_half_up``.

    ``round_half_up(2.5)`` is 3 and ``round_half_up(-2.5)`` is -3, where
    :func:`round` gives 2 and -2. As in janitor, *x* is scaled by
    ``10 ** digits``, nudged up by a half plus ``sqrt(.Machine$double.eps)``
    (about 1.5e-8), truncated, and scaled back, in R's double arithmetic.
    So a tie stored a hair low still rounds up: ``round_half_up(2.675, 2)``
    is 2.68, where :func:`round` gives 2.67. Negative ``digits`` round to
    tens, hundreds, and so on (``round_half_up(25, -1)`` is 30). Integer
    columns stay integers; missing values, NaN, and infinities pass through.
    """
    if isinstance(digits, bool) or not isinstance(digits, int):
        raise TypeError("round_half_up() digits must be an integer")
    return _func("round_half_up", _value(x), digits=digits)


def signif_half_up(x: Any, digits: int = 6) -> Expr:
    """Round *x* to *digits* significant digits, ties away from zero.

    janitor's ``signif_half_up``: each value is scaled so that *digits*
    digits sit before the decimal point, rounded with
    :func:`round_half_up`, and scaled back, so ``signif_half_up(12.5, 2)``
    is 13 and ``signif_half_up(0.00125, 2)`` is 0.0013. Zero, missing
    values, NaN, and infinities pass through. Integer columns stay integers
    (``signif_half_up(1250, 2)`` is 1300). As in janitor, a value too small
    to scale (a subnormal, below about 1e-308) comes out NaN.
    """
    if isinstance(digits, bool) or not isinstance(digits, int):
        raise TypeError("signif_half_up() digits must be an integer")
    return _func("signif_half_up", _value(x), digits=digits)


def round_to_fraction(x: Any, denominator: float, digits: int | str | None = None) -> Expr:
    """Round *x* to the nearest fraction with *denominator*, as janitor does.

    ``round_to_fraction(x, 4)`` rounds to quarters, so 1.3 becomes 1.25 and
    2.6 becomes 2.5. As in janitor, ``x * denominator`` is rounded with R's
    ``round()`` and divided back; an exact tie goes to the even number, so
    in quarters 0.125 rounds to 0 and 0.375 to 0.5. *denominator* is a
    number of at least 1. ``digits`` then rounds the result with
    :func:`round`: ``None`` (janitor's ``Inf``, the default) leaves it,
    an integer rounds to that many decimals, and ``"auto"`` uses
    ``ceiling(log10(denominator)) + 1``, so thirds show as 0.333.
    """
    if isinstance(denominator, bool) or not isinstance(denominator, (int, float)):
        raise TypeError("round_to_fraction() denominator must be a number")
    if not denominator >= 1:
        raise ValueError("round_to_fraction() denominator must be at least 1")
    if digits == "auto":
        digits = math.ceil(math.log10(denominator)) + 1
    elif isinstance(digits, float) and math.isinf(digits) and digits > 0:
        digits = None
    elif digits is not None and (isinstance(digits, bool) or not isinstance(digits, int)):
        raise TypeError('round_to_fraction() digits must be an integer, "auto", or None')
    return _func(
        "round_to_fraction", _value(x), denominator=float(denominator), digits=digits
    )


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


def _polars_flag(name: str, series: pl.Series) -> pl.Series:
    """One of is_na/is_nan/is_finite/is_infinite, computed natively."""
    dtype = series.dtype
    if dtype.is_float():
        if name == "is_na":
            return series.is_null() | series.is_nan().fill_null(False)
        return getattr(series, name)().fill_null(False)
    if name == "is_na":
        return series.is_null()
    if name == "is_finite" and dtype.is_numeric():
        return series.is_not_null()
    return pl.repeat(False, series.len(), dtype=pl.Boolean, eager=True)



def _native_flag(name: str, value: pl.Expr, dtype: Any) -> pl.Expr:
    """_polars_flag() as an expression, for a known input type."""
    if dtype.is_float():
        if name == "is_na":
            return value.is_null() | value.is_nan().fill_null(False)
        return getattr(value, name)().fill_null(False)
    if name == "is_na":
        return value.is_null()
    if name == "is_finite" and dtype.is_numeric():
        return value.is_not_null()
    # One False per row (not a literal, which an aggregate would see once).
    return value.is_null() & pl.lit(False)


def _flag_expr(name: str, value: pl.Expr) -> pl.Expr:
    from tidy3.expr import input_dtype

    dtype = input_dtype(value)
    if dtype is not None:
        return _native_flag(name, value, dtype)

    def batch(series: pl.Series) -> pl.Series:
        return _polars_flag(name, series)

    return value.map_batches(batch, return_dtype=pl.Boolean, is_elementwise=True)


# ── math ────────────────────────────────────────────────────────────────


def _nullify_nan(expr: pl.Expr) -> pl.Expr:
    """Treat NaN as missing so pmin/row sums match R, where NaN is NA.

    Only float columns hold NaN; text such as ``"nan"`` is left alone, so
    ``pmin(c("nan", "b"), c("c", "a"))`` is ``c("c", "a")`` as in R.
    """

    from tidy3.expr import input_dtype

    dtype = input_dtype(expr)
    if dtype is not None:
        return expr.fill_nan(None) if dtype.is_float() else expr

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
    from tidy3.expr import input_dtype

    known = input_dtype(value)
    if known is not None and known.is_integer() and digits >= 0:
        # Whole numbers stay as they are: no batch to run.
        return value
    if digits == 0 and known is not None and known.is_float():
        # Nothing is scaled, so Polars' ties-to-even is R's round(x):
        # checked against the candidate comparison on halves, huge
        # values, subnormals, infinities, and signed zero.
        return value.round(0, mode="half_to_even")

    def batch(series: pl.Series) -> pl.Series:
        return _round_batch(series, lambda v: _round_numpy(v, digits), keep_integers=digits >= 0)

    # Integers stay integers; floats stay floats.
    return value.map_batches(batch, return_dtype=pl.dtype_of(value), is_elementwise=True)


def _round_half_up_expr(value: pl.Expr, digits: int) -> pl.Expr:
    from tidy3.expr import input_dtype

    known = input_dtype(value)
    if known is not None and known.is_integer() and digits >= 0:
        return value

    def batch(series: pl.Series) -> pl.Series:
        return _round_batch(
            series, lambda v: _round_half_up_numpy(v, digits), keep_integers=digits >= 0
        )

    return value.map_batches(batch, return_dtype=pl.dtype_of(value), is_elementwise=True)


def _signif_half_up_expr(value: pl.Expr, digits: int) -> pl.Expr:
    def batch(series: pl.Series) -> pl.Series:
        return _round_batch(series, lambda v: _signif_half_up_numpy(v, digits))

    return value.map_batches(batch, return_dtype=pl.dtype_of(value), is_elementwise=True)


def _round_to_fraction_expr(value: pl.Expr, denominator: float, digits: int | None) -> pl.Expr:
    def batch(series: pl.Series) -> pl.Series:
        return _round_batch(
            series, lambda v: _round_to_fraction_numpy(v, denominator, digits)
        )

    return value.map_batches(batch, return_dtype=pl.dtype_of(value), is_elementwise=True)


def _round_batch(series: pl.Series, kernel: Any, *, keep_integers: bool = False) -> pl.Series:
    """Round one batch with a NumPy *kernel*, keeping nulls and the dtype.

    *keep_integers* skips integer columns, which the kernel would return
    unchanged.
    """
    if keep_integers and series.dtype.is_integer():
        return series
    # NumPy, not Polars arithmetic: Polars divides a column by a constant
    # through its reciprocal, so 3 / 10 would come out 0.30000000000000004.
    # (Polars expressions were tried: at these sizes each of the twenty
    # or so passes costs more in Polars than in NumPy.)
    values = series.cast(pl.Float64).to_numpy()
    rounded = pl.Series(series.name, kernel(values))
    if series.null_count():
        rounded = pl.select(
            pl.when(series.is_null()).then(None).otherwise(rounded)
        ).to_series()
    return rounded if rounded.dtype == series.dtype else rounded.cast(series.dtype)


_SQRT_EPS = math.sqrt(sys.float_info.epsilon)  # R's sqrt(.Machine$double.eps)


def _round_half_up_numpy(values: np.ndarray, digits: int) -> np.ndarray:
    """janitor's ``round_half_up`` on a float array, step for step.

    Each operation is one IEEE double operation in janitor's order, so the
    result is R's to the last bit: ``(|x| * 10^digits + 0.5) + sqrt(eps)``,
    truncated, divided by ``10^digits``, times the sign of *x*.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        scale = np.power(10.0, digits)
        scaled = np.abs(values) * scale + 0.5 + _SQRT_EPS
        return np.trunc(scaled) / scale * np.sign(values)


def _signif_half_up_numpy(values: np.ndarray, digits: int) -> np.ndarray:
    """janitor's ``signif_half_up`` on a float array, step for step.

    Nonzero finite values are multiplied by
    ``10^(digits - ceiling(log10(|x|)))``, rounded half up to a whole
    number, and divided by the same power; the rest are left as they are.
    """
    out = values.copy()
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        rows = np.flatnonzero((values != 0) & np.isfinite(values))
        x = values[rows]
        scale = np.power(10.0, digits - np.ceil(np.log10(np.abs(x))))
        out[rows] = _round_half_up_numpy(x * scale, 0) / scale
    return out


def _round_to_fraction_numpy(values: np.ndarray, denominator: float, digits: int | None) -> np.ndarray:
    """janitor's ``round_to_fraction``: R's ``round(x * d) / d``, then ``round(, digits)``."""
    with np.errstate(over="ignore", invalid="ignore"):
        # R's round(, 0) is rint: an exact tie goes to the even number.
        out = np.rint(values * denominator) / denominator
    return out if digits is None else _round_numpy(out, digits)


def _round_numpy(values: np.ndarray, digits: int) -> np.ndarray:
    """R's ``round`` on a float array; shared by both backends.

    Most values are settled by rounding the scaled value to the nearest
    whole number with ties to even, in three passes. A value whose scaled
    fraction is within a few units in the last place of one half, where
    R's comparison of the two candidates could go either way, and a value
    large enough for the keep rule, go through :func:`_round_candidates`.
    """
    x = np.asarray(values, dtype=float)
    if digits < 0 or digits > 308:
        return _round_candidates(x, digits)
    scale = 10.0**digits
    with np.errstate(all="ignore"):
        scaled = x * scale
        nearest = np.rint(scaled)
        # |scaled - nearest| is one half at a tie; within 16 eps relative
        # (at least 8 ulp of scaled) the candidates are nearly equidistant.
        unsettled = np.abs(scaled - nearest) >= 0.5 - (16 * np.finfo(float).eps) * np.abs(scaled)
        if digits > 0:
            unsettled |= np.abs(x) >= 10.0 ** (_ROUND_KEEP_DIGITS - digits - 1)
        rounded = nearest / scale
    if unsettled.any():
        rounded[unsettled] = _round_candidates(x[unsettled], digits)
    return rounded


def _round_candidates(values: np.ndarray, digits: int) -> np.ndarray:
    """R's ``round``, comparing both candidates for every value."""
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
            scaled = size * scale
            whole = np.floor(scaled)
            lower, upper = whole / scale, np.ceil(scaled) / scale
        else:
            scale = 10.0**-digits
            scaled = size / scale
            whole = np.floor(scaled)
            lower, upper = whole * scale, np.ceil(scaled) * scale
        down, up = size - lower, upper - size
        # whole is a non-negative whole number, so halving and doubling are
        # exact: a cheaper even test than fmod. Ties need finite values.
        even = np.floor(whole * 0.5) * 2 == whole
        nearer = np.where((down < up) | ((down == up) & even), lower, upper)
        keep = ~np.isfinite(size) | (size == 0)
        if digits > 0:
            # log10(size) + digits > 15 needs size near 10 ** (15 - digits);
            # only values within a factor of 10 of that pay for the log.
            near = size >= 10.0 ** (_ROUND_KEEP_DIGITS - digits - 1)
            if near.any():
                keep[near] |= np.log10(size[near]) + digits > _ROUND_KEEP_DIGITS
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
    if not min_count:
        # One horizontal kernel: ignore_nulls=False is R's na.rm = FALSE
        # (any missing value makes the row missing); a row with no values
        # has a missing mean and a zero sum.
        if operation == "mean":
            return pl.mean_horizontal(values, ignore_nulls=na_rm)
        return pl.sum_horizontal(values, ignore_nulls=na_rm).cast(pl.Float64)
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
    if name == "round_half_up":
        return _round_half_up_expr(args[0], int(kwargs["digits"]))
    if name == "signif_half_up":
        return _signif_half_up_expr(args[0], int(kwargs["digits"]))
    if name == "round_to_fraction":
        digits = kwargs.get("digits")
        return _round_to_fraction_expr(
            args[0], float(kwargs["denominator"]), None if digits is None else int(digits)
        )
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


