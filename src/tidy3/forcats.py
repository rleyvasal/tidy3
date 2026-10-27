"""Factor ordering and value changes for expressions and categorical vectors.

A factor is a string column with a list of levels and an ordered flag. In a
frame it is a Polars Enum, whose categories are the levels; the ordered flag
travels beside it in the frame's recorded levels. Factor mutations compute a
snapshot to discover the levels, and the result resumes as a lazy frame;
other mutations keep their usual lazy path.

The eager functions (``fct_rev(["a", "b"])``) return a pandas Categorical
for pandas and list inputs, Python's factor vector, and a Polars Enum Series
for a Polars Series; without pandas installed every input gives the Series.
"""

from __future__ import annotations

import math
from numbers import Real
import sys
from typing import Any
import warnings

import numpy as np
import polars as pl

from tidy3.expr import Expr, _func, col, median

__all__ = [
    "fct_relevel", "fct_reorder", "fct_infreq", "fct_rev",
    "fct_recode", "fct_collapse", "fct_lump_n", "fct_lump_prop", "fct_lump_min",
    "fct_drop", "fct_expand", "fct_inorder", "fct_reorder2", "first2", "last2",
    "fct_na_value_to_level", "fct_na_level_to_value", "as_factor",
]
FUNCTIONS = frozenset(__all__) - {"first2", "last2"}
_ORDERING_FUNCTIONS = frozenset({"fct_relevel", "fct_reorder", "fct_infreq", "fct_rev"})
# Calls whose values are their source's values; only the levels change.
_SAME_VALUES = _ORDERING_FUNCTIONS | {"fct_drop", "fct_expand", "fct_inorder", "fct_reorder2", "as_factor"}
_LUMPS = frozenset({"fct_lump_n", "fct_lump_prop", "fct_lump_min"})
# Calls that read only the levels, never the values: on a factor source every
# group gets the same result.
_LEVEL_ONLY = frozenset({
    "as_factor", "fct_rev", "fct_relevel", "fct_expand", "fct_recode", "fct_collapse",
    "fct_na_level_to_value", "fct_na_value_to_level",
})


def _column(value: Any) -> Any:
    return col(value) if isinstance(value, str) else value


def _dispatch(name: str, f: Any, *args: Any, **kwargs: Any) -> Any:
    if isinstance(f, (str, Expr, pl.Expr)):
        return _func(name, _column(f), *args, **kwargs)
    return _vector_result(_run(name, f, args, kwargs), f)


def _check_after(name: str, after: Any) -> None:
    if isinstance(after, bool) or not isinstance(after, (int, float)) or after < 0 or math.isnan(after) or (math.isfinite(after) and after != int(after)):
        raise ValueError(f"{name}() after must be a non-negative integer or infinity")


def _level_list(name: str, values: Any) -> list[str]:
    """Strings and lists of strings, flattened; None is no levels."""
    if values is None:
        return []
    if isinstance(values, str):
        return [values]
    if isinstance(values, (list, tuple)):
        return [level for value in values for level in _level_list(name, value)]
    raise TypeError(f"{name}() levels must be strings or lists of strings")


def fct_relevel(f: Any, *levels: Any, after: int | float = 0) -> Any:
    """Move named levels (or a callable's result) after *after* other levels.

    Unmentioned levels keep their order. ``after=float('inf')`` moves to
    the end. Unknown levels warn and are ignored, as in forcats.
    """
    _check_after("fct_relevel", after)
    if len(levels) == 1 and isinstance(levels[0], (list, tuple)):
        levels = tuple(levels[0])
    if not (len(levels) == 1 and callable(levels[0])) and not all(isinstance(level, str) for level in levels):
        raise TypeError("fct_relevel() levels must be strings or one callable")
    return _dispatch("fct_relevel", f, levels=list(levels), after=after)


def fct_reorder(
    f: Any, x: Any, fun: Any = median, *, na_rm: bool | None = None,
    default: float = math.inf, desc: bool = False, **kwargs: Any,
) -> Any:
    """Order levels by a summary of *x* (median by default).

    ``na_rm=None`` drops missing *x* with a warning; True drops silently.
    Empty levels use *default*. Ties keep the existing level order.
    *fun* accepts a tidy3 aggregation or a callable returning one number.
    """
    if not callable(fun):
        raise TypeError("fct_reorder() fun must be callable")
    if na_rm is not None and not isinstance(na_rm, bool):
        raise TypeError("fct_reorder() na_rm must be a boolean or None")
    if not isinstance(desc, bool):
        raise TypeError("fct_reorder() desc must be a boolean")
    return _dispatch("fct_reorder", f, _column(x), fun=fun, na_rm=na_rm, default=default, desc=desc, fun_kwargs=kwargs)


def fct_infreq(f: Any, w: Any = None, *, ordered: bool | None = None) -> Any:
    """Order levels by decreasing frequency, optionally weighted by *w*.

    Ties keep the existing level order; unused levels come last. ``ordered``
    preserves the input's status when None (Polars uses Enum order).
    """
    if ordered is not None and not isinstance(ordered, bool):
        raise TypeError("fct_infreq() ordered must be a boolean or None")
    args = () if w is None else (_column(w),)
    return _dispatch("fct_infreq", f, *args, ordered=ordered)


def fct_rev(f: Any) -> Any:
    """Reverse all levels, including unused ones, without changing values."""
    return _dispatch("fct_rev", f)


def fct_recode(f: Any, **levels: str) -> Any:
    """Rename levels with ``new="old"``; ``NULL="old"`` removes a level.

    Unmentioned levels stay unchanged. Unknown old levels warn. Use
    :func:`fct_collapse` to combine several old levels under one name.
    """
    if any(not isinstance(old, str) for old in levels.values()):
        raise TypeError("fct_recode() levels must be named strings")
    return _dispatch("fct_recode", f, levels=levels)


def fct_collapse(f: Any, *, other_level: str | None = None, **levels: Any) -> Any:
    """Combine levels with ``new=["old", ...]`` (or one old string).

    Unmentioned levels stay unchanged unless *other_level* is supplied;
    that level collects the remainder and is placed last.
    """
    if other_level is not None and not isinstance(other_level, str):
        raise TypeError("fct_collapse() other_level must be a string or None")
    normalized = {}
    for new, old in levels.items():
        old = [old] if isinstance(old, str) else old
        if not isinstance(old, (list, tuple)) or any(not isinstance(value, str) for value in old):
            raise TypeError("fct_collapse() levels must be strings or lists of strings")
        normalized[new] = list(old)
    return _dispatch("fct_collapse", f, levels=normalized, other_level=other_level)


def _lump_call(name: str, f: Any, value: Any, w: Any, other_level: str, **kwargs: Any) -> Any:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name}() threshold must be a finite number")
    if name == "fct_lump_min" and value < 0:
        raise ValueError("fct_lump_min() min must be non-negative")
    if not isinstance(other_level, str):
        raise TypeError(f"{name}() other_level must be a string")
    args = () if w is None else (_column(w),)
    return _dispatch(name, f, *args, threshold=value, other_level=other_level, **kwargs)


def fct_lump_n(
    f: Any, n: float, w: Any = None, *, other_level: str = "Other", ties_method: str = "min",
) -> Any:
    """Keep the *n* most frequent levels, or the least frequent if *n* < 0.

    Ties default to keeping all levels at the cutoff. ``ties_method`` accepts
    min, average, first, last, random, or max. Weights apply to rows.
    """
    if ties_method not in {"min", "average", "first", "last", "random", "max"}:
        raise ValueError("fct_lump_n() unknown ties_method")
    return _lump_call("fct_lump_n", f, n, w, other_level, ties_method=ties_method)


def fct_lump_prop(f: Any, prop: float, w: Any = None, *, other_level: str = "Other") -> Any:
    """Keep frequencies strictly above *prop*; if negative, keep <= abs(prop).

    The denominator includes missing rows (and their weights), as in forcats.
    A zero denominator with existing levels raises ValueError.
    """
    return _lump_call("fct_lump_prop", f, prop, w, other_level)


def fct_lump_min(f: Any, min: float, w: Any = None, *, other_level: str = "Other") -> Any:
    """Keep levels with frequency at least *min*, optionally weighted."""
    return _lump_call("fct_lump_min", f, min, w, other_level)


def fct_drop(f: Any, only: Any = None) -> Any:
    """Remove unused levels; with *only*, remove just those unused levels.

    Used levels in *only* stay. The ordered flag and remaining order are kept.
    """
    return _dispatch("fct_drop", f, only=None if only is None else _level_list("fct_drop", only))


def fct_expand(f: Any, *levels: Any, after: int | float = math.inf) -> Any:
    """Add levels after *after* existing ones (at the end by default).

    Levels may be strings or lists of strings. Levels already present, and
    repeats, are ignored. Values do not change.
    """
    _check_after("fct_expand", after)
    return _dispatch("fct_expand", f, levels=_level_list("fct_expand", list(levels)), after=after)


def fct_inorder(f: Any, *, ordered: bool | None = None) -> Any:
    """Order levels by first appearance; unused levels follow in their order.

    ``ordered=None`` keeps the input's ordered flag.
    """
    if ordered is not None and not isinstance(ordered, bool):
        raise TypeError("fct_inorder() ordered must be a boolean or None")
    return _dispatch("fct_inorder", f, ordered=ordered)


def _terminal(x: Any, y: Any, desc: bool) -> Any:
    xs, ys = _vector_series(x), _vector_series(y)
    keep = xs.is_not_null() & ys.is_not_null()
    if not bool(keep.any()):
        return None
    xs, ys = xs.filter(keep), ys.filter(keep)
    # arg_max and arg_min return the first row among ties, as R's order() does.
    return ys[int(xs.arg_max() if desc else xs.arg_min())]


def last2(x: Any, y: Any) -> Any:
    """The *y* value at the largest *x*: :func:`fct_reorder2`'s default summary.

    Rows missing either value are skipped; ties take the first row.
    """
    return _terminal(x, y, True)


def first2(x: Any, y: Any) -> Any:
    """The *y* value at the smallest *x*; ties take the first row."""
    return _terminal(x, y, False)


def fct_reorder2(
    f: Any, x: Any, y: Any, fun: Any = last2, *, na_rm: bool | None = None,
    default: float = -math.inf, desc: bool = True, **kwargs: Any,
) -> Any:
    """Order levels by a summary of *x* and *y*, largest first by default.

    ``fun(x, y)`` returns one number per level; the default :func:`last2`
    is the *y* value at the largest *x*, so lines in a plot and their legend
    end in the same order. Rows missing *x* or *y* are left out of the
    summaries with a warning (``na_rm=None``), silently (True), or passed to
    *fun* (False). Empty levels use *default*; ties keep the existing level
    order. Every row stays in the result; forcats also drops the missing rows
    from its result when it removes them.
    """
    if not callable(fun):
        raise TypeError("fct_reorder2() fun must be callable")
    if na_rm is not None and not isinstance(na_rm, bool):
        raise TypeError("fct_reorder2() na_rm must be a boolean or None")
    if not isinstance(desc, bool):
        raise TypeError("fct_reorder2() desc must be a boolean")
    return _dispatch(
        "fct_reorder2", f, _column(x), _column(y),
        fun=fun, na_rm=na_rm, default=default, desc=desc, fun_kwargs=kwargs,
    )


def fct_na_value_to_level(f: Any, level: str = "(Missing)") -> Any:
    """Turn missing values into the level *level*, added last if it is new.

    The level is added even when nothing is missing. forcats defaults to an
    ``NA`` level, which a tidy3 factor cannot hold, so the default is
    forcats' earlier ``"(Missing)"``.
    """
    if not isinstance(level, str):
        raise TypeError("fct_na_value_to_level() level must be a string")
    return _dispatch("fct_na_value_to_level", f, level=level)


def fct_na_level_to_value(f: Any, extra_levels: Any = None) -> Any:
    """Turn the values of *extra_levels* into missing values and drop those levels.

    Unknown levels are ignored. Without *extra_levels* nothing changes: a
    tidy3 factor never has an ``NA`` level for this to convert.
    """
    return _dispatch(
        "fct_na_level_to_value", f, extra_levels=_level_list("fct_na_level_to_value", extra_levels),
    )


def as_factor(x: Any) -> Any:
    """Make a factor, as forcats::as_factor does.

    Text gets levels in order of first appearance. Numbers get levels in
    numeric order and logical values get ``FALSE`` and ``TRUE``, written
    with tidy3's header text rules: ``2.0`` is ``"2"``, but numbers are
    never rounded or written in scientific notation as R does. Factors are
    unchanged. Dates and other types raise; convert them with
    ``as_character`` first.
    """
    return _dispatch("as_factor", x)


# ── the factor model ────────────────────────────────────────────────────


class Factor:
    """A Series of level names with its levels and ordered flag.

    Every value is one of the levels or null. The values keep their input
    dtype, String or an Enum, until a function changes them, so a function
    that only reorders levels never rewrites 100,000 values; ``to_enum``
    recasts an Enum straight to the new levels.
    """

    __slots__ = ("values", "levels", "ordered")

    def __init__(self, values: pl.Series, levels: list[str], ordered: bool) -> None:
        self.values = values
        self.levels = list(levels)
        self.ordered = bool(ordered)

    def __len__(self) -> int:
        return len(self.values)

    def with_levels(self, levels: list[str], ordered: bool | None = None) -> Factor:
        return Factor(self.values, levels, self.ordered if ordered is None else ordered)

    def used(self) -> set[str]:
        return set(self.values.drop_nulls().unique().to_list())

    def to_enum(self) -> pl.Series:
        dtype = self.values.dtype
        if isinstance(dtype, pl.Enum) and dtype.categories.to_list() == self.levels:
            return self.values
        return self.values.cast(pl.Enum(self.levels))

    def strings(self) -> pl.Series:
        """The values as a String Series, for a function that rewrites them."""
        return self.values if self.values.dtype == pl.String else self.values.cast(pl.String)

    def to_pandas(self) -> Any:
        import pandas as pd

        return pd.Categorical(self.values.to_list(), categories=self.levels, ordered=self.ordered)


def _pandas() -> Any:
    """The pandas module when something has imported it, else None."""
    return sys.modules.get("pandas")


def _scalar_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (float, np.floating)) and value != value:
        return True
    pd = _pandas()
    return pd is not None and (value is pd.NA or value is pd.NaT)


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (str, bytes)) or not hasattr(value, "__len__")


def _vector_series(value: Any) -> pl.Series:
    """*value* as a Polars Series; NaN is missing, as pandas and R mean it."""
    if isinstance(value, Factor):
        return value.values
    if isinstance(value, pl.Series):
        series = value
    else:
        pd = _pandas()
        if pd is not None and isinstance(value, pd.Series) and not isinstance(value.dtype, pd.CategoricalDtype):
            series = pl.from_pandas(value.reset_index(drop=True))
        else:
            items = [None if _scalar_missing(item) else item for item in ([value] if _is_scalar(value) else list(value))]
            present = [item for item in items if item is not None]
            numbers = all(isinstance(item, Real) and not isinstance(item, (bool, np.bool_)) for item in present)
            if present and numbers and not all(isinstance(item, (int, np.integer)) for item in present):
                items = [None if item is None else float(item) for item in items]
            try:
                series = pl.Series(items)
            except TypeError:
                raise TypeError("values mix types") from None
    return series.fill_nan(None) if series.dtype.is_float() else series


def _categorical_source(value: Any) -> tuple[pl.Series, list[Any], bool] | None:
    """(values, levels, ordered) for a factor-like input, else None."""
    if isinstance(value, Factor):
        return value.values, value.levels, value.ordered
    if isinstance(value, pl.Series):
        if isinstance(value.dtype, pl.Enum):
            return value, value.dtype.categories.to_list(), True
        if isinstance(value.dtype, pl.Categorical):
            return value, value.cat.get_categories().to_list(), False
        return None
    pd = _pandas()
    if pd is None:
        return None
    if isinstance(value, pd.Series) and isinstance(value.dtype, pd.CategoricalDtype):
        value = value.array
    if isinstance(value, pd.Categorical):
        values = pl.Series([None if _scalar_missing(item) else item for item in value])
        return values, value.categories.tolist(), bool(value.ordered)
    return None


def _is_factor(value: Any) -> bool:
    return _categorical_source(value) is not None


def _factor(value: Any, levels: list[str] | None = None, *, ordered: bool | None = None) -> Factor:
    found = _categorical_source(value)
    if found is not None:
        values, found_levels, found_ordered = found
        if not all(isinstance(level, str) for level in found_levels):
            raise TypeError("factor levels must be strings")
        flag = found_ordered if ordered is None else ordered
        return Factor(values, found_levels if levels is None else levels, flag)
    try:
        series = _vector_series(value)
    except TypeError:
        raise TypeError("factor input must be a character or categorical column") from None
    if series.dtype == pl.Null:
        series = series.cast(pl.String)
    if series.dtype != pl.String:
        raise TypeError("factor input must be a character or categorical column")
    categories = sorted(series.drop_nulls().unique().to_list()) if levels is None else levels
    return Factor(series, categories, False if ordered is None else ordered)


def _as_factor(value: Any) -> Factor:
    """forcats::as_factor levels: appearance for text, sorted for numbers and logicals."""
    if _is_factor(value):
        return _factor(value)
    try:
        series = _vector_series(value)
    except TypeError:
        raise TypeError("as_factor() input must be text, numbers, logical values, or a factor; "
                        "convert other types with as_character() first") from None
    if series.dtype == pl.Null:
        series = series.cast(pl.String)
    if series.dtype == pl.String:
        return Factor(series, series.drop_nulls().unique(maintain_order=True).to_list(), False)
    from tidy3.janitor import _header_text

    if series.dtype == pl.Boolean:
        labels = ["FALSE", "TRUE"]
        values = series.replace_strict({False: "FALSE", True: "TRUE"}, default=None, return_dtype=pl.String)
    elif series.dtype.is_numeric():
        distinct = series.drop_nulls().unique().sort().to_list()
        labels = [_header_text(item) for item in distinct]
        if len(set(labels)) != len(labels):
            raise ValueError("as_factor() numbers must have distinct text")
        values = (
            series.replace_strict(dict(zip(distinct, labels)), default=None, return_dtype=pl.String)
            if distinct else series.cast(pl.String)
        )
    else:
        raise TypeError("as_factor() input must be text, numbers, logical values, or a factor; "
                        "convert other types with as_character() first")
    return Factor(values, labels, False)


def _revalue(source: Factor, mapping: dict[str, str | None]) -> Factor:
    """Relabel levels, merging duplicates; a level mapped to None is removed."""
    renamed = [mapping.get(level, level) for level in source.levels]
    categories = list(dict.fromkeys(level for level in renamed if level is not None))
    values = source.values
    if any(old != new for old, new in zip(source.levels, renamed)):
        values = source.strings().replace_strict(source.levels, renamed, default=None, return_dtype=pl.String)
    return Factor(values, categories, source.ordered)


def _keep_levels(source: Factor, keep: list[bool], other_level: str) -> Factor:
    if all(keep):
        return source
    mapping = {level: other_level for level, kept in zip(source.levels, keep) if not kept}
    result = _revalue(source, mapping)
    return result.with_levels([level for level in result.levels if level != other_level] + [other_level])


def _level_parts(
    values: pl.Series, columns: dict[str, pl.Series], keep: pl.Series | None
) -> dict[str, pl.DataFrame]:
    """The rows of each level present, as one frame per level."""
    frame = pl.DataFrame({"__f": values, **columns})
    if keep is not None:
        frame = frame.filter(keep)
    frame = frame.drop_nulls("__f")
    return {key[0]: part for key, part in frame.partition_by("__f", maintain_order=True, as_dict=True).items()}


def _level_totals(source: Factor, weights: pl.Series) -> list[float]:
    """The weight of each level's rows; missing rows belong to no level."""
    frame = pl.DataFrame({"__f": source.values, "__w": weights}).drop_nulls("__f")
    totals = dict(frame.group_by("__f").agg(pl.col("__w").sum()).iter_rows())
    return [float(totals.get(level, 0.0)) for level in source.levels]


def _weights(name: str, length: int, args: tuple[Any, ...], *, finite: bool) -> pl.Series:
    if not args:
        return pl.repeat(1.0, length, dtype=pl.Float64, eager=True)
    weights = args[0]
    if weights.dtype == pl.Boolean:
        weights = weights.cast(pl.Int64)
    message = (
        f"{name}() weights must be finite, non-negative and non-missing" if finite
        else f"{name}() weights must be numeric, non-negative and non-missing"
    )
    if not weights.dtype.is_numeric():
        raise ValueError(message)
    weights = weights.cast(pl.Float64)
    if weights.null_count() or bool(weights.is_nan().any()) or bool((weights < 0).any()):
        raise ValueError(message)
    if finite and not bool(weights.is_finite().all()):
        raise ValueError(message)
    return weights


def _tidy3_summary(fun: Any) -> bool:
    return getattr(fun, "__module__", "") == "tidy3.expr"


def _summary(values: pl.Series, fun: Any, kwargs: dict[str, Any]) -> Any:
    if _tidy3_summary(fun):
        from tidy3.expr import to_polars

        result = pl.DataFrame({"x": values}).select(to_polars(fun(col("x"), **kwargs)).alias("x"))
        if result.height != 1 or isinstance(result.schema["x"], pl.List):
            raise ValueError("fct_reorder() fun must return one value per level")
        return result.item()
    result = fun(values, **kwargs)
    if not _is_scalar(result):
        raise ValueError("fct_reorder() fun must return one value per level")
    return result


def _score_order(scores: list[Any], desc: bool) -> list[int]:
    """Positions sorted by score, missing last, ties in their existing order."""
    present = [i for i, score in enumerate(scores) if not _scalar_missing(score)]
    absent = [i for i, score in enumerate(scores) if _scalar_missing(score)]
    present.sort(key=lambda i: float(scores[i]), reverse=desc)
    return present + absent


def _reorder_scores(
    source: Factor, x: pl.Series, keep: pl.Series | None, fun: Any, kwargs: dict[str, Any], default: Any
) -> list[Any]:
    """fct_reorder's summary of x for each level; empty levels take *default*."""
    if _tidy3_summary(fun):
        from tidy3.expr import to_polars

        frame = pl.DataFrame({"__f": source.values, "x": x})
        if keep is not None:
            frame = frame.filter(keep)
        summary = frame.drop_nulls("__f").group_by("__f").agg(
            to_polars(fun(col("x"), **kwargs)).alias("__s")
        )
        if isinstance(summary.schema["__s"], pl.List):
            raise ValueError("fct_reorder() fun must return one value per level")
        found = dict(summary.iter_rows())
        return [found.get(level, default) for level in source.levels]
    parts = _level_parts(source.values, {"x": x}, keep)
    return [
        _summary(parts[level]["x"], fun, kwargs) if level in parts else default
        for level in source.levels
    ]


def _reorder2_levels(source: Factor, x: pl.Series, y: pl.Series, kw: dict[str, Any]) -> list[str]:
    missing = x.is_null() | y.is_null()
    na_rm = kw["na_rm"]
    count = int(missing.sum())
    if count and na_rm is None:
        warnings.warn(
            f"fct_reorder2(): removing {count} missing value{'s' if count != 1 else ''}; set na_rm=True to silence",
            UserWarning, stacklevel=4,
        )
        na_rm = True
    parts = _level_parts(source.values, {"x": x, "y": y}, ~missing if na_rm else None)
    scores = []
    for level in source.levels:
        part = parts.get(level)
        if part is None:
            scores.append(kw["default"])
            continue
        result = kw["fun"](part["x"], part["y"], **kw["fun_kwargs"])
        if not _is_scalar(result):
            raise ValueError("fct_reorder2() fun must return one value per level")
        scores.append(result)
    return [source.levels[i] for i in _score_order(scores, kw["desc"])]


def _change_levels(name: str, source: Factor, args: tuple[Any, ...], kw: dict[str, Any]) -> Factor:
    levels = source.levels
    if name == "as_factor":
        return source
    if name == "fct_drop":
        used = source.used()
        only = kw["only"]
        kept = [level for level in levels if level in used or (only is not None and level not in only)]
        return source.with_levels(kept)
    if name == "fct_expand":
        new = [level for level in dict.fromkeys(kw["levels"]) if level not in levels]
        position = len(levels) if math.isinf(kw["after"]) else min(int(kw["after"]), len(levels))
        return source.with_levels(levels[:position] + new + levels[position:])
    if name == "fct_inorder":
        first = source.values.drop_nulls().unique(maintain_order=True).to_list()
        seen = set(first)
        order = first + [level for level in levels if level not in seen]
        ordered = source.ordered if kw["ordered"] is None else kw["ordered"]
        return source.with_levels(order, ordered)
    if name == "fct_reorder2":
        return source.with_levels(_reorder2_levels(source, args[0], args[1], kw))
    if name == "fct_na_value_to_level":
        level = kw["level"]
        new_levels = levels if level in levels else levels + [level]
        return Factor(source.strings().fill_null(level), new_levels, source.ordered)
    if name == "fct_na_level_to_value":
        removed = set(kw["extra_levels"])
        return _revalue(source, {level: None for level in levels if level in removed})
    if name in {"fct_recode", "fct_collapse"}:
        mapping_levels = kw["levels"]
        removed = [mapping_levels["NULL"]] if name == "fct_recode" and "NULL" in mapping_levels else []
        mapping: dict[str, str | None] = {}
        for new, old in mapping_levels.items():
            if name == "fct_recode" and new == "NULL":
                continue
            for level in ([old] if isinstance(old, str) else old):
                mapping[level] = new
        unknown = [level for level in mapping if level not in levels or level in removed]
        if unknown:
            warnings.warn(f"{name}(): unknown levels: {unknown}", UserWarning, stacklevel=4)
        mapping.update({level: None for level in removed})
        result = _revalue(source, mapping)
        if name == "fct_collapse" and kw["other_level"] is not None:
            keep = [level in mapping_levels for level in result.levels]
            return _keep_levels(result, keep, kw["other_level"])
        return result

    weights = _weights(name, len(source), args, finite=True)
    counts = _level_totals(source, weights)
    threshold = kw["threshold"]
    if name == "fct_lump_n":
        scores = pl.Series([count if threshold < 0 else -count for count in counts], dtype=pl.Float64)
        method = kw["ties_method"]
        if method == "last":
            ranks = scores.reverse().rank(method="ordinal").reverse()
        elif method == "first":
            ranks = scores.rank(method="ordinal")
        else:
            ranks = scores.rank(method=method)
        keep = [rank <= abs(threshold) for rank in ranks.to_list()]
    elif name == "fct_lump_prop":
        denominator = float(weights.sum())
        if denominator == 0 and counts:
            raise ValueError("fct_lump_prop() requires a positive total weight when levels exist")
        proportions = [count / denominator for count in counts]
        keep = [p <= -threshold for p in proportions] if threshold < 0 else [p > threshold for p in proportions]
    else:
        keep = [count >= threshold for count in counts]
    return _keep_levels(source, keep, kw["other_level"])


def _apply(name: str, source: Factor, args: tuple[Any, ...], kw: dict[str, Any]) -> Factor:
    """Run one factor function on *source*; *args* are Series of its length."""
    if name not in _ORDERING_FUNCTIONS:
        return _change_levels(name, source, args, kw)
    levels = source.levels
    ordered = source.ordered
    if name == "fct_rev":
        result = levels[::-1]
    elif name == "fct_relevel":
        requested = kw["levels"]
        if len(requested) == 1 and callable(requested[0]):
            requested = list(requested[0](levels.copy()))
        if not all(isinstance(level, str) for level in requested):
            raise TypeError("fct_relevel() callable must return character levels")
        if len(set(requested)) != len(requested):
            raise ValueError("fct_relevel() levels must be unique")
        unknown = [level for level in requested if level not in levels]
        if unknown:
            warnings.warn(f"fct_relevel(): unknown levels: {unknown}", UserWarning, stacklevel=3)
        requested = [level for level in requested if level in levels]
        other = [level for level in levels if level not in requested]
        position = len(other) if math.isinf(kw["after"]) else min(int(kw["after"]), len(other))
        result = other[:position] + requested + other[position:]
    elif name == "fct_infreq":
        scores = _level_totals(source, _weights("fct_infreq", len(source), args, finite=False))
        result = [levels[i] for i in sorted(range(len(levels)), key=lambda i: -scores[i])]
        ordered = ordered if kw["ordered"] is None else kw["ordered"]
    else:
        x = args[0]
        if not (x.dtype.is_numeric() or x.dtype == pl.Boolean) and not x.drop_nulls().is_empty():
            raise TypeError("fct_reorder() x must be numeric")
        if kw["na_rm"] is None and x.null_count():
            warnings.warn("fct_reorder(): removing missing x values; set na_rm=True to silence", UserWarning, stacklevel=3)
        keep = x.is_not_null() if kw["na_rm"] is not False else None
        scores = _reorder_scores(source, x, keep, kw["fun"], kw["fun_kwargs"], kw["default"])
        result = [levels[i] for i in _score_order(scores, kw["desc"])]
    return source.with_levels(result, ordered)


def _run(name: str, f: Any, args: tuple[Any, ...], kw: dict[str, Any]) -> Factor:
    """A factor function on evaluated inputs: the source, then extra vectors."""
    source = _as_factor(f) if name == "as_factor" else _factor(f)
    extras = []
    for value in args:
        series = _vector_series(value)
        if len(series) != len(source):
            raise ValueError(f"{name}() inputs must have the same length")
        extras.append(series)
    return _apply(name, source, tuple(extras), kw)


def _vector_result(result: Factor, like: Any) -> Any:
    """An Enum Series for Polars input; a pandas Categorical otherwise, when pandas exists."""
    if isinstance(like, pl.Series):
        return result.to_enum()
    try:
        import pandas  # noqa: F401
    except ImportError:
        return result.to_enum()
    return result.to_pandas()


# ── evaluation inside a frame ───────────────────────────────────────────


class _Context:
    """Where a factor expression runs: the groups, their order, and the
    ordered flags of the frame's Enum columns."""

    __slots__ = ("groups", "sort_groups", "flags")

    def __init__(self, groups: list[str] | None, sort_groups: bool, flags: dict[str, bool]) -> None:
        self.groups = list(groups) if groups else None
        self.sort_groups = sort_groups
        self.flags = flags

    def ungrouped(self) -> _Context:
        return _Context(None, self.sort_groups, self.flags)


def _column_value(frame: pl.DataFrame, name: str, ctx: _Context) -> Any:
    series = frame[name]
    if isinstance(series.dtype, pl.Enum):
        # An Enum without a recorded flag is ordered, as it is when collected to pandas.
        return Factor(series, series.dtype.categories.to_list(), ctx.flags.get(name, True))
    return series


def _arg(node: Any, frame: pl.DataFrame, ctx: _Context) -> Any:
    """One evaluated argument: a literal, a column, a nested factor, or a computed Series."""
    kind = node[0]
    if kind == "lit":
        return node[1]
    if kind == "col" or (kind == "name" and node[1] in frame.columns):
        return _column_value(frame, node[1], ctx)
    if kind == "func" and node[1] in FUNCTIONS:
        return _factor_node(node, frame, ctx)
    return _compile_value(node, frame, ctx)


def _factor_node(node: tuple, frame: pl.DataFrame, ctx: _Context) -> Factor:
    """A factor call on an ungrouped frame (or one group's rows)."""
    name, raw_args, raw_kwargs = node[1], node[2], node[3]
    args = [_arg(arg, frame, ctx) for arg in raw_args]
    kw = {key: _arg(value, frame, ctx) for key, value in raw_kwargs.items()}
    return _run(name, args[0], tuple(args[1:]), kw)


def _grouped_factor(node: tuple, frame: pl.DataFrame, ctx: _Context) -> Factor:
    """A factor call per group, the groups' levels joined in group order.

    Every group is computed at once: each group's level list lives in a
    table with one row per group and level, and each factor function is a
    group-wise transformation of that table (see :class:`_Grouped`). The
    result matches evaluating the call on each group's rows by itself and
    joining the groups' levels in group order, each adding levels not seen.
    """
    if frame.height == 0:
        return _factor_node(node, frame, ctx.ungrouped())
    result = _grouped_node(node, frame, frame.select(ctx.groups), ctx)
    return Factor(result.values, _g_union(result, ctx), result.ordered)


def _g_union(g: _Grouped, ctx: _Context) -> list[str]:
    """The groups' level lists joined in group order, each adding new levels."""
    if g.uniform is not None:
        return list(g.uniform)
    if ctx.sort_groups:
        ordered = g.levels.sort([*g.by, "__pos"], nulls_last=True)
    else:
        # Appearance order: the rows' first group first.
        order = g.keys.unique(maintain_order=True).with_row_index("__tidy3_gid")
        ordered = g.levels.join(order, on=g.by, how="left", nulls_equal=True).sort(["__tidy3_gid", "__pos"])
    return ordered["__level"].unique(maintain_order=True).to_list()


# ── group-wise factors ──────────────────────────────────────────────────


class _Grouped:
    """A factor whose levels are per group.

    ``keys`` holds the group columns of every row, ``values`` the rows' level
    names, and ``levels`` one row per group and level: the group columns,
    ``__level``, and ``__pos``, the level's position in its group's list.
    Nothing numbers the groups until the end; within a group, order is a
    sort of its own rows.
    """

    __slots__ = ("keys", "by", "values", "_levels", "ordered", "uniform")

    def __init__(
        self, keys: pl.DataFrame, values: pl.Series, levels: pl.DataFrame | None, ordered: bool,
        *, uniform: list[str] | None = None,
    ) -> None:
        self.keys = keys
        self.by = keys.columns
        self.values = values
        # Building the levels from text counts each level's rows; that
        # ``__count`` column travels with the levels while the values are
        # unchanged, so an unweighted fct_infreq or lump reads it instead of
        # passing over the rows again.
        self._levels = levels
        self.ordered = bool(ordered)
        # A factor source gives every group the same level list. A function
        # that reads only the levels then gives every group the same result,
        # so it runs once on that list; the table is built only when a
        # function reads the data.
        self.uniform = list(uniform) if uniform is not None else None

    @property
    def levels(self) -> pl.DataFrame:
        if self._levels is None:
            self._levels = _g_all_levels(self.keys.unique(), self.uniform or [])
        return self._levels

    @property
    def on(self) -> list[str]:
        """The join keys of the level table: the group columns and the level."""
        return [*self.by, "__level"]

    def rows(self, **columns: pl.Series) -> pl.DataFrame:
        """The rows as a frame: the group columns, ``__level`` (the value), and *columns*."""
        return self.keys.with_columns(__level=self.strings(), **columns)

    def strings(self) -> pl.Series:
        return self.values if self.values.dtype == pl.String else self.values.cast(pl.String)

    def with_levels(self, levels: pl.DataFrame, ordered: bool | None = None, *, counts: bool = False) -> _Grouped:
        """*levels* in their row order; ``counts`` keeps ``__count`` when the values did not change."""
        return _Grouped(self.keys, self.values, _g_order(levels, self.by, counts=counts), self.ordered if ordered is None else ordered)

    def groups(self) -> pl.DataFrame:
        """Every group, including those whose values are all missing."""
        return self.keys.unique()

    def level_groups(self) -> pl.DataFrame:
        """The groups that have at least one level."""
        return self.levels.select(self.by).unique()

    def join_levels(self, table: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
        """The level table with *columns* of *table* joined on group and level."""
        return self.levels.join(table.select([*self.on, *columns]), on=self.on, how="left", nulls_equal=True)


def _g_order(levels: pl.DataFrame, by: list[str], *, counts: bool = False) -> pl.DataFrame:
    """Fix the levels' current row order as ``__pos``.

    Positions are row numbers over the whole table: within a group they
    order its levels, which is all later sorts need.
    """
    kept = ["__count"] if counts and "__count" in levels.columns else []
    return levels.select(*by, "__level", *kept).with_row_index("__pos").with_columns(pl.col("__pos").cast(pl.Int64))


def _g_all_levels(groups: pl.DataFrame, levels: list[str]) -> pl.DataFrame:
    """Every level for every group, in level order."""
    table = pl.DataFrame({
        "__level": pl.Series(levels, dtype=pl.String),
        "__pos": pl.Series(range(len(levels)), dtype=pl.Int64),
    })
    return groups.join(table, how="cross")


def _g_lengths(name: str, keys: pl.DataFrame, series: pl.Series) -> pl.Series:
    if len(series) != keys.height:
        raise ValueError(f"{name}() inputs must have the same length")
    return series


def _g_source(name: str, value: Any, keys: pl.DataFrame) -> _Grouped:
    """*value* as a group-wise factor: a factor keeps all its levels in
    every group; text gives each group the levels present in it, sorted."""
    if isinstance(value, _Grouped):
        return value
    if _is_factor(value):
        factor = _factor(value)
        _g_lengths(name, keys, factor.values)
        return _Grouped(keys, factor.values, None, factor.ordered, uniform=factor.levels)
    try:
        series = _vector_series(value)
    except TypeError:
        raise TypeError("factor input must be a character or categorical column") from None
    if series.dtype == pl.Null:
        series = series.cast(pl.String)
    if series.dtype != pl.String:
        raise TypeError("factor input must be a character or categorical column")
    _g_lengths(name, keys, series)
    by = keys.columns
    counts = (
        keys.with_columns(__level=series).drop_nulls("__level")
        .group_by([*by, "__level"]).agg(pl.len().cast(pl.Float64).alias("__count"))
        .sort("__level")
    )
    return _Grouped(keys, series, _g_order(counts, by, counts=True), False)


def _g_as_factor(value: Any, keys: pl.DataFrame) -> _Grouped:
    """forcats::as_factor per group: appearance order for text, sorted for numbers."""
    if isinstance(value, _Grouped) or _is_factor(value):
        return _g_source("as_factor", value, keys)
    try:
        series = _vector_series(value)
    except TypeError:
        raise TypeError("as_factor() input must be text, numbers, logical values, or a factor; "
                        "convert other types with as_character() first") from None
    _g_lengths("as_factor", keys, series)
    by = keys.columns
    if series.dtype == pl.Null:
        series = series.cast(pl.String)
    if series.dtype == pl.String:
        first = (
            keys.with_columns(__level=series).with_row_index("__row")
            .drop_nulls("__level").group_by([*by, "__level"]).agg(pl.col("__row").min())
            .sort("__row")
        )
        return _Grouped(keys, series, _g_order(first, by), False)
    from tidy3.janitor import _header_text

    if series.dtype == pl.Boolean:
        values = series.replace_strict({False: "FALSE", True: "TRUE"}, default=None, return_dtype=pl.String)
        return _Grouped(keys, values, None, False, uniform=["FALSE", "TRUE"])
    if series.dtype.is_numeric():
        distinct = series.drop_nulls().unique().sort().to_list()
        labels = [_header_text(item) for item in distinct]
        if len(set(labels)) != len(labels):
            raise ValueError("as_factor() numbers must have distinct text")
        if not distinct:
            empty = keys.clear().with_columns(__level=pl.lit(None, dtype=pl.String))
            return _Grouped(keys, series.cast(pl.String), _g_order(empty, by), False)
        pairs = keys.with_columns(__number=series).drop_nulls("__number").unique().sort("__number")
        mapping = dict(zip(distinct, labels))
        levels = pairs.select(*by, __level=pl.col("__number").replace_strict(mapping, default=None, return_dtype=pl.String))
        values = series.replace_strict(mapping, default=None, return_dtype=pl.String)
        return _Grouped(keys, values, _g_order(levels, by), False)
    raise TypeError("as_factor() input must be text, numbers, logical values, or a factor; "
                    "convert other types with as_character() first")


def _g_revalue(g: _Grouped, mapping: dict[str, str | None]) -> _Grouped:
    """Relabel levels in every group, merging duplicates; None removes a level."""
    names = g.levels["__level"].unique().to_list()
    full = {name: mapping.get(name, name) for name in names}
    if all(old == new for old, new in full.items()):
        return g
    old, new = list(full), [full[name] for name in full]
    values = g.strings().replace_strict(old, new, default=None, return_dtype=pl.String)
    levels = (
        g.levels.sort("__pos")
        .with_columns(pl.col("__level").replace_strict(old, new, default=None, return_dtype=pl.String))
        .drop_nulls("__level")
        .unique(subset=g.on, keep="first", maintain_order=True)
    )
    return _Grouped(g.keys, values, _g_order(levels, g.by), g.ordered)


def _g_keep_levels(g: _Grouped, pairs: pl.DataFrame, other_level: str) -> _Grouped:
    """Levels not kept become *other_level*, placed last in the groups that changed.

    *pairs* is the level table with a boolean ``__keep`` column.
    """
    if bool(pairs["__keep"].all()):
        return g
    pairs = pairs.sort("__pos").with_columns(
        __changed=(~pl.col("__keep")).any().over(g.by),
        __new=pl.when(pl.col("__keep")).then(pl.col("__level")).otherwise(pl.lit(other_level)),
    )
    values = g.rows().join(
        pairs.select(*g.on, "__new"), on=g.on, how="left", nulls_equal=True, maintain_order="left"
    )["__new"]
    levels = (
        pairs.select(*g.by, "__changed", __level=pl.col("__new"))
        .unique(subset=g.on, keep="first", maintain_order=True)
        .with_columns(__band=((pl.col("__level") == other_level) & pl.col("__changed")).cast(pl.Int64))
        .with_row_index("__pos")
        .sort(["__band", "__pos"])
    )
    return _Grouped(g.keys, values, _g_order(levels, g.by), g.ordered)


def _g_distinct_lists(table: pl.DataFrame, by: list[str]) -> list[list[str]]:
    """The distinct per-group lists of ``__level`` in *table*, ordered by ``__index``."""
    if table.is_empty():
        return []
    lists = (
        table.sort([*by, "__index"], nulls_last=True).group_by(by, maintain_order=True)
        .agg(pl.col("__level"))["__level"].to_list()
    )
    return [list(item) for item in dict.fromkeys(tuple(item) for item in lists)]


def _g_unknown(g: _Grouped, requested: pl.DataFrame) -> list[list[str]]:
    """Requested levels absent from a group, one list per distinct list.

    *requested* has the group columns, ``__level`` and ``__index``.
    """
    absent = requested.join(g.levels.select(g.on), on=g.on, how="anti", nulls_equal=True)
    return _g_distinct_lists(absent, g.by)


def _g_requested(groups: pl.DataFrame, levels: list[str]) -> pl.DataFrame:
    """The same requested levels for every group in *groups*, in order."""
    table = pl.DataFrame({
        "__level": pl.Series(levels, dtype=pl.String),
        "__index": pl.Series(range(len(levels)), dtype=pl.Int64),
    })
    return groups.join(table, how="cross")


def _g_sort_scores(pairs: pl.DataFrame, desc: bool) -> pl.DataFrame:
    """Each group's levels by ``__key``, missing last, ties in their existing order."""
    return pairs.sort("__pos").sort("__key", descending=desc, nulls_last=True, maintain_order=True)


def _g_summaries(
    values: pl.DataFrame, by: list[str], fun: Any, kwargs: dict[str, Any], message: str, pair: bool
) -> pl.DataFrame:
    """A summary per group and level: *by* columns and ``__score``.

    A tidy3 aggregation (or :func:`last2`/:func:`first2`) runs as one Polars
    aggregation; any other callable runs once per group and level on the
    Series collected for it.
    """
    if pair and fun in {last2, first2}:
        # The y value at the largest (smallest) x among rows with both values;
        # a level whose rows all lack one has no value, so it sorts last.
        pick = pl.col("x").arg_max() if fun is last2 else pl.col("x").arg_min()
        both = values.filter(~pl.col("__missing")).group_by(by).agg(pl.col("y").get(pick).cast(pl.Float64).alias("__score"))
        return values.select(by).unique().join(both, on=by, how="left", nulls_equal=True)
    if not pair and _tidy3_summary(fun):
        from tidy3.expr import to_polars

        summary = values.group_by(by).agg(to_polars(fun(col("x"), **kwargs)).alias("__score"))
        if isinstance(summary.schema["__score"], pl.List):
            raise ValueError(message)
        return summary.with_columns(pl.col("__score").cast(pl.Float64))
    collected = values.group_by(by, maintain_order=True).agg(pl.col("x"), *([pl.col("y")] if pair else []))
    scores = []
    for index, x in enumerate(collected["x"]):
        result = fun(x, collected["y"][index], **kwargs) if pair else _summary(x, fun, kwargs)
        if not _is_scalar(result):
            raise ValueError(message)
        scores.append(result)
    return collected.select(by).with_columns(
        __score=pl.Series([None if _scalar_missing(s) else float(s) for s in scores], dtype=pl.Float64)
    )


def _g_scored(g: _Grouped, scores: pl.DataFrame, default: Any) -> pl.DataFrame:
    """The level table with ``__key``: the level's score, or *default* without one."""
    fallback = pl.lit(None if _scalar_missing(default) else float(default), dtype=pl.Float64)
    return g.join_levels(scores.with_columns(__found=pl.lit(True)), ["__score", "__found"]).with_columns(
        __key=pl.when(pl.col("__found")).then(pl.col("__score")).otherwise(fallback)
    )


def _g_totals(g: _Grouped, weights: pl.Series | None) -> pl.DataFrame:
    """The level table with ``__count``: the weight of each level's rows in its group.

    *weights* None means one per row, read from the counts recorded when the
    levels were built where there are any.
    """
    if weights is None and "__count" in g.levels.columns:
        return g.levels
    if weights is None:
        weights = pl.repeat(1.0, g.keys.height, dtype=pl.Float64, eager=True)
    counts = g.rows(__w=weights).drop_nulls("__level").group_by(g.on).agg(pl.col("__w").sum().alias("__count"))
    return g.join_levels(counts, ["__count"]).with_columns(pl.col("__count").fill_null(0.0))


def _g_change_levels(name: str, g: _Grouped, args: tuple[pl.Series, ...], kw: dict[str, Any]) -> _Grouped:
    if name == "as_factor":
        return g
    if name == "fct_drop":
        used = g.rows().drop_nulls("__level").unique().with_columns(__used=pl.lit(True))
        pairs = g.join_levels(used, ["__used"]).with_columns(pl.col("__used").fill_null(False))
        only = kw["only"]
        keep = pl.col("__used") if only is None else pl.col("__used") | ~pl.col("__level").is_in(list(only))
        return g.with_levels(pairs.filter(keep).sort("__pos"), counts=True)
    if name == "fct_expand":
        new = list(dict.fromkeys(kw["levels"]))
        if not new:
            return g
        counts = g.levels.group_by(g.by).agg(pl.len().alias("__n"))
        groups = g.groups().join(counts, on=g.by, how="left", nulls_equal=True).with_columns(
            pl.col("__n").fill_null(0).cast(pl.Int64)
        )
        after = kw["after"]
        groups = groups.with_columns(
            __at=pl.col("__n") if math.isinf(after) else pl.min_horizontal(pl.lit(int(after)), pl.col("__n"))
        )
        additions = (
            _g_requested(groups.select(g.by), new)
            .join(g.levels.select(g.on), on=g.on, how="anti", nulls_equal=True)
            .select(*g.by, "__level", __band=pl.lit(1, dtype=pl.Int64), __key=pl.col("__index"))
        )
        existing = g.levels.sort("__pos").with_columns(
            __local=(pl.col("__level").cum_count().over(g.by) - 1).cast(pl.Int64)
        ).join(groups.select(*g.by, "__at"), on=g.by, how="left", nulls_equal=True).select(
            *g.by, "__level",
            __band=pl.when(pl.col("__local") < pl.col("__at")).then(0).otherwise(2).cast(pl.Int64),
            __key=pl.col("__pos"),
        )
        return g.with_levels(pl.concat([existing, additions]).sort(["__band", "__key"]))
    if name == "fct_inorder":
        first = g.rows().with_row_index("__row").drop_nulls("__level").group_by(g.on).agg(pl.col("__row").min())
        pairs = g.join_levels(first, ["__row"]).sort(["__row", "__pos"], nulls_last=True)
        return g.with_levels(pairs, g.ordered if kw["ordered"] is None else kw["ordered"], counts=True)
    if name == "fct_reorder2":
        x, y = args
        missing = x.is_null() | y.is_null()
        na_rm = kw["na_rm"]
        count = int(missing.sum())
        if count and na_rm is None:
            warnings.warn(
                f"fct_reorder2(): removing {count} missing value{'s' if count != 1 else ''}; set na_rm=True to silence",
                UserWarning, stacklevel=5,
            )
            na_rm = True
        values = g.rows(x=x, y=y, __missing=missing).drop_nulls("__level")
        if na_rm:
            values = values.filter(~pl.col("__missing"))
        scores = _g_summaries(values, g.on, kw["fun"], kw["fun_kwargs"], "fct_reorder2() fun must return one value per level", pair=True)
        return g.with_levels(_g_sort_scores(_g_scored(g, scores, kw["default"]), kw["desc"]), counts=True)
    if name == "fct_na_value_to_level":
        level = kw["level"]
        have = g.levels.filter(pl.col("__level") == level).select(g.by)
        additions = g.groups().join(have, on=g.by, how="anti", nulls_equal=True).with_columns(
            __level=pl.lit(level), __pos=pl.lit(len(g.levels) + 1, dtype=pl.Int64)
        )
        levels = pl.concat([g.levels.select(*g.by, "__level", "__pos"), additions.select(*g.by, "__level", "__pos")]).sort("__pos")
        return _Grouped(g.keys, g.strings().fill_null(level), _g_order(levels, g.by), g.ordered)
    if name == "fct_na_level_to_value":
        removed = set(kw["extra_levels"])
        return _g_revalue(g, {level: None for level in removed})
    if name in {"fct_recode", "fct_collapse"}:
        mapping_levels = kw["levels"]
        removed = [mapping_levels["NULL"]] if name == "fct_recode" and "NULL" in mapping_levels else []
        mapping: dict[str, str | None] = {}
        for new, old in mapping_levels.items():
            if name == "fct_recode" and new == "NULL":
                continue
            for level in ([old] if isinstance(old, str) else old):
                mapping[level] = new
        requested = _g_requested(g.level_groups(), list(mapping))
        absent = requested.join(g.levels.select(g.on), on=g.on, how="anti", nulls_equal=True)
        unknown = pl.concat([absent, requested.filter(pl.col("__level").is_in(removed))]).unique()
        for levels in _g_distinct_lists(unknown, g.by):
            warnings.warn(f"{name}(): unknown levels: {levels}", UserWarning, stacklevel=5)
        mapping.update({level: None for level in removed})
        result = _g_revalue(g, mapping)
        if name == "fct_collapse" and kw["other_level"] is not None:
            pairs = result.levels.with_columns(__keep=pl.col("__level").is_in(list(mapping_levels)))
            return _g_keep_levels(result, pairs, kw["other_level"])
        return result

    weights = _weights(name, g.keys.height, args, finite=True) if args else None
    pairs = _g_totals(g, weights)
    threshold = kw["threshold"]
    if name == "fct_lump_n":
        score = pl.col("__count") if threshold < 0 else -pl.col("__count")
        method = kw["ties_method"]
        if method == "last":
            pairs = pairs.sort("__pos", descending=True).with_columns(__rank=score.rank(method="ordinal").over(g.by))
        else:
            pairs = pairs.sort("__pos").with_columns(
                __rank=score.rank(method="ordinal" if method == "first" else method).over(g.by)
            )
        pairs = pairs.with_columns(__keep=pl.col("__rank") <= abs(threshold))
    elif name == "fct_lump_prop":
        row_weights = pl.repeat(1.0, g.keys.height, dtype=pl.Float64, eager=True) if weights is None else weights
        totals = g.rows(__w=row_weights).group_by(g.by).agg(pl.col("__w").sum().alias("__total"))
        pairs = pairs.join(totals, on=g.by, how="left", nulls_equal=True)
        if bool((pairs["__total"] == 0).any()):
            raise ValueError("fct_lump_prop() requires a positive total weight when levels exist")
        proportion = pl.col("__count") / pl.col("__total")
        pairs = pairs.with_columns(__keep=proportion <= -threshold if threshold < 0 else proportion > threshold)
    else:
        pairs = pairs.with_columns(__keep=pl.col("__count") >= threshold)
    return _g_keep_levels(g, pairs, kw["other_level"])


def _g_apply(name: str, g: _Grouped, args: tuple[pl.Series, ...], kw: dict[str, Any]) -> _Grouped:
    """One factor function on every group at once; *args* are Series of the rows' length."""
    if g.uniform is not None and name in _LEVEL_ONLY:
        # Every group has the same levels and the function reads only the
        # levels: the groups' results are identical, so compute them once.
        result = _apply(name, Factor(g.values, g.uniform, g.ordered), args, kw)
        return _Grouped(g.keys, result.values, None, result.ordered, uniform=result.levels)
    if name not in _ORDERING_FUNCTIONS:
        return _g_change_levels(name, g, args, kw)
    ordered = g.ordered
    if name == "fct_rev":
        return g.with_levels(g.levels.sort("__pos", descending=True), counts=True)
    if name == "fct_relevel":
        requested = kw["levels"]
        if len(requested) == 1 and callable(requested[0]):
            function = requested[0]
            per_group = (
                g.levels.sort("__pos").group_by(g.by, maintain_order=True).agg(pl.col("__level"))
                .with_row_index("__g")
            )
            numbers, names, indexes = [], [], []
            for number, levels in zip(per_group["__g"], per_group["__level"]):
                chosen = list(function(levels.to_list()))
                if not all(isinstance(level, str) for level in chosen):
                    raise TypeError("fct_relevel() callable must return character levels")
                if len(set(chosen)) != len(chosen):
                    raise ValueError("fct_relevel() levels must be unique")
                numbers.extend([number] * len(chosen))
                names.extend(chosen)
                indexes.extend(range(len(chosen)))
            table = pl.DataFrame({
                "__g": pl.Series(numbers, dtype=per_group.schema["__g"]),
                "__level": pl.Series(names, dtype=pl.String),
                "__index": pl.Series(indexes, dtype=pl.Int64),
            }).join(per_group.select(*g.by, "__g"), on="__g", how="left").drop("__g")
        else:
            if not all(isinstance(level, str) for level in requested):
                raise TypeError("fct_relevel() callable must return character levels")
            if len(set(requested)) != len(requested):
                raise ValueError("fct_relevel() levels must be unique")
            table = _g_requested(g.level_groups(), list(requested))
        for unknown in _g_unknown(g, table):
            warnings.warn(f"fct_relevel(): unknown levels: {unknown}", UserWarning, stacklevel=5)
        after = kw["after"]
        pairs = g.join_levels(table, ["__index"]).sort("__pos").with_columns(
            __other=pl.col("__index").is_null()
        ).with_columns(
            __other_rank=pl.col("__other").cast(pl.Int64).cum_sum().over(g.by) - 1,
            __n_other=pl.col("__other").cast(pl.Int64).sum().over(g.by),
        )
        position = pl.col("__n_other") if math.isinf(after) else pl.min_horizontal(pl.lit(int(after)), pl.col("__n_other"))
        before = pl.col("__other") & (pl.col("__other_rank") < position)
        pairs = pairs.with_columns(
            __band=pl.when(before).then(0).when(pl.col("__other")).then(2).otherwise(1),
            __key=pl.when(pl.col("__other")).then(pl.col("__other_rank")).otherwise(pl.col("__index")),
        ).sort(["__band", "__key"])
        return g.with_levels(pairs, counts=True)
    if name == "fct_infreq":
        weights = _weights("fct_infreq", g.keys.height, args, finite=False) if args else None
        pairs = _g_totals(g, weights).sort(["__count", "__pos"], descending=[True, False])
        return g.with_levels(pairs, ordered if kw["ordered"] is None else kw["ordered"], counts=weights is None)
    x = args[0]
    if not (x.dtype.is_numeric() or x.dtype == pl.Boolean) and not x.drop_nulls().is_empty():
        raise TypeError("fct_reorder() x must be numeric")
    if kw["na_rm"] is None and x.null_count():
        warnings.warn("fct_reorder(): removing missing x values; set na_rm=True to silence", UserWarning, stacklevel=5)
    values = g.rows(x=x).drop_nulls("__level")
    if kw["na_rm"] is not False:
        values = values.filter(pl.col("x").is_not_null())
    scores = _g_summaries(values, g.on, kw["fun"], kw["fun_kwargs"], "fct_reorder() fun must return one value per level", pair=False)
    return g.with_levels(_g_sort_scores(_g_scored(g, scores, kw["default"]), kw["desc"]), counts=True)


def _g_arg(node: Any, frame: pl.DataFrame, keys: pl.DataFrame, ctx: _Context) -> Any:
    """One argument of a grouped factor call, with dplyr's per-group semantics."""
    kind = node[0]
    if kind == "lit":
        return node[1]
    if kind == "col" or (kind == "name" and node[1] in frame.columns):
        return _column_value(frame, node[1], ctx)
    if kind == "func" and node[1] in FUNCTIONS:
        return _grouped_node(node, frame, keys, ctx)
    return _compile_value(node, frame, ctx)


def _grouped_node(node: tuple, frame: pl.DataFrame, keys: pl.DataFrame, ctx: _Context) -> _Grouped:
    """A factor call evaluated for every group at once; *keys* are the rows' group columns."""
    name, raw_args, raw_kwargs = node[1], node[2], node[3]
    args = [_g_arg(arg, frame, keys, ctx) for arg in raw_args]
    kw = {key: _g_arg(value, frame, keys, ctx) for key, value in raw_kwargs.items()}
    source = _g_as_factor(args[0], keys) if name == "as_factor" else _g_source(name, args[0], keys)
    extras = tuple(
        _g_lengths(name, keys, value.values if isinstance(value, _Grouped) else _vector_series(value))
        for value in args[1:]
    )
    return _g_apply(name, source, extras, kw)


def _replace_children(node: tuple, children: list[Any]) -> tuple:
    """*node* with its children (as :func:`_expression_children` lists them) replaced."""
    kind = node[0]
    if kind == "bin":
        return (kind, node[1], children[0], children[1])
    if kind in {"neg", "not", "desc"}:
        return (kind, children[0])
    if kind == "func":
        n_args = len(node[2])
        return (kind, node[1], tuple(children[:n_args]), dict(zip(node[3].keys(), children[n_args:])))
    if kind == "call":
        rest = iter(children)
        base = next(rest)
        args = tuple(Expr(next(rest)) if isinstance(value, Expr) else value for value in node[3])
        kwargs = {
            key: Expr(next(rest)) if isinstance(value, Expr) else value for key, value in node[4].items()
        }
        return (kind, node[1], base, args, kwargs)
    if kind == "case_when":
        rest = iter(children)
        cases = [(next(rest), next(rest)) for _ in node[1]]
        return (kind, cases, next(rest))
    return node


def _lift_factors(node: tuple, frame: pl.DataFrame, ctx: _Context) -> tuple[tuple, pl.DataFrame]:
    """Compute the factor calls inside *node* as Enum columns of *frame*.

    Each becomes a column reference, so the rest of the expression compiles
    to Polars as usual.
    """
    from tidy3.verbs import _expression_children

    children = _expression_children(node)
    if not children:
        return node, frame
    replaced = []
    changed = False
    for child in children:
        if isinstance(child, tuple) and child and child[0] == "func" and child[1] in FUNCTIONS:
            result = _grouped_factor(child, frame, ctx) if ctx.groups else _factor_node(child, frame, ctx)
            name = f"__tidy3_factor_{frame.width}"
            frame = frame.with_columns(result.to_enum().alias(name))
            replaced.append(("col", name))
            changed = True
        elif contains_factor(child):
            lifted, frame = _lift_factors(child, frame, ctx)
            replaced.append(lifted)
            changed = True
        else:
            replaced.append(child)
    if not changed:
        return node, frame
    return _replace_children(node, replaced), frame


def _compile_value(node: tuple, frame: pl.DataFrame, ctx: _Context) -> pl.Series:
    """Evaluate a non-factor expression on *frame* with dplyr's window semantics."""
    from tidy3.verbs import _grouped_polars_value

    node, frame = _lift_factors(node, frame, ctx)
    expr = _grouped_polars_value(Expr(node), ctx.groups)
    if not isinstance(expr, pl.Expr):
        expr = pl.lit(expr)
    return frame.lazy().select(expr.alias("__tidy3_value")).collect().to_series()


def _evaluate(value: Expr, frame: pl.DataFrame, ctx: _Context) -> Factor | pl.Series:
    """A stage expression containing factor calls: a Factor, or a Series when
    the outermost call is not a factor function (``as_character(fct_rev(x))``)."""
    node = value.node
    if node[0] == "func" and node[1] in FUNCTIONS:
        return _grouped_factor(node, frame, ctx) if ctx.groups else _factor_node(node, frame, ctx)
    return _compile_value(node, frame, ctx)


def contains_factor(value: Any) -> bool:
    node = value.node if isinstance(value, Expr) else value
    if not isinstance(node, tuple) or not node or not isinstance(node[0], str):
        return False
    if node[0] == "func" and node[1] in FUNCTIONS:
        return True
    from tidy3.verbs import _expression_children

    return any(contains_factor(child) for child in _expression_children(node))


def prepare_polars(
    tf: Any, lf: pl.LazyFrame, stage: dict[str, Any], groups: list[str] | None,
    *, sort_groups: bool, ordered_flags: dict[str, bool] | None = None,
) -> tuple[pl.LazyFrame, dict[str, Any], dict[str, bool]]:
    """Compute factor assignments once; preserve other columns' Polars dtypes.

    The third result maps each produced factor column to its ordered flag.
    Polars Enums have no unordered state, so the flag has to travel beside
    the level names.
    """
    factors = {name: value for name, value in stage.items() if contains_factor(value)}
    if not factors:
        return lf, stage, {}
    from tidy3.frame import note_data_read
    from tidy3.verbs import _column_references

    note_data_read("fct_*() reads factor levels from the data")
    required = set(groups or []) | set().union(*(_column_references(value) for value in factors.values()))
    sources = {name: _level_source(value) for name, value in factors.items()}
    schema = lf.collect_schema()
    lazy = bool(required) and all(
        source is not None and _text_like(lf, source, schema) for source in sources.values()
    )
    if lazy:
        # Only the columns the factors read are collected, and the factor
        # values are their source values, so the result is a cast of those
        # columns rather than every row copied back.
        data = lf.select(sorted(required, key=list(schema.names()).index)).collect()
    else:
        data = lf.collect()
    flags = {column: bool(getattr(levels, "ordered", False))
             for column, levels in tf._category_levels.items()}
    flags.update(ordered_flags or {})
    ctx = _Context(groups, sort_groups, flags)
    results = {name: _evaluate(value, data, ctx) for name, value in factors.items()}
    rest = {name: value for name, value in stage.items() if name not in factors}
    ordered: dict[str, bool] = {}
    columns = []
    if lazy and all(isinstance(result, Factor) for result in results.values()):
        from tidy3.expr import to_polars

        for name, result in results.items():
            ordered[name] = result.ordered
            columns.append(to_polars(sources[name]).cast(pl.Enum(result.levels)).alias(name))
        return lf.with_columns(columns), rest, ordered
    if lazy:
        data = lf.collect()
    for name, result in results.items():
        if isinstance(result, Factor):
            ordered[name] = result.ordered
            columns.append(result.to_enum().alias(name))
        else:
            columns.append(result.alias(name))
    return data.with_columns(columns).lazy(), rest, ordered


def _level_source(value: Any) -> Any:
    """The expression a chain of fct_* calls reorders, or None.

    The ordering calls, fct_drop, fct_expand, fct_inorder, fct_reorder2, and
    as_factor change the levels and never the values, so
    ``fct_rev(fct_infreq(species))`` has the values of ``species``. (A
    numeric as_factor source is not text, so it never takes this path.)
    """
    node = value.node if isinstance(value, Expr) else None
    if not isinstance(node, tuple) or node[0] != "func" or node[1] not in _SAME_VALUES:
        return None
    while isinstance(node, tuple) and node[0] == "func" and node[1] in _SAME_VALUES:
        node = node[2][0]
    if isinstance(node, tuple) and node and node[0] == "func" and node[1] in FUNCTIONS:
        return None
    from tidy3.verbs import _expression_children

    if any(contains_factor(child) for child in _expression_children(node)):
        return None
    return Expr(node)


def _text_like(lf: pl.LazyFrame, source: Any, schema: Any) -> bool:
    """Text and factor sources keep their values as level names."""
    from tidy3.expr import to_polars

    try:
        dtype = lf.select(to_polars(source)).collect_schema().dtypes()[0]
    except pl.exceptions.PolarsError:
        return False
    return dtype == pl.String or isinstance(dtype, (pl.Enum, pl.Categorical))
