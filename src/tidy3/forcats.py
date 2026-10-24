"""Factor ordering and value changes for expressions and categorical vectors.

Polars factor mutations compute a snapshot to discover Enum levels. The
result resumes as a lazy frame; other mutations keep their usual lazy path.
"""

from __future__ import annotations

from contextvars import ContextVar
import math
from numbers import Real
from typing import Any
import warnings

import numpy as np
import pandas as pd
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
SORT_GROUPS: ContextVar[bool] = ContextVar("tidy3_factor_sort_groups", default=True)


def _column(value: Any) -> Any:
    return col(value) if isinstance(value, str) else value


def _dispatch(name: str, f: Any, *args: Any, **kwargs: Any) -> Any:
    if isinstance(f, (str, Expr, pl.Expr)):
        return _func(name, _column(f), *args, **kwargs)
    values = tuple(pd.Series(value) if not isinstance(value, pd.Series) else value for value in args)
    return eval_pandas(name, (f, *values), kwargs).array


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
    x, y = pd.Series(x).reset_index(drop=True), pd.Series(y).reset_index(drop=True)
    keep = (x.notna() & y.notna()).to_numpy()
    if not keep.any():
        return None
    values = x[keep].to_numpy()
    # argmax and argmin return the first row among ties, as R's order() does.
    return y[keep].iloc[int(values.argmax() if desc else values.argmin())]


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


def _revalue(source: pd.Series, mapping: dict[str, str | None]) -> pd.Series:
    """Relabel categorical codes, merging duplicates and preserving missing rows."""
    renamed = [mapping.get(level, level) for level in source.cat.categories]
    categories = list(dict.fromkeys(level for level in renamed if level is not None))
    lookup = {level: code for code, level in enumerate(categories)}
    codes = np.array([lookup.get(level, -1) for level in renamed] + [-1], dtype=np.int64)
    result = pd.Categorical.from_codes(
        codes[source.cat.codes.to_numpy()], categories=categories, ordered=source.cat.ordered,
    )
    return pd.Series(result, index=source.index, name=source.name)


def _keep_levels(source: pd.Series, keep: Any, other_level: str) -> pd.Series:
    if all(keep):
        return source
    mapping = {level: other_level for level, kept in zip(source.cat.categories, keep) if not kept}
    result = _revalue(source, mapping)
    categories = [level for level in result.cat.categories if level != other_level] + [other_level]
    return result.cat.reorder_categories(categories)


def _reorder2_levels(source: pd.Series, args: tuple[Any, ...], kw: dict[str, Any]) -> list[str]:
    x, y = args
    missing = (x.isna() | y.isna()).to_numpy()
    na_rm = kw["na_rm"]
    if missing.any() and na_rm is None:
        count = int(missing.sum())
        warnings.warn(
            f"fct_reorder2(): removing {count} missing value{'s' if count != 1 else ''}; set na_rm=True to silence",
            UserWarning, stacklevel=4,
        )
        na_rm = True
    keep = ~missing if na_rm else np.ones(len(source), dtype=bool)
    codes = source.cat.codes.to_numpy()[keep]
    x, y = x[keep], y[keep]
    scores = []
    for code in range(len(source.cat.categories)):
        rows = codes == code
        if not rows.any():
            scores.append(kw["default"])
            continue
        result = kw["fun"](x[rows], y[rows], **kw["fun_kwargs"])
        if not pd.api.types.is_scalar(result):
            raise ValueError("fct_reorder2() fun must return one value per level")
        scores.append(result)
    order = pd.Series(scores, dtype="Float64").sort_values(
        ascending=not kw["desc"], na_position="last", kind="stable"
    ).index
    return [source.cat.categories[i] for i in order]


def _change_levels(name: str, source: pd.Series, args: tuple[Any, ...], kw: dict[str, Any]) -> pd.Series:
    categories = source.cat.categories.tolist()
    if name == "as_factor":
        return source
    if name == "fct_drop":
        used = set(source.cat.codes[source.cat.codes >= 0].tolist())
        only = kw["only"]
        kept = [level for code, level in enumerate(categories)
                if code in used or (only is not None and level not in only)]
        return source.cat.set_categories(kept)
    if name == "fct_expand":
        new = [level for level in dict.fromkeys(kw["levels"]) if level not in categories]
        position = len(categories) if math.isinf(kw["after"]) else min(int(kw["after"]), len(categories))
        return source.cat.set_categories(categories[:position] + new + categories[position:])
    if name == "fct_inorder":
        codes = source.cat.codes.to_numpy()
        first = pd.unique(codes[codes >= 0]).tolist()
        order = first + [code for code in range(len(categories)) if code not in set(first)]
        ordered = source.cat.ordered if kw["ordered"] is None else kw["ordered"]
        return source.cat.reorder_categories([categories[code] for code in order], ordered=ordered)
    if name == "fct_reorder2":
        return source.cat.reorder_categories(_reorder2_levels(source, args, kw))
    if name == "fct_na_value_to_level":
        level = kw["level"]
        result = source if level in categories else source.cat.add_categories([level])
        return result.fillna(level)
    if name == "fct_na_level_to_value":
        removed = set(kw["extra_levels"])
        return source.cat.set_categories([level for level in categories if level not in removed])
    if name in {"fct_recode", "fct_collapse"}:
        levels = kw["levels"]
        removed = [levels["NULL"]] if name == "fct_recode" and "NULL" in levels else []
        mapping = {}
        for new, old in levels.items():
            if name == "fct_recode" and new == "NULL":
                continue
            for level in ([old] if isinstance(old, str) else old):
                mapping[level] = new
        unknown = [level for level in mapping if level not in source.cat.categories or level in removed]
        if unknown:
            warnings.warn(f"{name}(): unknown levels: {unknown}", UserWarning, stacklevel=4)
        mapping.update({level: None for level in removed})
        result = _revalue(source, mapping)
        if name == "fct_collapse" and kw["other_level"] is not None:
            return _keep_levels(result, result.cat.categories.isin(levels), kw["other_level"])
        return result

    weights = args[0] if args else pd.Series(1.0, index=source.index)
    if (not pd.api.types.is_numeric_dtype(weights) or weights.isna().any()
            or not np.isfinite(weights.to_numpy(dtype=float)).all() or (weights < 0).any()):
        raise ValueError(f"{name}() weights must be finite, non-negative and non-missing")
    counts = weights.groupby(source, observed=False, sort=False).sum().reindex(source.cat.categories, fill_value=0)
    threshold = kw["threshold"]
    if name == "fct_lump_n":
        scores = counts if threshold < 0 else -counts
        method = kw["ties_method"]
        if method == "last":
            ranks = scores.iloc[::-1].rank(method="first").iloc[::-1]
        elif method == "random":
            ranks = scores.iloc[np.random.permutation(len(scores))].rank(method="first").reindex(scores.index)
        else:
            ranks = scores.rank(method=method)
        keep = ranks <= abs(threshold)
    elif name == "fct_lump_prop":
        denominator = weights.sum()
        if denominator == 0 and len(counts):
            raise ValueError("fct_lump_prop() requires a positive total weight when levels exist")
        proportions = counts / denominator
        keep = proportions <= -threshold if threshold < 0 else proportions > threshold
    else:
        keep = counts >= threshold
    return _keep_levels(source, keep, kw["other_level"])


def _factor(value: Any, levels: list[str] | None = None, *, ordered: bool | None = None) -> pd.Series:
    if isinstance(value, pl.Series):
        if isinstance(value.dtype, pl.Enum):
            levels = value.dtype.categories.to_list()
        value = value.to_pandas()
    source = value if isinstance(value, pd.Series) else pd.Series(value)
    if isinstance(source.dtype, pd.CategoricalDtype):
        if not all(isinstance(level, str) for level in source.cat.categories):
            raise TypeError("factor levels must be strings")
        if levels is not None:
            flag = source.cat.ordered if ordered is None else ordered
            return pd.Series(pd.Categorical(source, categories=levels, ordered=flag), index=source.index, name=source.name)
        return source
    # Checking the distinct values is enough, and far cheaper than every row.
    distinct = source.dropna().unique().tolist()
    if not all(isinstance(value, str) for value in distinct):
        raise TypeError("factor input must be a character or categorical column")
    categories = sorted(distinct) if levels is None else levels
    flag = False if ordered is None else ordered
    return pd.Series(pd.Categorical(source, categories=categories, ordered=flag), index=source.index, name=source.name)


def _summary(values: pd.Series, fun: Any, kwargs: dict[str, Any]) -> Any:
    if getattr(fun, "__module__", "") == "tidy3.expr":
        from tidy3.pandas_engine import eval_expr

        result = eval_expr(fun(col("x"), **kwargs), pd.DataFrame({"x": values}), None, "agg")
    else:
        result = fun(values, **kwargs)
    if not pd.api.types.is_scalar(result):
        raise ValueError("fct_reorder() fun must return one value per level")
    return result


def _apply(name: str, source: pd.Series, args: tuple[Any, ...], kw: dict[str, Any]) -> pd.Series:
    if name not in _ORDERING_FUNCTIONS:
        return _change_levels(name, source, args, kw)
    levels = source.cat.categories.tolist()
    ordered = source.cat.ordered
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
        weights = args[0] if args else pd.Series(1.0, index=source.index)
        if not pd.api.types.is_numeric_dtype(weights) or weights.isna().any() or (weights < 0).any():
            raise ValueError("fct_infreq() weights must be numeric, non-negative and non-missing")
        scores = [weights[source == level].sum() for level in levels]
        result = [levels[i] for i in sorted(range(len(levels)), key=lambda i: -scores[i])]
        ordered = ordered if kw["ordered"] is None else kw["ordered"]
    else:
        x = args[0]
        if not pd.api.types.is_numeric_dtype(x) and not x.dropna().empty:
            raise TypeError("fct_reorder() x must be numeric")
        missing = x.isna()
        if kw["na_rm"] is None and missing.any():
            warnings.warn("fct_reorder(): removing missing x values; set na_rm=True to silence", UserWarning, stacklevel=3)
        scores = []
        for level in levels:
            values = x[source == level]
            if kw["na_rm"] is not False:
                values = values.dropna()
            scores.append(kw["default"] if values.empty else _summary(values, kw["fun"], kw["fun_kwargs"]))
        order = pd.Series(scores, dtype="Float64").sort_values(ascending=not kw["desc"], na_position="last", kind="stable").index
        result = [levels[i] for i in order]
    return source.cat.reorder_categories(result, ordered=ordered)


def _as_factor(value: Any) -> pd.Series:
    """forcats::as_factor levels: appearance for text, sorted for numbers and logicals."""
    if isinstance(value, pl.Series):
        if isinstance(value.dtype, (pl.Enum, pl.Categorical)):
            return _factor(value)
        value = value.to_pandas()
    source = value if isinstance(value, pd.Series) else pd.Series(value)
    if isinstance(source.dtype, pd.CategoricalDtype):
        return _factor(source)
    distinct = pd.unique(source.dropna()).tolist()
    if all(isinstance(item, str) for item in distinct):
        return _factor(source, distinct)
    from tidy3.janitor import _header_text

    if all(isinstance(item, (bool, np.bool_)) for item in distinct):
        values, labels = [False, True], ["FALSE", "TRUE"]
    elif all(isinstance(item, (Real, np.number)) and not isinstance(item, (bool, np.bool_)) for item in distinct):
        values = sorted(distinct)
        labels = [_header_text(item) for item in values]
        if len(set(labels)) != len(labels):
            raise ValueError("as_factor() numbers must have distinct text")
    else:
        raise TypeError("as_factor() input must be text, numbers, logical values, or a factor; "
                        "convert other types with as_character() first")
    lookup = {item: code for code, item in enumerate(values)}
    codes = np.array([-1 if pd.isna(item) else lookup[item] for item in source.tolist()], dtype=np.int64)
    return pd.Series(pd.Categorical.from_codes(codes, categories=labels), index=source.index, name=source.name)


def eval_pandas(name: str, args: tuple[Any, ...], kw: dict[str, Any]) -> pd.Series:
    source = _as_factor(args[0]) if name == "as_factor" else _factor(args[0])
    extra = []
    for value in args[1:]:
        series = value if isinstance(value, pd.Series) else pd.Series(value)
        if len(series) != len(source):
            raise ValueError(f"{name}() inputs must have the same length")
        extra.append(series.set_axis(source.index))
    return _apply(name, source, tuple(extra), kw)


def evaluate_grouped(node: tuple, df: pd.DataFrame, groups: list[str], mode: str) -> pd.Series:
    from tidy3.pandas_engine import eval_expr

    grouped = df.groupby(groups, sort=SORT_GROUPS.get(), dropna=False, observed=True)
    if len(df) and _columns_only(node):
        name, raw_args, raw_kwargs = node[1], node[2], node[3]
        args = tuple(eval_expr(Expr(arg), df, None, mode) for arg in raw_args)
        kw = {key: eval_expr(Expr(value), df, None, mode) for key, value in raw_kwargs.items()}
        if not (name == "fct_relevel" and len(kw["levels"]) == 1 and callable(kw["levels"][0])):
            source = _factor(args[0])
            # ngroup numbers groups in the order the loop below visits them.
            group_ids = grouped.ngroup().to_numpy()
            levels, ordered = _reordered_levels(
                name, source, args[1:], kw, group_ids, text=not _is_factor(args[0])
            )
            categorical = pd.Categorical(source.astype(object), categories=levels, ordered=ordered)
            return pd.Series(categorical, index=df.index)
    parts = grouped.indices.values()
    levels: list[str] = []
    values = pd.Series(index=pd.RangeIndex(len(df)), dtype=object)
    found = False
    ordered = False
    for positions in parts:
        part = eval_expr(Expr(node), df.iloc[positions], None, mode)
        levels.extend(level for level in part.cat.categories if level not in levels)
        ordered = part.cat.ordered
        values.iloc[positions] = part.astype(object).to_numpy()
        found = True
    if not found:
        return eval_expr(Expr(node), df, None, mode)
    return pd.Series(pd.Categorical(values, categories=levels, ordered=ordered), index=df.index)


def _columns_only(node: tuple) -> bool:
    """A single fct_* call on columns: the vectorized grouped path applies.

    Nested factor calls or computed arguments use the per-group loop.
    """
    if not (isinstance(node, tuple) and node[0] == "func" and node[1] in _ORDERING_FUNCTIONS):
        return False
    columns = all(isinstance(arg, tuple) and arg[0] in {"col", "name"} for arg in node[2])
    literals = all(isinstance(value, tuple) and value[0] == "lit" for value in node[3].values())
    return bool(node[2]) and columns and literals


def _is_factor(value: Any) -> bool:
    if isinstance(value, pl.Series):
        return isinstance(value.dtype, (pl.Enum, pl.Categorical))
    return isinstance(getattr(value, "dtype", None), pd.CategoricalDtype)


def _reordered_levels(
    name: str,
    source: pd.Series,
    args: tuple[Any, ...],
    kw: dict[str, Any],
    group_ids: Any,
    *,
    text: bool,
) -> tuple[list[str], bool]:
    """Each group's reordered levels, joined in group order: the final levels.

    The same rules as :func:`_apply` run on one group at a time, computed for
    all groups at once. A text source gives each group the levels present in
    it, in sorted order; a factor source gives every group all its levels.
    Groups are joined in ``group_ids`` order, each adding levels not seen yet,
    so the result matches applying ``_apply`` group by group.
    """
    categories = source.cat.categories.tolist()
    codes = source.cat.codes.to_numpy()
    groups = np.asarray(group_ids)
    count = len(categories)
    if text:
        present = codes >= 0
        pairs = pd.DataFrame({"gid": groups[present], "code": codes[present]}).drop_duplicates()
    else:
        unique_groups = pd.unique(groups)
        pairs = pd.DataFrame({
            "gid": np.repeat(unique_groups, count),
            "code": np.tile(np.arange(count), len(unique_groups)),
        })
    pairs = pairs.sort_values(["gid", "code"], kind="stable").reset_index(drop=True)
    ordered = bool(source.cat.ordered)
    if name == "fct_rev":
        pairs["key"] = -pairs["code"]
        keys = ["gid", "key"]
    elif name == "fct_relevel":
        requested = list(kw["levels"])
        if not all(isinstance(level, str) for level in requested):
            raise TypeError("fct_relevel() callable must return character levels")
        if len(set(requested)) != len(requested):
            raise ValueError("fct_relevel() levels must be unique")
        rank = {categories.index(level): i for i, level in enumerate(requested) if level in categories}
        missing_per_group = _unknown_requested(pairs, rank, requested)
        for unknown in missing_per_group:
            warnings.warn(f"fct_relevel(): unknown levels: {unknown}", UserWarning, stacklevel=4)
        chosen = pairs["code"].map(rank)
        other = chosen.isna()
        pairs["other_rank"] = other.groupby(pairs["gid"]).cumsum().where(other) - 1
        others = other.groupby(pairs["gid"]).transform("sum")
        after = kw["after"]
        position = others if math.isinf(after) else np.minimum(int(after), others)
        before = other & (pairs["other_rank"] < position)
        pairs["band"] = np.where(before, 0, np.where(other, 2, 1))
        pairs["key"] = np.where(other, pairs["other_rank"], chosen)
        keys = ["gid", "band", "key"]
    elif name == "fct_infreq":
        weights = args[0] if args else pd.Series(1.0, index=source.index)
        if not pd.api.types.is_numeric_dtype(weights) or weights.isna().any() or (weights < 0).any():
            raise ValueError("fct_infreq() weights must be numeric, non-negative and non-missing")
        present = codes >= 0
        sums = (
            pd.DataFrame({"gid": groups[present], "code": codes[present], "w": np.asarray(weights)[present]})
            .groupby(["gid", "code"], sort=False)["w"].sum()
        )
        pairs["score"] = pd.MultiIndex.from_frame(pairs[["gid", "code"]]).map(sums).fillna(0).astype(float)
        pairs["key"] = -pairs["score"]
        keys = ["gid", "key"]
        ordered = ordered if kw["ordered"] is None else kw["ordered"]
    else:
        x = args[0]
        if not pd.api.types.is_numeric_dtype(x) and not x.dropna().empty:
            raise TypeError("fct_reorder() x must be numeric")
        if kw["na_rm"] is None and x.isna().any():
            warnings.warn("fct_reorder(): removing missing x values; set na_rm=True to silence", UserWarning, stacklevel=4)
        present = codes >= 0
        values = pd.DataFrame({"gid": groups[present], "code": codes[present], "x": np.asarray(x)[present]})
        if kw["na_rm"] is not False:
            values = values[values["x"].notna()]
        scores = _pair_summaries(values, kw["fun"], kw["fun_kwargs"])
        score = pd.MultiIndex.from_frame(pairs[["gid", "code"]]).map(scores)
        filled = pd.Series(score, dtype="Float64").where(
            pd.Series(pd.MultiIndex.from_frame(pairs[["gid", "code"]]).isin(scores.index)),
            kw["default"],
        )
        pairs["key"] = filled.to_numpy(dtype=float, na_value=np.nan)
        pairs = pairs.sort_values(
            ["gid", "key"], ascending=[True, not kw["desc"]], na_position="last", kind="stable"
        )
        keys = None
    if keys is not None:
        pairs = pairs.sort_values(keys, kind="stable")
    order = pd.unique(pairs["code"].to_numpy())
    return [categories[code] for code in order], ordered


def _unknown_requested(pairs: pd.DataFrame, rank: dict[int, int], requested: list[str]) -> list[list[str]]:
    """The unknown-level lists _apply would warn about, one per distinct list."""
    codes = {position: code for code, position in rank.items()}
    groups = pairs["gid"].drop_duplicates()
    missing = pd.DataFrame({
        level: ~groups.isin(pairs.loc[pairs["code"] == codes.get(i, -1), "gid"]).to_numpy()
        for i, level in enumerate(requested)
    }).drop_duplicates()
    return [
        [level for level, absent in zip(requested, row) if absent]
        for row in missing.itertuples(index=False, name=None) if any(row)
    ]


def _pair_summaries(values: pd.DataFrame, fun: Any, kwargs: dict[str, Any]) -> pd.Series:
    """fct_reorder's summary of x for each (group, level) pair present."""
    if values.empty:
        return pd.Series(dtype=float, index=pd.MultiIndex.from_arrays([[], []], names=["gid", "code"]))
    if getattr(fun, "__module__", "") == "tidy3.expr":
        from tidy3.pandas_engine import eval_expr

        result = eval_expr(fun(col("x"), **kwargs), values, ["gid", "code"], "agg")
        if isinstance(result, pd.Series) and list(result.index.names) == ["gid", "code"]:
            return result
    return pd.Series(
        {key: _summary(group, fun, kwargs) for key, group in values.groupby(["gid", "code"], sort=False)["x"]},
        dtype=object,
    ).rename_axis(["gid", "code"])


def contains_factor(value: Any) -> bool:
    node = value.node if isinstance(value, Expr) else value
    if not isinstance(node, tuple) or not node or not isinstance(node[0], str):
        return False
    if node[0] == "func" and node[1] in FUNCTIONS:
        return True
    from tidy3.verbs import _expression_children

    return any(contains_factor(child) for child in _expression_children(node))


def _lit(node: Any) -> Any:
    if isinstance(node, tuple) and node and node[0] == "lit":
        return node[1]
    return None


def _column_name(node: Any) -> str | None:
    if isinstance(node, tuple) and node and node[0] in {"col", "name"} and isinstance(node[1], str):
        return node[1]
    return None


def _group_key_dtype(dtype: Any) -> bool:
    return dtype in {pl.String, pl.Boolean, pl.Date} or dtype.is_integer() or isinstance(dtype, pl.Datetime)


def _native_spec(value: Any, schema: Any, groups: list[str] | None) -> dict[str, Any] | None:
    """A single fct_* on string columns, or None when pandas still has to run.

    Nested factor calls, callables, and Enum inputs keep the pandas path so
    unused levels and per-group Python summaries stay exact.
    """
    node = value.node if isinstance(value, Expr) else None
    if not isinstance(node, tuple) or node[0] != "func" or node[1] not in _ORDERING_FUNCTIONS:
        return None
    name, args, kwargs = node[1], node[2], node[3]
    if len(args) < 1:
        return None
    source = _column_name(args[0])
    if source is None or source not in schema or schema[source] != pl.String:
        return None
    for group in groups or []:
        if group not in schema or not _group_key_dtype(schema[group]):
            return None
    extra = None
    if name == "fct_infreq":
        if args[1:]:
            extra = _column_name(args[1])
            if extra is None or extra not in schema:
                return None
        if _lit(kwargs.get("ordered")) not in {None, True, False}:
            return None
    elif name == "fct_reorder":
        extra = _column_name(args[1]) if len(args) > 1 else None
        if extra is None or extra not in schema:
            return None
        fun = _lit(kwargs.get("fun"))
        from tidy3.expr import max as expr_max
        from tidy3.expr import mean, min as expr_min
        from tidy3.expr import sum as expr_sum

        aggs = {
            median: lambda column: pl.col(column).median(),
            mean: lambda column: pl.col(column).mean(),
            expr_sum: lambda column: pl.col(column).sum(),
            expr_min: lambda column: pl.col(column).min(),
            expr_max: lambda column: pl.col(column).max(),
        }
        if fun not in aggs or _lit(kwargs.get("fun_kwargs")) or not isinstance(_lit(kwargs.get("desc")), bool):
            return None
        if _lit(kwargs.get("na_rm")) not in {None, True, False}:
            return None
    elif name == "fct_relevel":
        requested = _lit(kwargs.get("levels"))
        after = _lit(kwargs.get("after"))
        if not isinstance(requested, list) or any(not isinstance(level, str) for level in requested):
            return None
        if len(set(requested)) != len(requested):
            return None
        if isinstance(after, bool) or not isinstance(after, (int, float)):
            return None
    elif name != "fct_rev":
        return None
    return {"name": name, "source": source, "extra": extra, "kwargs": kwargs}


def _attach_codes(pairs: pl.DataFrame, source: str, categories: list[str]) -> pl.DataFrame:
    codes = pl.DataFrame({"__level": categories, "__tidy3_code": list(range(len(categories)))})
    return pairs.join(codes, left_on=source, right_on="__level", how="left")


def _with_group_ids(pairs: pl.DataFrame, data: pl.DataFrame, groups: list[str], sort_groups: bool) -> pl.DataFrame:
    if not groups:
        return pairs.with_columns(__tidy3_gid=pl.lit(0, dtype=pl.UInt32))
    keys = data.select(groups).unique(maintain_order=True)
    if sort_groups:
        keys = keys.sort(groups)
    keys = keys.with_row_index("__tidy3_gid")
    return pairs.join(keys, on=groups, how="left")


def _native_levels(
    data: pl.DataFrame, spec: dict[str, Any], groups: list[str] | None, sort_groups: bool
) -> tuple[list[str], bool] | None:
    """Level order for one string factor, using the same rules as :func:`_reordered_levels`."""
    source = spec["source"]
    groups = list(groups or [])
    if groups and any(data[group].null_count() for group in groups):
        return None
    extra = spec["extra"]
    if extra is not None:
        column = data[extra]
        if not (column.dtype.is_integer() or column.dtype.is_float()) or column.null_count():
            return None
        # pandas skips NaN in median/mean/sum; Polars aggregates it.
        if column.dtype.is_float() and bool(column.is_nan().any()):
            return None
        if spec["name"] == "fct_infreq" and bool((column < 0).any()):
            return None
    present = data.drop_nulls(subset=[source])
    categories = sorted(present[source].unique().to_list())
    if not categories:
        return None
    name = spec["name"]
    kwargs = spec["kwargs"]
    if name == "fct_relevel":
        requested = _lit(kwargs["levels"])
        after = _lit(kwargs["after"])
        known = [level for level in requested if level in categories]
        pairs = present.select([*groups, source]).unique()
        pairs = _attach_codes(pairs, source, categories)
        pairs = _with_group_ids(pairs, data, groups, sort_groups)
        pairs = pairs.sort(["__tidy3_gid", "__tidy3_code"])
        other = ~pl.col(source).is_in(known) if known else pl.lit(True)
        rank = {level: index for index, level in enumerate(requested) if level in categories}
        pairs = pairs.with_columns(__other=other)
        pairs = pairs.with_columns(
            __other_rank=pl.col("__other").cast(pl.Int64).cum_sum().over("__tidy3_gid", order_by="__tidy3_code") - 1,
            __n_other=pl.col("__other").cast(pl.Int64).sum().over("__tidy3_gid"),
            __req=pl.col(source).replace_strict(rank, default=None),
        )
        position = pl.col("__n_other") if math.isinf(after) else pl.min_horizontal(pl.lit(int(after)), pl.col("__n_other"))
        before = pl.col("__other") & (pl.col("__other_rank") < position)
        pairs = pairs.with_columns(
            __band=pl.when(before).then(0).when(pl.col("__other")).then(2).otherwise(1),
            __key=pl.when(pl.col("__other")).then(pl.col("__other_rank")).otherwise(pl.col("__req")),
        ).sort(["__tidy3_gid", "__band", "__key"], maintain_order=True)
        codes = {categories.index(level): index for index, level in enumerate(requested) if level in categories}
        warning_rows = pairs.select(gid=pl.col("__tidy3_gid"), code=pl.col("__tidy3_code")).to_pandas()
        for unknown in _unknown_requested(warning_rows, codes, requested):
            warnings.warn(f"fct_relevel(): unknown levels: {unknown}", UserWarning, stacklevel=4)
        ordered = False
    else:
        group_keys = [*groups, source]
        if name == "fct_rev":
            pairs = present.select(group_keys).unique()
            pairs = _attach_codes(pairs, source, categories)
            pairs = _with_group_ids(pairs, data, groups, sort_groups)
            pairs = pairs.sort(["__tidy3_gid", "__tidy3_code"], descending=[False, True])
            ordered = False
        elif name == "fct_infreq":
            score = pl.len() if extra is None else pl.col(extra).sum()
            pairs = present.group_by(group_keys).agg(score.alias("__tidy3_score"))
            pairs = _attach_codes(pairs, source, categories)
            pairs = _with_group_ids(pairs, data, groups, sort_groups)
            pairs = pairs.sort(
                ["__tidy3_gid", "__tidy3_score", "__tidy3_code"],
                descending=[False, True, False],
            )
            flag = _lit(kwargs.get("ordered"))
            ordered = False if flag is None else bool(flag)
        else:
            from tidy3.expr import max as expr_max
            from tidy3.expr import mean, min as expr_min
            from tidy3.expr import sum as expr_sum

            agg = {
                median: lambda column: pl.col(column).median(),
                mean: lambda column: pl.col(column).mean(),
                expr_sum: lambda column: pl.col(column).sum(),
                expr_min: lambda column: pl.col(column).min(),
                expr_max: lambda column: pl.col(column).max(),
            }[_lit(kwargs["fun"])]
            pairs = present.group_by(group_keys).agg(agg(extra).alias("__tidy3_score"))
            pairs = _attach_codes(pairs, source, categories)
            pairs = _with_group_ids(pairs, data, groups, sort_groups)
            pairs = pairs.sort(
                ["__tidy3_gid", "__tidy3_score", "__tidy3_code"],
                descending=[False, bool(_lit(kwargs["desc"])), False],
            )
            ordered = False
    return pairs[source].unique(maintain_order=True).to_list(), ordered


def _apply_native_factors(
    lf: pl.LazyFrame, factors: dict[str, Any], groups: list[str] | None, sort_groups: bool
) -> tuple[pl.LazyFrame, dict[str, Any], dict[str, bool]]:
    """Cast string factors in Polars. Anything else stays in *factors* for pandas."""
    schema = lf.collect_schema()
    specs = []
    for name, value in factors.items():
        spec = _native_spec(value, schema, groups)
        if spec is not None:
            specs.append((name, spec))
    if not specs:
        return lf, factors, {}
    needed = set(groups or [])
    for _, spec in specs:
        needed.add(spec["source"])
        if spec["extra"]:
            needed.add(spec["extra"])
    data = lf.select([name for name in schema.names() if name in needed]).collect()
    exprs = []
    ordered: dict[str, bool] = {}
    for name, spec in specs:
        levels = _native_levels(data, spec, groups, sort_groups)
        if levels is None:
            continue
        level_names, flag = levels
        exprs.append(pl.col(spec["source"]).cast(pl.String).cast(pl.Enum(level_names)).alias(name))
        ordered[name] = flag
    if exprs:
        lf = lf.with_columns(exprs)
    remaining = {name: value for name, value in factors.items() if name not in ordered}
    return lf, remaining, ordered


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

    note_data_read("fct_*() reads factor levels from the data")
    lf, factors, native_ordered = _apply_native_factors(lf, factors, groups, sort_groups)
    if not factors:
        rest = {name: value for name, value in stage.items() if name not in native_ordered}
        return lf, rest, native_ordered
    from tidy3.pandas_engine import eval_expr
    from tidy3.verbs import _column_references

    required = set(groups or []) | set().union(*(_column_references(value) for value in factors.values()))
    sources = {name: _level_source(value) for name, value in factors.items()}
    schema = lf.collect_schema()
    lazy = bool(required) and all(
        source is not None and _text_like(lf, source, schema) for source in sources.values()
    )
    if lazy:
        # Only the columns the factors read go to pandas, and the factor
        # values are their source values, so the result is a cast of those
        # columns rather than every row copied back through Python lists.
        pdf = lf.select(sorted(required, key=list(schema.names()).index)).collect().to_pandas()
    else:
        data = lf.collect()
        pdf = data.select([name for name in data.columns if name in required]).to_pandas()
        if not required:
            pdf = pd.DataFrame(index=pd.RangeIndex(data.height))
    flags = {column: bool(getattr(levels, "ordered", False))
             for column, levels in tf._category_levels.items()}
    flags.update(ordered_flags or {})
    flags.update(native_ordered)
    for column, flag in flags.items():
        if column in pdf and isinstance(pdf[column].dtype, pd.CategoricalDtype):
            # An Enum collected to pandas is always ordered. Reapply the
            # current flag, retaining levels from earlier assignment stages.
            pdf[column] = pdf[column].cat.as_ordered() if flag else pdf[column].cat.as_unordered()
    token = SORT_GROUPS.set(sort_groups)
    try:
        results = {name: eval_expr(value, pdf, groups, "window") for name, value in factors.items()}
    finally:
        SORT_GROUPS.reset(token)
    columns = []
    ordered = dict(native_ordered)
    rest = {
        name: value for name, value in stage.items()
        if name not in factors and name not in native_ordered
    }
    if lazy and all(isinstance(value.dtype, pd.CategoricalDtype) for value in results.values()):
        from tidy3.expr import to_polars

        for name, value in results.items():
            ordered[name] = bool(value.cat.ordered)
            levels = pl.Enum(value.cat.categories.tolist())
            columns.append(to_polars(sources[name]).cast(pl.String).cast(levels).alias(name))
        return lf.with_columns(columns), rest, ordered
    if lazy:
        data = lf.collect()
    for name, value in results.items():
        if isinstance(value, pd.Series) and isinstance(value.dtype, pd.CategoricalDtype):
            ordered[name] = bool(value.cat.ordered)
            columns.append(pl.Series(name, value.astype(object).where(value.notna(), None).tolist(), dtype=pl.Enum(value.cat.categories.tolist())))
        else:
            columns.append(pl.Series(name, value))
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
