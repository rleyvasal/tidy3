"""Factor ordering for tidy3 expressions and categorical vectors.

Polars factor mutations compute a snapshot to discover Enum levels. The
result resumes as a lazy frame; other mutations keep their usual lazy path.
"""

from __future__ import annotations

from contextvars import ContextVar
import math
from typing import Any
import warnings

import pandas as pd
import polars as pl

from tidy3.expr import Expr, _func, col, median

__all__ = ["fct_relevel", "fct_reorder", "fct_infreq", "fct_rev"]
FUNCTIONS = frozenset(__all__)
SORT_GROUPS: ContextVar[bool] = ContextVar("tidy3_factor_sort_groups", default=True)


def _column(value: Any) -> Any:
    return col(value) if isinstance(value, str) else value


def _dispatch(name: str, f: Any, *args: Any, **kwargs: Any) -> Any:
    if isinstance(f, (str, Expr, pl.Expr)):
        return _func(name, _column(f), *args, **kwargs)
    source = _factor(f)
    values = tuple(pd.Series(value) if not isinstance(value, pd.Series) else value for value in args)
    return eval_pandas(name, (source, *values), kwargs).array


def fct_relevel(f: Any, *levels: Any, after: int | float = 0) -> Any:
    """Move named levels (or a callable's result) after *after* other levels.

    Unmentioned levels keep their order. ``after=float('inf')`` moves to
    the end. Unknown levels warn and are ignored, as in forcats.
    """
    if isinstance(after, bool) or not isinstance(after, (int, float)) or after < 0 or math.isnan(after) or (math.isfinite(after) and after != int(after)):
        raise ValueError("fct_relevel() after must be a non-negative integer or infinity")
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


def _factor(value: Any, levels: list[str] | None = None) -> pd.Series:
    if isinstance(value, pl.Series):
        if isinstance(value.dtype, pl.Enum):
            levels = value.dtype.categories.to_list()
        value = value.to_pandas()
    source = value if isinstance(value, pd.Series) else pd.Series(value)
    if isinstance(source.dtype, pd.CategoricalDtype):
        if not all(isinstance(level, str) for level in source.cat.categories):
            raise TypeError("factor levels must be strings")
        if levels is not None:
            return pd.Series(pd.Categorical(source, categories=levels, ordered=source.cat.ordered), index=source.index, name=source.name)
        return source
    present = source.dropna()
    if not all(isinstance(value, str) for value in present):
        raise TypeError("factor input must be a character or categorical column")
    categories = sorted(present.unique().tolist()) if levels is None else levels
    return pd.Series(pd.Categorical(source, categories=categories), index=source.index, name=source.name)


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


def eval_pandas(name: str, args: tuple[Any, ...], kw: dict[str, Any]) -> pd.Series:
    source = _factor(args[0])
    extra = []
    for value in args[1:]:
        series = value if isinstance(value, pd.Series) else pd.Series(value)
        if len(series) != len(source):
            raise ValueError(f"{name}() inputs must have the same length")
        extra.append(series.set_axis(source.index))
    return _apply(name, source, tuple(extra), kw)


def evaluate_grouped(node: tuple, df: pd.DataFrame, groups: list[str], mode: str) -> pd.Series:
    from tidy3.pandas_engine import eval_expr

    parts = df.groupby(groups, sort=SORT_GROUPS.get(), dropna=False, observed=True).indices.values()
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


def contains_factor(value: Any) -> bool:
    node = value.node if isinstance(value, Expr) else value
    if not isinstance(node, tuple) or not node or not isinstance(node[0], str):
        return False
    if node[0] == "func" and node[1] in FUNCTIONS:
        return True
    from tidy3.verbs import _expression_children

    return any(contains_factor(child) for child in _expression_children(node))


def prepare_polars(tf: Any, lf: pl.LazyFrame, stage: dict[str, Any], groups: list[str] | None, *, sort_groups: bool) -> tuple[pl.LazyFrame, dict[str, Any]]:
    """Compute factor assignments once; preserve other columns' Polars dtypes."""
    factors = {name: value for name, value in stage.items() if contains_factor(value)}
    if not factors:
        return lf, stage
    from tidy3.pandas_engine import eval_expr
    from tidy3.verbs import _column_references

    data = lf.collect()
    required = set(groups or []) | set().union(*(_column_references(value) for value in factors.values()))
    pdf = data.select([name for name in data.columns if name in required]).to_pandas()
    if not required:
        pdf = pd.DataFrame(index=pd.RangeIndex(data.height))
    for column, levels in tf._category_levels.items():
        if column in pdf and not isinstance(data.schema[column], pl.Enum):
            pdf[column] = _factor(pdf[column], levels)
    token = SORT_GROUPS.set(sort_groups)
    try:
        results = {name: eval_expr(value, pdf, groups, "window") for name, value in factors.items()}
    finally:
        SORT_GROUPS.reset(token)
    columns = []
    for name, value in results.items():
        if isinstance(value, pd.Series) and isinstance(value.dtype, pd.CategoricalDtype):
            columns.append(pl.Series(name, value.astype(object).where(value.notna(), None).tolist(), dtype=pl.Enum(value.cat.categories.tolist())))
        else:
            columns.append(pl.Series(name, value))
    return data.with_columns(columns).lazy(), {name: value for name, value in stage.items() if name not in factors}
