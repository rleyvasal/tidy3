"""Factor ordering for tidy3 expressions and categorical vectors.

Polars factor mutations compute a snapshot to discover Enum levels. The
result resumes as a lazy frame; other mutations keep their usual lazy path.
"""

from __future__ import annotations

from contextvars import ContextVar
import math
from typing import Any
import warnings

import numpy as np
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
    if not (isinstance(node, tuple) and node[0] == "func" and node[1] in FUNCTIONS):
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
        missing_per_group = _unknown_requested(pairs, rank, requested, categories)
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


def _unknown_requested(pairs: pd.DataFrame, rank: dict[int, int], requested: list[str], categories: list[str]) -> list[list[str]]:
    """The unknown-level lists _apply would warn about, one per distinct list."""
    present = pairs.groupby("gid", sort=False)["code"].agg(frozenset)
    seen: dict[tuple[str, ...], None] = {}
    for codes in present:
        unknown = tuple(level for level in requested if level not in categories or categories.index(level) not in codes)
        if unknown:
            seen.setdefault(unknown, None)
    return [list(unknown) for unknown in seen]


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


def prepare_polars(tf: Any, lf: pl.LazyFrame, stage: dict[str, Any], groups: list[str] | None, *, sort_groups: bool) -> tuple[pl.LazyFrame, dict[str, Any], dict[str, bool]]:
    """Compute factor assignments once; preserve other columns' Polars dtypes.

    The third result maps each produced factor column to its ordered flag.
    Polars Enums have no unordered state, so the flag has to travel beside
    the level names.
    """
    factors = {name: value for name, value in stage.items() if contains_factor(value)}
    if not factors:
        return lf, stage, {}
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
    for column, levels in tf._category_levels.items():
        if column in pdf:
            # An Enum collected to pandas is always ordered. Reapply the
            # recorded flag so fct_* can preserve an unordered factor.
            pdf[column] = _factor(
                pdf[column], list(levels), ordered=bool(getattr(levels, "ordered", False))
            )
    token = SORT_GROUPS.set(sort_groups)
    try:
        results = {name: eval_expr(value, pdf, groups, "window") for name, value in factors.items()}
    finally:
        SORT_GROUPS.reset(token)
    columns = []
    ordered: dict[str, bool] = {}
    rest = {name: value for name, value in stage.items() if name not in factors}
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

    fct_relevel, fct_reorder, fct_infreq, and fct_rev change the levels and
    never the values, so ``fct_rev(fct_infreq(species))`` has the values of
    ``species``.
    """
    node = value.node if isinstance(value, Expr) else None
    if not isinstance(node, tuple) or node[0] != "func" or node[1] not in FUNCTIONS:
        return None
    while isinstance(node, tuple) and node[0] == "func" and node[1] in FUNCTIONS:
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
