"""Eager pandas backend: expression evaluator + verb operations.

Exists for 1:1 engine comparisons (e.g. against datar, which is
pandas-only): the same tidy3 pipeline runs on either engine via
``tidy(df, backend="pandas")``.

Evaluation modes mirror dplyr:
- ``"window"`` (mutate/filter): aggregates broadcast within the group,
  cumulative ops restart per group.
- ``"agg"`` (summarise): aggregates reduce to one row per group.
"""

from __future__ import annotations

import operator
from typing import Any

import numpy as np
import pandas as pd

from tidy3.expr import Expr
from tidy3.join_spec import JoinSpec

_BIN = {
    "+": operator.add, "-": operator.sub, "*": operator.mul,
    "/": operator.truediv, "//": operator.floordiv, "%": operator.mod,
    "**": operator.pow, "==": operator.eq, "!=": operator.ne,
    "<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge,
    "&": operator.and_, "|": operator.or_,
}

# polars method name → pandas reduction name
_AGG = {
    "mean": "mean", "sum": "sum", "min": "min", "max": "max",
    "median": "median", "std": "std", "var": "var", "count": "count",
    "n_unique": "nunique", "first": "first", "last": "last",
    "any": "any", "all": "all",
}

# polars method name → pandas per-row/window method name
_WINDOW = {
    "cum_sum": "cumsum", "cumsum": "cumsum", "cum_max": "cummax",
    "cum_min": "cummin", "cum_prod": "cumprod", "shift": "shift",
    "diff": "diff", "rank": "rank",
}

_ELEM = {
    "abs": lambda s: s.abs(),
    "round": lambda s, d=0: s.round(d),
    "sqrt": lambda s: np.sqrt(s),
    "log": lambda s: np.log(s),
    "log10": lambda s: np.log10(s),
    "exp": lambda s: np.exp(s),
    "floor": lambda s: np.floor(s),
    "ceil": lambda s: np.ceil(s),
    "is_null": lambda s: s.isna(),
    "is_not_null": lambda s: s.notna(),
    "fill_null": lambda s, v: s.fillna(v),
    "alias": lambda s, name: s,  # naming handled by the verbs
}

_GB_KW = {"dropna": False, "observed": True}


def _keys(df: pd.DataFrame, groups: list[str]) -> list[pd.Series]:
    return [df[g] for g in groups]


def _grouped(obj, df: pd.DataFrame, groups: list[str]):
    return obj.groupby(_keys(df, groups), **_GB_KW)


def eval_expr(e: Any, df: pd.DataFrame, groups: list[str] | None, mode: str) -> Any:
    """Evaluate an expression against *df*. Non-Expr values pass through."""
    if not isinstance(e, Expr):
        return e
    return _ev(e.node, df, groups or None, mode)


def _ev(node: tuple, df: pd.DataFrame, groups: list[str] | None, mode: str) -> Any:
    kind = node[0]
    if kind == "col":
        return df[node[1]]
    if kind == "lit":
        return node[1]
    if kind == "pl":
        raise NotImplementedError(
            "raw polars expressions cannot run on the pandas backend; "
            "use col()/helpers or backend='polars'"
        )
    if kind == "n":
        if groups:
            ones = pd.Series(1, index=df.index)
            g = _grouped(ones, df, groups)
            return g.transform("size") if mode == "window" else g.size()
        return len(df)
    if kind == "desc":
        raise ValueError("desc() is only valid inside arrange()")
    if kind == "neg":
        return -_ev(node[1], df, groups, mode)
    if kind == "not":
        return ~_ev(node[1], df, groups, mode)
    if kind == "bin":
        return _BIN[node[1]](_ev(node[2], df, groups, mode), _ev(node[3], df, groups, mode))
    if kind == "horizontal":
        _, operation, columns = node
        if not columns:
            identities = {"sum": 0, "any": False, "all": True}
            return identities.get(operation)
        values = df[list(columns)]
        if operation in {
            "sum",
            "mean",
            "min",
            "max",
            "median",
            "std",
            "any",
            "all",
        }:
            return getattr(values, operation)(axis=1)
        if operation == "first":
            return values.iloc[:, 0]
        if operation == "last":
            return values.iloc[:, -1]
        raise ValueError(f"unknown horizontal operation: {operation}")
    if kind == "column_set":
        _, columns, as_list = node
        values = df[list(columns)]
        if as_list:
            return pd.Series(values.values.tolist(), index=df.index)
        return pd.Series(values.to_dict(orient="records"), index=df.index)
    if kind == "func":
        return _ev_func(node, df, groups, mode)
    if kind == "case_when":
        _, cases, default = node
        result = _as_series(_ev(default, df, groups, mode), df.index)
        for condition_node, value_node in reversed(cases):
            condition = _as_series(
                _ev(condition_node, df, groups, mode), df.index
            )
            value = _as_series(_ev(value_node, df, groups, mode), df.index)
            result = result.mask(condition.fillna(False).astype(bool), value)
        return result.infer_objects()
    if kind == "call":
        return _ev_call(node, df, groups, mode)
    raise ValueError(f"unknown expression node: {node!r}")


def _as_series(value: Any, index: pd.Index) -> pd.Series:
    if isinstance(value, pd.Series):
        return value.reindex(index)
    return pd.Series(value, index=index)


def _ev_func(
    node: tuple,
    df: pd.DataFrame,
    groups: list[str] | None,
    mode: str,
) -> Any:
    _, name, raw_args, raw_kwargs = node
    descending = bool(
        name
        in {"row_number", "min_rank", "dense_rank", "percent_rank", "cume_dist", "ntile"}
        and raw_args
        and raw_args[0][0] == "desc"
    )
    if descending:
        raw_args = (raw_args[0][1], *raw_args[1:])
    args = tuple(_ev(arg, df, groups, mode) for arg in raw_args)
    kwargs = {key: _ev(value, df, groups, mode) for key, value in raw_kwargs.items()}

    if name == "cur_group_id":
        group_names = tuple(kwargs.get("groups", ()))
        if not group_names:
            return pd.Series(1, index=df.index, dtype="Int64")
        # Match the stable group ordering used by tidy3's grouped operations.
        return (
            df.groupby(list(group_names), sort=True, **_GB_KW)
            .ngroup()
            .add(1)
            .astype("Int64")
        )

    if name == "n_groups":
        group_names = tuple(kwargs.get("groups", ()))
        count = (
            int(df.groupby(list(group_names), sort=True, **_GB_KW).ngroups)
            if group_names
            else 1
        )
        return pd.Series(count, index=df.index, dtype="Int64")

    if name == "n_distinct":
        if len(args) == 1:
            value = _as_series(args[0], df.index)
        else:
            values = pd.concat(
                [_as_series(argument, df.index) for argument in args], axis=1
            )
            missing = values.isna().any(axis=1)
            value = values.apply(
                lambda row: tuple(
                    None if pd.isna(item) else item for item in row
                ),
                axis=1,
            )
            if bool(kwargs["na_rm"]):
                value = value.mask(missing)
        dropna = bool(kwargs["na_rm"])
        if groups:
            grouped = _grouped(value, df, groups)
            if mode == "agg":
                return grouped.nunique(dropna=dropna)
            return grouped.transform(lambda series: series.nunique(dropna=dropna))
        return value.nunique(dropna=dropna)

    if name == "nth":
        value = _as_series(args[0], df.index)
        order = args[1]
        position = int(kwargs["n"])
        default = kwargs["default"]
        na_rm = bool(kwargs["na_rm"])

        def choose(positions):
            selected = value.iloc[positions]
            if isinstance(order, pd.Series):
                ordering = order.iloc[positions]
                ranked = ordering.sort_values(kind="stable", na_position="last")
                selected = selected.loc[ranked.index]
            if na_rm:
                selected = selected.dropna()
            index = position - 1 if position > 0 else position
            if position == 0 or abs(index) >= len(selected) + int(index < 0):
                return default
            try:
                return selected.iloc[index]
            except IndexError:
                return default

        if groups:
            grouped = df.groupby(list(groups), sort=False, **_GB_KW)
            values = [choose(pos) for pos in grouped.indices.values()]
            result = pd.Series(values, index=grouped.size().index)
            if mode == "agg":
                return result
            broadcast = pd.Series(index=df.index, dtype="object")
            for positions, selected in zip(grouped.indices.values(), values):
                broadcast.iloc[positions] = selected
            return broadcast.infer_objects()
        return choose(np.arange(len(df)))

    if name == "near":
        return (args[0] - args[1]).abs() <= float(kwargs["tolerance"])

    if name == "na_if":
        value = _as_series(args[0], df.index)
        return value.mask(value == args[1])

    if name == "between":
        value, left, right = args
        bounds = kwargs["bounds"]
        lower = value >= left if bounds[0] == "[" else value > left
        upper = value <= right if bounds[1] == "]" else value < right
        return lower & upper

    if name == "consecutive_id":
        values = pd.concat(
            [_as_series(value, df.index) for value in args], axis=1
        )
        previous = (
            values.groupby(_keys(df, groups), **_GB_KW).shift()
            if groups
            else values.shift()
        )
        equal = values.eq(previous) | (values.isna() & previous.isna())
        changed = ~equal.all(axis=1)
        if groups:
            first = df.groupby(list(groups), sort=False, **_GB_KW).cumcount() == 0
            changed = changed | first
            return changed.groupby(_keys(df, groups), **_GB_KW).cumsum().astype(int)
        if len(changed):
            changed.iloc[0] = True
        return changed.cumsum().astype(int)

    if name in {
        "row_number",
        "min_rank",
        "dense_rank",
        "percent_rank",
        "cume_dist",
        "ntile",
        "lead",
        "lag",
        "cummean",
        "cumall",
        "cumany",
    }:
        if mode != "window":
            raise ValueError(f"{name}() is not valid inside summarise()")
        if name == "row_number" and not args:
            if groups:
                return df.groupby(list(groups), sort=False, **_GB_KW).cumcount() + 1
            return pd.Series(np.arange(1, len(df) + 1), index=df.index)

        value = _as_series(args[0], df.index)
        grouped = _grouped(value, df, groups) if groups else None
        if name in {"row_number", "min_rank", "dense_rank"}:
            method = {
                "row_number": "first",
                "min_rank": "min",
                "dense_rank": "dense",
            }[name]
            if grouped is not None:
                rank = grouped.rank(
                    method=method,
                    na_option="keep",
                    ascending=not descending,
                )
            else:
                rank = value.rank(
                    method=method,
                    na_option="keep",
                    ascending=not descending,
                )
            return rank.astype("Int64")
        if name in {"percent_rank", "cume_dist", "ntile"}:
            method = "max" if name == "cume_dist" else (
                "first" if name == "ntile" else "min"
            )
            rank = (
                grouped.rank(method=method, na_option="keep")
                if grouped is not None
                else value.rank(
                    method=method,
                    na_option="keep",
                    ascending=not descending,
                )
            )
            if grouped is not None and descending:
                rank = grouped.rank(
                    method=method, na_option="keep", ascending=False
                )
            count = (
                grouped.transform("count")
                if grouped is not None
                else pd.Series(value.count(), index=df.index)
            )
            if name == "percent_rank":
                return (rank - 1) / (count - 1)
            if name == "cume_dist":
                return rank / count
            return (
                np.floor((rank - 1) * int(kwargs["n"]) / count) + 1
            ).astype("Int64")
        if name in {"lead", "lag"}:
            periods = -int(kwargs["n"]) if name == "lead" else int(kwargs["n"])
            default = kwargs["default"]
            order = args[1]
            if isinstance(order, pd.Series):
                result = pd.Series(index=df.index, dtype="object")
                position_groups = (
                    df.groupby(list(groups), sort=False, **_GB_KW).indices.values()
                    if groups
                    else [np.arange(len(df))]
                )
                for positions in position_groups:
                    values = value.iloc[positions]
                    ordering = order.iloc[positions]
                    ranked = ordering.sort_values(
                        kind="stable", na_position="last"
                    )
                    shifted = values.loc[ranked.index].shift(
                        periods, fill_value=default
                    )
                    result.loc[shifted.index] = shifted
                return result.infer_objects()
            if grouped is not None:
                return grouped.shift(periods, fill_value=default)
            return value.shift(periods, fill_value=default)
        if name == "cummean":
            missing_seen = (
                _grouped(value.isna(), df, groups).cummax()
                if groups
                else value.isna().cummax()
            )
            if grouped is not None:
                result = grouped.expanding().mean()
                result = result.droplevel(list(range(len(groups))))
                if df.index.is_unique:
                    return result.reindex(df.index).mask(missing_seen)
                positioned = pd.Series(value.to_numpy(), index=range(len(value)))
                keys = [
                    pd.Series(df[group].to_numpy(), index=positioned.index)
                    for group in groups
                ]
                result = positioned.groupby(keys, **_GB_KW).expanding().mean()
                result = result.droplevel(list(range(len(groups)))).sort_index()
                result.index = df.index
                return result.mask(missing_seen.to_numpy())
            return value.expanding().mean().mask(missing_seen)
        operation = "cummin" if name == "cumall" else "cummax"
        if grouped is not None:
            return grouped.transform(operation)
        return getattr(value, operation)()

    if name == "coalesce":
        result = _as_series(args[0], df.index)
        for value in args[1:]:
            result = result.combine_first(_as_series(value, df.index))
        return result.infer_objects()

    if name == "if_else":
        condition = _as_series(args[0], df.index)
        if not isinstance(args[1], pd.Series) and not isinstance(args[2], pd.Series):
            missing = condition.isna()
            values = np.where(
                condition.fillna(False).to_numpy(dtype=bool), args[1], args[2]
            )
            result = pd.Series(values, index=df.index)
            if missing.any():
                missing_value = kwargs["missing"]
                if isinstance(missing_value, pd.Series):
                    result = result.mask(missing, missing_value.reindex(df.index))
                else:
                    result = result.mask(missing, missing_value)
            return result
        true = _as_series(args[1], df.index)
        false = _as_series(args[2], df.index)
        result = false.mask(condition.fillna(False).astype(bool), true)
        missing = condition.isna()
        if missing.any():
            result = result.mask(
                missing, _as_series(kwargs["missing"], df.index)
            )
        return result.infer_objects()

    raise ValueError(f"unknown expression function: {name}")


def _ev_call(node: tuple, df: pd.DataFrame, groups: list[str] | None, mode: str) -> Any:
    _, name, base, args, kwargs = node
    # dplyr nesting: inside an aggregation, inner aggregates broadcast per
    # group (window); only the outermost call reduces. mean(x - mean(x))
    base_mode = "window" if name in _AGG else mode
    b = _ev(base, df, groups, base_mode)
    args = tuple(eval_expr(a, df, groups, base_mode) for a in args)
    kwargs = {k: eval_expr(v, df, groups, base_mode) for k, v in kwargs.items()}

    if name in _AGG:
        pdname = _AGG[name]
        na_rm = bool(kwargs.pop("na_rm", False))
        if not isinstance(b, pd.Series):
            b = pd.Series(b, index=df.index)
        if groups:
            g = _grouped(b, df, groups)
            if pdname in {"first", "last"} and not na_rm:
                index = 0 if pdname == "first" else -1
                if mode == "window":
                    return g.transform(lambda series: series.iloc[index])
                return g.apply(lambda series: series.iloc[index])
            if mode == "window":
                result = g.transform(pdname)
                if na_rm:
                    return result
                missing = g.transform(lambda series: series.isna().any())
            else:
                result = getattr(g, pdname)()
                if na_rm:
                    return result
                missing = g.apply(lambda series: series.isna().any())
            if pdname == "any":
                return result.mask(~result.astype(bool) & missing)
            if pdname == "all":
                return result.mask(result.astype(bool) & missing)
            return result.mask(missing)
        # ungrouped → scalar (broadcasts in window mode, one row in agg mode)
        if pdname == "first":
            values = b.dropna() if na_rm else b
            return values.iloc[0] if len(values) else np.nan
        if pdname == "last":
            values = b.dropna() if na_rm else b
            return values.iloc[-1] if len(values) else np.nan
        result = getattr(b, pdname)()
        if na_rm or not b.isna().any():
            return result
        if pdname == "any" and bool(result):
            return True
        if pdname == "all" and not bool(result):
            return False
        return np.nan

    if name in _WINDOW:
        if mode != "window":
            raise ValueError(f"{name}() is not valid inside summarise()")
        pdname = _WINDOW[name]
        if not isinstance(b, pd.Series):
            b = pd.Series(b, index=df.index)
        if groups:
            result = getattr(_grouped(b, df, groups), pdname)(*args, **kwargs)
            if name in {"cum_sum", "cumsum", "cum_max", "cum_min", "cum_prod"}:
                missing_seen = _grouped(b.isna(), df, groups).cummax()
                result = result.mask(missing_seen)
            return result
        result = getattr(b, pdname)(*args, **kwargs)
        if name in {"cum_sum", "cumsum", "cum_max", "cum_min", "cum_prod"}:
            result = result.mask(b.isna().cummax())
        return result

    if name in _ELEM:
        return _ELEM[name](b, *args, **kwargs)

    raise NotImplementedError(
        f".{name}() is not supported on the pandas backend "
        "(full polars API available with backend='polars')"
    )


# ── verb operations (all take/return plain pandas DataFrames) ───────────


def do_filter(df: pd.DataFrame, predicates: tuple, groups: list[str] | None) -> pd.DataFrame:
    mask = eval_expr(predicates[0], df, groups, "window")
    for p in predicates[1:]:
        mask = mask & eval_expr(p, df, groups, "window")
    if not isinstance(mask, pd.Series):
        mask = pd.Series(mask, index=df.index)
    mask = mask.fillna(False).astype(bool)  # dplyr/polars drop NA predicates
    return df[mask].reset_index(drop=True)


def do_filter_out(
    df: pd.DataFrame, predicates: tuple, groups: list[str] | None
) -> pd.DataFrame:
    """Drop rows matching every predicate; missing predicates are retained."""
    mask = eval_expr(predicates[0], df, groups, "window")
    for predicate in predicates[1:]:
        mask = mask & eval_expr(predicate, df, groups, "window")
    if not isinstance(mask, pd.Series):
        mask = pd.Series(mask, index=df.index)
    keep = ~mask.fillna(False).astype(bool)
    return df[keep].reset_index(drop=True)


_NO_FAST_WINDOW = object()


def _fast_grouped_window(
    expr: Any,
    df: pd.DataFrame,
    grouped: Any,
    groups: list[str],
) -> Any:
    node = expr.node if isinstance(expr, Expr) else None
    if node is None:
        return _NO_FAST_WINDOW
    if (
        node[0] == "call"
        and node[1] in _AGG
        and not node[3]
        and set(node[4]) <= {"na_rm"}
        and node[4].get("na_rm", False)
    ):
        base = node[2]
        if base[0] == "col":
            return grouped[base[1]].transform(_AGG[node[1]])
    if node[0] != "func":
        return _NO_FAST_WINDOW
    _, name, args, kwargs = node
    if name == "row_number" and not args:
        return grouped.cumcount() + 1
    if not args or args[0][0] != "col":
        return _NO_FAST_WINDOW
    column = args[0][1]
    series_group = grouped[column]
    literal_kwargs = {
        key: value[1] if value[0] == "lit" else None
        for key, value in kwargs.items()
    }
    if name in {"lead", "lag"}:
        if len(args) > 1 and not (
            args[1][0] == "lit" and args[1][1] is None
        ):
            return _NO_FAST_WINDOW
        periods = int(literal_kwargs["n"])
        periods = -periods if name == "lead" else periods
        return series_group.shift(periods, fill_value=literal_kwargs["default"])
    if name in {"row_number", "min_rank", "dense_rank"}:
        method = {
            "row_number": "first",
            "min_rank": "min",
            "dense_rank": "dense",
        }[name]
        return series_group.rank(method=method, na_option="keep").astype("Int64")
    if name == "cummean":
        result = series_group.expanding().mean()
        result = result.droplevel(list(range(len(groups))))
        if df.index.is_unique:
            missing_seen = series_group.transform(
                lambda series: series.isna().cummax()
            )
            return result.reindex(df.index).mask(missing_seen)
    return _NO_FAST_WINDOW


def do_mutate(df: pd.DataFrame, kwargs: dict, groups: list[str] | None) -> pd.DataFrame:
    # parallel semantics like polars with_columns: all RHS see the input df.
    # assign() (not copy-then-set) so copy-on-write shares unchanged columns.
    grouped = (
        df.groupby(list(groups), sort=False, **_GB_KW) if groups else None
    )
    new = {}
    for name, value in kwargs.items():
        result = (
            _fast_grouped_window(value, df, grouped, groups)
            if grouped is not None
            else _NO_FAST_WINDOW
        )
        new[name] = (
            eval_expr(value, df, groups, "window")
            if result is _NO_FAST_WINDOW
            else result
        )
    return df.assign(**new)


def _has_agg(x) -> bool:
    """True if the node tree contains an aggregation (n(), mean(), …)."""
    if isinstance(x, Expr):
        return _has_agg(x.node)
    if not (isinstance(x, tuple) and x and isinstance(x[0], str)):
        return False
    if (
        x[0] == "n"
        or (x[0] == "call" and x[1] in _AGG)
        or (x[0] == "func" and x[1] in {"n_distinct", "nth"})
    ):
        return True
    if x[0] == "call":
        _, _, base, args, kwargs = x
        return (
            _has_agg(base)
            or any(_has_agg(a) for a in args)
            or any(_has_agg(v) for v in kwargs.values())
        )
    return any(_has_agg(p) for p in x[1:])


def _fast_summarise(
    df: pd.DataFrame,
    kwargs: dict,
    groups: list[str],
    *,
    observed: bool = True,
) -> pd.DataFrame | None:
    """Single groupby().agg() pass when every kwarg is a plain aggregation.

    This matches hand-written pandas performance (one key factorization for
    all aggregates). Returns None when any expression needs the general
    evaluator (e.g. aggregates nested inside the aggregated expression).
    """
    named: dict[str, tuple[str, str]] = {}
    extra: dict[str, pd.Series] = {}
    for k, v in kwargs.items():
        node = v.node if isinstance(v, Expr) else None
        if node is None:
            return None
        if node[0] == "n":
            extra.setdefault("__tidy3_ones", pd.Series(1, index=df.index))
            named[k] = ("__tidy3_ones", "size")
            continue
        if not (
            node[0] == "call"
            and node[1] in _AGG
            and not node[3]
            and set(node[4]) <= {"na_rm"}
            and node[4].get("na_rm", False)
        ):
            return None
        base = node[2]
        if _has_agg(base):
            return None  # e.g. mean(x - mean(x)): inner mean is per-group
        if base[0] == "col":
            src = base[1]
        else:
            src = f"__tidy3_b_{k}"
            try:
                s = _ev(base, df, None, "window")
            except NotImplementedError:
                return None
            if not isinstance(s, pd.Series):
                return None
            extra[src] = s
        named[k] = (src, _AGG[node[1]])
    work = df.assign(**extra) if extra else df
    out = (
        work.groupby(
            list(groups),
            sort=False,
            dropna=False,
            observed=observed,
        )
        .agg(**{k: pd.NamedAgg(column=c, aggfunc=f) for k, (c, f) in named.items()})
        .reset_index()
    )
    return out


def do_summarise(
    df: pd.DataFrame,
    kwargs: dict,
    groups: list[str] | None,
    *,
    sort_groups: bool = True,
    observed: bool = True,
) -> pd.DataFrame:
    if groups:
        fast = _fast_summarise(
            df, kwargs, groups, observed=observed
        )
        if fast is not None:
            if sort_groups:
                fast = fast.sort_values(
                    list(groups), kind="stable", na_position="last"
                ).reset_index(drop=True)
            return fast
    vals = {k: eval_expr(v, df, groups, "agg") for k, v in kwargs.items()}
    if groups:
        series = {k: v for k, v in vals.items() if isinstance(v, pd.Series)}
        if series:
            out = pd.DataFrame(series)
        else:  # all-scalar aggs still need the group index
            out = pd.DataFrame(index=_grouped(pd.Series(1, index=df.index), df, groups).size().index)
        for k, v in vals.items():
            if k not in series:
                out[k] = v
        out = out[list(vals.keys())].reset_index()
        if sort_groups:
            out = out.sort_values(
                list(groups), kind="stable", na_position="last"
            ).reset_index(drop=True)
        else:
            marker = "__tidy3_group_order"
            while marker in out.columns:
                marker += "_"
            order = df.loc[:, list(groups)].drop_duplicates().assign(
                **{marker: lambda frame: range(len(frame))}
            )
            out = (
                out.merge(order, on=list(groups), how="left", sort=False)
                .sort_values(marker, kind="stable")
                .drop(columns=marker)
                .reset_index(drop=True)
            )
        return out
    return pd.DataFrame({k: [v] for k, v in vals.items()})


def _reframe_values(value: Any) -> list[Any]:
    if isinstance(value, pd.Series):
        return value.tolist()
    if isinstance(value, (pd.Index, np.ndarray, list, tuple)):
        return list(value)
    return [value]


def do_reframe(
    df: pd.DataFrame,
    kwargs: dict,
    groups: list[str] | None,
    *,
    sort_groups: bool = True,
) -> pd.DataFrame:
    """Return an arbitrary number of rows per group, always ungrouped."""
    pieces = []
    for piece in _group_pieces(df, groups):
        values = {
            name: _reframe_values(eval_expr(expr, piece, None, "agg"))
            for name, expr in kwargs.items()
        }
        lengths = [len(value) for value in values.values()]
        if not lengths:
            continue
        non_scalar = {length for length in lengths if length != 1}
        if len(non_scalar) > 1:
            raise ValueError("reframe() outputs must have compatible sizes")
        size = next(iter(non_scalar), 1)
        if size == 0 and any(length > 1 for length in lengths):
            raise ValueError("reframe() outputs must have compatible sizes")
        expanded = {
            name: value * size if len(value) == 1 and size != 1 else value
            for name, value in values.items()
        }
        if groups:
            keys = {group: [piece.iloc[0][group]] * size for group in groups}
            expanded = {**keys, **expanded}
        pieces.append(pd.DataFrame(expanded))
    if pieces:
        output = pd.concat(pieces, ignore_index=True)
        if groups and sort_groups:
            output = output.sort_values(
                list(groups), kind="stable", na_position="last"
            ).reset_index(drop=True)
        return output
    return pd.DataFrame(columns=[*(groups or []), *kwargs.keys()])


def do_select(df: pd.DataFrame, cols: tuple) -> pd.DataFrame:
    bad = [c for c in cols if not isinstance(c, str)]
    if bad:
        raise TypeError(
            "pandas backend select() takes column names; "
            "use mutate() for computed columns"
        )
    return df[list(cols)]


def do_arrange(df: pd.DataFrame, keys: tuple) -> pd.DataFrame:
    by: list[str] = []
    asc: list[bool] = []
    tmp: dict[str, Any] = {}
    for i, k in enumerate(keys):
        node = k.node if isinstance(k, Expr) else k
        if isinstance(node, tuple) and node[0] == "desc":
            inner = node[1]
            if inner[0] == "col":
                by.append(inner[1])
            else:
                name = f"__tidy3_sort_{i}"
                tmp[name] = _ev(inner, df, None, "window")
                by.append(name)
            asc.append(False)
        elif isinstance(k, str):
            by.append(k)
            asc.append(True)
        elif isinstance(k, Expr):
            name = f"__tidy3_sort_{i}"
            tmp[name] = _ev(node, df, None, "window")
            by.append(name)
            asc.append(True)
        else:
            raise TypeError(f"arrange(): unsupported key {k!r} on pandas backend")
    work = df.assign(**tmp) if tmp else df
    out = work.sort_values(by, ascending=asc, kind="stable")
    if tmp:
        out = out.drop(columns=list(tmp))
    return out.reset_index(drop=True)


def do_distinct(df: pd.DataFrame, cols: tuple) -> pd.DataFrame:
    return df.drop_duplicates(subset=list(cols) or None).reset_index(drop=True)


def do_head(df: pd.DataFrame, n: int, groups: list[str] | None) -> pd.DataFrame:
    if groups:
        return df.groupby(list(groups), **_GB_KW).head(n).reset_index(drop=True)
    return df.head(n).reset_index(drop=True)


def _group_pieces(df: pd.DataFrame, groups: list[str] | None):
    if not groups:
        return [df]
    grouped = df.groupby(list(groups), sort=False, **_GB_KW)
    return [df.iloc[positions] for positions in grouped.indices.values()]


def do_slice(
    df: pd.DataFrame, positions: tuple[int, ...], groups: list[str] | None
) -> pd.DataFrame:
    if not any(positions):
        return df.iloc[0:0].copy().reset_index(drop=True)
    positive = [position - 1 for position in positions if position > 0]
    excluded = {abs(position) - 1 for position in positions if position < 0}
    pieces = []
    for group in _group_pieces(df, groups):
        if positive:
            valid = [position for position in positive if position < len(group)]
            pieces.append(group.iloc[valid])
        else:
            keep = [position for position in range(len(group)) if position not in excluded]
            pieces.append(group.iloc[keep])
    if not pieces:
        return df.iloc[0:0].copy().reset_index(drop=True)
    return pd.concat(pieces, ignore_index=True)


def do_slice_size(
    df: pd.DataFrame,
    n: int | None,
    prop: float | None,
    groups: list[str] | None,
    *,
    tail: bool,
) -> pd.DataFrame:
    pieces = []
    for group in _group_pieces(df, groups):
        value = n if n is not None else prop
        size = _sample_size(len(group), value, fraction=prop is not None)
        pieces.append(group.tail(size) if tail and size else group.head(size))
    if not pieces:
        return df.iloc[0:0].copy().reset_index(drop=True)
    return pd.concat(pieces, ignore_index=True)


def do_slice_extreme(
    df: pd.DataFrame,
    order_by: Any,
    n: int | None,
    prop: float | None,
    groups: list[str] | None,
    *,
    largest: bool,
    with_ties: bool,
    na_rm: bool,
) -> pd.DataFrame:
    order = df[order_by] if isinstance(order_by, str) else eval_expr(
        order_by, df, groups, "window"
    )
    if not isinstance(order, pd.Series):
        order = pd.Series(order, index=df.index)
    marker = "__tidy3_order"
    while marker in df.columns:
        marker += "_"
    work = df.assign(**{marker: order})
    pieces = []
    for group in _group_pieces(work, groups):
        if na_rm:
            group = group[group[marker].notna()]
        group = group.sort_values(
            marker, ascending=not largest, na_position="last", kind="stable"
        )
        value = n if n is not None else prop
        size = _sample_size(len(group), value, fraction=prop is not None)
        if size == 0:
            pieces.append(group.head(0))
            continue
        if with_ties and size < len(group):
            threshold = group[marker].iloc[size - 1]
            if pd.isna(threshold):
                chosen = group
            elif largest:
                chosen = group[group[marker] >= threshold]
            else:
                chosen = group[group[marker] <= threshold]
            pieces.append(chosen)
        else:
            pieces.append(group.head(size))
    if not pieces:
        return df.iloc[0:0].copy().reset_index(drop=True)
    return pd.concat(pieces, ignore_index=True).drop(columns=marker)


def _sample_size(length: int, value: int | float, *, fraction: bool) -> int:
    if fraction:
        raw = length * (value if value >= 0 else 1 + value)
        size = int(raw)  # dplyr: truncate proportions towards zero
    else:
        size = int(value if value >= 0 else length + value)
    return min(length, max(0, size))


def _sample_groups(
    df: pd.DataFrame,
    groups: list[str],
    size,
    seed: int | None,
) -> pd.DataFrame:
    """Sample each group by positional index without losing group columns."""
    pieces = []
    grouped = df.groupby(list(groups), sort=False, **_GB_KW)
    for positions in grouped.indices.values():
        group = df.iloc[positions]
        pieces.append(group.sample(n=size(len(group)), random_state=seed))
    if not pieces:
        return df.iloc[0:0].copy().reset_index(drop=True)
    return pd.concat(pieces, ignore_index=True)


def do_sample_n(df: pd.DataFrame, n: int, seed: int | None, groups: list[str] | None) -> pd.DataFrame:
    if groups:
        return _sample_groups(
            df,
            groups,
            lambda length: _sample_size(length, n, fraction=False),
            seed,
        )
    size = _sample_size(len(df), n, fraction=False)
    return df.sample(n=size, random_state=seed).reset_index(drop=True)


def do_sample_frac(df: pd.DataFrame, frac: float, seed: int | None, groups: list[str] | None) -> pd.DataFrame:
    if groups:
        return _sample_groups(
            df,
            groups,
            lambda length: _sample_size(length, frac, fraction=True),
            seed,
        )
    size = _sample_size(len(df), frac, fraction=True)
    return df.sample(n=size, random_state=seed).reset_index(drop=True)


def do_slice_sample(
    df: pd.DataFrame,
    n: int | None,
    prop: float | None,
    weight_by: Any,
    replace: bool,
    seed: int | None,
    groups: list[str] | None,
) -> pd.DataFrame:
    weights = None
    if weight_by is not None:
        weights = (
            df[weight_by]
            if isinstance(weight_by, str)
            else eval_expr(weight_by, df, groups, "window")
        )
        if not isinstance(weights, pd.Series):
            weights = pd.Series(weights, index=df.index)
    pieces = []
    for group in _group_pieces(df, groups):
        value = n if n is not None else prop
        size = _sample_size(len(group), value, fraction=prop is not None)
        if replace:
            if n is not None:
                size = max(0, n if n >= 0 else len(group) + n)
            else:
                factor = prop if prop >= 0 else 1.0 + prop
                size = max(0, int(len(group) * factor))
        group_weights = weights.loc[group.index] if weights is not None else None
        pieces.append(
            group.sample(
                n=size,
                replace=replace,
                weights=group_weights,
                random_state=seed,
            )
        )
    if not pieces:
        return df.iloc[0:0].copy().reset_index(drop=True)
    return pd.concat(pieces, ignore_index=True)


def do_count(
    df: pd.DataFrame,
    cols: tuple,
    name: str,
    *,
    wt: Any = None,
    sort: bool = False,
    observed: bool = True,
) -> pd.DataFrame:
    values = None
    if wt is not None:
        values = df[wt] if isinstance(wt, str) else eval_expr(wt, df, None, "window")
        if not isinstance(values, pd.Series):
            values = pd.Series(values, index=df.index)
    if cols:
        if values is None:
            counts = df.groupby(
                list(cols),
                sort=False,
                dropna=False,
                observed=observed,
            ).size()
        else:
            counts = values.groupby(
                _keys(df, list(cols)),
                sort=False,
                dropna=False,
                observed=observed,
            ).sum(min_count=0)
        out = counts.reset_index(name=name)
        out = out.sort_values(
            list(cols), kind="stable", na_position="last"
        ).reset_index(drop=True)
    else:
        value = len(df) if values is None else values.sum()
        out = pd.DataFrame({name: [value]})
    if sort:
        out = out.sort_values(name, ascending=False, kind="stable").reset_index(drop=True)
    return out


def do_add_count(
    df: pd.DataFrame,
    cols: tuple[str, ...],
    name: str,
    *,
    wt: Any = None,
    sort: bool = False,
) -> pd.DataFrame:
    """Add a group count without collapsing rows."""
    if wt is None:
        values = pd.Series(1, index=df.index)
        counts = (
            _grouped(values, df, list(cols)).transform("size")
            if cols
            else len(df)
        )
    else:
        values = df[wt] if isinstance(wt, str) else eval_expr(wt, df, None, "window")
        if not isinstance(values, pd.Series):
            values = pd.Series(values, index=df.index)
        counts = (
            _grouped(values, df, list(cols)).transform("sum")
            if cols
            else values.sum()
        )
    out = df.assign(**{name: counts})
    if sort:
        out = out.sort_values(name, ascending=False, kind="stable")
    return out.reset_index(drop=True)


def do_join(
    df: pd.DataFrame, right: pd.DataFrame, on, how: str, **kwargs
) -> pd.DataFrame:
    if how == "outer":
        left_order = "__tidy3_left_order"
        right_order = "__tidy3_right_order"
        occupied = set(df.columns) | set(right.columns)
        while left_order in occupied:
            left_order += "_"
        occupied.add(left_order)
        while right_order in occupied:
            right_order += "_"
        left = df.assign(**{left_order: range(len(df))})
        other = right.assign(**{right_order: range(len(right))})
        params = {
            "how": how,
            "suffixes": ("", "_right"),
            "on": on,
            **kwargs,
        }
        return (
            left.merge(other, **params)
            .sort_values(
                [left_order, right_order],
                kind="stable",
                na_position="last",
            )
            .drop(columns=[left_order, right_order])
            .reset_index(drop=True)
        )
    params = {"how": how, "suffixes": ("", "_right"), **kwargs}
    if how != "cross":
        params["on"] = on
    return df.merge(right, **params)


def do_filter_join(
    df: pd.DataFrame, right: pd.DataFrame, on, *, anti: bool
) -> pd.DataFrame:
    """Semi/anti join while preserving every left column and its row order."""
    keys = [on] if isinstance(on, str) else list(on)
    marker = "__tidy3_match"
    while marker in df.columns or marker in right.columns:
        marker += "_"
    matches = right[keys].drop_duplicates().assign(**{marker: True})
    merged = df.merge(matches, on=keys, how="left", sort=False)
    mask = merged[marker].isna() if anti else merged[marker].notna()
    return merged.loc[mask, list(df.columns)].reset_index(drop=True)


def do_rows(
    df: pd.DataFrame,
    right: pd.DataFrame,
    keys: list[str],
    operation: str,
    *,
    conflict: str = "error",
    unmatched: str = "error",
) -> pd.DataFrame:
    """SQL-like row insertion, update, patch, upsert, and deletion."""
    columns = list(df.columns)
    extra = [column for column in right.columns if column not in columns]
    if extra:
        raise ValueError(f"{operation}(): y has columns absent from x: {extra}")
    missing_keys = [key for key in keys if key not in columns or key not in right.columns]
    if missing_keys:
        raise KeyError(f"{operation}(): key columns not found: {missing_keys}")
    if operation in {"rows_update", "rows_patch", "rows_upsert"}:
        if right.duplicated(keys).any():
            raise ValueError(f"{operation}(): y keys must be unique")

    marker = "__tidy3_match"
    while marker in columns or marker in right.columns:
        marker += "_"
    x_keys = df[keys].drop_duplicates().assign(**{marker: True})
    checked = right.merge(x_keys, on=keys, how="left", sort=False)
    matched = checked[marker].notna()

    if operation == "rows_append":
        additions = right
    elif operation == "rows_insert":
        if conflict == "error" and matched.any():
            raise ValueError("rows_insert(): y contains keys that already exist in x")
        additions = checked.loc[~matched, list(right.columns)]
    elif operation in {"rows_update", "rows_patch", "rows_delete"}:
        if unmatched == "error" and (~matched).any():
            raise ValueError(f"{operation}(): y contains keys absent from x")
        right = checked.loc[matched, list(right.columns)]
    elif operation != "rows_upsert":
        raise ValueError(f"unknown row operation: {operation}")

    if operation in {"rows_append", "rows_insert"}:
        return pd.concat([df, additions], ignore_index=True, sort=False)[columns]
    if operation == "rows_delete":
        return do_filter_join(df, right[keys], keys, anti=True)[columns]

    update_columns = [column for column in right.columns if column not in keys]
    update_marker = marker + "_update"
    payload = right.assign(**{update_marker: True})
    merged = df.merge(
        payload,
        on=keys,
        how="left",
        suffixes=("", "__tidy3_y"),
        sort=False,
    )
    for column in update_columns:
        incoming = f"{column}__tidy3_y"
        if operation == "rows_patch":
            merged[column] = merged[column].where(merged[column].notna(), merged[incoming])
        else:
            merged[column] = merged[column].where(
                merged[update_marker].isna(), merged[incoming]
            )
    updated = merged[columns]
    if operation != "rows_upsert":
        return updated.reset_index(drop=True)

    additions = checked.loc[~matched, list(right.columns)]
    return pd.concat([updated, additions], ignore_index=True, sort=False)[columns]


def do_join_by(
    df: pd.DataFrame,
    right: pd.DataFrame,
    spec: JoinSpec,
    how: str,
    *,
    suffix: str = "_right",
    keep: bool = False,
    na_matches: str = "na",
    multiple: str = "all",
    unmatched: str = "drop",
    relationship: str | None = None,
) -> pd.DataFrame:
    """Execute equality/inequality/rolling/overlap join specifications."""
    left_columns = list(df.columns)
    right_columns = list(right.columns)
    for condition in spec.conditions:
        if condition.left not in left_columns:
            raise KeyError(f"join_by(): left column not found: {condition.left!r}")
        if condition.right not in right_columns:
            raise KeyError(f"join_by(): right column not found: {condition.right!r}")

    occupied = [*left_columns, *right_columns]
    left_id = "__tidy3_left_row"
    while left_id in occupied:
        left_id += "_"
    right_id = "__tidy3_right_row"
    while right_id in occupied or right_id == left_id:
        right_id += "_"
    right_names = {}
    for column in right_columns:
        name = f"__tidy3_y_{column}"
        while name in occupied or name in right_names.values():
            name += "_"
        right_names[column] = name

    equality = spec.equality
    left_data = df.copy(deep=False)
    right_data = right.copy(deep=False)
    for condition in equality:
        left_key = left_data[condition.left]
        right_key = right_data[condition.right]
        if left_key.isna().all() or right_key.isna().all():
            left_data = left_data.copy(deep=False)
            right_data = right_data.copy(deep=False)
            left_data[condition.left] = left_key.astype("object")
            right_data[condition.right] = right_key.astype("object")

    left = left_data.reset_index(drop=True).assign(**{left_id: range(len(df))})
    y = right_data.rename(columns=right_names).reset_index(drop=True)
    y[right_id] = range(len(y))
    if equality:
        candidates = left.merge(
            y,
            left_on=[condition.left for condition in equality],
            right_on=[right_names[condition.right] for condition in equality],
            how="inner",
            sort=False,
        )
        if na_matches == "never" and not candidates.empty:
            valid = pd.Series(True, index=candidates.index)
            for condition in equality:
                valid &= candidates[condition.left].notna()
                valid &= candidates[right_names[condition.right]].notna()
            candidates = candidates[valid]
    else:
        candidates = left.merge(y, how="cross", sort=False)

    operators = {
        ">": lambda a, b: a > b,
        ">=": lambda a, b: a >= b,
        "<": lambda a, b: a < b,
        "<=": lambda a, b: a <= b,
    }
    for condition in spec.inequality:
        mask = operators[condition.operator](
            candidates[condition.left], candidates[right_names[condition.right]]
        )
        candidates = candidates[mask.fillna(False)]

    rolling = next((condition for condition in spec.conditions if condition.rolling), None)
    if rolling is not None and not candidates.empty:
        target = candidates[right_names[rolling.right]]
        grouped = target.groupby(candidates[left_id], sort=False)
        extreme = (
            grouped.transform("max")
            if rolling.operator in {">", ">="}
            else grouped.transform("min")
        )
        candidates = candidates[target == extreme]

    left_counts = candidates[left_id].value_counts()
    right_counts = candidates[right_id].value_counts()
    if relationship in {"one-to-one", "many-to-one"} and (left_counts > 1).any():
        raise ValueError(
            f"join relationship {relationship!r} violated: "
            "a row in x matches multiple rows in y"
        )
    if relationship in {"one-to-one", "one-to-many"} and (right_counts > 1).any():
        raise ValueError(
            f"join relationship {relationship!r} violated: "
            "a row in y matches multiple rows in x"
        )

    all_matched_left = set(candidates[left_id])
    all_matched_right = set(candidates[right_id])
    if unmatched == "error":
        if how in {"inner", "right"} and not set(left[left_id]).issubset(
            all_matched_left
        ):
            raise ValueError("join unmatched='error': rows in x have no match in y")
        if how in {"inner", "left"} and not set(y[right_id]).issubset(
            all_matched_right
        ):
            raise ValueError("join unmatched='error': rows in y have no match in x")

    if multiple != "all":
        candidates = candidates.drop_duplicates(
            subset=[left_id], keep="last" if multiple == "last" else "first"
        )

    matched_left = candidates[[left_id]].drop_duplicates()
    if how in {"semi", "anti"}:
        marked = left.merge(
            matched_left.assign(__tidy3_matched=True), on=left_id, how="left"
        )
        keep = marked["__tidy3_matched"].notna()
        if how == "anti":
            keep = ~keep
        return marked.loc[keep, left_columns].reset_index(drop=True)

    matched_right = candidates[[right_id]].drop_duplicates()
    pieces = [candidates]
    if how in {"left", "full"}:
        missing_left = left.merge(matched_left, on=left_id, how="left", indicator=True)
        missing_left = missing_left[missing_left["_merge"] == "left_only"].drop(
            columns="_merge"
        )
        for column in [*right_names.values(), right_id]:
            missing_left[column] = np.nan
        pieces.append(missing_left)
    if how in {"right", "full"}:
        missing_right = y.merge(matched_right, on=right_id, how="left", indicator=True)
        missing_right = missing_right[missing_right["_merge"] == "left_only"].drop(
            columns="_merge"
        )
        for column in [*left_columns, left_id]:
            missing_right[column] = np.nan
        pieces.append(missing_right)
    combined = pd.concat(pieces, ignore_index=True, sort=False)
    if how == "right":
        combined = combined.sort_values([right_id, left_id], kind="stable")
    else:
        combined = combined.sort_values([left_id, right_id], kind="stable")

    equality_map = {condition.left: condition.right for condition in equality}
    equality_right = {condition.right for condition in equality}
    output: dict[str, Any] = {}
    for column in left_columns:
        if column in equality_map and not keep:
            output[column] = combined[column].combine_first(
                combined[right_names[equality_map[column]]]
            )
        else:
            output[column] = combined[column]
    for column in right_columns:
        if column in equality_right and not keep:
            continue
        name = column + suffix if column in left_columns else column
        output[name] = combined[right_names[column]]
    return pd.DataFrame(output).reset_index(drop=True)
