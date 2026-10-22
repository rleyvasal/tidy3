"""pandas evaluation of tidy3 expressions, used only by the factor fallback.

tidy3 runs on Polars. Factor calls that :func:`tidy3.forcats.prepare_polars`
cannot compute natively (nested calls, callables, NaN values) evaluate their
arguments here on a pandas copy of the needed columns.

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

_BIN = {
    "+": operator.add, "-": operator.sub, "*": operator.mul,
    "/": operator.truediv, "//": operator.floordiv, "%": operator.mod,
    "**": operator.pow, "==": operator.eq, "!=": operator.ne,
    "<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge,
    "&": operator.and_, "|": operator.or_,
}

_COMPARE = frozenset({"==", "!=", "<", "<=", ">", ">="})


def _is_missing(value: Any) -> Any:
    if isinstance(value, pd.Series):
        return value.isna()
    return value is None or value is pd.NA or (
        isinstance(value, float) and value != value
    )


def _compare(op: str, left: Any, right: Any) -> Any:
    """Compare like R and Polars: a missing operand gives a missing result.

    NumPy comparisons give False for NaN/None; ``filter`` would drop those
    rows either way, but ``mutate`` and ``when_any`` must see NA.
    """
    result = _BIN[op](left, right)
    if not isinstance(result, pd.Series):
        return result
    missing = _is_missing(left) | _is_missing(right)
    if isinstance(missing, pd.Series):
        if not missing.any():
            return result
    elif not missing:
        return result
    return result.astype("boolean").mask(missing, pd.NA)


def _as_logical(value: Any) -> Any:
    if isinstance(value, pd.Series):
        return value if str(value.dtype) == "boolean" else value.astype("boolean")
    return pd.NA if _is_missing(value) is True else value


def _logic(op: str, left: Any, right: Any) -> Any:
    """``&`` / ``|`` with three-valued logic (TRUE | NA is TRUE, as in R)."""
    if not isinstance(left, pd.Series) and not isinstance(right, pd.Series):
        return _BIN[op](left, right)
    result = _BIN[op](_as_logical(left), _as_logical(right))
    if isinstance(result, pd.Series) and not result.isna().any():
        return result.astype(bool)
    return result


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
    # Object columns: fill without fillna's deprecated silent downcast.
    "fill_null": lambda s, v: (
        s.mask(s.isna(), v).infer_objects() if s.dtype == object else s.fillna(v)
    ),
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
    if kind == "name":  # a bare notebook name: column first (dplyr)
        return df[node[1]] if node[1] in df.columns else _ev(node[2], df, groups, mode)
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
        if node[1] in {"+", "-"}:
            from tidy3.dates import Period, shift_pandas

            left_period = (
                node[2][1]
                if node[2][0] == "lit" and isinstance(node[2][1], Period)
                else None
            )
            right_period = (
                node[3][1]
                if node[3][0] == "lit" and isinstance(node[3][1], Period)
                else None
            )
            if left_period is not None or right_period is not None:
                if left_period is not None and right_period is not None:
                    raise TypeError(
                        "add months() or years() to a date, not to another period"
                    )
                if left_period is not None:
                    if node[1] == "-":
                        raise TypeError(
                            "cannot subtract a date from months() or years()"
                        )
                    side = _ev(node[3], df, groups, mode)
                    if not isinstance(side, pd.Series):
                        side = pd.Series(side, index=df.index)
                    return shift_pandas(
                        side, _resolve_period(left_period, df, groups, mode)
                    )
                period = right_period if node[1] == "+" else -right_period
                side = _ev(node[2], df, groups, mode)
                if not isinstance(side, pd.Series):
                    side = pd.Series(side, index=df.index)
                return shift_pandas(side, _resolve_period(period, df, groups, mode))
        left = _ev(node[2], df, groups, mode)
        right = _ev(node[3], df, groups, mode)
        if node[1] in _COMPARE:
            return _compare(node[1], left, right)
        if node[1] in {"&", "|"}:
            return _logic(node[1], left, right)
        return _BIN[node[1]](left, right)
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


def _resolve_period(period: Any, df: pd.DataFrame, groups: list[str] | None, mode: str) -> Any:
    """Evaluate column counts in ``months(col("k"))`` against *df*."""
    import polars as pl

    from tidy3.dates import Period

    def count(value: Any) -> Any:
        if isinstance(value, Expr):
            return _as_series(_ev(value.node, df, groups, mode), df.index)
        if isinstance(value, pl.Expr):
            raise TypeError(
                "months() and years() take col() on the pandas backend, not a Polars expression"
            )
        return value

    return Period(months=count(period.months), years=count(period.years))


def _as_series(value: Any, index: pd.Index) -> pd.Series:
    if isinstance(value, pd.Series):
        return value.reindex(index)
    return pd.Series(value, index=index)


def _eval_column_function(
    name: str, raw_args: tuple, args: tuple, kwargs: dict, df: pd.DataFrame
) -> Any:
    """Run a helper, conversion, or date function from tidy3's own modules."""
    from tidy3.convert import FUNCTIONS as _CONVERT_FUNCTIONS
    from tidy3.convert import eval_pandas as _eval_convert
    from tidy3.dates import FUNCTIONS as _DATE_FUNCTIONS
    from tidy3.dates import eval_pandas as _eval_date
    from tidy3.helpers import eval_pandas as _eval_helper

    if name in _CONVERT_FUNCTIONS:
        from tidy3.convert import _OUTPUT

        source = raw_args[0][1] if raw_args and raw_args[0][0] in {"col", "name"} else None
        column = source or _OUTPUT.get() or "<expression>"
        return _eval_convert(name, args, kwargs, df, column=column)
    if name in _DATE_FUNCTIONS:
        return _eval_date(name, args, kwargs, df)
    return _eval_helper(name, args, kwargs, df)


def _nth_per_group(
    value: pd.Series, order: Any, grouped: Any, position: int, default: Any, na_rm: bool
) -> list[Any]:
    """nth() for every group at once: the same choice as choose() per group.

    Rows are sorted by group (then by order_by, missing last, ties in row
    order); missing values are dropped when na_rm; the row at the requested
    rank is picked, or the default when the group is too short.
    """
    gid = grouped.ngroup().to_numpy()
    groups = len(grouped.size())
    work = pd.DataFrame({"gid": gid, "row": np.arange(len(value))})
    keys = ["gid", "row"]
    if isinstance(order, pd.Series):
        work["order"] = order.to_numpy()
        keys = ["gid", "order", "row"]
    work = work.sort_values(keys, na_position="last", kind="stable")
    if na_rm:
        work = work[value.iloc[work["row"].to_numpy()].notna().to_numpy()]
    rank = work.groupby("gid", sort=False).cumcount().to_numpy()
    length = np.bincount(work["gid"].to_numpy(), minlength=groups)
    index = position - 1 if position > 0 else position
    target = np.full(groups, index) if index >= 0 else length + index
    valid = (position != 0) & ((index < length) if index >= 0 else (-index <= length))
    member = work["gid"].to_numpy()
    hit = valid[member] & (rank == target[member])
    out = [default] * groups
    picked = value.iloc[work["row"].to_numpy()[hit]]
    for group, item in zip(member[hit].tolist(), picked.tolist()):
        out[group] = item
    return out


def _ev_func(
    node: tuple,
    df: pd.DataFrame,
    groups: list[str] | None,
    mode: str,
) -> Any:
    _, name, raw_args, raw_kwargs = node
    from tidy3.forcats import FUNCTIONS as _FACTOR_FUNCTIONS
    from tidy3.forcats import eval_pandas as _eval_factor, evaluate_grouped

    if name in _FACTOR_FUNCTIONS and groups:
        return evaluate_grouped(node, df, groups, mode)
    if name == "order_by":
        # Evaluate on rows sorted by the key (stable, missing last), then
        # line the results back up with the original rows by index.
        raw_order, raw_expr = raw_args
        descending = raw_order[0] == "desc"
        order = _as_series(
            _ev(raw_order[1] if descending else raw_order, df, groups, mode), df.index
        )
        ordered = df.loc[
            order.sort_values(
                ascending=not descending, kind="stable", na_position="last"
            ).index
        ]
        value = _ev(raw_expr, ordered, groups, mode)
        return _as_series(value, ordered.index).reindex(df.index)
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

    if name in _FACTOR_FUNCTIONS:
        return _eval_factor(name, args, kwargs)

    if name.startswith("str_"):
        from tidy3.stringr import eval_pandas

        if args and not any(isinstance(arg, pd.Series) for arg in args):
            out = eval_pandas(name, args, kwargs, pd.DataFrame(index=pd.RangeIndex(1)), None, mode)
            return out.iloc[0] if isinstance(out, pd.Series) else out
        return eval_pandas(name, args, kwargs, df, groups, mode)

    from tidy3.helpers import FUNCTIONS as _HELPER_FUNCTIONS
    from tidy3.convert import FUNCTIONS as _CONVERT_FUNCTIONS
    from tidy3.dates import FUNCTIONS as _DATE_FUNCTIONS

    if name in _HELPER_FUNCTIONS or name in _CONVERT_FUNCTIONS or name in _DATE_FUNCTIONS:
        if name == "add_with_rollback":
            kwargs = {**kwargs, "period": _resolve_period(kwargs["period"], df, groups, mode)}
        if args and not any(isinstance(arg, pd.Series) for arg in args):
            # A single value in, such as mean(hp) inside summarise(), is a
            # single value out: round(mean(hp), 1) is one number per group.
            out = _eval_column_function(
                name, raw_args, args, kwargs, pd.DataFrame(index=pd.RangeIndex(1))
            )
            return out.iloc[0] if isinstance(out, pd.Series) else out
        return _eval_column_function(name, raw_args, args, kwargs, df)

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
            values = _nth_per_group(value, order, grouped, position, default, na_rm)
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

    if name == "must_match":
        matched = _as_series(args[1], df.index).fillna(False).astype(bool)
        unmatched_count = int((~matched).sum())
        if unmatched_count:
            raise ValueError(f"{args[2]} {unmatched_count} value(s) had no match.")
        return args[0]

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
        min_count = kwargs.pop("min_count", None)
        if min_count is not None and name != "sum":
            raise ValueError("min_count is only supported by sum()")
        if not isinstance(b, pd.Series):
            b = pd.Series(b, index=df.index)

        def _finite(series: pd.Series) -> pd.Series:
            numeric = pd.to_numeric(series, errors="coerce")
            return pd.Series(
                np.isfinite(numeric.to_numpy(dtype=float, na_value=np.nan)), index=series.index
            )

        def _observed(series: pd.Series) -> float:
            return float(_finite(series).sum())

        if groups:
            # Native group reductions throughout: a Python call per group
            # costs seconds with tens of thousands of groups.
            g = _grouped(b, df, groups)
            if pdname in {"first", "last"} and not na_rm:
                if mode == "window":
                    return g.transform(pdname, skipna=False)
                return getattr(g, pdname)(skipna=False)
            if mode == "window":
                result = g.transform(pdname)
                if not na_rm:
                    missing = _grouped(b.isna(), df, groups).transform("any")
                    if pdname == "any":
                        result = result.mask(~result.astype(bool) & missing)
                    elif pdname == "all":
                        result = result.mask(result.astype(bool) & missing)
                    else:
                        result = result.mask(missing)
                if min_count is not None:
                    counts = _grouped(_finite(b), df, groups).transform("sum").astype(float)
                    result = result.mask(counts < min_count)
                return result
            result = getattr(g, pdname)()
            if not na_rm:
                missing = _grouped(b.isna(), df, groups).any()
                if pdname == "any":
                    result = result.mask(~result.astype(bool) & missing)
                elif pdname == "all":
                    result = result.mask(result.astype(bool) & missing)
                else:
                    result = result.mask(missing)
            if min_count is not None:
                counts = _grouped(_finite(b), df, groups).sum().astype(float)
                result = result.mask(counts < min_count)
            return result
        # ungrouped → scalar (broadcasts in window mode, one row in agg mode)
        if pdname == "first":
            values = b.dropna() if na_rm else b
            return values.iloc[0] if len(values) else np.nan
        if pdname == "last":
            values = b.dropna() if na_rm else b
            return values.iloc[-1] if len(values) else np.nan
        result = getattr(b, pdname)()
        if not na_rm and b.isna().any():
            if not (pdname == "any" and bool(result)):
                if not (pdname == "all" and not bool(result)):
                    result = np.nan
        if min_count is not None and _observed(b) < min_count:
            return np.nan
        return result

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
