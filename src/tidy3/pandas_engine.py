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
    if kind == "call":
        return _ev_call(node, df, groups, mode)
    raise ValueError(f"unknown expression node: {node!r}")


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
        if not isinstance(b, pd.Series):
            b = pd.Series(b, index=df.index)
        if groups:
            g = _grouped(b, df, groups)
            if mode == "window":
                return g.transform(pdname)
            return getattr(g, pdname)()
        # ungrouped → scalar (broadcasts in window mode, one row in agg mode)
        if pdname == "first":
            return b.iloc[0]
        if pdname == "last":
            return b.iloc[-1]
        return getattr(b, pdname)()

    if name in _WINDOW:
        if mode != "window":
            raise ValueError(f"{name}() is not valid inside summarise()")
        pdname = _WINDOW[name]
        if not isinstance(b, pd.Series):
            b = pd.Series(b, index=df.index)
        if groups:
            return getattr(_grouped(b, df, groups), pdname)(*args, **kwargs)
        return getattr(b, pdname)(*args, **kwargs)

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


def do_mutate(df: pd.DataFrame, kwargs: dict, groups: list[str] | None) -> pd.DataFrame:
    # parallel semantics like polars with_columns: all RHS see the input df.
    # assign() (not copy-then-set) so copy-on-write shares unchanged columns.
    new = {k: eval_expr(v, df, groups, "window") for k, v in kwargs.items()}
    return df.assign(**new)


def _has_agg(x) -> bool:
    """True if the node tree contains an aggregation (n(), mean(), …)."""
    if isinstance(x, Expr):
        return _has_agg(x.node)
    if not (isinstance(x, tuple) and x and isinstance(x[0], str)):
        return False
    if x[0] == "n" or (x[0] == "call" and x[1] in _AGG):
        return True
    if x[0] == "call":
        _, _, base, args, kwargs = x
        return (
            _has_agg(base)
            or any(_has_agg(a) for a in args)
            or any(_has_agg(v) for v in kwargs.values())
        )
    return any(_has_agg(p) for p in x[1:])


def _fast_summarise(df: pd.DataFrame, kwargs: dict, groups: list[str]) -> pd.DataFrame | None:
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
        if not (node[0] == "call" and node[1] in _AGG and not node[3] and not node[4]):
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
        work.groupby(list(groups), sort=False, **_GB_KW)
        .agg(**{k: pd.NamedAgg(column=c, aggfunc=f) for k, (c, f) in named.items()})
        .reset_index()
    )
    return out


def do_summarise(df: pd.DataFrame, kwargs: dict, groups: list[str] | None) -> pd.DataFrame:
    if groups:
        fast = _fast_summarise(df, kwargs, groups)
        if fast is not None:
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
        return out
    return pd.DataFrame({k: [v] for k, v in vals.items()})


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


def do_sample_n(df: pd.DataFrame, n: int, seed: int | None, groups: list[str] | None) -> pd.DataFrame:
    if groups:
        out = df.groupby(list(groups), group_keys=False, **_GB_KW).apply(
            lambda g: g.sample(n=n if n < len(g) else len(g), random_state=seed)
        )
        return out.reset_index(drop=True)
    return df.sample(n=n if n < len(df) else len(df), random_state=seed).reset_index(drop=True)


def do_sample_frac(df: pd.DataFrame, frac: float, seed: int | None, groups: list[str] | None) -> pd.DataFrame:
    if groups:
        out = df.groupby(list(groups), group_keys=False, **_GB_KW).sample(
            frac=frac, random_state=seed
        )
        return out.reset_index(drop=True)
    return df.sample(frac=frac, random_state=seed).reset_index(drop=True)


def do_count(df: pd.DataFrame, cols: tuple, name: str) -> pd.DataFrame:
    if cols:
        return (
            df.groupby(list(cols), sort=False, **_GB_KW)
            .size()
            .reset_index(name=name)
        )
    return pd.DataFrame({name: [len(df)]})


def do_join(df: pd.DataFrame, right: pd.DataFrame, on, how: str, **kwargs) -> pd.DataFrame:
    return df.merge(right, on=on, how=how, suffixes=("", "_right"), **kwargs)
