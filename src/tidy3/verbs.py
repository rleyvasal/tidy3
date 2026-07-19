"""dplyr-style verbs as pipeable objects (``df >> filter(...)``).

Each verb dispatches on the frame's backend: polars (lazy — expressions are
compiled to ``pl.Expr``) or pandas (eager — expressions are evaluated by
``tidy3.pandas_engine``).
"""

from __future__ import annotations

from typing import Any, Callable

import polars as pl

from tidy3.expr import Expr, to_polars


def _plx(v: Any) -> Any:
    """Compile a tidy3 Expr for polars; pass everything else through."""
    return to_polars(v) if isinstance(v, Expr) else v


def _pe():
    from tidy3 import pandas_engine

    return pandas_engine


class Verb:
    """Pipeable verb: ``tidy_frame >> verb`` via ``__rrshift__``."""

    __slots__ = ("_fn", "name")

    def __init__(self, fn: Callable[..., Any], name: str = "verb"):
        self._fn = fn
        self.name = name

    def __rrshift__(self, other: Any) -> Any:
        from tidy3.frame import TidyFrame

        if not isinstance(other, TidyFrame):
            raise TypeError(
                f"tidy3 pipe expects TidyFrame on the left of >>, got {type(other).__name__}"
            )
        return self._fn(other)

    def __repr__(self) -> str:
        return f"<tidy3.Verb {self.name}>"


def _windowed(expr: Any, groups: list[str] | None) -> Any:
    """dplyr grouped semantics on polars: evaluate the expression per group."""
    if groups and isinstance(expr, pl.Expr):
        return expr.over(groups)
    return expr


def filter(*predicates: Any) -> Verb:  # noqa: A001
    """Keep rows matching all predicates (AND).

    After ``group_by`` the predicate is evaluated per group (dplyr window
    semantics), so ``filter(col("x") > mean("x"))`` compares within groups.
    """
    if not predicates:
        raise TypeError("filter() requires at least one predicate")

    def _apply(tf):
        if tf._backend == "pandas":
            return tf._with_pdf(
                _pe().do_filter(tf._pdf, predicates, tf._groups), groups=tf._groups
            )
        expr = _plx(predicates[0])
        for p in predicates[1:]:
            expr = expr & _plx(p)
        return tf._with_lf(tf._lf.filter(_windowed(expr, tf._groups)), groups=tf._groups)

    return Verb(_apply, "filter")


def mutate(**kwargs: Any) -> Verb:
    """Add or overwrite columns.

    After ``group_by`` each expression is evaluated per group (dplyr window
    semantics): aggregates broadcast within the group, ``cum_sum`` restarts
    per group, etc.
    """
    if not kwargs:
        raise TypeError("mutate() requires at least one assignment")

    def _apply(tf):
        if tf._backend == "pandas":
            return tf._with_pdf(
                _pe().do_mutate(tf._pdf, kwargs, tf._groups), groups=tf._groups
            )
        exprs = {k: _windowed(_plx(v), tf._groups) for k, v in kwargs.items()}
        return tf._with_lf(tf._lf.with_columns(**exprs), groups=tf._groups)

    return Verb(_apply, "mutate")


def transmute(**kwargs: Any) -> Verb:
    """Keep only newly defined columns (plus groups if set)."""
    if not kwargs:
        raise TypeError("transmute() requires at least one assignment")

    def _apply(tf):
        keep = list(kwargs.keys())
        if tf._groups:
            keep = list(dict.fromkeys([*tf._groups, *keep]))
        if tf._backend == "pandas":
            pdf = _pe().do_mutate(tf._pdf, kwargs, tf._groups)
            return tf._with_pdf(pdf[keep], groups=tf._groups)
        exprs = {k: _windowed(_plx(v), tf._groups) for k, v in kwargs.items()}
        lf = tf._lf.with_columns(**exprs)
        return tf._with_lf(lf.select(keep), groups=tf._groups)

    return Verb(_apply, "transmute")


def select(*cols: Any) -> Verb:
    """Select columns by name or expression.

    Like dplyr, grouping columns are always kept (prepended when missing).
    """
    if not cols:
        raise TypeError("select() requires at least one column")

    def _apply(tf):
        sel = list(cols)
        if tf._groups:
            named = {c for c in sel if isinstance(c, str)}
            sel = [*(g for g in tf._groups if g not in named), *sel]
        if tf._backend == "pandas":
            return tf._with_pdf(_pe().do_select(tf._pdf, tuple(sel)), groups=tf._groups)
        return tf._with_lf(tf._lf.select(*[_plx(c) for c in sel]), groups=tf._groups)

    return Verb(_apply, "select")


def drop(*cols: str) -> Verb:
    def _apply(tf):
        if tf._groups:
            bad = [c for c in cols if c in tf._groups]
            if bad:
                raise ValueError(
                    f"drop(): cannot drop grouping column(s) {bad}; ungroup() first"
                )
        if tf._backend == "pandas":
            return tf._with_pdf(tf._pdf.drop(columns=list(cols)), groups=tf._groups)
        return tf._with_lf(tf._lf.drop(*cols), groups=tf._groups)

    return Verb(_apply, "drop")


def rename(**kwargs: str) -> Verb:
    """Rename columns: ``rename(new=old)`` (dplyr style)."""
    # dplyr: rename(new_name = old_name) → mapping new←old
    # polars/pandas: {old: new}
    mapping = {old: new for new, old in kwargs.items()}

    def _apply(tf):
        groups = [mapping.get(g, g) for g in tf._groups] if tf._groups else None
        if tf._backend == "pandas":
            return tf._with_pdf(tf._pdf.rename(columns=mapping), groups=groups)
        return tf._with_lf(tf._lf.rename(mapping), groups=groups)

    return Verb(_apply, "rename")


def arrange(*keys: Any) -> Verb:
    if not keys:
        raise TypeError("arrange() requires at least one key")

    def _apply(tf):
        if tf._backend == "pandas":
            return tf._with_pdf(_pe().do_arrange(tf._pdf, keys), groups=tf._groups)
        return tf._with_lf(tf._lf.sort(*[_plx(k) for k in keys]), groups=tf._groups)

    return Verb(_apply, "arrange")


def distinct(*cols: str) -> Verb:
    def _apply(tf):
        if tf._backend == "pandas":
            return tf._with_pdf(_pe().do_distinct(tf._pdf, cols), groups=tf._groups)
        if cols:
            return tf._with_lf(tf._lf.unique(subset=list(cols)), groups=tf._groups)
        return tf._with_lf(tf._lf.unique(), groups=tf._groups)

    return Verb(_apply, "distinct")


def group_by(*cols: str) -> Verb:
    if not cols:
        raise TypeError("group_by() requires at least one column")

    def _apply(tf):
        if tf._backend == "pandas":
            return tf._with_pdf(tf._pdf, groups=list(cols))
        return tf._with_lf(tf._lf, groups=list(cols))

    return Verb(_apply, "group_by")


def ungroup() -> Verb:
    def _apply(tf):
        if tf._backend == "pandas":
            return tf._with_pdf(tf._pdf, groups=None)
        return tf._with_lf(tf._lf, groups=None)

    return Verb(_apply, "ungroup")


def summarise(**kwargs: Any) -> Verb:
    """Aggregate; uses current ``group_by`` if set."""
    if not kwargs:
        raise TypeError("summarise() requires at least one aggregation")

    def _apply(tf):
        if tf._backend == "pandas":
            return tf._with_pdf(
                _pe().do_summarise(tf._pdf, kwargs, tf._groups), groups=None
            )
        named = []
        for name, expr in kwargs.items():
            e = _plx(expr)
            if isinstance(e, pl.Expr):
                named.append(e.alias(name))
            else:
                named.append(pl.lit(e).alias(name))
        if tf._groups:
            lf = tf._lf.group_by(tf._groups).agg(named)
            return tf._with_lf(lf, groups=None)
        lf = tf._lf.select(named)
        return tf._with_lf(lf, groups=None)

    return Verb(_apply, "summarise")


summarize = summarise


def count(*cols: str, name: str = "n") -> Verb:
    """Count rows, optionally by columns."""

    def _apply(tf):
        if tf._backend == "pandas":
            return tf._with_pdf(_pe().do_count(tf._pdf, cols, name), groups=None)
        if cols:
            lf = tf._lf.group_by(list(cols)).len(name)
            return tf._with_lf(lf, groups=None)
        lf = tf._lf.select(pl.len().alias(name))
        return tf._with_lf(lf, groups=None)

    return Verb(_apply, "count")


def head(n: int = 10) -> Verb:
    """First *n* rows — per group when grouped (dplyr ``slice_head``)."""

    def _apply(tf):
        if tf._backend == "pandas":
            return tf._with_pdf(_pe().do_head(tf._pdf, n, tf._groups), groups=tf._groups)
        if tf._groups:
            pred = pl.int_range(pl.len()).over(tf._groups) < n
            return tf._with_lf(tf._lf.filter(pred), groups=tf._groups)
        return tf._with_lf(tf._lf.head(n), groups=tf._groups)

    return Verb(_apply, "head")


slice_head = head


def sample_n(n: int, *, seed: int | None = None) -> Verb:
    """Random *n* rows (per group when grouped), lazy on polars.

    Polars: a shuffled-index filter, so the plan never materializes the
    whole dataset and original row order is preserved.
    """

    def _apply(tf):
        if tf._backend == "pandas":
            return tf._with_pdf(
                _pe().do_sample_n(tf._pdf, n, seed, tf._groups), groups=tf._groups
            )
        pred = pl.int_range(pl.len()).shuffle(seed=seed) < n
        return tf._with_lf(tf._lf.filter(_windowed(pred, tf._groups)), groups=tf._groups)

    return Verb(_apply, "sample_n")


def sample_frac(frac: float, *, seed: int | None = None) -> Verb:
    """Random fraction of rows (per group when grouped), lazy on polars."""

    def _apply(tf):
        if tf._backend == "pandas":
            return tf._with_pdf(
                _pe().do_sample_frac(tf._pdf, frac, seed, tf._groups), groups=tf._groups
            )
        pred = pl.int_range(pl.len()).shuffle(seed=seed) < pl.len() * frac
        return tf._with_lf(tf._lf.filter(_windowed(pred, tf._groups)), groups=tf._groups)

    return Verb(_apply, "sample_frac")


def _right_frame(right: Any, backend: str) -> Any:
    """Resolve the right side of a join for the given backend."""
    from tidy3.frame import TidyFrame, tidy

    if backend == "pandas":
        if isinstance(right, TidyFrame):
            return right.collect(as_="pandas")
        import pandas as pd

        if isinstance(right, pd.DataFrame):
            return right
        return tidy(right, backend="pandas")._pdf
    return right._lf if isinstance(right, TidyFrame) else tidy(right)._lf


def left_join(right: Any, *, on: str | list[str] | None = None, **kwargs: Any) -> Verb:
    def _apply(tf):
        r = _right_frame(right, tf._backend)
        if tf._backend == "pandas":
            return tf._with_pdf(
                _pe().do_join(tf._pdf, r, on, "left", **kwargs), groups=tf._groups
            )
        return tf._with_lf(tf._lf.join(r, on=on, how="left", **kwargs), groups=tf._groups)

    return Verb(_apply, "left_join")


def inner_join(right: Any, *, on: str | list[str] | None = None, **kwargs: Any) -> Verb:
    def _apply(tf):
        r = _right_frame(right, tf._backend)
        if tf._backend == "pandas":
            return tf._with_pdf(
                _pe().do_join(tf._pdf, r, on, "inner", **kwargs), groups=tf._groups
            )
        return tf._with_lf(tf._lf.join(r, on=on, how="inner", **kwargs), groups=tf._groups)

    return Verb(_apply, "inner_join")


def collect(as_: str = "polars") -> Verb:
    """Materialize — returns pandas/polars DataFrame (not TidyFrame)."""

    def _apply(tf):
        return tf.collect(as_=as_)

    return Verb(_apply, "collect")


def peek(n: int | None = None) -> Verb:
    """Print a preview mid-pipe and pass the frame through.

    Useful while exploring a long pipe in one cell::

        tidy(df) >> filter(...) >> peek() >> mutate(...)
    """

    def _apply(tf):
        from tidy3.options import get_options

        rows = get_options().preview_rows if n is None else n
        print(tf.preview(rows))
        return tf

    return Verb(_apply, "peek")
