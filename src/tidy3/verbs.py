"""dplyr-style verbs as pipeable objects (``df >> filter(...)``)."""

from __future__ import annotations

from typing import Any, Callable

import polars as pl


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


def filter(*predicates: Any) -> Verb:  # noqa: A001
    """Keep rows matching all predicates (AND)."""
    if not predicates:
        raise TypeError("filter() requires at least one predicate")

    def _apply(tf):
        expr = predicates[0]
        for p in predicates[1:]:
            expr = expr & p
        return tf._with_lf(tf._lf.filter(expr), groups=tf._groups)

    return Verb(_apply, "filter")


def mutate(**kwargs: Any) -> Verb:
    """Add or overwrite columns."""
    if not kwargs:
        raise TypeError("mutate() requires at least one assignment")

    def _apply(tf):
        return tf._with_lf(tf._lf.with_columns(**kwargs), groups=tf._groups)

    return Verb(_apply, "mutate")


def transmute(**kwargs: Any) -> Verb:
    """Keep only newly defined columns (plus groups if set)."""
    if not kwargs:
        raise TypeError("transmute() requires at least one assignment")

    def _apply(tf):
        lf = tf._lf.with_columns(**kwargs)
        keep = list(kwargs.keys())
        if tf._groups:
            keep = list(dict.fromkeys([*tf._groups, *keep]))
        return tf._with_lf(lf.select(keep), groups=tf._groups)

    return Verb(_apply, "transmute")


def select(*cols: Any) -> Verb:
    """Select columns by name or expression."""
    if not cols:
        raise TypeError("select() requires at least one column")

    def _apply(tf):
        return tf._with_lf(tf._lf.select(*cols), groups=tf._groups)

    return Verb(_apply, "select")


def drop(*cols: str) -> Verb:
    def _apply(tf):
        return tf._with_lf(tf._lf.drop(*cols), groups=tf._groups)

    return Verb(_apply, "drop")


def rename(**kwargs: str) -> Verb:
    """Rename columns: ``rename(new=old)`` (dplyr style)."""
    # dplyr: rename(new_name = old_name) → mapping new←old
    # polars: rename({old: new})
    mapping = {old: new for new, old in kwargs.items()}

    def _apply(tf):
        return tf._with_lf(tf._lf.rename(mapping), groups=tf._groups)

    return Verb(_apply, "rename")


def arrange(*keys: Any) -> Verb:
    if not keys:
        raise TypeError("arrange() requires at least one key")

    def _apply(tf):
        return tf._with_lf(tf._lf.sort(*keys), groups=tf._groups)

    return Verb(_apply, "arrange")


def distinct(*cols: str) -> Verb:
    def _apply(tf):
        if cols:
            return tf._with_lf(tf._lf.unique(subset=list(cols)), groups=tf._groups)
        return tf._with_lf(tf._lf.unique(), groups=tf._groups)

    return Verb(_apply, "distinct")


def group_by(*cols: str) -> Verb:
    if not cols:
        raise TypeError("group_by() requires at least one column")

    def _apply(tf):
        return tf._with_lf(tf._lf, groups=list(cols))

    return Verb(_apply, "group_by")


def ungroup() -> Verb:
    def _apply(tf):
        return tf._with_lf(tf._lf, groups=None)

    return Verb(_apply, "ungroup")


def summarise(**kwargs: Any) -> Verb:
    """Aggregate; uses current ``group_by`` if set."""
    if not kwargs:
        raise TypeError("summarise() requires at least one aggregation")

    def _apply(tf):
        aggs = list(kwargs.values())
        names = list(kwargs.keys())
        # Ensure named aggregations
        named = []
        for name, expr in zip(names, aggs):
            if isinstance(expr, pl.Expr):
                named.append(expr.alias(name))
            else:
                named.append(pl.lit(expr).alias(name))
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
        if cols:
            lf = tf._lf.group_by(list(cols)).len(name)
            return tf._with_lf(lf, groups=None)
        lf = tf._lf.select(pl.len().alias(name))
        return tf._with_lf(lf, groups=None)

    return Verb(_apply, "count")


def head(n: int = 10) -> Verb:
    def _apply(tf):
        return tf._with_lf(tf._lf.head(n), groups=tf._groups)

    return Verb(_apply, "head")


slice_head = head


def sample_n(n: int, *, seed: int | None = None) -> Verb:
    def _apply(tf):
        # sample needs collect for LazyFrame in some polars versions —
        # use random filter approx or collect-safe path via head after shuffle.
        # Polars LazyFrame has no sample until collect; collect then re-lazy
        # for sample_n is OK for plot prep (explicit row limit).
        df = tf._lf.collect()
        if n >= df.height:
            out = df
        else:
            out = df.sample(n=n, seed=seed, shuffle=True)
        return tf._with_lf(out.lazy(), groups=tf._groups)

    return Verb(_apply, "sample_n")


def sample_frac(frac: float, *, seed: int | None = None) -> Verb:
    def _apply(tf):
        df = tf._lf.collect()
        out = df.sample(fraction=frac, seed=seed, shuffle=True)
        return tf._with_lf(out.lazy(), groups=tf._groups)

    return Verb(_apply, "sample_frac")


def left_join(right: Any, *, on: str | list[str] | None = None, **kwargs: Any) -> Verb:
    from tidy3.frame import TidyFrame, tidy

    def _apply(tf):
        r = right._lf if isinstance(right, TidyFrame) else tidy(right)._lf
        return tf._with_lf(tf._lf.join(r, on=on, how="left", **kwargs), groups=tf._groups)

    return Verb(_apply, "left_join")


def inner_join(right: Any, *, on: str | list[str] | None = None, **kwargs: Any) -> Verb:
    from tidy3.frame import TidyFrame, tidy

    def _apply(tf):
        r = right._lf if isinstance(right, TidyFrame) else tidy(right)._lf
        return tf._with_lf(tf._lf.join(r, on=on, how="inner", **kwargs), groups=tf._groups)

    return Verb(_apply, "inner_join")


def collect(as_: str = "polars") -> Verb:
    """Materialize — returns pandas/polars DataFrame (not TidyFrame)."""

    def _apply(tf):
        return tf.collect(as_=as_)

    return Verb(_apply, "collect")
