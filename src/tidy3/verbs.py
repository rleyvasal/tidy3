"""dplyr-style verbs as pipeable objects (``df >> filter(...)``).

Each verb dispatches on the frame's backend: polars (lazy — expressions are
compiled to ``pl.Expr``) or pandas (eager — expressions are evaluated by
``tidy3.pandas_engine``).
"""

from __future__ import annotations

import math
import random
import re
from typing import Any, Callable

import polars as pl

from tidy3.expr import Expr, to_polars
from tidy3.join_spec import JoinSpec, join_by
from tidy3.tidyselect import (
    AcrossSpec,
    ColumnSet,
    ColumnwisePredicate,
    HorizontalSpec,
    Selector,
    _group_rows_token,
    _reset_group_rows,
    resolve_selection,
)


def _plx(v: Any) -> Any:
    """Compile a tidy3 Expr for polars; pass everything else through."""
    return to_polars(v) if isinstance(v, Expr) else v


def _pl_expr(v: Any) -> pl.Expr:
    value = _plx(v)
    return value if isinstance(value, pl.Expr) else pl.lit(value)


def _dplyr_sort_expressions(
    lf: pl.LazyFrame, values: list[Any]
) -> list[pl.Expr]:
    """Build sort keys with R's missing-last treatment, including NaN."""
    expressions: list[pl.Expr] = []
    for index, value in enumerate(values):
        expression = pl.col(value) if isinstance(value, str) else _pl_expr(value)
        name = f"__tidy3_sort_schema_{index}"
        dtype = lf.select(expression.alias(name)).collect_schema()[name]
        expressions.append(
            expression.fill_nan(None) if dtype.is_float() else expression
        )
    return expressions


def _pe():
    from tidy3 import pandas_engine

    return pandas_engine


class Verb:
    """Pipeable verb: ``tidy_frame >> verb`` via ``__rrshift__``.

    Typed so Pylance/Pyright treat ``frame >> select(...)`` as ``TidyFrame``.
    """

    __slots__ = ("_fn", "name")

    def __init__(self, fn: Callable[..., Any], name: str = "verb"):
        self._fn = fn
        self.name = name

    def __rrshift__(self, other: Any) -> "TidyFrame":
        from tidy3.frame import TidyFrame

        if not isinstance(other, TidyFrame):
            cls = type(other)
            if cls.__module__ == "tidy3.frame" and cls.__name__ == "TidyFrame":
                # A remote re-seed / %autoreload creates a new TidyFrame class
                # identity while frames already in the notebook still use the
                # old one. Re-wrap backend data in the current class at this seam.
                try:
                    other = TidyFrame(
                        other._data,
                        groups=other._groups,
                        rowwise=getattr(other, "_rowwise", False),
                        group_drop=getattr(other, "_group_drop", True),
                        category_levels=getattr(
                            other, "_category_levels", None
                        ),
                        select_base=getattr(other, "_select_base", None),
                    )
                except (AttributeError, TypeError) as e:
                    raise TypeError(
                        "tidy3 pipe received an incompatible TidyFrame from "
                        "another loaded copy — restart the kernel "
                        "(%restart / Restart Kernel) and re-import tidy3"
                    ) from e
            else:
                raise TypeError(
                    "tidy3 pipe expects TidyFrame on the left of >>, "
                    f"got {cls.__name__!r}. "
                    "If you used a documentation placeholder like "
                    "slice_max(...), pass real arguments "
                    '(e.g. slice_max(order_by="hp", n=1)). '
                    "After editing tidy3 source, restart the kernel so "
                    "pipes and verbs share one TidyFrame class."
                )
        return self._fn(other)

    # Also support ``verb(frame)`` call style for type checkers / tooling.
    def __call__(self, other: Any) -> "TidyFrame":
        return self.__rrshift__(other)

    def __repr__(self) -> str:
        return f"<tidy3.Verb {self.name}>"


def _windowed(expr: Any, groups: list[str] | None) -> Any:
    """dplyr grouped semantics on polars: evaluate the expression per group."""
    if groups and isinstance(expr, pl.Expr):
        return expr.over(groups)
    return expr


def _resolved_value(tf: Any, value: Any) -> Any:
    if isinstance(value, (ColumnSet, ColumnwisePredicate, HorizontalSpec)):
        return value.expand(tf)
    return value


def _expanded_assignments(
    tf: Any,
    specs: tuple[Any, ...],
    assignments: dict[str, Any],
    verb_name: str,
) -> dict[str, Any]:
    from tidy3.masking import NamedAssign

    expanded: dict[str, Any] = {}
    for spec in specs:
        if isinstance(spec, NamedAssign):
            # From R-style mutate(`new col` = expr) → __tidy3_assign__(...)
            if spec.name in expanded or spec.name in assignments:
                raise ValueError(
                    f"{verb_name}() defines column {spec.name!r} more than once"
                )
            expanded[spec.name] = _resolved_value(tf, spec.value)
            continue
        if not isinstance(spec, AcrossSpec):
            raise TypeError(
                f"{verb_name}() positional arguments must come from across() "
                f"or a named assignment (got {type(spec).__name__})"
            )
        for name, value in spec.expand(tf).items():
            if name in expanded or name in assignments:
                raise ValueError(f"{verb_name}() defines column {name!r} more than once")
            expanded[name] = value
    expanded.update(
        {name: _resolved_value(tf, value) for name, value in assignments.items()}
    )
    return expanded


def _column_references(value: Any) -> set[str]:
    """Return column names referenced by a backend-neutral expression."""
    if not isinstance(value, Expr):
        return set()
    references: set[str] = set()

    def visit(node: tuple) -> None:
        if node[0] == "col":
            references.add(node[1])
            return
        for child in _expression_children(node):
            visit(child)

    visit(value.node)
    return references


def _assignment_stages(assignments: dict[str, Any]) -> list[dict[str, Any]]:
    """Batch independent assignments while preserving dplyr dependencies."""
    stages: list[dict[str, Any]] = []
    current: dict[str, Any] = {}
    for name, value in assignments.items():
        if _column_references(value) & set(current):
            stages.append(current)
            current = {}
        current[name] = value
    if current:
        stages.append(current)
    return stages


def _sequential_summary_assignments(
    assignments: dict[str, Any],
) -> dict[str, Any]:
    """Inline earlier summaries so later expressions can reference them."""

    def replacement_node(value: Any) -> tuple:
        if isinstance(value, Expr):
            return value.node
        if isinstance(value, pl.Expr):
            return ("pl", value)
        return ("lit", value)

    def rewrite(node: tuple, prior: dict[str, Any]) -> tuple:
        kind = node[0]
        if kind == "col" and node[1] in prior:
            return replacement_node(prior[node[1]])
        if kind == "bin":
            return (kind, node[1], rewrite(node[2], prior), rewrite(node[3], prior))
        if kind in {"neg", "not", "desc"}:
            return (kind, rewrite(node[1], prior))
        if kind == "call":
            references = _column_references(Expr(node[2]))
            if references and references <= set(prior):
                rewritten_base = rewrite(node[2], prior)
                if node[1] in {
                    "sum",
                    "mean",
                    "min",
                    "max",
                    "median",
                    "first",
                    "last",
                    "any",
                    "all",
                }:
                    return rewritten_base
                if node[1] in {"std", "var"}:
                    return ("lit", float("nan"))
            return (
                kind,
                node[1],
                rewrite(node[2], prior),
                tuple(
                    Expr(rewrite(value.node, prior))
                    if isinstance(value, Expr)
                    else value
                    for value in node[3]
                ),
                {
                    key: Expr(rewrite(value.node, prior))
                    if isinstance(value, Expr)
                    else value
                    for key, value in node[4].items()
                },
            )
        if kind == "func":
            return (
                kind,
                node[1],
                tuple(rewrite(value, prior) for value in node[2]),
                {key: rewrite(value, prior) for key, value in node[3].items()},
            )
        if kind == "case_when":
            return (
                kind,
                tuple(
                    (rewrite(condition, prior), rewrite(value, prior))
                    for condition, value in node[1]
                ),
                rewrite(node[2], prior),
            )
        return node

    resolved: dict[str, Any] = {}
    for name, value in assignments.items():
        resolved[name] = (
            Expr(rewrite(value.node, resolved))
            if isinstance(value, Expr)
            else value
        )
    return resolved


def _mutate_column_order(
    result: Any,
    input_columns: list[str],
    assignments: dict[str, Any],
    groups: list[str] | None,
    *,
    keep: str,
    before: Any,
    after: Any,
) -> list[str]:
    referenced = (
        set().union(
            *(_column_references(value) for value in assignments.values())
        )
        if assignments
        else set()
    )
    if keep == "all":
        retained = set(input_columns)
    elif keep == "used":
        retained = set(input_columns) & referenced
    elif keep == "unused":
        retained = set(input_columns) - referenced
    else:
        retained = set()
    retained |= set(groups or []) | (set(assignments) & set(input_columns))
    base = [name for name in input_columns if name in retained]
    new_columns = [name for name in assignments if name not in input_columns]
    ordered = [*base, *new_columns]
    anchor_spec = before if before is not None else after
    if anchor_spec is not None and new_columns:
        anchors = resolve_selection(result, [anchor_spec])
        if len(anchors) != 1:
            raise ValueError("mutate(): before/after must select one column")
        anchor = anchors[0]
        if anchor not in base:
            raise ValueError("mutate(): before/after cannot select a new column")
        position = base.index(anchor) + int(after is not None)
        ordered = [*base[:position], *new_columns, *base[position:]]
    return ordered


def _operation_groups(tf: Any, by: Any, verb_name: str) -> tuple[list[str] | None, bool]:
    """Resolve transient ``by=`` groups and report whether they were supplied."""
    if by is None:
        return tf._groups, False
    if tf._groups or tf._rowwise:
        raise ValueError(
            f"{verb_name}(): by= cannot be used on a grouped or rowwise frame"
        )
    groups = resolve_selection(tf, [by])
    return groups or None, True


def _group_context(tf: Any, groups: list[str] | None):
    """Create a metadata-only view used while expanding tidy-select specs."""
    if groups == tf._groups:
        return tf
    if tf._backend == "pandas":
        return tf._with_pdf(tf._pdf, groups=groups, rowwise=False)
    return tf._with_lf(tf._lf, groups=groups, rowwise=False)


_AGGREGATE_CALLS = {
    "mean",
    "sum",
    "min",
    "max",
    "median",
    "std",
    "var",
    "count",
    "n_unique",
    "first",
    "last",
    "any",
    "all",
}

_WINDOW_CALLS = {
    "cum_sum",
    "cumsum",
    "cum_max",
    "cum_min",
    "cum_prod",
    "shift",
    "diff",
    "rank",
}

_ELEMENT_CALLS = {
    "abs",
    "alias",
    "ceil",
    "exp",
    "fill_null",
    "floor",
    "is_not_null",
    "is_null",
    "log",
    "log10",
    "round",
    "sqrt",
}

_WINDOW_FUNCTIONS = {
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
    "n_distinct",
    "nth",
    "consecutive_id",
}


def _expression_children(node: tuple) -> list[tuple]:
    kind = node[0]
    if kind == "bin":
        return [node[2], node[3]]
    if kind in {"neg", "not", "desc"}:
        return [node[1]]
    if kind == "call":
        return [
            node[2],
            *[value.node for value in node[3] if isinstance(value, Expr)],
            *[
                value.node
                for value in node[4].values()
                if isinstance(value, Expr)
            ],
        ]
    if kind == "func":
        return [*node[2], *node[3].values()]
    if kind == "case_when":
        return [
            *[
                child
                for condition, value in node[1]
                for child in (condition, value)
            ],
            node[2],
        ]
    return []


def _is_aggregate_node(node: tuple) -> bool:
    return (
        node[0] == "n"
        or (node[0] == "call" and node[1] in _AGGREGATE_CALLS)
        or (node[0] == "func" and node[1] in {"n_distinct", "nth"})
    )


def _requires_group_window(node: tuple) -> bool:
    if _is_aggregate_node(node):
        return True
    if node[0] == "call" and node[1] in _WINDOW_CALLS:
        return True
    if node[0] == "call" and node[1] not in _ELEMENT_CALLS:
        return True
    if node[0] == "func" and node[1] in _WINDOW_FUNCTIONS:
        return True
    return any(_requires_group_window(child) for child in _expression_children(node))


def _grouped_polars_value(value: Any, groups: list[str] | None) -> Any:
    compiled = _plx(value)
    if not groups:
        return compiled
    if isinstance(value, Expr) and not _requires_group_window(value.node):
        return compiled
    return _windowed(compiled, groups)


def _aggregate_cache_plan(
    assignments: dict[str, Any], occupied: list[str]
) -> tuple[dict[str, Expr], dict[str, Any], list[str]]:
    """Extract reusable leaf aggregates from grouped mutate expressions."""
    occurrences: dict[str, dict[str, Any]] = {}

    def visit(node: tuple, *, root: bool) -> bool:
        children = _expression_children(node)
        child_has_aggregate = any(
            [visit(child, root=False) for child in children]
        )
        aggregate = _is_aggregate_node(node)
        if aggregate and not child_has_aggregate:
            key = repr(node)
            info = occurrences.setdefault(
                key, {"node": node, "count": 0, "nested": False}
            )
            info["count"] += 1
            info["nested"] = info["nested"] or not root
        return aggregate or child_has_aggregate

    for value in assignments.values():
        if isinstance(value, Expr):
            visit(value.node, root=True)

    selected = {
        key: info
        for key, info in occurrences.items()
        if info["nested"] or info["count"] > 1
    }
    if not selected:
        return {}, assignments, []

    temp_names: dict[str, str] = {}
    used = [*occupied, *assignments]
    for index, key in enumerate(selected, start=1):
        name = _temp_column([*used, *temp_names.values()], f"__tidy3_agg_{index}")
        temp_names[key] = name

    def rewrite(node: tuple) -> tuple:
        key = repr(node)
        if key in temp_names:
            return ("col", temp_names[key])
        kind = node[0]
        if kind == "bin":
            return (kind, node[1], rewrite(node[2]), rewrite(node[3]))
        if kind in {"neg", "not", "desc"}:
            return (kind, rewrite(node[1]))
        if kind == "call":
            return (
                kind,
                node[1],
                rewrite(node[2]),
                tuple(
                    Expr(rewrite(value.node)) if isinstance(value, Expr) else value
                    for value in node[3]
                ),
                {
                    key: Expr(rewrite(value.node))
                    if isinstance(value, Expr)
                    else value
                    for key, value in node[4].items()
                },
            )
        if kind == "func":
            return (
                kind,
                node[1],
                tuple(rewrite(value) for value in node[2]),
                {key: rewrite(value) for key, value in node[3].items()},
            )
        if kind == "case_when":
            return (
                kind,
                tuple(
                    (rewrite(condition), rewrite(value))
                    for condition, value in node[1]
                ),
                rewrite(node[2]),
            )
        return node

    cached = {
        temp_names[key]: Expr(info["node"]) for key, info in selected.items()
    }
    rewritten = {
        name: Expr(rewrite(value.node)) if isinstance(value, Expr) else value
        for name, value in assignments.items()
    }
    return cached, rewritten, list(cached)


def _expr_column_names(value: Any) -> set[str]:
    """Column names referenced by a tidy3 Expr / polars Expr / plain value."""
    names: set[str] = set()

    def walk(node: Any) -> None:
        if not isinstance(node, tuple) or not node:
            return
        kind = node[0]
        if kind == "col":
            names.add(str(node[1]))
        elif kind == "lit":
            return
        elif kind == "pl":
            try:
                names.update(node[1].meta.root_names())
            except Exception:
                pass
        elif kind in {"bin", "neg", "not", "desc"}:
            for child in node[1:]:
                if isinstance(child, tuple):
                    walk(child)
        elif kind == "call":
            # ("call", name, base_node, args, kwargs)
            walk(node[2])
            for arg in node[3] or ():
                if isinstance(arg, Expr):
                    walk(arg.node)
                elif isinstance(arg, tuple):
                    walk(arg)
            for val in (node[4] or {}).values():
                if isinstance(val, Expr):
                    walk(val.node)
                elif isinstance(val, tuple):
                    walk(val)
        elif kind == "n":
            return
        else:
            for child in node[1:]:
                if isinstance(child, tuple):
                    walk(child)

    if isinstance(value, Expr):
        walk(value.node)
    elif isinstance(value, pl.Expr):
        try:
            names.update(value.meta.root_names())
        except Exception:
            pass
    elif isinstance(value, str):
        names.add(value)
    return names


def _predicate_column_names(predicates: tuple[Any, ...]) -> set[str]:
    names: set[str] = set()
    for predicate in predicates:
        names |= _expr_column_names(predicate)
    return names


def _base_columns(data: Any) -> list[str]:
    if isinstance(data, pl.LazyFrame):
        return data.collect_schema().names()
    return [str(c) for c in data.columns]


def _maybe_widen_for_select_base(
    tf: Any, needed: set[str] | None = None
) -> tuple[Any, bool]:
    """Evaluate a row verb on the pre-select frame when a select is still open.

    Rewrites ``select(...) >> verb(...)`` into ``verb(...) >> select(...)`` for
    row-preserving-column-set verbs (filter, arrange, slice_min/max). Always
    operating on the wide base keeps stacked verbs correct (each updates the
    base snapshot).

    Returns ``(frame, widened)`` — re-project only when *widened* is True.
    """
    base = getattr(tf, "_select_base", None)
    if base is None:
        return tf, False
    pre = base["data"]
    pre_names = set(_base_columns(pre))
    if needed and not needed <= pre_names:
        # Column never existed (even before select) — let the normal path error.
        return tf, False
    if tf._backend == "pandas":
        return (
            tf._with_pdf(pre, groups=tf._groups, select_base=base),
            True,
        )
    return tf._with_lf(pre, groups=tf._groups, select_base=base), True


# Back-compat alias used by filter path during transition.
def _maybe_widen_for_filter(
    tf: Any, predicates: tuple[Any, ...]
) -> tuple[Any, bool]:
    return _maybe_widen_for_select_base(tf, _predicate_column_names(predicates))


def _reproject_after_select_base(tf: Any, result: Any) -> Any:
    """Re-apply a deferred select projection after a wide-frame row verb."""
    base = getattr(tf, "_select_base", None)
    if base is None:
        return result
    groups = None if result._groups is None else list(result._groups)
    if result._backend == "pandas":
        sel = base["sel"]
        mapping = base["mapping"]
        pdf = result._pdf[sel].rename(columns=mapping)
        # Keep select_base pointed at the wide frame for later row verbs.
        new_base = {**base, "data": result._pdf}
        return result._with_pdf(pdf, groups=groups, select_base=new_base)
    expressions = base["expressions"]
    lf = result._lf.select(expressions)
    new_base = {**base, "data": result._lf}
    return result._with_lf(lf, groups=groups, select_base=new_base)


def _reproject_after_filter(tf: Any, result: Any) -> Any:
    return _reproject_after_select_base(tf, result)


def _finish_select_base_verb(
    wide: Any, out: Any, *, widened: bool
) -> Any:
    """Preserve select_base and re-project when the verb ran on a wide frame."""
    if not widened:
        # Still keep the open select snapshot for a later filter/arrange/slice.
        return out
    return _reproject_after_select_base(wide, out)


def filter(*predicates: Any, by: Any = None) -> Verb:  # noqa: A001
    """Keep rows matching all predicates (AND).

    After ``group_by`` the predicate is evaluated per group (dplyr window
    semantics), so ``filter(col("x") > mean("x"))`` compares within groups.

    ``select(...) >> filter(...)`` is rewritten when the filter needs columns
    that select dropped (same result as ``filter >> select``).
    """
    if not predicates:
        raise TypeError("filter() requires at least one predicate")

    def _apply(tf):
        wide, widened = _maybe_widen_for_filter(tf, predicates)
        groups, transient = _operation_groups(wide, by, "filter")
        context = _group_context(wide, groups)
        resolved = tuple(
            _resolved_value(context, predicate) for predicate in predicates
        )
        # Preserve select_base so a later filter can still widen if needed.
        # When we widen, reproject updates base["data"] to the filtered wide plan.
        keep_base = wide._select_base

        if wide._backend == "pandas":
            work = wide._pdf
            marker = None
            if wide._rowwise:
                marker = _temp_column(_frame_columns(wide), "__tidy3_rowwise")
                work = work.assign(**{marker: range(len(work))})
                groups = [marker]
            pdf = _pe().do_filter(work, resolved, groups)
            if marker is not None:
                pdf = pdf.drop(columns=marker)
            out = wide._with_pdf(
                pdf,
                groups=None if transient else wide._groups,
                select_base=keep_base,
            )
            return _reproject_after_filter(wide, out) if widened else out
        expr = _plx(resolved[0])
        for p in resolved[1:]:
            expr = expr & _plx(p)
        if wide._rowwise:
            marker = _temp_column(_frame_columns(wide), "__tidy3_rowwise")
            lf = (
                wide._lf.with_row_index(marker)
                .filter(_pl_expr(expr).over(marker))
                .drop(marker)
            )
        else:
            lf = wide._lf.filter(_windowed(expr, groups))
        out = wide._with_lf(
            lf,
            groups=None if transient else wide._groups,
            select_base=keep_base,
        )
        return _reproject_after_filter(wide, out) if widened else out

    return Verb(_apply, "filter")


def filter_out(*predicates: Any, by: Any = None) -> Verb:
    """Drop rows matching all predicates; retain rows where they are null."""
    if not predicates:
        raise TypeError("filter_out() requires at least one predicate")

    def _apply(tf):
        wide, widened = _maybe_widen_for_filter(tf, predicates)
        groups, transient = _operation_groups(wide, by, "filter_out")
        context = _group_context(wide, groups)
        resolved = tuple(
            _resolved_value(context, predicate) for predicate in predicates
        )
        keep_base = wide._select_base

        if wide._backend == "pandas":
            work = wide._pdf
            marker = None
            if wide._rowwise:
                marker = _temp_column(_frame_columns(wide), "__tidy3_rowwise")
                work = work.assign(**{marker: range(len(work))})
                groups = [marker]
            pdf = _pe().do_filter_out(work, resolved, groups)
            if marker is not None:
                pdf = pdf.drop(columns=marker)
            out = wide._with_pdf(
                pdf,
                groups=None if transient else wide._groups,
                select_base=keep_base,
            )
            return _reproject_after_filter(wide, out) if widened else out
        expr = _plx(resolved[0])
        for predicate in resolved[1:]:
            expr = expr & _plx(predicate)
        if wide._rowwise:
            marker = _temp_column(_frame_columns(wide), "__tidy3_rowwise")
            lf = wide._lf.with_row_index(marker)
            expr = _pl_expr(expr).over(marker)
        else:
            marker = None
            lf = wide._lf
            expr = _windowed(expr, groups)
        lf = lf.filter((~expr).fill_null(True))
        if marker is not None:
            lf = lf.drop(marker)
        out = wide._with_lf(
            lf,
            groups=None if transient else wide._groups,
            select_base=keep_base,
        )
        return _reproject_after_filter(wide, out) if widened else out

    return Verb(_apply, "filter_out")


def mutate(
    *specs: Any,
    by: Any = None,
    keep: str = "all",
    before: Any = None,
    after: Any = None,
    **kwargs: Any,
) -> Verb:
    """Add or overwrite columns.

    After ``group_by`` each expression is evaluated per group (dplyr window
    semantics): aggregates broadcast within the group, ``cum_sum`` restarts
    per group, etc.
    """
    if not specs and not kwargs:
        raise TypeError("mutate() requires at least one assignment")
    if keep not in {"all", "used", "unused", "none"}:
        raise ValueError("mutate() keep must be all, used, unused, or none")
    if before is not None and after is not None:
        raise ValueError("mutate() accepts only one of before= or after=")

    def _apply(tf):
        input_columns = _frame_columns(tf)
        groups, transient = _operation_groups(tf, by, "mutate")
        context = _group_context(tf, groups)
        assignments = _expanded_assignments(context, specs, kwargs, "mutate")
        deleted_so_far: set[str] = set()
        for name, value in assignments.items():
            if value is None:
                deleted_so_far.add(name)
                continue
            missing = _column_references(value) & deleted_so_far
            if missing:
                raise KeyError(
                    "mutate() expression references columns deleted earlier "
                    f"in the call: {sorted(missing)}"
                )
        deletions = [name for name, value in assignments.items() if value is None]
        grouped_deletions = [
            name for name in deletions if name in (tf._groups or [])
        ]
        if grouped_deletions:
            raise ValueError(
                f"mutate() cannot delete grouping columns: {grouped_deletions}"
            )
        assignments = {
            name: value for name, value in assignments.items() if value is not None
        }
        input_columns = [name for name in input_columns if name not in deletions]
        stages = _assignment_stages(assignments)
        if tf._backend == "pandas":
            work = tf._pdf
            marker = None
            if tf._rowwise:
                marker = _temp_column(_frame_columns(tf), "__tidy3_rowwise")
                work = work.assign(**{marker: range(len(work))})
                groups = [marker]
            for stage in stages:
                cached, stage, cache_columns = (
                    _aggregate_cache_plan(stage, list(work.columns))
                    if groups and not tf._rowwise
                    else ({}, stage, [])
                )
                if cached:
                    work = _pe().do_mutate(work, cached, groups)
                work = _pe().do_mutate(work, stage, groups)
                if cache_columns:
                    work = work.drop(columns=cache_columns)
            pdf = work
            if marker is not None:
                pdf = pdf.drop(columns=marker)
            if deletions:
                pdf = pdf.drop(columns=deletions, errors="ignore")
            result = tf._with_pdf(
                pdf, groups=None if transient else tf._groups
            )
            ordered = _mutate_column_order(
                result,
                input_columns,
                assignments,
                groups,
                keep=keep,
                before=before,
                after=after,
            )
            return tf._with_pdf(
                pdf.loc[:, ordered],
                groups=None if transient else tf._groups,
            )
        if tf._rowwise:
            marker = _temp_column(_frame_columns(tf), "__tidy3_rowwise")
            lf = tf._lf.with_row_index(marker)
            for stage in stages:
                exprs = {k: _pl_expr(v).over(marker) for k, v in stage.items()}
                lf = lf.with_columns(**exprs)
            lf = lf.drop(marker)
        else:
            lf = tf._lf
            for stage in stages:
                cached, stage, cache_columns = (
                    _aggregate_cache_plan(stage, lf.collect_schema().names())
                    if groups
                    else ({}, stage, [])
                )
                if cached:
                    lf = lf.with_columns(
                        **{
                            name: _grouped_polars_value(value, groups)
                            for name, value in cached.items()
                        }
                    )
                exprs = {
                    k: _grouped_polars_value(v, groups)
                    for k, v in stage.items()
                }
                lf = lf.with_columns(**exprs)
                if cache_columns:
                    lf = lf.drop(cache_columns)
        if deletions:
            existing = lf.collect_schema().names()
            lf = lf.drop(*[name for name in deletions if name in existing])
        result = tf._with_lf(
            lf, groups=None if transient else tf._groups
        )
        ordered = _mutate_column_order(
            result,
            input_columns,
            assignments,
            groups,
            keep=keep,
            before=before,
            after=after,
        )
        return tf._with_lf(
            lf.select(ordered), groups=None if transient else tf._groups
        )

    return Verb(_apply, "mutate")


def transmute(*specs: Any, by: Any = None, **kwargs: Any) -> Verb:
    """Keep only newly defined columns (plus groups if set)."""
    verb = mutate(*specs, by=by, keep="none", **kwargs)
    verb.name = "transmute"
    return verb


def select(*cols: Any, **renames: Any) -> Verb:
    """Select columns by name or expression.

    Like dplyr, grouping columns are always kept (prepended when missing).

    A following ``filter()`` / ``filter_out()`` / ``arrange()`` /
    ``slice_min()`` / ``slice_max()`` may still reference columns dropped
    here; tidy3 rewrites that to verb-then-select (row ops before projection).
    """
    if not cols and not renames:
        raise TypeError("select() requires at least one column")

    def _apply(tf):
        from tidy3.masking import NamedAssign

        ordered: list[Any] = []
        seen_sources: set[str] = set()
        mapping: dict[str, str] = {}
        for spec in cols:
            if isinstance(spec, NamedAssign):
                # select(`new name` = old) from R-style rewrite
                resolved = resolve_selection(tf, [spec.value])
                if len(resolved) != 1:
                    raise ValueError(
                        f"select(): rename {spec.name!r} must select one column"
                    )
                old = resolved[0]
                mapping[old] = spec.name
                if old not in seen_sources:
                    ordered.append(old)
                    seen_sources.add(old)
                continue
            if isinstance(spec, (Expr, pl.Expr)):
                ordered.append(spec)
                continue
            for source in resolve_selection(tf, [spec]):
                if source not in seen_sources:
                    ordered.append(source)
                    seen_sources.add(source)
        for new, spec in renames.items():
            resolved = resolve_selection(tf, [spec])
            if len(resolved) != 1:
                raise ValueError(f"select(): rename {new!r} must select one column")
            old = resolved[0]
            mapping[old] = new
            if old not in seen_sources:
                ordered.append(old)
                seen_sources.add(old)
        if tf._groups:
            ordered = [
                *(g for g in tf._groups if g not in seen_sources),
                *ordered,
            ]
        sel = [value for value in ordered if isinstance(value, str)]
        computed = [value for value in ordered if not isinstance(value, str)]
        output_names = [mapping.get(name, name) for name in sel]
        if len(set(output_names)) != len(output_names):
            raise ValueError("select() produced duplicate column names")
        groups = [mapping.get(g, g) for g in tf._groups] if tf._groups else None
        if tf._backend == "pandas":
            if computed:
                raise TypeError(
                    "pandas backend select() does not support computed columns; "
                    "use mutate()"
                )
            pdf = tf._pdf[sel].rename(columns=mapping)
            # Snapshot pre-projection rows so a later filter can widen.
            select_base = {
                "data": tf._pdf,
                "sel": list(sel),
                "mapping": dict(mapping),
                "expressions": None,
            }
            return tf._with_pdf(pdf, groups=groups, select_base=select_base)
        expressions = [
            (
                pl.col(value).alias(mapping[value])
                if isinstance(value, str) and value in mapping
                else pl.col(value)
                if isinstance(value, str)
                else _plx(value)
            )
            for value in ordered
        ]
        select_base = {
            "data": tf._lf,
            "sel": list(sel),
            "mapping": dict(mapping),
            "expressions": expressions,
        }
        return tf._with_lf(
            tf._lf.select(expressions),
            groups=groups,
            select_base=select_base,
        )

    return Verb(_apply, "select")


def drop(*cols: Any) -> Verb:
    def _apply(tf):
        selected = resolve_selection(tf, cols)
        if tf._groups:
            bad = [c for c in selected if c in tf._groups]
            if bad:
                raise ValueError(
                    f"drop(): cannot drop grouping column(s) {bad}; ungroup() first"
                )
        if tf._backend == "pandas":
            return tf._with_pdf(tf._pdf.drop(columns=selected), groups=tf._groups)
        return tf._with_lf(tf._lf.drop(*selected), groups=tf._groups)

    return Verb(_apply, "drop")


def rename(*specs: Any, **kwargs: str) -> Verb:
    """Rename individual columns (dplyr ``rename``).

    Syntax is always **new = old** (tidyverse)::

        rename(power=hp)
        rename(power="hp")
        rename(`horse power` = hp)          # Jupyter R-style
        rename(**{"horse power": "hp"})     # plain Python

    For bulk / functional renames see :func:`rename_with` and :func:`set_names`.
    """

    def _apply(tf):
        from tidy3.masking import NamedAssign

        # dplyr: rename(new_name = old_name) → polars/pandas {old: new}
        mapping: dict[str, str] = {}
        for spec in specs:
            if not isinstance(spec, NamedAssign):
                raise TypeError(
                    "rename() positional args must be named assignments "
                    "(`new` = old)"
                )
            old = spec.value
            if not isinstance(old, str):
                resolved = resolve_selection(tf, [old])
                if len(resolved) != 1:
                    raise ValueError(
                        f"rename(): {spec.name!r} must map from one column"
                    )
                old = resolved[0]
            mapping[old] = spec.name
        for new, old in kwargs.items():
            mapping[str(old)] = new
        if not mapping:
            return tf
        groups = [mapping.get(g, g) for g in tf._groups] if tf._groups else None
        category_levels = {
            mapping.get(name, name): levels
            for name, levels in tf._category_levels.items()
        }
        if tf._backend == "pandas":
            return tf._with_pdf(
                tf._pdf.rename(columns=mapping),
                groups=groups,
                category_levels=category_levels,
            )
        return tf._with_lf(
            tf._lf.rename(mapping),
            groups=groups,
            category_levels=category_levels,
        )

    return Verb(_apply, "rename")


def _frame_columns(tf: Any) -> list[str]:
    if tf._backend == "pandas":
        return [str(column) for column in tf._pdf.columns]
    return tf._lf.collect_schema().names()


def relocate(
    *cols: Any,
    before: Any = None,
    after: Any = None,
) -> Verb:
    """Move named columns, optionally before or after another column."""
    if not cols:
        raise TypeError("relocate() requires at least one column")
    if before is not None and after is not None:
        raise ValueError("relocate() accepts only one of before= or after=")

    def _apply(tf):
        columns = _frame_columns(tf)
        moving = resolve_selection(tf, cols)
        anchor_spec = before if before is not None else after
        anchor = None
        if anchor_spec is not None:
            anchors = resolve_selection(tf, [anchor_spec])
            if len(anchors) != 1:
                raise ValueError("relocate(): before/after must select one column")
            anchor = anchors[0]
        if anchor in moving:
            raise ValueError("relocate(): anchor cannot also be a moved column")
        remaining = [column for column in columns if column not in moving]
        if before is not None:
            position = remaining.index(anchor)
        elif after is not None:
            position = remaining.index(anchor) + 1
        else:
            position = 0
        ordered = [*remaining[:position], *moving, *remaining[position:]]
        if tf._backend == "pandas":
            return tf._with_pdf(tf._pdf[ordered], groups=tf._groups)
        return tf._with_lf(tf._lf.select(ordered), groups=tf._groups)

    return Verb(_apply, "relocate")


def rename_with(fn: Callable[[str], str], *cols: Any) -> Verb:
    """Rename columns with a function (dplyr ``rename_with``).

    Parameters
    ----------
    fn:
        ``str -> str`` applied to each selected name (e.g. ``str.upper``,
        ``lambda s: s.replace(".", "_")``).
    *cols:
        Optional tidyselect; defaults to **all** columns.

    Examples
    --------
    >>> cars >> rename_with(str.upper)
    >>> cars >> rename_with(str.lower, starts_with("x"))
    >>> cars >> rename_with(lambda s: s.replace(" ", "_"))
    """
    if not callable(fn):
        raise TypeError("rename_with() fn must be callable")

    def _apply(tf):
        columns = _frame_columns(tf)
        selected = resolve_selection(tf, cols) if cols else columns
        mapping = {column: fn(column) for column in selected}
        if any(not isinstance(value, str) or not value for value in mapping.values()):
            raise TypeError("rename_with() fn must return non-empty strings")
        renamed = [mapping.get(column, column) for column in columns]
        if len(set(renamed)) != len(renamed):
            raise ValueError("rename_with() produced duplicate column names")
        groups = [mapping.get(group, group) for group in tf._groups] if tf._groups else None
        category_levels = {
            mapping.get(name, name): levels
            for name, levels in tf._category_levels.items()
        }
        if tf._backend == "pandas":
            return tf._with_pdf(
                tf._pdf.rename(columns=mapping),
                groups=groups,
                category_levels=category_levels,
            )
        return tf._with_lf(
            tf._lf.rename(mapping),
            groups=groups,
            category_levels=category_levels,
        )

    return Verb(_apply, "rename_with")


def set_names(*names: Any) -> Verb:
    """Replace **all** column names in order (rlang ``set_names`` / pandas).

    This is the bulk form for end-of-pipeline renames before ML handoff::

        features >> set_names(["x0", "x1", "x2"])
        features >> set_names("x0", "x1", "x2")
        features >> set_names([f"f{i}" for i in range(ncol(features))])

    A single callable renames every column (same as ``rename_with(fn)``)::

        features >> set_names(str.upper)

    Length must match the number of columns. For dropping names entirely into
    a matrix, use ``to_numpy()`` / ``to_numpy(columns=...)`` — arrays have no
    column names.

    See also :func:`rename` (individual ``new=old``) and :func:`rename_with`.
    """
    # Normalize arguments
    if len(names) == 1 and callable(names[0]) and not isinstance(names[0], type):
        return rename_with(names[0])

    if len(names) == 1 and isinstance(names[0], (list, tuple)):
        new_names = [str(x) for x in names[0]]
    else:
        new_names = [str(x) for x in names]

    if not new_names:
        raise TypeError(
            "set_names() requires new names, e.g. set_names('a', 'b') "
            "or set_names(['a', 'b'])"
        )
    if any(not n for n in new_names):
        raise ValueError("set_names() names must be non-empty strings")

    def _apply(tf):
        columns = _frame_columns(tf)
        if len(new_names) != len(columns):
            raise ValueError(
                f"set_names() got {len(new_names)} name(s) but frame has "
                f"{len(columns)} column(s)"
            )
        if len(set(new_names)) != len(new_names):
            raise ValueError("set_names() produced duplicate column names")
        mapping = dict(zip(columns, new_names))
        groups = [mapping.get(g, g) for g in tf._groups] if tf._groups else None
        category_levels = {
            mapping.get(name, name): levels
            for name, levels in tf._category_levels.items()
        }
        if tf._backend == "pandas":
            pdf = tf._pdf.copy()
            pdf.columns = new_names
            return tf._with_pdf(pdf, groups=groups, category_levels=category_levels)
        # Polars: rename via mapping old→new
        return tf._with_lf(
            tf._lf.rename(mapping),
            groups=groups,
            category_levels=category_levels,
        )

    return Verb(_apply, "set_names")


def _column_position(columns: list[str], value: str | int, verb: str) -> str:
    if isinstance(value, str):
        if value not in columns:
            raise KeyError(f"{verb}(): column not found: {value!r}")
        return value
    if not isinstance(value, int) or isinstance(value, bool) or value == 0:
        raise TypeError(f"{verb}(): column must be a name or non-zero integer")
    index = value - 1 if value > 0 else len(columns) + value
    if index < 0 or index >= len(columns):
        raise IndexError(f"{verb}(): column position out of range: {value}")
    return columns[index]


def pull(var: str | int = -1, *, name: str | int | None = None) -> Verb:
    """Materialize and extract one backend-native Series."""

    def _apply(tf):
        columns = _frame_columns(tf)
        column = _column_position(columns, var, "pull")
        name_column = (
            _column_position(columns, name, "pull") if name is not None else None
        )
        if tf._backend == "pandas":
            out = tf._pdf[column].copy()
            if name_column is not None:
                out.index = tf._pdf[name_column].to_numpy()
            return out
        selected = [column] if name_column is None else [column, name_column]
        out = tf._lf.select(selected).collect()
        if name_column is None:
            return out.get_column(column)
        pdf = out.to_pandas()
        series = pdf[column]
        series.index = pdf[name_column].to_numpy()
        return series

    return Verb(_apply, "pull")


def glimpse(n: int = 10) -> Verb:
    """Print a compact column-oriented preview and pass the frame through."""
    if not isinstance(n, int) or isinstance(n, bool) or n < 0:
        raise ValueError("glimpse() n must be a non-negative integer")

    def _apply(tf):
        preview = tf.preview(n)
        try:
            n_rows = len(tf)
        except Exception:
            n_rows = None
        if tf._backend == "pandas":
            print(f"Rows: {n_rows if n_rows is not None else len(tf._pdf)}")
            print(f"Columns: {tf.width}")
            for column, dtype in tf._pdf.dtypes.items():
                values = ", ".join(repr(value) for value in preview[column].tolist())
                print(f"$ {column} <{dtype}> {values}")
        else:
            schema = tf._lf.collect_schema()
            print(f"Rows: {n_rows if n_rows is not None else '?'}")
            print(f"Columns: {len(schema)}")
            for column, dtype in schema.items():
                values = ", ".join(
                    repr(value) for value in preview.get_column(column).to_list()
                )
                print(f"$ {column} <{dtype}> {values}")
        return tf

    return Verb(_apply, "glimpse")

def arrange(*keys: Any, by_group: bool = False) -> Verb:
    """Order rows. Keys may reference columns dropped by a prior ``select()``."""
    if not keys:
        raise TypeError("arrange() requires at least one key")
    if not isinstance(by_group, bool):
        raise TypeError("arrange() by_group must be a boolean")

    def _apply(tf):
        needed: set[str] = set()
        for key in keys:
            needed |= _expr_column_names(key)
        if by_group and tf._groups:
            needed |= set(tf._groups)
        wide, widened = _maybe_widen_for_select_base(tf, needed)
        effective_keys = (
            tuple([*(wide._groups or []), *keys]) if by_group else keys
        )
        keep_base = wide._select_base
        if wide._backend == "pandas":
            out = wide._with_pdf(
                _pe().do_arrange(wide._pdf, effective_keys),
                groups=wide._groups,
                select_base=keep_base,
            )
            return _finish_select_base_verb(wide, out, widened=widened)
        expressions = []
        descending = []
        for key in effective_keys:
            node = key.node if isinstance(key, Expr) else None
            if node is not None and node[0] == "desc":
                expressions.append(to_polars(Expr(node[1])))
                descending.append(True)
            else:
                expressions.append(_plx(key))
                descending.append(False)
        expressions = _dplyr_sort_expressions(wide._lf, expressions)
        out = wide._with_lf(
            wide._lf.sort(
                expressions,
                descending=descending,
                nulls_last=True,
                maintain_order=True,
            ),
            groups=wide._groups,
            select_base=keep_base,
        )
        return _finish_select_base_verb(wide, out, widened=widened)

    return Verb(_apply, "arrange")


def distinct(
    *cols: Any,
    keep_all: bool = False,
    maintain_order: bool = True,
    **computed: Any,
) -> Verb:
    """Keep unique rows, optionally creating keys before deduplication."""
    if not isinstance(keep_all, bool):
        raise TypeError("distinct() keep_all must be a boolean")
    if not isinstance(maintain_order, bool):
        raise TypeError("distinct() maintain_order must be a boolean")

    def _apply(tf):
        from tidy3.masking import NamedAssign

        # Fold NamedAssign positionals into computed kwargs
        computed_map = dict(computed)
        plain_cols: list[Any] = []
        for spec in cols:
            if isinstance(spec, NamedAssign):
                computed_map[spec.name] = spec.value
            else:
                plain_cols.append(spec)
        work = tf >> mutate(**computed_map) if computed_map else tf
        columns = _frame_columns(work)
        if plain_cols:
            selected = resolve_selection(work, plain_cols)
        elif computed_map:
            selected = []
        else:
            selected = columns
        selected = [*selected, *computed_map]
        keys = list(dict.fromkeys([*(work._groups or []), *selected]))
        if work._backend == "pandas":
            output = work._pdf.drop_duplicates(subset=keys).reset_index(drop=True)
            if (plain_cols or computed_map) and not keep_all:
                output = output.loc[:, keys]
            return work._with_pdf(output, groups=work._groups)
        lf = work._lf.unique(
            subset=keys,
            keep="first",
            maintain_order=maintain_order,
        )
        if (plain_cols or computed_map) and not keep_all:
            lf = lf.select(keys)
        return work._with_lf(
            lf,
            groups=work._groups,
        )

    return Verb(_apply, "distinct")


def _slice_positions(rows: tuple[Any, ...]) -> tuple[int, ...]:
    positions: list[int] = []
    for value in rows:
        if isinstance(value, int) and not isinstance(value, bool):
            positions.append(value)
            continue
        if isinstance(value, (str, bytes)):
            raise TypeError("slice() positions must be integers")
        try:
            values = list(value)
        except TypeError as e:
            raise TypeError("slice() positions must be integers") from e
        if any(not isinstance(item, int) or isinstance(item, bool) for item in values):
            raise TypeError("slice() positions must be integers")
        positions.extend(values)
    signs = {1 if position > 0 else -1 for position in positions if position}
    if len(signs) > 1:
        raise ValueError("slice() cannot mix positive and negative positions")
    return tuple(positions)


def _temp_column(columns: list[str], base: str) -> str:
    name = base
    while name in columns:
        name += "_"
    return name


def slice(*rows: Any, by: Any = None) -> Verb:  # noqa: A001
    """Select 1-based row positions per group; negative positions exclude."""
    positions = _slice_positions(rows)

    def _apply(tf):
        groups, transient = _operation_groups(tf, by, "slice")
        if tf._backend == "pandas":
            return tf._with_pdf(
                _pe().do_slice(tf._pdf, positions, groups),
                groups=None if transient else tf._groups,
            )
        columns = _frame_columns(tf)
        row_name = _temp_column(columns, "__tidy3_slice_row")
        global_name = _temp_column([*columns, row_name], "__tidy3_slice_global")
        group_name = _temp_column(
            [*columns, row_name, global_name], "__tidy3_slice_group"
        )
        order_name = _temp_column(
            [*columns, row_name, global_name, group_name], "__tidy3_slice_order"
        )
        local_index = pl.int_range(pl.len())
        if groups:
            local_index = local_index.over(groups)
        base = tf._lf.with_row_index(global_name).with_columns(
            local_index.alias(row_name)
        )
        if groups:
            base = base.with_columns(
                pl.col(global_name).min().over(groups).alias(group_name)
            )
        else:
            base = base.with_columns(pl.lit(0).alias(group_name))

        positive = [position - 1 for position in positions if position > 0]
        excluded = [abs(position) - 1 for position in positions if position < 0]
        if positive:
            selector = pl.DataFrame(
                {row_name: positive, order_name: list(range(len(positive)))}
            ).lazy()
            lf = (
                base.join(
                    selector,
                    on=row_name,
                    how="inner",
                    maintain_order="left_right",
                )
                .sort([group_name, order_name])
                .drop(row_name, global_name, group_name, order_name)
            )
        elif excluded:
            lf = base.filter(~pl.col(row_name).is_in(excluded)).drop(
                row_name, global_name, group_name
            )
        else:
            lf = base.head(0).drop(row_name, global_name, group_name)
        return tf._with_lf(lf, groups=None if transient else tf._groups)

    return Verb(_apply, "slice")


def _slice_size_args(
    n: int | None, prop: float | None
) -> tuple[int | None, float | None]:
    if n is not None and prop is not None:
        raise ValueError("supply only one of n= or prop=")
    if n is None and prop is None:
        n = 1
    if n is not None and (not isinstance(n, int) or isinstance(n, bool)):
        raise TypeError("n must be an integer")
    if prop is not None:
        if isinstance(prop, bool) or not isinstance(prop, (int, float)):
            raise TypeError("prop must be a finite number")
        prop = float(prop)
        if not math.isfinite(prop):
            raise ValueError("prop must be a finite number")
    return n, prop


def _slice_target(n: int | None, prop: float | None) -> pl.Expr:
    if n is not None:
        return pl.lit(n) if n >= 0 else pl.len() + n
    factor = prop if prop >= 0 else 1.0 + prop
    return (pl.len() * factor).floor()


def _slice_local_index(groups: list[str] | None) -> pl.Expr:
    index = pl.int_range(pl.len())
    return index.over(groups) if groups else index


def _slice_group_len(groups: list[str] | None) -> pl.Expr:
    return pl.len().over(groups) if groups else pl.len()


def slice_head(
    *, n: int | None = None, prop: float | None = None, by: Any = None
) -> Verb:
    n, prop = _slice_size_args(n, prop)

    def _apply(tf):
        groups, transient = _operation_groups(tf, by, "slice_head")
        if tf._backend == "pandas":
            pdf = _pe().do_slice_size(
                tf._pdf, n, prop, groups, tail=False
            )
            return tf._with_pdf(pdf, groups=None if transient else tf._groups)
        target = _windowed(_slice_target(n, prop), groups)
        predicate = _slice_local_index(groups) < target
        return tf._with_lf(
            tf._lf.filter(predicate), groups=None if transient else tf._groups
        )

    return Verb(_apply, "slice_head")


def slice_tail(
    *, n: int | None = None, prop: float | None = None, by: Any = None
) -> Verb:
    n, prop = _slice_size_args(n, prop)

    def _apply(tf):
        groups, transient = _operation_groups(tf, by, "slice_tail")
        if tf._backend == "pandas":
            pdf = _pe().do_slice_size(tf._pdf, n, prop, groups, tail=True)
            return tf._with_pdf(pdf, groups=None if transient else tf._groups)
        target = _windowed(_slice_target(n, prop), groups)
        predicate = _slice_local_index(groups) >= (
            _slice_group_len(groups) - target
        )
        return tf._with_lf(
            tf._lf.filter(predicate), groups=None if transient else tf._groups
        )

    return Verb(_apply, "slice_tail")


def _slice_extreme(
    order_by: Any,
    *,
    n: int | None,
    prop: float | None,
    with_ties: bool,
    na_rm: bool,
    largest: bool,
    verb_name: str,
    by: Any,
) -> Verb:
    n, prop = _slice_size_args(n, prop)

    def _apply(tf):
        needed = _expr_column_names(order_by)
        # by= may name columns dropped by a prior select; include them for widen.
        if by is not None:
            if isinstance(by, str):
                needed.add(by)
            elif isinstance(by, (list, tuple)):
                for item in by:
                    needed |= _expr_column_names(item) if not isinstance(item, str) else {item}
            else:
                needed |= _expr_column_names(by)
        wide, widened = _maybe_widen_for_select_base(tf, needed)
        groups, transient = _operation_groups(wide, by, verb_name)
        keep_base = wide._select_base
        if wide._backend == "pandas":
            pdf = _pe().do_slice_extreme(
                wide._pdf,
                order_by,
                n,
                prop,
                groups,
                largest=largest,
                with_ties=with_ties,
                na_rm=na_rm,
            )
            out = wide._with_pdf(
                pdf,
                groups=None if transient else wide._groups,
                select_base=keep_base,
            )
            return _finish_select_base_verb(wide, out, widened=widened)
        columns = _frame_columns(wide)
        order_name = _temp_column(columns, "__tidy3_slice_order")
        global_name = _temp_column([*columns, order_name], "__tidy3_slice_global")
        group_name = _temp_column(
            [*columns, order_name, global_name], "__tidy3_slice_group"
        )
        if order_by is Ellipsis:
            raise TypeError(
                f"{verb_name}() got Ellipsis (...) as order_by — that is only a "
                "docs placeholder. Pass a column name or expression, e.g. "
                f'{verb_name}(order_by="hp", n=1)'
            )
        value = pl.col(order_by) if isinstance(order_by, str) else _plx(order_by)
        if not isinstance(value, pl.Expr):
            raise TypeError(
                f"{verb_name}() order_by must be a column or expression, "
                f"got {type(order_by).__name__}"
            )
        lf = wide._lf.with_columns(value.alias(order_name))
        cleanup = [order_name]
        if groups:
            lf = lf.with_row_index(global_name).with_columns(
                pl.col(global_name).min().over(groups).alias(group_name)
            )
            cleanup.extend([global_name, group_name])
        if na_rm:
            lf = lf.filter(pl.col(order_name).is_not_null())
        target = _windowed(_slice_target(n, prop), groups)
        if with_ties:
            rank = pl.col(order_name).rank(method="min", descending=largest)
            null_rank = pl.col(order_name).count() + 1
            rank = _windowed(rank, groups).fill_null(
                _windowed(null_rank, groups)
            )
            lf = lf.filter(rank <= target)
            sort_columns = [*([group_name] if groups else []), order_name]
            lf = lf.sort(
                sort_columns,
                descending=[False] * bool(groups) + [largest],
                nulls_last=True,
                maintain_order=True,
            )
        else:
            sort_columns = [*([group_name] if groups else []), order_name]
            lf = lf.sort(
                sort_columns,
                descending=[False] * bool(groups) + [largest],
                nulls_last=True,
                maintain_order=True,
            )
            lf = lf.filter(_slice_local_index(groups) < target)
        out = wide._with_lf(
            lf.drop(*cleanup),
            groups=None if transient else wide._groups,
            select_base=keep_base,
        )
        return _finish_select_base_verb(wide, out, widened=widened)

    return Verb(_apply, verb_name)


def slice_min(
    order_by: Any,
    *,
    n: int | None = None,
    prop: float | None = None,
    with_ties: bool = True,
    na_rm: bool = False,
    by: Any = None,
) -> Verb:
    """Rows with the smallest ``order_by`` values (ties optional).

    ``order_by`` may reference columns dropped by a prior ``select()``.
    """
    return _slice_extreme(
        order_by,
        n=n,
        prop=prop,
        with_ties=with_ties,
        na_rm=na_rm,
        largest=False,
        verb_name="slice_min",
        by=by,
    )


def slice_max(
    order_by: Any,
    *,
    n: int | None = None,
    prop: float | None = None,
    with_ties: bool = True,
    na_rm: bool = False,
    by: Any = None,
) -> Verb:
    """Rows with the largest ``order_by`` values (ties optional).

    ``order_by`` may reference columns dropped by a prior ``select()``.
    """
    return _slice_extreme(
        order_by,
        n=n,
        prop=prop,
        with_ties=with_ties,
        na_rm=na_rm,
        largest=True,
        verb_name="slice_max",
        by=by,
    )


def rowwise(*cols: Any) -> Verb:
    """Group a frame one row at a time, optionally preserving identifier columns."""

    def _apply(tf):
        identifiers = resolve_selection(tf, cols) if cols else None
        if tf._backend == "pandas":
            return tf._with_pdf(
                tf._pdf, groups=identifiers, rowwise=True
            )
        return tf._with_lf(tf._lf, groups=identifiers, rowwise=True)

    return Verb(_apply, "rowwise")


def group_by(
    *cols: Any,
    add: bool = False,
    drop: bool | None = None,
    **computed: Any,
) -> Verb:
    if not cols and not computed:
        raise TypeError("group_by() requires at least one column")
    if not isinstance(add, bool):
        raise TypeError("group_by() add must be a boolean")
    if drop is not None and not isinstance(drop, bool):
        raise TypeError("group_by() drop must be a boolean or None")

    def _apply(tf):
        from tidy3.masking import NamedAssign

        context = _group_context(tf, None)
        computed_map = dict(computed)
        plain_cols: list[Any] = []
        for spec in cols:
            if isinstance(spec, NamedAssign):
                computed_map[spec.name] = spec.value
            else:
                plain_cols.append(spec)
        assignments = {
            name: _resolved_value(context, value)
            for name, value in computed_map.items()
        }
        if tf._backend == "pandas":
            pdf = tf._pdf
            for stage in _assignment_stages(assignments):
                pdf = _pe().do_mutate(pdf, stage, None)
            selected = resolve_selection(
                tf._with_pdf(pdf, groups=None, rowwise=False), plain_cols
            )
            groups = list(
                dict.fromkeys(
                    [
                        *((tf._groups or []) if add else []),
                        *selected,
                        *computed_map,
                    ]
                )
            )
            group_drop = (
                tf._group_drop if drop is None and tf._groups else
                True if drop is None else drop
            )
            return tf._with_pdf(
                pdf,
                groups=groups,
                rowwise=False,
                group_drop=group_drop,
            )
        lf = tf._lf
        for stage in _assignment_stages(assignments):
            lf = lf.with_columns(
                **{name: _pl_expr(value) for name, value in stage.items()}
            )
        selected = resolve_selection(
            tf._with_lf(lf, groups=None, rowwise=False), plain_cols
        )
        groups = list(
            dict.fromkeys(
                [
                    *((tf._groups or []) if add else []),
                    *selected,
                    *computed_map,
                ]
            )
        )
        group_drop = (
            tf._group_drop if drop is None and tf._groups else
            True if drop is None else drop
        )
        return tf._with_lf(
            lf,
            groups=groups,
            rowwise=False,
            group_drop=group_drop,
        )

    return Verb(_apply, "group_by")


def ungroup(*cols: Any) -> Verb:
    def _apply(tf):
        if cols:
            selected = resolve_selection(tf, cols)
            not_grouped = [name for name in selected if name not in (tf._groups or [])]
            if not_grouped:
                raise ValueError(
                    f"ungroup() columns are not grouping columns: {not_grouped}"
                )
            groups = [name for name in (tf._groups or []) if name not in selected]
            groups = groups or None
        else:
            groups = None
        if tf._backend == "pandas":
            return tf._with_pdf(tf._pdf, groups=groups, rowwise=False)
        return tf._with_lf(tf._lf, groups=groups, rowwise=False)

    return Verb(_apply, "ungroup")


def with_groups(cols: Any, fn: Any) -> Verb:
    """Run a callable with temporary grouping, then restore prior groups."""
    if not callable(fn):
        raise TypeError("with_groups() fn must be callable")
    selected = (cols,) if isinstance(cols, str) else tuple(cols)
    if not selected:
        raise TypeError("with_groups() requires at least one grouping column")

    def _apply(tf):
        original_groups = tf._groups
        grouped = tf >> group_by(*selected)
        result = fn(grouped)
        if not hasattr(result, "_backend"):
            raise TypeError("with_groups() fn must return a TidyFrame")
        if result._backend == "pandas":
            return result._with_pdf(
                result._pdf,
                groups=original_groups,
                rowwise=False,
                group_drop=tf._group_drop,
            )
        return result._with_lf(
            result._lf,
            groups=original_groups,
            rowwise=False,
            group_drop=tf._group_drop,
        )

    return Verb(_apply, "with_groups")


def _grouped_parts(
    tf: Any, cols: tuple[Any, ...]
) -> list[tuple[dict[str, Any], Any, tuple[int, ...]]]:
    """Materialize stable group partitions for Python group callbacks."""
    from tidy3.frame import tidy

    names = resolve_selection(tf, cols) if cols else list(tf._groups or [])
    if not names:
        raise ValueError("group workflow requires at least one grouping column")
    pdf = tf.collect(as_="pandas")
    parts: list[tuple[dict[str, Any], Any, tuple[int, ...]]] = []
    grouping_key = names[0] if len(names) == 1 else names
    for key, positions in pdf.groupby(grouping_key, sort=False, dropna=False).groups.items():
        key_tuple = key if isinstance(key, tuple) else (key,)
        subset = pdf.loc[positions].reset_index(drop=True)
        key_map = dict(zip(names, key_tuple))
        row_positions = tuple(int(index) for index in pdf.index.get_indexer(positions))
        parts.append((key_map, tidy(subset, backend=tf._backend), row_positions))
    return parts


def group_split(*cols: Any) -> Verb:
    """Split a grouped frame into a list of ungrouped ``TidyFrame`` objects."""

    def _apply(tf):
        return [part for _, part, _ in _grouped_parts(tf, tuple(cols))]

    return Verb(_apply, "group_split")


def group_map(fn: Any, *cols: Any) -> Verb:
    """Apply ``fn(.x, .y)`` to each group and return the callback results."""
    if not callable(fn):
        raise TypeError("group_map() fn must be callable")

    def _apply(tf):
        outputs = []
        for key, part, rows in _grouped_parts(tf, tuple(cols)):
            token = _group_rows_token(rows)
            try:
                outputs.append(fn(part, key))
            finally:
                _reset_group_rows(token)
        return outputs

    return Verb(_apply, "group_map")


def group_modify(fn: Any, *cols: Any) -> Verb:
    """Apply a frame-returning callback per group and bind the results."""
    if not callable(fn):
        raise TypeError("group_modify() fn must be callable")

    def _apply(tf):
        from tidy3.frame import tidy

        outputs = []
        for key, part, rows in _grouped_parts(tf, tuple(cols)):
            token = _group_rows_token(rows)
            try:
                result = fn(part, key)
            finally:
                _reset_group_rows(token)
            if hasattr(result, "collect"):
                outputs.append(result.collect(as_="pandas"))
            else:
                import pandas as pd

                outputs.append(pd.DataFrame(result))
        if not outputs:
            return tidy({}, backend=tf._backend)
        import pandas as pd

        return tidy(pd.concat(outputs, ignore_index=True), backend=tf._backend)

    return Verb(_apply, "group_modify")


def group_nest(*cols: Any, name: str = "data") -> Verb:
    """Nest each group's non-key columns into a list column.

    Polars builds a lazy ``group_by().agg(struct)`` plan (same shape as
    ``tidyr.nest``) — typically faster than building Python/pandas nested
    DataFrames. The pandas path uses one groupby pass and stores nested
    DataFrames for ergonomic row access. Both return an ungrouped frame
    with one row per group.

    For counts only, prefer ``count()`` / ``tally()``; nesting full row
    payloads is more work than ``groupby.size()``.
    """
    if not isinstance(name, str) or not name:
        raise TypeError("group_nest() name must be a non-empty string")

    def _apply(tf):
        from tidy3.frame import tidy

        keys = resolve_selection(tf, cols) if cols else list(tf._groups or [])
        if not keys:
            raise ValueError("group_nest() requires grouping columns or group_by()")
        key_set = set(keys)
        nested_cols = [
            column for column in _frame_columns(tf) if column not in key_set
        ]

        if tf._backend == "pandas":
            import pandas as pd

            pdf = tf._pdf
            grouping_key = keys[0] if len(keys) == 1 else keys
            # Pre-slice nested columns once; avoid per-group drop(columns=keys).
            if nested_cols:
                body = pdf.loc[:, nested_cols]
            else:
                body = None
            rows: list[dict[str, Any]] = []
            for key, positions in pdf.groupby(
                grouping_key, sort=False, dropna=False, observed=True
            ).groups.items():
                key_tuple = key if isinstance(key, tuple) else (key,)
                row = dict(zip(keys, key_tuple))
                if body is None:
                    row[name] = pd.DataFrame(index=range(len(positions)))
                else:
                    # positions is an Index into the original frame.
                    row[name] = body.loc[positions].reset_index(drop=True)
                rows.append(row)
            if not rows:
                empty = {key: pd.Series(dtype=pdf[key].dtype) for key in keys}
                empty[name] = pd.Series(dtype=object)
                return tidy(pd.DataFrame(empty), backend="pandas")
            return tidy(pd.DataFrame(rows), backend="pandas")

        if nested_cols:
            # exclude() keeps the plan free of a Python column list rebuild
            # when the schema is wide; maintain_order matches dplyr appearance.
            if len(keys) == 1:
                payload = pl.struct(pl.exclude(keys[0]))
            else:
                payload = pl.struct(pl.exclude(keys))
            agg = payload.alias(name)
        else:
            # Preserve group size with a list of nulls (no non-key columns).
            agg = pl.repeat(None, pl.len()).alias(name)
        lf = tf._lf.group_by(keys, maintain_order=True).agg(agg)
        return tf._with_lf(lf, groups=None, rowwise=False)

    return Verb(_apply, "group_nest")


def _complete_polars_empty_groups(
    tf: Any,
    result: pl.LazyFrame,
    groups: list[str],
    assignments: dict[str, Any],
) -> pl.LazyFrame:
    """Add factor levels omitted by Polars group_by when drop=False."""
    schema = tf._lf.collect_schema()
    grids: list[pl.LazyFrame] = []
    for name in groups:
        levels = tf._category_levels.get(name)
        if levels is not None:
            grid = pl.DataFrame({name: levels}).lazy().with_columns(
                pl.col(name).cast(schema[name])
            )
        else:
            grid = tf._lf.select(name).unique(maintain_order=True)
        grids.append(grid)
    grid = grids[0]
    for other in grids[1:]:
        grid = grid.join(other, how="cross")

    marker = _temp_column(
        [*result.collect_schema().names(), *groups],
        "__tidy3_observed_group",
    )
    defaults_frame = tf._lf.limit(0).select(
        *(
            (_plx(expr) if isinstance(_plx(expr), pl.Expr) else pl.lit(expr))
            .alias(name)
            for name, expr in assignments.items()
        )
    ).collect()
    defaults = defaults_frame.row(0, named=True)
    joined = grid.join(
        result.with_columns(pl.lit(True).alias(marker)),
        on=groups,
        how="left",
        maintain_order="left",
    )
    return joined.select(
        *groups,
        *(
            pl.when(pl.col(marker).is_null())
            .then(pl.lit(defaults[name]))
            .otherwise(pl.col(name))
            .alias(name)
            for name in assignments
        ),
    )


def summarise(
    *specs: Any,
    by: Any = None,
    groups: str | None = None,
    **kwargs: Any,
) -> Verb:
    """Aggregate; uses current ``group_by`` if set."""
    if not specs and not kwargs:
        raise TypeError("summarise() requires at least one aggregation")
    if groups not in {None, "drop_last", "drop", "keep", "rowwise"}:
        raise ValueError(
            "summarise() groups must be drop_last, drop, keep, rowwise, or None"
        )
    if by is not None and groups not in {None, "drop"}:
        raise ValueError("summarise() groups cannot be used with by=")
    requested_groups = groups

    def _apply(tf):
        operation_groups, transient = _operation_groups(tf, by, "summarise")
        input_groups = list(operation_groups or [])
        grouping_policy = (
            "keep" if tf._rowwise and requested_groups is None
            else "drop_last" if requested_groups is None
            else requested_groups
        )
        if transient or grouping_policy == "drop":
            result_groups = None
        elif grouping_policy == "keep":
            result_groups = input_groups or None
        elif grouping_policy == "rowwise":
            result_groups = input_groups or None
        else:
            result_groups = input_groups[:-1] or None
        result_rowwise = grouping_policy == "rowwise"
        context = _group_context(tf, operation_groups)
        assignments = _expanded_assignments(context, specs, kwargs, "summarise")
        assignments = _sequential_summary_assignments(assignments)
        if tf._rowwise:
            marker = _temp_column(_frame_columns(tf), "__tidy3_rowwise")
            keys = [*(tf._groups or []), marker]
            if tf._backend == "pandas":
                work = tf._pdf.assign(**{marker: range(len(tf._pdf))})
                pdf = _pe().do_reframe(
                    work, assignments, keys, sort_groups=False
                ).drop(columns=marker)
                return tf._with_pdf(
                    pdf, groups=result_groups, rowwise=result_rowwise
                )
            named = []
            for name, expr in assignments.items():
                named.append(_pl_expr(expr).alias(name))
            lf = (
                tf._lf.with_row_index(marker)
                .group_by(keys, maintain_order=True)
                .agg(named)
                .drop(marker)
            )
            return tf._with_lf(
                lf, groups=result_groups, rowwise=result_rowwise
            )
        if tf._backend == "pandas":
            return tf._with_pdf(
                _pe().do_summarise(
                    tf._pdf,
                    assignments,
                    operation_groups,
                    sort_groups=not transient,
                    observed=tf._group_drop or transient,
                ),
                groups=result_groups,
                rowwise=result_rowwise,
            )
        named = []
        for name, expr in assignments.items():
            e = _plx(expr)
            if isinstance(e, pl.Expr):
                named.append(e.alias(name))
            else:
                named.append(pl.lit(e).alias(name))
        if operation_groups:
            lf = tf._lf.group_by(
                operation_groups, maintain_order=transient
            ).agg(named)
            if not transient and not tf._group_drop:
                lf = _complete_polars_empty_groups(
                    tf, lf, input_groups, assignments
                )
            if not transient:
                lf = lf.sort(
                    _dplyr_sort_expressions(lf, list(operation_groups)),
                    nulls_last=True,
                    maintain_order=True,
                )
            return tf._with_lf(
                lf, groups=result_groups, rowwise=result_rowwise
            )
        lf = tf._lf.select(named)
        return tf._with_lf(
            lf, groups=result_groups, rowwise=result_rowwise
        )

    return Verb(_apply, "summarise")


summarize = summarise


def reframe(*specs: Any, by: Any = None, **kwargs: Any) -> Verb:
    """Return zero or more rows per group; the result is always ungrouped."""
    if not specs and not kwargs:
        raise TypeError("reframe() requires at least one expression")

    def _apply(tf):
        groups, transient = _operation_groups(tf, by, "reframe")
        context = _group_context(tf, groups)
        assignments = _expanded_assignments(context, specs, kwargs, "reframe")
        if tf._backend == "pandas":
            work = tf._pdf
            marker = None
            if tf._rowwise:
                marker = _temp_column(_frame_columns(tf), "__tidy3_rowwise")
                work = work.assign(**{marker: range(len(work))})
                groups = [*(tf._groups or []), marker]
            pdf = _pe().do_reframe(
                work,
                assignments,
                groups,
                sort_groups=not transient and marker is None,
            )
            if marker is not None:
                pdf = pdf.drop(columns=marker)
            return tf._with_pdf(pdf, groups=None, rowwise=False)

        named = [_pl_expr(expr).alias(name) for name, expr in assignments.items()]
        lf = tf._lf
        groups = list(groups or [])
        marker = None
        if tf._rowwise:
            marker = _temp_column(_frame_columns(tf), "__tidy3_rowwise")
            lf = lf.with_row_index(marker)
            groups.append(marker)
        if groups:
            lf = lf.group_by(groups, maintain_order=True).agg(named)
            if not transient and marker is None:
                lf = lf.sort(
                    _dplyr_sort_expressions(lf, groups),
                    nulls_last=True,
                    maintain_order=True,
                )
        else:
            lf = lf.select(named)
        schema = lf.collect_schema()
        list_columns = [
            name
            for name in assignments
            if schema[name].base_type() == pl.List
        ]
        if list_columns:
            lf = lf.explode(list_columns, empty_as_null=False)
        if marker is not None:
            lf = lf.drop(marker)
        return tf._with_lf(lf, groups=None, rowwise=False)

    return Verb(_apply, "reframe")


def _count_name(tf: Any, requested: str | None, groups: list[str]) -> str:
    if tf._backend == "pandas":
        columns = list(tf._pdf.columns)
    else:
        columns = tf._lf.collect_schema().names()
    if requested is None:
        name = "n"
        while name in columns:
            name += "n"
    elif not isinstance(requested, str) or not requested:
        raise TypeError("count name must be a non-empty string or None")
    else:
        name = requested
    if name in groups:
        raise ValueError(f"count name {name!r} conflicts with a grouping column")
    return name


def _count_rows(
    tf: Any,
    cols: tuple[str, ...],
    *,
    wt: Any,
    sort: bool,
    name: str | None,
    result_groups: list[str] | None,
    drop: bool,
):
    eff = list(dict.fromkeys([*(tf._groups or []), *cols]))
    out_name = _count_name(tf, name, eff)
    if tf._backend == "pandas":
        pdf = _pe().do_count(
            tf._pdf,
            tuple(eff),
            out_name,
            wt=wt,
            sort=sort,
            observed=drop,
        )
        return tf._with_pdf(pdf, groups=result_groups)

    if wt is None:
        agg = pl.len()
    elif isinstance(wt, str):
        agg = pl.col(wt).sum()
    else:
        agg = _plx(wt)
        if not isinstance(agg, pl.Expr):
            raise TypeError("count wt must be a column name or expression")
        agg = agg.sum()
    if eff:
        lf = tf._lf.group_by(eff).agg(agg.alias(out_name))
        if not drop:
            lf = _complete_polars_empty_groups(
                tf, lf, eff, {out_name: agg}
            )
    else:
        lf = tf._lf.select(agg.alias(out_name))
    if eff:
        group_keys = _dplyr_sort_expressions(lf, eff)
        if sort:
            lf = lf.sort(
                [pl.col(out_name), *group_keys],
                descending=[True, *([False] * len(group_keys))],
                nulls_last=True,
                maintain_order=True,
            )
        else:
            lf = lf.sort(
                group_keys, nulls_last=True, maintain_order=True
            )
    elif sort:
        lf = lf.sort(out_name, descending=True, maintain_order=True)
    return tf._with_lf(lf, groups=result_groups)


def count(
    *cols: str,
    wt: Any = None,
    sort: bool = False,
    name: str | None = None,
    drop: bool | None = None,
) -> Verb:
    """Count rows per group (dplyr): existing ``group_by`` groups plus *cols*.

    ``group_by("a") >> count()`` ≡ R's ``group_by(a) %>% tally()``.
    Existing groups are used for the calculation and preserved on the result.
    """
    if drop is not None and not isinstance(drop, bool):
        raise TypeError("count() drop must be a boolean or None")

    def _apply(tf):
        groups = list(tf._groups) if tf._groups else None
        effective_drop = (
            tf._group_drop if drop is None and tf._groups else
            True if drop is None else drop
        )
        return _count_rows(
            tf,
            cols,
            wt=wt,
            sort=sort,
            name=name,
            result_groups=groups,
            drop=effective_drop,
        )

    return Verb(_apply, "count")


def tally(*, wt: Any = None, sort: bool = False, name: str | None = None) -> Verb:
    """Count rows using existing groups, with summarise-style group dropping."""

    def _apply(tf):
        groups = list(tf._groups[:-1]) if tf._groups and len(tf._groups) > 1 else None
        return _count_rows(
            tf,
            (),
            wt=wt,
            sort=sort,
            name=name,
            result_groups=groups,
            drop=tf._group_drop,
        )

    return Verb(_apply, "tally")


def _add_count_rows(
    tf: Any,
    cols: tuple[str, ...],
    *,
    wt: Any,
    sort: bool,
    name: str | None,
):
    eff = list(dict.fromkeys([*(tf._groups or []), *cols]))
    out_name = _count_name(tf, name, eff)
    if tf._backend == "pandas":
        pdf = _pe().do_add_count(
            tf._pdf, tuple(eff), out_name, wt=wt, sort=sort
        )
        return tf._with_pdf(pdf, groups=tf._groups)
    if wt is None:
        value = pl.len()
    elif isinstance(wt, str):
        value = pl.col(wt).sum()
    else:
        value = _plx(wt)
        if not isinstance(value, pl.Expr):
            raise TypeError("add_count wt must be a column name or expression")
        value = value.sum()
    if eff:
        value = value.over(eff)
    lf = tf._lf.with_columns(value.alias(out_name))
    if sort:
        lf = lf.sort(out_name, descending=True, maintain_order=True)
    return tf._with_lf(lf, groups=tf._groups)


def add_count(
    *cols: str,
    wt: Any = None,
    sort: bool = False,
    name: str | None = None,
) -> Verb:
    """Add counts for existing groups plus *cols*, without collapsing rows."""

    def _apply(tf):
        return _add_count_rows(tf, cols, wt=wt, sort=sort, name=name)

    return Verb(_apply, "add_count")


def add_tally(
    *, wt: Any = None, sort: bool = False, name: str | None = None
) -> Verb:
    """Add counts for the current groups, without collapsing rows."""

    def _apply(tf):
        return _add_count_rows(tf, (), wt=wt, sort=sort, name=name)

    return Verb(_apply, "add_tally")


def head(n: int = 10) -> Verb:
    """First *n* rows — per group when grouped (dplyr ``slice_head``)."""
    verb = slice_head(n=n)
    verb.name = "head"
    return verb


def sample_n(n: int, *, seed: int | None = None) -> Verb:
    """Random *n* rows (per group when grouped), lazy on polars.

    Polars: a shuffled-index filter, so the plan never materializes the
    whole dataset and original row order is preserved.
    """

    if not isinstance(n, int) or isinstance(n, bool):
        raise TypeError("sample_n() n must be an integer")

    def _apply(tf):
        if tf._backend == "pandas":
            return tf._with_pdf(
                _pe().do_sample_n(tf._pdf, n, seed, tf._groups), groups=tf._groups
            )
        target = pl.lit(n) if n >= 0 else pl.len() + n
        pred = pl.int_range(pl.len()).shuffle(seed=seed) < target
        return tf._with_lf(tf._lf.filter(_windowed(pred, tf._groups)), groups=tf._groups)

    return Verb(_apply, "sample_n")


def sample_frac(frac: float, *, seed: int | None = None) -> Verb:
    """Random fraction per group, truncating fractional row counts toward zero."""

    if isinstance(frac, bool) or not isinstance(frac, (int, float)):
        raise TypeError("sample_frac() frac must be a finite number")
    frac = float(frac)
    if not math.isfinite(frac):
        raise ValueError("sample_frac() frac must be a finite number")

    def _apply(tf):
        if tf._backend == "pandas":
            return tf._with_pdf(
                _pe().do_sample_frac(tf._pdf, frac, seed, tf._groups), groups=tf._groups
            )
        factor = frac if frac >= 0 else 1.0 + frac
        target = (pl.len() * factor).floor()
        pred = pl.int_range(pl.len()).shuffle(seed=seed) < target
        return tf._with_lf(tf._lf.filter(_windowed(pred, tf._groups)), groups=tf._groups)

    return Verb(_apply, "sample_frac")


def slice_sample(
    *,
    n: int | None = None,
    prop: float | None = None,
    weight_by: Any = None,
    replace: bool = False,
    seed: int | None = None,
    by: Any = None,
) -> Verb:
    """Random rows, per group when grouped (dplyr ``slice_sample``)."""
    n, prop = _slice_size_args(n, prop)

    def _apply(tf):
        groups, transient = _operation_groups(tf, by, "slice_sample")
        if tf._backend == "pandas":
            pdf = _pe().do_slice_sample(
                tf._pdf,
                n,
                prop,
                weight_by,
                replace,
                seed,
                groups,
            )
            return tf._with_pdf(pdf, groups=None if transient else tf._groups)
        if weight_by is not None:
            columns = _frame_columns(tf)
            row_name = _temp_column(columns, "__tidy3_sample_row")
            weight_name = _temp_column(
                [*columns, row_name], "__tidy3_sample_weight"
            )
            key_name = _temp_column(
                [*columns, row_name, weight_name], "__tidy3_sample_key"
            )
            seed_value = (
                seed if seed is not None else random.randrange(2**32)
            )
            weight = (
                pl.col(weight_by)
                if isinstance(weight_by, str)
                else _plx(weight_by)
            )
            if not isinstance(weight, pl.Expr):
                raise TypeError(
                    "slice_sample() weight_by must be a column or expression"
                )
            base = tf._lf.with_row_index(row_name).with_columns(
                weight.cast(pl.Float64).alias(weight_name)
            )
            invalid = base.filter(
                pl.col(weight_name).is_null()
                | ~pl.col(weight_name).is_finite()
                | (pl.col(weight_name) < 0)
            )
            base = _pl_guard_no_rows(
                base,
                invalid,
                "slice_sample() weights must be finite, non-missing, and non-negative",
            )
            target = _windowed(_slice_target(n, prop), groups)
            positive = _windowed(
                (pl.col(weight_name) > 0).sum(), groups
            )
            if not replace:
                too_few = base.filter(positive < target)
                base = _pl_guard_no_rows(
                    base,
                    too_few,
                    "slice_sample() cannot take a larger sample than the number of positive weights",
                )
                modulus = float(2**53 - 1)
                uniform = (
                    (
                        pl.col(row_name).hash(seed=seed_value)
                        % (2**53 - 1)
                    ).cast(pl.Float64)
                    + 1.0
                ) / modulus
                key = (-uniform.log() / pl.col(weight_name)).alias(key_name)
                lf = base.with_columns(key)
                rank = _windowed(
                    pl.col(key_name).rank(method="ordinal"), groups
                )
                lf = lf.filter(rank <= target)
                sort_keys = [*(groups or []), key_name]
                lf = lf.sort(sort_keys, maintain_order=True)
                return tf._with_lf(
                    lf.select(columns),
                    groups=None if transient else tf._groups,
                )

            no_positive = base.filter((positive == 0) & (target > 0))
            base = _pl_guard_no_rows(
                base,
                no_positive,
                "slice_sample() has too few positive weights",
            )

            group_keys = list(groups or [])
            dummy = None
            if not group_keys:
                dummy = _temp_column(
                    [*columns, row_name, weight_name, key_name],
                    "__tidy3_sample_group",
                )
                base = base.with_columns(pl.lit(0).alias(dummy))
                group_keys = [dummy]
            cumulative = _temp_column(
                [*columns, row_name, weight_name, key_name, *group_keys],
                "__tidy3_sample_cumulative",
            )
            total = _temp_column(
                [*columns, row_name, weight_name, key_name, cumulative],
                "__tidy3_sample_total",
            )
            length = _temp_column(
                [*columns, row_name, weight_name, key_name, cumulative, total],
                "__tidy3_sample_length",
            )
            draw = _temp_column(
                [
                    *columns,
                    row_name,
                    weight_name,
                    key_name,
                    cumulative,
                    total,
                    length,
                ],
                "__tidy3_sample_draw",
            )
            base = base.with_columns(
                pl.col(weight_name).cum_sum().over(group_keys).alias(cumulative)
            )
            group_info = base.group_by(
                group_keys, maintain_order=True
            ).agg(
                pl.col(weight_name).sum().alias(total),
                pl.len().alias(length),
            )
            if n is not None:
                target_count = (
                    pl.lit(n) if n >= 0 else pl.col(length) + n
                )
            else:
                factor = prop if prop >= 0 else 1.0 + prop
                target_count = (pl.col(length) * factor).floor()
            group_info = group_info.with_columns(
                pl.int_ranges(0, target_count).alias(draw)
            ).explode(draw, empty_as_null=False)
            modulus = float(2**53 - 1)
            uniform = (
                (
                    pl.struct([*group_keys, draw]).hash(seed=seed_value)
                    % (2**53 - 1)
                ).cast(pl.Float64)
                + 1.0
            ) / modulus
            draws = group_info.with_columns(
                (uniform * pl.col(total)).alias(key_name)
            )
            candidates = (
                draws.join(base, on=group_keys, how="inner")
                .filter(pl.col(cumulative) > pl.col(key_name))
                .sort([*group_keys, draw, cumulative], maintain_order=True)
                .unique(
                    subset=[*group_keys, draw],
                    keep="first",
                    maintain_order=True,
                )
                .sort([*group_keys, draw], maintain_order=True)
            )
            return tf._with_lf(
                candidates.select(columns),
                groups=None if transient else tf._groups,
            )
        if not replace:
            context = _group_context(tf, groups)
            if n is not None:
                result = sample_n(n, seed=seed)._fn(context)
            else:
                result = sample_frac(prop, seed=seed)._fn(context)
            if transient:
                return result._with_lf(result._lf, groups=None, rowwise=False)
            return result

        columns = _frame_columns(tf)
        row_name = _temp_column(columns, "__tidy3_sample_row")
        value_columns = [column for column in columns if column not in (groups or [])]
        base = tf._lf
        if not value_columns:
            base = base.with_row_index(row_name)
            value_columns = [row_name]
        sample_args = (
            {"n": n if n >= 0 else pl.len() + n}
            if n is not None
            else {"fraction": prop if prop >= 0 else 1.0 + prop}
        )
        sampled = pl.struct(value_columns).sample(
            **sample_args,
            with_replacement=True,
            shuffle=True,
            seed=seed,
        )
        struct_name = _temp_column([*columns, row_name], "__tidy3_sample")
        if groups:
            lf = (
                base.group_by(groups, maintain_order=True)
                .agg(sampled.alias(struct_name))
                .explode(struct_name, empty_as_null=False)
                .unnest(struct_name)
            )
        else:
            lf = (
                base.select(sampled.alias(struct_name))
                .explode(struct_name, empty_as_null=False)
                .unnest(struct_name)
            )
        if row_name in lf.collect_schema().names():
            lf = lf.drop(row_name)
        return tf._with_lf(
            lf.select(columns), groups=None if transient else tf._groups
        )

    return Verb(_apply, "slice_sample")


def _right_frame(right: Any, backend: str) -> Any:
    """Resolve the right side of a join for the given backend."""
    from tidy3.frame import TidyFrame, tidy

    cls = type(right)
    is_tf = isinstance(right, TidyFrame) or (
        cls.__module__ == "tidy3.frame" and cls.__name__ == "TidyFrame"
    )
    if backend == "pandas":
        if is_tf:
            return right.collect(as_="pandas")
        import pandas as pd

        if isinstance(right, pd.DataFrame):
            return right
        return tidy(right, backend="pandas")._pdf
    if is_tf:
        if right.backend == "polars":
            return right._lf
        return pl.from_pandas(right.collect(as_="pandas")).lazy()
    return tidy(right)._lf


def _join_keys(left: Any, right: Any, on: str | list[str] | None) -> str | list[str]:
    """Resolve dplyr-style natural join keys when ``on`` is omitted."""
    if on is not None:
        return on
    if isinstance(left, pl.LazyFrame):
        left_cols = left.collect_schema().names()
        right_cols = set(right.collect_schema().names())
    else:
        left_cols = list(left.columns)
        right_cols = set(right.columns)
    common = [col for col in left_cols if col in right_cols]
    if not common:
        raise ValueError("join has no common columns; supply on= explicitly")
    return common


def _join_by_predicate(condition: Any, right_names: dict[str, str]) -> pl.Expr:
    left = pl.col(condition.left)
    right = pl.col(right_names[condition.right])
    return {
        ">": left > right,
        ">=": left >= right,
        "<": left < right,
        "<=": left <= right,
    }[condition.operator]


def _polars_join_by(
    tf: Any,
    right: pl.LazyFrame,
    spec: JoinSpec,
    how: str,
    *,
    suffix: str,
    keep: bool,
    na_matches: str,
    multiple: str,
    unmatched: str,
    relationship: str | None,
):
    left_columns = _frame_columns(tf)
    right_columns = right.collect_schema().names()
    for condition in spec.conditions:
        if condition.left not in left_columns:
            raise KeyError(f"join_by(): left column not found: {condition.left!r}")
        if condition.right not in right_columns:
            raise KeyError(f"join_by(): right column not found: {condition.right!r}")

    equality = spec.equality
    left_schema = tf._lf.collect_schema()
    right_schema = right.collect_schema()
    left_frame = tf._lf
    right_frame = right
    for condition in equality:
        left_type = left_schema[condition.left]
        right_type = right_schema[condition.right]
        if left_type == pl.Null and right_type != pl.Null:
            left_frame = left_frame.with_columns(
                pl.col(condition.left).cast(right_type)
            )
        elif right_type == pl.Null and left_type != pl.Null:
            right_frame = right_frame.with_columns(
                pl.col(condition.right).cast(left_type)
            )
    left_schema = left_frame.collect_schema()
    right_schema = right_frame.collect_schema()

    occupied = [*left_columns, *right_columns]
    left_id = _temp_column(occupied, "__tidy3_left_row")
    right_id = _temp_column([*occupied, left_id], "__tidy3_right_row")
    right_names: dict[str, str] = {}
    for column in right_columns:
        right_names[column] = _temp_column(
            [*occupied, left_id, right_id, *right_names.values()],
            f"__tidy3_y_{column}",
        )

    left = left_frame.with_row_index(left_id)
    y = right_frame.with_row_index(right_id).rename(right_names)
    inequalities = [
        _join_by_predicate(condition, right_names) for condition in spec.inequality
    ]
    if equality:
        candidates = left.join(
            y,
            left_on=[condition.left for condition in equality],
            right_on=[right_names[condition.right] for condition in equality],
            how="inner",
            nulls_equal=na_matches == "na",
            coalesce=False,
            maintain_order="left",
        )
        if inequalities:
            predicate = inequalities[0]
            for condition in inequalities[1:]:
                predicate = predicate & condition
            candidates = candidates.filter(predicate)
    else:
        candidates = left.join_where(y, *inequalities)

    rolling = next((condition for condition in spec.conditions if condition.rolling), None)
    if rolling is not None:
        target = pl.col(right_names[rolling.right])
        extreme = (
            target.max().over(left_id)
            if rolling.operator in {">", ">="}
            else target.min().over(left_id)
        )
        candidates = candidates.filter(target == extreme)

    guards: list[tuple[pl.LazyFrame, str]] = []
    if relationship in {"one-to-one", "many-to-one"}:
        guards.append(
            (
                candidates.group_by(left_id)
                .len()
                .filter(pl.col("len") > 1),
                f"join relationship {relationship!r} violated: a row in x matches multiple rows in y",
            )
        )
    if relationship in {"one-to-one", "one-to-many"}:
        guards.append(
            (
                candidates.group_by(right_id)
                .len()
                .filter(pl.col("len") > 1),
                f"join relationship {relationship!r} violated: a row in y matches multiple rows in x",
            )
        )

    all_matched_left = candidates.select(left_id).unique()
    all_matched_right = candidates.select(right_id).unique()
    if unmatched == "error":
        if how in {"inner", "right"}:
            guards.append(
                (
                    left.join(all_matched_left, on=left_id, how="anti"),
                    "join unmatched='error': rows in x have no match in y",
                )
            )
        if how in {"inner", "left"}:
            guards.append(
                (
                    y.join(all_matched_right, on=right_id, how="anti"),
                    "join unmatched='error': rows in y have no match in x",
                )
            )

    if multiple != "all":
        candidates = candidates.unique(
            subset=[left_id],
            keep="last" if multiple == "last" else "first",
            maintain_order=True,
        )

    matched_left = candidates.select(left_id).unique()
    if how in {"semi", "anti"}:
        lf = left.join(
            matched_left,
            on=left_id,
            how="semi" if how == "semi" else "anti",
            maintain_order="left",
        ).drop(left_id)
        for violations, message in guards:
            lf = _pl_guard_no_rows(lf, violations, message)
        return tf._with_lf(lf, groups=tf._groups)

    matched_right = candidates.select(right_id).unique()
    pieces = [candidates]
    if how in {"left", "full"}:
        missing_left = left.join(
            matched_left, on=left_id, how="anti", maintain_order="left"
        ).with_columns(
            *[
                pl.lit(None).cast(right_schema[column]).alias(right_names[column])
                for column in right_columns
            ],
            pl.lit(None).cast(pl.UInt32).alias(right_id),
        )
        pieces.append(missing_left)
    if how in {"right", "full"}:
        missing_right = y.join(
            matched_right, on=right_id, how="anti", maintain_order="left"
        ).with_columns(
            *[
                pl.lit(None).cast(left_schema[column]).alias(column)
                for column in left_columns
            ],
            pl.lit(None).cast(pl.UInt32).alias(left_id),
        )
        pieces.append(missing_right)
    combined = pl.concat(pieces, how="diagonal_relaxed")
    combined = combined.sort(
        [right_id, left_id] if how == "right" else [left_id, right_id],
        nulls_last=True,
        maintain_order=True,
    )

    equality_map = {condition.left: condition.right for condition in equality}
    equality_right = {condition.right for condition in equality}
    output = []
    for column in left_columns:
        if column in equality_map and not keep:
            output.append(
                pl.coalesce(
                    pl.col(column), pl.col(right_names[equality_map[column]])
                ).alias(column)
            )
        else:
            output.append(pl.col(column))
    for column in right_columns:
        if column in equality_right and not keep:
            continue
        name = column + suffix if column in left_columns else column
        output.append(pl.col(right_names[column]).alias(name))
    lf = combined.select(output)
    for violations, message in guards:
        lf = _pl_guard_no_rows(lf, violations, message)
    return tf._with_lf(lf, groups=tf._groups)


def _join_by_operation(
    tf: Any,
    right: Any,
    spec: JoinSpec,
    how: str,
    kwargs: dict[str, Any],
):
    params = dict(kwargs)
    suffix = params.pop("suffix", "_right")
    keep_value = params.pop("keep", None)
    keep = bool(keep_value) if keep_value is not None else False
    na_matches = params.pop("na_matches", "na")
    multiple = params.pop("multiple", "all")
    unmatched = params.pop("unmatched", "drop")
    relationship = params.pop("relationship", None)
    if not isinstance(suffix, str) or not suffix:
        raise TypeError("join suffix must be a non-empty string")
    if keep_value is not None and not isinstance(keep_value, bool):
        raise TypeError("join keep must be True, False, or None")
    if na_matches not in {"na", "never"}:
        raise ValueError("join na_matches must be 'na' or 'never'")
    if multiple not in {"all", "any", "first", "last"}:
        raise ValueError("join multiple must be 'all', 'any', 'first', or 'last'")
    if unmatched not in {"drop", "error"}:
        raise ValueError("join unmatched must be 'drop' or 'error'")
    allowed_relationships = {
        None,
        "one-to-one",
        "one-to-many",
        "many-to-one",
        "many-to-many",
    }
    if relationship not in allowed_relationships:
        raise ValueError(
            "join relationship must be one-to-one, one-to-many, "
            "many-to-one, many-to-many, or None"
        )
    if params:
        raise TypeError(
            f"unsupported join arguments: {sorted(params)}"
        )
    resolved = _right_frame(right, tf._backend)
    if tf._backend == "pandas":
        pdf = _pe().do_join_by(
            tf._pdf,
            resolved,
            spec,
            how,
            suffix=suffix,
            keep=keep,
            na_matches=na_matches,
            multiple=multiple,
            unmatched=unmatched,
            relationship=relationship,
        )
        return tf._with_pdf(pdf, groups=tf._groups)
    return _polars_join_by(
        tf,
        resolved,
        spec,
        how,
        suffix=suffix,
        keep=keep,
        na_matches=na_matches,
        multiple=multiple,
        unmatched=unmatched,
        relationship=relationship,
    )


def _mutating_join(
    right: Any,
    *,
    on: str | list[str] | None,
    how: str,
    pandas_how: str | None = None,
    verb_name: str,
    **kwargs: Any,
) -> Verb:
    def _apply(tf):
        if isinstance(on, JoinSpec):
            return _join_by_operation(tf, right, on, how, kwargs)
        r = _right_frame(right, tf._backend)
        keys = _join_keys(tf._pdf if tf._backend == "pandas" else tf._lf, r, on)
        if not kwargs:
            if tf._backend == "pandas":
                return tf._with_pdf(
                    _pe().do_join(tf._pdf, r, keys, pandas_how or how),
                    groups=tf._groups,
                )
            params: dict[str, Any] = {
                "nulls_equal": True,
                "maintain_order": (
                    "right_left"
                    if how == "right"
                    else "left_right"
                    if how == "full"
                    else "left"
                ),
            }
            if how == "full":
                params["coalesce"] = True
            lf = tf._lf.join(r, on=keys, how=how, **params)
            left_columns = tf._lf.collect_schema().names()
            right_columns = r.collect_schema().names()
            key_columns = [keys] if isinstance(keys, str) else list(keys)
            ordered = [*left_columns]
            for column in right_columns:
                if column in key_columns:
                    continue
                ordered.append(
                    f"{column}_right" if column in left_columns else column
                )
            return tf._with_lf(lf.select(ordered), groups=tf._groups)
        key_names = [keys] if isinstance(keys, str) else list(keys)
        return _join_by_operation(tf, r, join_by(*key_names), how, kwargs)

    return Verb(_apply, verb_name)


def _join_argument(on: Any, by: Any) -> Any:
    if on is not None and by is not None:
        raise ValueError("supply only one of on= or by=")
    return by if by is not None else on


def left_join(
    right: Any,
    *,
    on: str | list[str] | JoinSpec | None = None,
    by: str | list[str] | JoinSpec | None = None,
    **kwargs: Any,
) -> Verb:
    return _mutating_join(
        right,
        on=_join_argument(on, by),
        how="left",
        verb_name="left_join",
        **kwargs,
    )


def inner_join(
    right: Any,
    *,
    on: str | list[str] | JoinSpec | None = None,
    by: str | list[str] | JoinSpec | None = None,
    **kwargs: Any,
) -> Verb:
    return _mutating_join(
        right,
        on=_join_argument(on, by),
        how="inner",
        verb_name="inner_join",
        **kwargs,
    )


def right_join(
    right: Any,
    *,
    on: str | list[str] | JoinSpec | None = None,
    by: str | list[str] | JoinSpec | None = None,
    **kwargs: Any,
) -> Verb:
    return _mutating_join(
        right,
        on=_join_argument(on, by),
        how="right",
        verb_name="right_join",
        **kwargs,
    )


def full_join(
    right: Any,
    *,
    on: str | list[str] | JoinSpec | None = None,
    by: str | list[str] | JoinSpec | None = None,
    **kwargs: Any,
) -> Verb:
    return _mutating_join(
        right,
        on=_join_argument(on, by),
        how="full",
        pandas_how="outer",
        verb_name="full_join",
        **kwargs,
    )


def nest_join(
    right: Any,
    *,
    on: str | list[str] | None = None,
    by: str | list[str] | None = None,
    name: str = "data",
    keep: bool = False,
    na_matches: str = "na",
) -> Verb:
    """Attach matching right-hand rows as a nested list-column.

    On pandas, nested cells are DataFrames (same representation as
    ``group_nest`` / ``nest``). On Polars they are list-of-struct columns
    from a lazy ``group_by().agg(struct)`` plan joined back to the left.
    """
    if on is not None and by is not None:
        raise ValueError("supply only one of on= or by=")
    if not isinstance(name, str) or not name:
        raise TypeError("nest_join() name must be a non-empty string")
    if na_matches not in {"na", "never"}:
        raise ValueError("na_matches must be 'na' or 'never'")
    join_keys = by if by is not None else on

    def _apply(tf):
        columns = _frame_columns(tf)
        if name in columns:
            raise ValueError(f"nest_join() output column exists: {name!r}")
        r = _right_frame(right, tf._backend)
        keys = _join_keys(
            tf._pdf if tf._backend == "pandas" else tf._lf,
            r,
            join_keys,
        )
        key_columns = [keys] if isinstance(keys, str) else list(keys)
        if tf._backend == "pandas":
            import pandas as pd

            nested_columns = [
                column
                for column in r.columns
                if keep or column not in key_columns
            ]
            empty_nested = pd.DataFrame(columns=nested_columns)
            # Pre-slice once; build nested DataFrames via group positions
            # (no to_dict(records) Python row materialization).
            if nested_columns:
                body = r.loc[:, nested_columns]
            else:
                body = pd.DataFrame(index=r.index)
            grouping_key = (
                key_columns[0] if len(key_columns) == 1 else key_columns
            )
            rows: list[dict[str, Any]] = []
            for key, positions in r.groupby(
                grouping_key,
                sort=False,
                dropna=na_matches == "never",
                observed=True,
            ).groups.items():
                key_tuple = key if isinstance(key, tuple) else (key,)
                row = dict(zip(key_columns, key_tuple))
                row[name] = body.loc[positions].reset_index(drop=True)
                rows.append(row)
            nested = pd.DataFrame(rows, columns=[*key_columns, name])
            output = tf._pdf.merge(
                nested, on=key_columns, how="left", sort=False
            )

            def _as_nested(value: Any) -> pd.DataFrame:
                if isinstance(value, pd.DataFrame):
                    return value
                return empty_nested.copy()

            output[name] = output[name].map(_as_nested)
            return tf._with_pdf(output, groups=tf._groups, rowwise=False)

        right_columns = r.collect_schema().names()
        nested_columns = [
            column
            for column in right_columns
            if keep or column not in key_columns
        ]
        if nested_columns:
            if len(key_columns) == 1 and set(nested_columns) == set(
                right_columns
            ) - set(key_columns):
                payload = pl.struct(pl.exclude(key_columns[0]))
            else:
                payload = pl.struct(nested_columns)
            nested = r.group_by(key_columns, maintain_order=True).agg(
                payload.alias(name)
            )
        else:
            nested = r.group_by(key_columns, maintain_order=True).agg(
                pl.repeat(None, pl.len()).alias(name)
            )
        output = tf._lf.join(
            nested,
            on=key_columns,
            how="left",
            nulls_equal=na_matches == "na",
            maintain_order="left",
        )
        dtype = output.collect_schema()[name]
        output = output.with_columns(
            pl.col(name).fill_null(pl.lit([], dtype=dtype)).alias(name)
        )
        return tf._with_lf(output, groups=tf._groups, rowwise=False)

    return Verb(_apply, "nest_join")


def _filter_join(
    right: Any,
    *,
    on: str | list[str] | None,
    anti: bool,
    verb_name: str,
    na_matches: str,
) -> Verb:
    def _apply(tf):
        params = {"na_matches": na_matches}
        if isinstance(on, JoinSpec):
            return _join_by_operation(
                tf, right, on, "anti" if anti else "semi", params
            )
        r = _right_frame(right, tf._backend)
        keys = _join_keys(tf._pdf if tf._backend == "pandas" else tf._lf, r, on)
        if na_matches == "na":
            if tf._backend == "pandas":
                pdf = _pe().do_filter_join(tf._pdf, r, keys, anti=anti)
                return tf._with_pdf(pdf, groups=tf._groups)
            lf = tf._lf.join(
                r,
                on=keys,
                how="anti" if anti else "semi",
                nulls_equal=True,
                maintain_order="left",
            )
            return tf._with_lf(lf, groups=tf._groups)
        key_names = [keys] if isinstance(keys, str) else list(keys)
        return _join_by_operation(
            tf,
            r,
            join_by(*key_names),
            "anti" if anti else "semi",
            params,
        )

    return Verb(_apply, verb_name)


def semi_join(
    right: Any,
    *,
    on: str | list[str] | JoinSpec | None = None,
    by: str | list[str] | JoinSpec | None = None,
    na_matches: str = "na",
) -> Verb:
    return _filter_join(
        right,
        on=_join_argument(on, by),
        anti=False,
        verb_name="semi_join",
        na_matches=na_matches,
    )


def anti_join(
    right: Any,
    *,
    on: str | list[str] | JoinSpec | None = None,
    by: str | list[str] | JoinSpec | None = None,
    na_matches: str = "na",
) -> Verb:
    return _filter_join(
        right,
        on=_join_argument(on, by),
        anti=True,
        verb_name="anti_join",
        na_matches=na_matches,
    )


def cross_join(right: Any, **kwargs: Any) -> Verb:
    """Return the Cartesian product of the left and right frames."""

    def _apply(tf):
        r = _right_frame(right, tf._backend)
        if tf._backend == "pandas":
            pdf = _pe().do_join(tf._pdf, r, None, "cross", **kwargs)
            return tf._with_pdf(pdf, groups=tf._groups)
        params = dict(kwargs)
        params.setdefault("maintain_order", "left")
        return tf._with_lf(
            tf._lf.join(r, how="cross", **params), groups=tf._groups
        )

    return Verb(_apply, "cross_join")


def bind_rows(*others: Any, id: str | None = None) -> Verb:  # noqa: A002
    """Append frames, taking the union of their columns."""
    if not others:
        raise TypeError("bind_rows() requires at least one frame")
    if id is not None and (not isinstance(id, str) or not id):
        raise TypeError("bind_rows() id must be a non-empty string or None")

    def _apply(tf):
        frames = [tf._pdf if tf._backend == "pandas" else tf._lf]
        frames.extend(_right_frame(other, tf._backend) for other in others)
        if tf._backend == "pandas":
            import pandas as pd

            if id is not None:
                frames = [
                    frame.assign(**{id: str(index)})[
                        [id, *(column for column in frame.columns if column != id)]
                    ]
                    for index, frame in enumerate(frames, start=1)
                ]
            pdf = pd.concat(frames, ignore_index=True, sort=False)
            return tf._with_pdf(pdf, groups=tf._groups)
        if id is not None:
            frames = [
                frame.with_columns(pl.lit(str(index)).alias(id)).select(
                    id,
                    pl.exclude(id),
                )
                for index, frame in enumerate(frames, start=1)
            ]
        return tf._with_lf(
            pl.concat(frames, how="diagonal_relaxed"), groups=tf._groups
        )

    return Verb(_apply, "bind_rows")


def _repair_bind_names(names: list[str]) -> list[str]:
    """Apply the default vctrs-style unique repair used by bind_cols()."""
    bases = [re.sub(r"\.\.\.[0-9]+$", "", name) for name in names]
    counts: dict[str, int] = {}
    for name in bases:
        counts[name] = counts.get(name, 0) + 1
    repaired = [
        f"{name}...{position}"
        if not name or counts[name] > 1
        else name
        for position, name in enumerate(bases, start=1)
    ]
    return repaired


def _common_recycled_size(sizes: list[int]) -> int:
    non_scalar = {size for size in sizes if size != 1}
    if len(non_scalar) > 1:
        raise ValueError(
            "bind_cols() inputs must have the same row count or one row"
        )
    return next(iter(non_scalar)) if non_scalar else 1


def bind_cols(*others: Any) -> Verb:
    """Combine frames column-wise with unique names and size-one recycling."""
    if not others:
        raise TypeError("bind_cols() requires at least one frame")

    def _apply(tf):
        frames = [tf._pdf if tf._backend == "pandas" else tf._lf]
        frames.extend(_right_frame(other, tf._backend) for other in others)
        column_sets = [
            list(frame.columns)
            if tf._backend == "pandas"
            else frame.collect_schema().names()
            for frame in frames
        ]
        all_columns = [column for columns in column_sets for column in columns]
        repaired = _repair_bind_names(all_columns)
        repaired_sets: list[list[str]] = []
        offset = 0
        for columns in column_sets:
            repaired_sets.append(repaired[offset : offset + len(columns)])
            offset += len(columns)
        groups = None
        if tf._groups:
            left_mapping = dict(zip(column_sets[0], repaired_sets[0]))
            groups = [left_mapping[name] for name in tf._groups]
        if tf._backend == "pandas":
            import pandas as pd
            import numpy as np

            sizes = [len(frame) for frame in frames]
            target = _common_recycled_size(sizes)
            recycled = []
            for frame, size, names in zip(frames, sizes, repaired_sets):
                work = frame.copy(deep=False)
                work.columns = names
                if size == 1 and target != 1:
                    if target == 0:
                        work = work.iloc[:0]
                    else:
                        work = work.take(np.zeros(target, dtype=np.intp))
                recycled.append(work.reset_index(drop=True))
            pdf = pd.concat(
                recycled, axis=1
            )
            return tf._with_pdf(pdf, groups=groups)
        sizes = [
            int(frame.select(pl.len()).collect().item()) for frame in frames
        ]
        target = _common_recycled_size(sizes)
        renamed = [
            frame.rename(dict(zip(columns, names)))
            for frame, columns, names in zip(
                frames, column_sets, repaired_sets
            )
        ]
        base_index = next(
            (index for index, size in enumerate(sizes) if size == target), 0
        )
        horizontal = [renamed[base_index]]
        for index, (frame, size, names) in enumerate(
            zip(renamed, sizes, repaired_sets)
        ):
            if index == base_index:
                continue
            if size == target:
                horizontal.append(frame)
                continue
            if target == 0:
                horizontal.append(frame.head(0))
                continue
            row = frame.collect().row(0, named=True)
            horizontal.append(
                horizontal[0].with_columns(
                    *(pl.lit(row[name]).alias(name) for name in names)
                ).select(names)
            )
        lf = pl.concat(horizontal, how="horizontal_extend").select(repaired)
        return tf._with_lf(
            lf, groups=groups
        )

    return Verb(_apply, "bind_cols")


def _set_frames(tf: Any, right: Any) -> tuple[Any, Any, list[str]]:
    left = tf._pdf if tf._backend == "pandas" else tf._lf
    other = _right_frame(right, tf._backend)
    left_columns = (
        list(left.columns)
        if tf._backend == "pandas"
        else left.collect_schema().names()
    )
    right_columns = (
        list(other.columns)
        if tf._backend == "pandas"
        else other.collect_schema().names()
    )
    if set(left_columns) != set(right_columns):
        raise ValueError("set operations require frames with the same columns")
    if right_columns != left_columns:
        other = other[left_columns] if tf._backend == "pandas" else other.select(left_columns)
    return left, other, left_columns


def _set_operation(right: Any, operation: str) -> Verb:
    def _apply(tf):
        left, other, columns = _set_frames(tf, right)
        if tf._backend == "pandas":
            import pandas as pd

            left_unique = left.drop_duplicates()
            right_unique = other.drop_duplicates()
            if operation == "union":
                out = pd.concat([left, other], ignore_index=True).drop_duplicates()
            elif operation == "union_all":
                out = pd.concat([left, other], ignore_index=True)
            elif operation == "intersect":
                out = _pe().do_filter_join(left_unique, right_unique, columns, anti=False)
            elif operation == "setdiff":
                out = _pe().do_filter_join(left_unique, right_unique, columns, anti=True)
            else:
                left_only = _pe().do_filter_join(left_unique, right_unique, columns, anti=True)
                right_only = _pe().do_filter_join(right_unique, left_unique, columns, anti=True)
                out = pd.concat([left_only, right_only], ignore_index=True)
            return tf._with_pdf(out.reset_index(drop=True), groups=tf._groups)
        if operation == "union_all":
            lf = pl.concat([left, other], how="vertical_relaxed")
        elif operation == "union":
            lf = pl.concat([left, other], how="vertical_relaxed").unique(
                keep="first", maintain_order=True
            )
        else:
            left_unique = left.unique(keep="first", maintain_order=True)
            right_unique = other.unique(keep="first", maintain_order=True)
            join_how = "semi" if operation == "intersect" else "anti"
            left_part = left_unique.join(
                right_unique,
                on=columns,
                how=join_how,
                nulls_equal=True,
                maintain_order="left",
            )
            if operation == "symdiff":
                right_part = right_unique.join(
                    left_unique,
                    on=columns,
                    how="anti",
                    nulls_equal=True,
                    maintain_order="left",
                )
                lf = pl.concat([left_part, right_part], how="vertical_relaxed")
            else:
                lf = left_part
        return tf._with_lf(lf, groups=tf._groups)

    return Verb(_apply, operation)


def union(right: Any) -> Verb:  # noqa: A001
    return _set_operation(right, "union")


def union_all(right: Any) -> Verb:
    return _set_operation(right, "union_all")


def intersect(right: Any) -> Verb:
    return _set_operation(right, "intersect")


def setdiff(right: Any) -> Verb:
    return _set_operation(right, "setdiff")


def symdiff(right: Any) -> Verb:
    return _set_operation(right, "symdiff")


def setequal(right: Any) -> Verb:
    """Return whether two frames contain the same distinct rows."""

    def _apply(tf):
        left, other, columns = _set_frames(tf, right)
        if tf._backend != "pandas":
            left = left.collect().to_pandas()
            other = other.collect().to_pandas()
        left = left[columns].drop_duplicates(ignore_index=True)
        other = other[columns].drop_duplicates(ignore_index=True)
        if len(left) != len(other):
            return False
        import pandas as pd

        left_hash = pd.util.hash_pandas_object(left, index=False).sort_values().to_numpy()
        right_hash = pd.util.hash_pandas_object(other, index=False).sort_values().to_numpy()
        return bool((left_hash == right_hash).all())

    return Verb(_apply, "setequal")


def _row_keys(right: Any, by: str | list[str] | None) -> list[str]:
    columns = (
        list(right.columns)
        if not isinstance(right, pl.LazyFrame)
        else right.collect_schema().names()
    )
    if by is None:
        if not columns:
            raise ValueError("rows operation y must contain at least one column")
        return [str(columns[0])]
    keys = [by] if isinstance(by, str) else list(by)
    if not keys or any(not isinstance(key, str) for key in keys):
        raise TypeError("by must be a column name or non-empty list of names")
    return keys


def _validate_policy(value: str, name: str) -> str:
    if value not in {"error", "ignore"}:
        raise ValueError(f"{name} must be 'error' or 'ignore'")
    return value


def _pl_assert_rows(lf: pl.LazyFrame, condition: pl.Expr, message: str) -> pl.LazyFrame:
    """Raise at collect-time while keeping row operations lazy."""

    def validate(value: bool) -> bool:
        if not value:
            raise ValueError(message)
        return True

    return lf.filter(
        condition.map_elements(validate, return_dtype=pl.Boolean, skip_nulls=False)
    )


def _pl_guard_no_rows(
    lf: pl.LazyFrame, violations: pl.LazyFrame, message: str
) -> pl.LazyFrame:
    """Attach a lazy assertion that *violations* must contain no rows."""
    schema = lf.collect_schema()
    occupied = schema.names()
    marker = _temp_column(occupied, "__tidy3_validation")
    count_name = _temp_column([*occupied, marker], "__tidy3_violation_count")

    def validate(count: int) -> bool:
        if count:
            raise ValueError(message)
        return True

    sentinel = violations.select(pl.len().alias(count_name)).select(
        *[pl.lit(None).cast(dtype).alias(name) for name, dtype in schema.items()],
        pl.col(count_name)
        .map_elements(validate, return_dtype=pl.Boolean, skip_nulls=False)
        .alias(marker),
    )
    actual = lf.with_columns(pl.lit(False).alias(marker))
    return (
        pl.concat([actual, sentinel], how="vertical")
        .filter(~pl.col(marker))
        .drop(marker)
    )


def _pl_row_inputs(tf: Any, right: Any, operation: str):
    y = _right_frame(right, "polars")
    x_columns = _frame_columns(tf)
    y_columns = y.collect_schema().names()
    extra = [column for column in y_columns if column not in x_columns]
    if extra:
        raise ValueError(f"{operation}(): y has columns absent from x: {extra}")
    return y, x_columns, y_columns


def _pl_match_y(
    y: pl.LazyFrame,
    x: pl.LazyFrame,
    keys: list[str],
    columns: list[str],
) -> tuple[pl.LazyFrame, str]:
    marker = _temp_column(columns, "__tidy3_match")
    x_keys = x.select(keys).unique().with_columns(pl.lit(True).alias(marker))
    return (
        y.join(
            x_keys,
            on=keys,
            how="left",
            nulls_equal=True,
            maintain_order="left",
        ),
        marker,
    )


def _rows_operation(
    right: Any,
    *,
    by: str | list[str] | None,
    operation: str,
    conflict: str = "error",
    unmatched: str = "error",
) -> Verb:
    conflict = _validate_policy(conflict, "conflict")
    unmatched = _validate_policy(unmatched, "unmatched")

    def _apply(tf):
        y = _right_frame(right, tf._backend)
        keys = _row_keys(y, by)
        if tf._backend == "pandas":
            pdf = _pe().do_rows(
                tf._pdf,
                y,
                keys,
                operation,
                conflict=conflict,
                unmatched=unmatched,
            )
            return tf._with_pdf(pdf, groups=tf._groups)

        y, x_columns, y_columns = _pl_row_inputs(tf, right, operation)
        missing_keys = [
            key for key in keys if key not in x_columns or key not in y_columns
        ]
        if missing_keys:
            raise KeyError(f"{operation}(): key columns not found: {missing_keys}")
        if operation in {"rows_update", "rows_patch", "rows_upsert"}:
            y = _pl_assert_rows(
                y,
                pl.struct(keys).n_unique() == pl.len(),
                f"{operation}(): y keys must be unique",
            )

        if operation == "rows_append":
            additions = y
        else:
            checked, marker = _pl_match_y(y, tf._lf, keys, [*x_columns, *y_columns])
            is_match = pl.col(marker).fill_null(False)
            if operation == "rows_insert":
                if conflict == "error":
                    checked = _pl_assert_rows(
                        checked,
                        (~is_match).all(),
                        "rows_insert(): y contains keys that already exist in x",
                    )
                additions = checked.filter(~is_match).drop(marker)
            elif operation in {"rows_update", "rows_patch", "rows_delete"}:
                if unmatched == "error":
                    checked = _pl_assert_rows(
                        checked,
                        is_match.all(),
                        f"{operation}(): y contains keys absent from x",
                    )
                y = checked.filter(is_match).drop(marker)

        def aligned(frame: pl.LazyFrame, frame_columns: list[str]) -> pl.LazyFrame:
            return frame.select(
                *[
                    pl.col(column)
                    if column in frame_columns
                    else pl.lit(None).alias(column)
                    for column in x_columns
                ]
            )

        if operation in {"rows_append", "rows_insert"}:
            additions = aligned(additions, y_columns)
            lf = pl.concat([tf._lf, additions], how="vertical_relaxed")
            return tf._with_lf(lf, groups=tf._groups)
        if operation == "rows_delete":
            lf = tf._lf.join(
                y.select(keys),
                on=keys,
                how="anti",
                nulls_equal=True,
                maintain_order="left",
            )
            return tf._with_lf(lf, groups=tf._groups)

        update_columns = [column for column in y_columns if column not in keys]
        occupied = [*x_columns, *y_columns]
        marker = _temp_column(occupied, "__tidy3_update_match")
        rename_map: dict[str, str] = {}
        for column in update_columns:
            incoming = _temp_column([*occupied, *rename_map.values()], f"__tidy3_y_{column}")
            rename_map[column] = incoming
        payload = y.rename(rename_map).with_columns(pl.lit(True).alias(marker))
        joined = tf._lf.join(
            payload,
            on=keys,
            how="left",
            nulls_equal=True,
            maintain_order="left",
        )
        expressions = []
        for column in x_columns:
            if column not in rename_map:
                expressions.append(pl.col(column))
            elif operation == "rows_patch":
                expressions.append(
                    pl.coalesce(pl.col(column), pl.col(rename_map[column])).alias(column)
                )
            else:
                expressions.append(
                    pl.when(pl.col(marker).fill_null(False))
                    .then(pl.col(rename_map[column]))
                    .otherwise(pl.col(column))
                    .alias(column)
                )
        updated = joined.select(expressions)
        if operation == "rows_upsert":
            checked, match_marker = _pl_match_y(
                y, tf._lf, keys, [*x_columns, *y_columns, marker]
            )
            additions = checked.filter(
                ~pl.col(match_marker).fill_null(False)
            ).drop(match_marker)
            additions = aligned(additions, y_columns)
            updated = pl.concat([updated, additions], how="vertical_relaxed")
        return tf._with_lf(updated, groups=tf._groups)

    return Verb(_apply, operation)


def rows_insert(
    right: Any,
    *,
    by: str | list[str] | None = None,
    conflict: str = "error",
) -> Verb:
    return _rows_operation(
        right, by=by, operation="rows_insert", conflict=conflict
    )


def rows_append(right: Any) -> Verb:
    return _rows_operation(right, by=None, operation="rows_append")


def rows_update(
    right: Any,
    *,
    by: str | list[str] | None = None,
    unmatched: str = "error",
) -> Verb:
    return _rows_operation(
        right, by=by, operation="rows_update", unmatched=unmatched
    )


def rows_patch(
    right: Any,
    *,
    by: str | list[str] | None = None,
    unmatched: str = "error",
) -> Verb:
    return _rows_operation(
        right, by=by, operation="rows_patch", unmatched=unmatched
    )


def rows_upsert(
    right: Any, *, by: str | list[str] | None = None
) -> Verb:
    return _rows_operation(right, by=by, operation="rows_upsert")


def rows_delete(
    right: Any,
    *,
    by: str | list[str] | None = None,
    unmatched: str = "error",
) -> Verb:
    return _rows_operation(
        right, by=by, operation="rows_delete", unmatched=unmatched
    )


def collect(
    as_: str = "polars",
    *,
    columns: Any = None,
    arrow_backed: bool = False,
    engine: Any = "auto",
    dtype: Any = None,
    order: str = "fortran",
    writable: bool = False,
    allow_copy: bool = True,
) -> Verb:
    """Materialize — returns pandas/polars DataFrame (not TidyFrame)."""

    def _apply(tf):
        return tf.collect(
            as_=as_,
            columns=columns,
            arrow_backed=arrow_backed,
            engine=engine,
            dtype=dtype,
            order=order,
            writable=writable,
            allow_copy=allow_copy,
        )

    return Verb(_apply, "collect")


def to_numpy(
    *,
    columns: Any = None,
    dtype: Any = None,
    order: str = "fortran",
    writable: bool = False,
    allow_copy: bool = True,
    engine: Any = "auto",
) -> Verb:
    """Materialize selected columns as a NumPy matrix."""

    def _apply(tf):
        return tf.to_numpy(
            columns=columns,
            dtype=dtype,
            order=order,
            writable=writable,
            allow_copy=allow_copy,
            engine=engine,
        )

    return Verb(_apply, "to_numpy")


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
