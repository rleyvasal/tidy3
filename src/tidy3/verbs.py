"""dplyr-style verbs as pipeable objects (``df >> filter(...)``).

Each verb dispatches on the frame's backend: polars (lazy — expressions are
compiled to ``pl.Expr``) or pandas (eager — expressions are evaluated by
``tidy3.pandas_engine``).
"""

from __future__ import annotations

import math
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
    resolve_selection,
)


def _plx(v: Any) -> Any:
    """Compile a tidy3 Expr for polars; pass everything else through."""
    return to_polars(v) if isinstance(v, Expr) else v


def _pl_expr(v: Any) -> pl.Expr:
    value = _plx(v)
    return value if isinstance(value, pl.Expr) else pl.lit(value)


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
            cls = type(other)
            if cls.__module__ == "tidy3.frame" and cls.__name__ == "TidyFrame":
                # A remote re-seed creates a new TidyFrame class identity while
                # frames already stored in the notebook still use the old one.
                # Re-wrap its backend data in the current class at this seam.
                try:
                    other = TidyFrame(
                        other._data,
                        groups=other._groups,
                        rowwise=getattr(other, "_rowwise", False),
                    )
                except (AttributeError, TypeError) as e:
                    raise TypeError(
                        "tidy3 pipe received an incompatible TidyFrame from "
                        "another loaded copy"
                    ) from e
            else:
                raise TypeError(
                    "tidy3 pipe expects TidyFrame on the left of >>, "
                    f"got {cls.__name__}"
                )
        return self._fn(other)

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
    expanded: dict[str, Any] = {}
    for spec in specs:
        if not isinstance(spec, AcrossSpec):
            raise TypeError(f"{verb_name}() positional arguments must come from across()")
        for name, value in spec.expand(tf).items():
            if name in expanded or name in assignments:
                raise ValueError(f"{verb_name}() defines column {name!r} more than once")
            expanded[name] = value
    expanded.update(
        {name: _resolved_value(tf, value) for name, value in assignments.items()}
    )
    return expanded


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


def filter(*predicates: Any, by: Any = None) -> Verb:  # noqa: A001
    """Keep rows matching all predicates (AND).

    After ``group_by`` the predicate is evaluated per group (dplyr window
    semantics), so ``filter(col("x") > mean("x"))`` compares within groups.
    """
    if not predicates:
        raise TypeError("filter() requires at least one predicate")

    def _apply(tf):
        groups, transient = _operation_groups(tf, by, "filter")
        context = _group_context(tf, groups)
        resolved = tuple(
            _resolved_value(context, predicate) for predicate in predicates
        )
        if tf._backend == "pandas":
            work = tf._pdf
            marker = None
            if tf._rowwise:
                marker = _temp_column(_frame_columns(tf), "__tidy3_rowwise")
                work = work.assign(**{marker: range(len(work))})
                groups = [marker]
            pdf = _pe().do_filter(work, resolved, groups)
            if marker is not None:
                pdf = pdf.drop(columns=marker)
            return tf._with_pdf(pdf, groups=None if transient else tf._groups)
        expr = _plx(resolved[0])
        for p in resolved[1:]:
            expr = expr & _plx(p)
        if tf._rowwise:
            marker = _temp_column(_frame_columns(tf), "__tidy3_rowwise")
            lf = (
                tf._lf.with_row_index(marker)
                .filter(_pl_expr(expr).over(marker))
                .drop(marker)
            )
        else:
            lf = tf._lf.filter(_windowed(expr, groups))
        return tf._with_lf(lf, groups=None if transient else tf._groups)

    return Verb(_apply, "filter")


def filter_out(*predicates: Any, by: Any = None) -> Verb:
    """Drop rows matching all predicates; retain rows where they are null."""
    if not predicates:
        raise TypeError("filter_out() requires at least one predicate")

    def _apply(tf):
        groups, transient = _operation_groups(tf, by, "filter_out")
        context = _group_context(tf, groups)
        resolved = tuple(
            _resolved_value(context, predicate) for predicate in predicates
        )
        if tf._backend == "pandas":
            work = tf._pdf
            marker = None
            if tf._rowwise:
                marker = _temp_column(_frame_columns(tf), "__tidy3_rowwise")
                work = work.assign(**{marker: range(len(work))})
                groups = [marker]
            pdf = _pe().do_filter_out(work, resolved, groups)
            if marker is not None:
                pdf = pdf.drop(columns=marker)
            return tf._with_pdf(pdf, groups=None if transient else tf._groups)
        expr = _plx(resolved[0])
        for predicate in resolved[1:]:
            expr = expr & _plx(predicate)
        if tf._rowwise:
            marker = _temp_column(_frame_columns(tf), "__tidy3_rowwise")
            lf = tf._lf.with_row_index(marker)
            expr = _pl_expr(expr).over(marker)
        else:
            marker = None
            lf = tf._lf
            expr = _windowed(expr, groups)
        lf = lf.filter((~expr).fill_null(True))
        if marker is not None:
            lf = lf.drop(marker)
        return tf._with_lf(lf, groups=None if transient else tf._groups)

    return Verb(_apply, "filter_out")


def mutate(*specs: Any, by: Any = None, **kwargs: Any) -> Verb:
    """Add or overwrite columns.

    After ``group_by`` each expression is evaluated per group (dplyr window
    semantics): aggregates broadcast within the group, ``cum_sum`` restarts
    per group, etc.
    """
    if not specs and not kwargs:
        raise TypeError("mutate() requires at least one assignment")

    def _apply(tf):
        groups, transient = _operation_groups(tf, by, "mutate")
        context = _group_context(tf, groups)
        assignments = _expanded_assignments(context, specs, kwargs, "mutate")
        if tf._backend == "pandas":
            work = tf._pdf
            marker = None
            if tf._rowwise:
                marker = _temp_column(_frame_columns(tf), "__tidy3_rowwise")
                work = work.assign(**{marker: range(len(work))})
                groups = [marker]
            pdf = _pe().do_mutate(work, assignments, groups)
            if marker is not None:
                pdf = pdf.drop(columns=marker)
            return tf._with_pdf(pdf, groups=None if transient else tf._groups)
        if tf._rowwise:
            marker = _temp_column(_frame_columns(tf), "__tidy3_rowwise")
            lf = tf._lf.with_row_index(marker)
            exprs = {k: _pl_expr(v).over(marker) for k, v in assignments.items()}
            lf = lf.with_columns(**exprs).drop(marker)
        else:
            exprs = {
                k: _windowed(_plx(v), groups) for k, v in assignments.items()
            }
            lf = tf._lf.with_columns(**exprs)
        return tf._with_lf(lf, groups=None if transient else tf._groups)

    return Verb(_apply, "mutate")


def transmute(*specs: Any, by: Any = None, **kwargs: Any) -> Verb:
    """Keep only newly defined columns (plus groups if set)."""
    if not specs and not kwargs:
        raise TypeError("transmute() requires at least one assignment")

    def _apply(tf):
        groups, transient = _operation_groups(tf, by, "transmute")
        context = _group_context(tf, groups)
        assignments = _expanded_assignments(context, specs, kwargs, "transmute")
        keep = list(assignments.keys())
        if groups:
            keep = list(dict.fromkeys([*groups, *keep]))
        if tf._backend == "pandas":
            work = tf._pdf
            if tf._rowwise:
                marker = _temp_column(_frame_columns(tf), "__tidy3_rowwise")
                work = work.assign(**{marker: range(len(work))})
                groups = [marker]
            pdf = _pe().do_mutate(work, assignments, groups)
            return tf._with_pdf(
                pdf[keep], groups=None if transient else tf._groups
            )
        if tf._rowwise:
            marker = _temp_column(_frame_columns(tf), "__tidy3_rowwise")
            lf = tf._lf.with_row_index(marker)
            exprs = {k: _pl_expr(v).over(marker) for k, v in assignments.items()}
            lf = lf.with_columns(**exprs)
        else:
            exprs = {
                k: _windowed(_plx(v), groups) for k, v in assignments.items()
            }
            lf = tf._lf.with_columns(**exprs)
        return tf._with_lf(
            lf.select(keep), groups=None if transient else tf._groups
        )

    return Verb(_apply, "transmute")


def select(*cols: Any, **renames: Any) -> Verb:
    """Select columns by name or expression.

    Like dplyr, grouping columns are always kept (prepended when missing).
    """
    if not cols and not renames:
        raise TypeError("select() requires at least one column")

    def _apply(tf):
        ordered: list[Any] = []
        seen_sources: set[str] = set()
        for spec in cols:
            if isinstance(spec, (Expr, pl.Expr)):
                ordered.append(spec)
                continue
            for source in resolve_selection(tf, [spec]):
                if source not in seen_sources:
                    ordered.append(source)
                    seen_sources.add(source)
        mapping: dict[str, str] = {}
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
            return tf._with_pdf(pdf, groups=groups)
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
        return tf._with_lf(tf._lf.select(expressions), groups=groups)

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
    """Rename selected columns with ``fn``; all columns when none are named."""
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
        if tf._backend == "pandas":
            return tf._with_pdf(tf._pdf.rename(columns=mapping), groups=groups)
        return tf._with_lf(tf._lf.rename(mapping), groups=groups)

    return Verb(_apply, "rename_with")


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
        if tf._backend == "pandas":
            print(f"Rows: {len(tf._pdf)}")
            print(f"Columns: {len(tf._pdf.columns)}")
            for column, dtype in tf._pdf.dtypes.items():
                values = ", ".join(repr(value) for value in preview[column].tolist())
                print(f"$ {column} <{dtype}> {values}")
        else:
            schema = tf._lf.collect_schema()
            print("Rows: ? (lazy)")
            print(f"Columns: {len(schema)}")
            for column, dtype in schema.items():
                values = ", ".join(
                    repr(value) for value in preview.get_column(column).to_list()
                )
                print(f"$ {column} <{dtype}> {values}")
        return tf

    return Verb(_apply, "glimpse")


def arrange(*keys: Any) -> Verb:
    if not keys:
        raise TypeError("arrange() requires at least one key")

    def _apply(tf):
        if tf._backend == "pandas":
            return tf._with_pdf(_pe().do_arrange(tf._pdf, keys), groups=tf._groups)
        expressions = []
        descending = []
        for key in keys:
            node = key.node if isinstance(key, Expr) else None
            if node is not None and node[0] == "desc":
                expressions.append(to_polars(Expr(node[1])))
                descending.append(True)
            else:
                expressions.append(_plx(key))
                descending.append(False)
        return tf._with_lf(
            tf._lf.sort(expressions, descending=descending), groups=tf._groups
        )

    return Verb(_apply, "arrange")


def distinct(*cols: str) -> Verb:
    def _apply(tf):
        if tf._backend == "pandas":
            return tf._with_pdf(_pe().do_distinct(tf._pdf, cols), groups=tf._groups)
        if cols:
            return tf._with_lf(
                tf._lf.unique(subset=list(cols), keep="first", maintain_order=True),
                groups=tf._groups,
            )
        return tf._with_lf(
            tf._lf.unique(keep="first", maintain_order=True), groups=tf._groups
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
        groups, transient = _operation_groups(tf, by, verb_name)
        if tf._backend == "pandas":
            pdf = _pe().do_slice_extreme(
                tf._pdf,
                order_by,
                n,
                prop,
                groups,
                largest=largest,
                with_ties=with_ties,
                na_rm=na_rm,
            )
            return tf._with_pdf(pdf, groups=None if transient else tf._groups)
        columns = _frame_columns(tf)
        order_name = _temp_column(columns, "__tidy3_slice_order")
        global_name = _temp_column([*columns, order_name], "__tidy3_slice_global")
        group_name = _temp_column(
            [*columns, order_name, global_name], "__tidy3_slice_group"
        )
        value = pl.col(order_by) if isinstance(order_by, str) else _plx(order_by)
        if not isinstance(value, pl.Expr):
            raise TypeError(f"{verb_name}() order_by must be a column or expression")
        lf = tf._lf.with_columns(value.alias(order_name))
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
        return tf._with_lf(
            lf.drop(*cleanup), groups=None if transient else tf._groups
        )

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


def group_by(*cols: str) -> Verb:
    if not cols:
        raise TypeError("group_by() requires at least one column")

    def _apply(tf):
        if tf._backend == "pandas":
            return tf._with_pdf(tf._pdf, groups=list(cols), rowwise=False)
        return tf._with_lf(tf._lf, groups=list(cols), rowwise=False)

    return Verb(_apply, "group_by")


def ungroup() -> Verb:
    def _apply(tf):
        if tf._backend == "pandas":
            return tf._with_pdf(tf._pdf, groups=None, rowwise=False)
        return tf._with_lf(tf._lf, groups=None, rowwise=False)

    return Verb(_apply, "ungroup")


def summarise(*specs: Any, by: Any = None, **kwargs: Any) -> Verb:
    """Aggregate; uses current ``group_by`` if set."""
    if not specs and not kwargs:
        raise TypeError("summarise() requires at least one aggregation")

    def _apply(tf):
        groups, transient = _operation_groups(tf, by, "summarise")
        context = _group_context(tf, groups)
        assignments = _expanded_assignments(context, specs, kwargs, "summarise")
        if tf._rowwise:
            marker = _temp_column(_frame_columns(tf), "__tidy3_rowwise")
            keys = [*(tf._groups or []), marker]
            result_groups = list(tf._groups) if tf._groups else None
            if tf._backend == "pandas":
                work = tf._pdf.assign(**{marker: range(len(tf._pdf))})
                pdf = _pe().do_reframe(work, assignments, keys).drop(
                    columns=marker
                )
                return tf._with_pdf(
                    pdf, groups=result_groups, rowwise=False
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
            return tf._with_lf(lf, groups=result_groups, rowwise=False)
        if tf._backend == "pandas":
            return tf._with_pdf(
                _pe().do_summarise(tf._pdf, assignments, groups),
                groups=None,
                rowwise=False,
            )
        named = []
        for name, expr in assignments.items():
            e = _plx(expr)
            if isinstance(e, pl.Expr):
                named.append(e.alias(name))
            else:
                named.append(pl.lit(e).alias(name))
        if groups:
            lf = tf._lf.group_by(groups, maintain_order=transient).agg(named)
            return tf._with_lf(lf, groups=None, rowwise=False)
        lf = tf._lf.select(named)
        return tf._with_lf(lf, groups=None, rowwise=False)

    return Verb(_apply, "summarise")


summarize = summarise


def reframe(*specs: Any, by: Any = None, **kwargs: Any) -> Verb:
    """Return zero or more rows per group; the result is always ungrouped."""
    if not specs and not kwargs:
        raise TypeError("reframe() requires at least one expression")

    def _apply(tf):
        groups, _ = _operation_groups(tf, by, "reframe")
        context = _group_context(tf, groups)
        assignments = _expanded_assignments(context, specs, kwargs, "reframe")
        if tf._backend == "pandas":
            work = tf._pdf
            marker = None
            if tf._rowwise:
                marker = _temp_column(_frame_columns(tf), "__tidy3_rowwise")
                work = work.assign(**{marker: range(len(work))})
                groups = [*(tf._groups or []), marker]
            pdf = _pe().do_reframe(work, assignments, groups)
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
):
    eff = list(dict.fromkeys([*(tf._groups or []), *cols]))
    out_name = _count_name(tf, name, eff)
    if tf._backend == "pandas":
        pdf = _pe().do_count(tf._pdf, tuple(eff), out_name, wt=wt, sort=sort)
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
    else:
        lf = tf._lf.select(agg.alias(out_name))
    if sort:
        lf = lf.sort(out_name, descending=True)
    return tf._with_lf(lf, groups=result_groups)


def count(
    *cols: str,
    wt: Any = None,
    sort: bool = False,
    name: str | None = None,
) -> Verb:
    """Count rows per group (dplyr): existing ``group_by`` groups plus *cols*.

    ``group_by("a") >> count()`` ≡ R's ``group_by(a) %>% tally()``.
    Existing groups are used for the calculation and preserved on the result.
    """

    def _apply(tf):
        groups = list(tf._groups) if tf._groups else None
        return _count_rows(
            tf, cols, wt=wt, sort=sort, name=name, result_groups=groups
        )

    return Verb(_apply, "count")


def tally(*, wt: Any = None, sort: bool = False, name: str | None = None) -> Verb:
    """Count rows using existing groups, with summarise-style group dropping."""

    def _apply(tf):
        groups = list(tf._groups[:-1]) if tf._groups and len(tf._groups) > 1 else None
        return _count_rows(
            tf, (), wt=wt, sort=sort, name=name, result_groups=groups
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
            raise NotImplementedError(
                "slice_sample() weight_by currently requires backend='pandas'"
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
            return tf._with_lf(
                tf._lf.join(r, on=keys, how=how, **params), groups=tf._groups
            )
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


def bind_cols(*others: Any) -> Verb:
    """Combine frames column-wise; duplicate column names are rejected."""
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
        if len(set(all_columns)) != len(all_columns):
            raise ValueError("bind_cols() inputs must have unique column names")
        if tf._backend == "pandas":
            import pandas as pd

            sizes = {len(frame) for frame in frames}
            if len(sizes) > 1:
                raise ValueError("bind_cols() inputs must have the same row count")
            pdf = pd.concat(
                [frame.reset_index(drop=True) for frame in frames], axis=1
            )
            return tf._with_pdf(pdf, groups=tf._groups)
        return tf._with_lf(
            pl.concat(frames, how="horizontal_extend"), groups=tf._groups
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
