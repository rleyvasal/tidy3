"""Janitor-style cleaning helpers.

``remove_empty`` with ``which="rows"`` stays lazy on Polars. Deciding which
columns are empty or constant, and reading a header row for ``row_to_names``,
looks at the current values, so those results discover their schema from the
data. Blank strings are kept: only missing values count as empty. Nothing
here prints janitor's status messages.
"""

from __future__ import annotations

import math
import sys
import warnings
from datetime import date, datetime
from typing import Any

import numpy as np
import polars as pl

from tidy3.verbs import Verb

_WHICH = frozenset({"rows", "cols"})
_ROUNDING = frozenset({"half to even", "half up"})
_RETURNS = frozenset({"all", "match", "mismatch"})
_BIND = frozenset({"bind_rows", "rbind"})


def tabyl(
    column: Any, column2: Any = None, column3: Any = None, *, show_na: bool = True,
    show_missing_levels: bool = True,
) -> Any:
    """One-variable frequencies, or two-variable tables of numeric counts.

    ``frame >> tabyl("x")`` selects one column; a vector is tabulated
    immediately, using its name or ``"x"``. Missing rows are shown by default
    and add ``valid_percent`` (proportions excluding missing rows). Unused
    factor levels have zero counts unless ``show_missing_levels=False``.
    ``frame >> tabyl("x", "y")`` makes one column per value of y. Empty
    combinations have zero counts. ``frame >> tabyl("x", "y", "z")`` returns
    a dict of two-variable tables keyed by z's values as text, in level or
    sorted order, missing (``"NA_"``) last; every table shows all x and y
    values unless ``show_missing_levels=False``. Existing groups are
    ignored. The small count table is computed once.
    """
    from tidy3.expr import Expr
    from tidy3.frame import note_data_read, tidy
    from tidy3.tidyselect import Selector, resolve_selection
    from tidy3.verbs import arrange, count

    _bool("tabyl() show_na", show_na)
    _bool("tabyl() show_missing_levels", show_missing_levels)
    if not isinstance(column, (str, Expr, pl.Expr, Selector)):
        if column2 is not None or column3 is not None:
            raise TypeError("tabyl() with two or three variables requires frame column selections")
        import pandas as pd

        name = str(getattr(column, "name", None) or "x")
        source = tidy(column.rename(name).to_frame()) if isinstance(column, pl.Series) else tidy(pd.DataFrame({name: column}))
        return source >> tabyl(
            name, show_na=show_na, show_missing_levels=show_missing_levels,
        )

    if column3 is not None and column2 is None:
        raise TypeError("tabyl() with a third variable also requires a second")

    def _apply(tf):
        if column3 is not None:
            return _tabyl_three_way(tf, [column, column2, column3], show_na, show_missing_levels)
        if column2 is not None:
            return _tabyl_two_way(tf, column, column2, show_na, show_missing_levels)
        selected = resolve_selection(tf, [column])
        if len(selected) != 1:
            raise ValueError("tabyl() requires exactly one column")
        name = selected[0]
        if name in {"n", "percent", "valid_percent"}:
            raise ValueError(f"tabyl() column {name!r} conflicts with an output name; rename it first")
        dtype = tf._schema()[name]
        if dtype.base_type() in {pl.List, pl.Array, pl.Struct, pl.Object, pl.Binary}:
            raise TypeError("tabyl() requires a scalar numeric, text, factor, boolean, or date column")
        levels = {name: tf._category_levels[name]} if name in tf._category_levels else {}
        source = tf._with_lf(tf._lf.select(name), groups=None, rowwise=False, category_levels=levels)
        missing = _missing_expr(name, dtype)
        if not show_na:
            source = source._with_lf(source._lf.filter(~missing), rowwise=False)
        keep_levels = show_missing_levels and (bool(levels) or isinstance(dtype, (pl.Enum, pl.Categorical)))
        counted = source >> count(name, name="n", drop=not keep_levels)
        if levels:
            counted = counted >> arrange(name)
        else:
            counted = counted._with_lf(counted._lf.sort(name, nulls_last=True), rowwise=False)
        note_data_read("tabyl() reads frequency counts to choose percentage columns")
        snapshot = counted._compute()
        data = snapshot.collect()
        total = data["n"].sum()
        expressions = [(pl.col("n") / _per_row(pl.lit(total))).alias("percent")]
        if data.select(missing.any()).item():
            valid_total = data.filter(~missing)["n"].sum()
            expressions.append(
                pl.when(missing).then(None)
                .otherwise(pl.col("n") / _per_row(pl.lit(valid_total))).alias("valid_percent")
            )
        result = data.with_columns(expressions)
        return snapshot._with_lf(result.lazy(), rowwise=False, tabyl=("one_way", result))

    return Verb(_apply, "tabyl")


def _tabyl_axis(tf: Any, data: pl.DataFrame, name: str, show_missing_levels: bool) -> pl.Series:
    """Distinct axis values, including unused factor levels when requested."""
    from tidy3.verbs import arrange

    values = data[name]
    levels = tf._category_levels.get(name)
    if levels is None and isinstance(values.dtype, pl.Enum):
        levels = values.dtype.categories.to_list()
    elif levels is None and values.dtype == pl.Categorical:
        levels = values.cat.get_categories().to_list()
    if show_missing_levels and levels is not None:
        missing = data.filter(_missing_expr(name, values.dtype))[name].unique()
        axis = pl.concat([pl.Series(name, levels, dtype=values.dtype), missing])
    else:
        axis = values.unique()
    if values.dtype.is_float() and name not in tf._category_levels:
        return axis.sort(nulls_last=True)
    view = tf._with_lf(axis.to_frame().lazy(), rowwise=False)
    return (view >> arrange(name)).collect()[name]


def _tabyl_column_label(value: Any) -> str:
    if value is None:
        return "NA_"
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Inf" if value > 0 else "-Inf"
    return "emptystring_" if value == "" else _header_text(value)


def _tabyl_names(tf: Any, selections: list[Any]) -> list[str]:
    from tidy3.tidyselect import resolve_selection

    names = []
    for selection in selections:
        selected = resolve_selection(tf, [selection])
        if len(selected) != 1:
            raise ValueError("tabyl() requires exactly one column for each variable")
        names.append(selected[0])
    if len(set(names)) != len(names):
        raise ValueError(f"tabyl() requires {'two' if len(names) == 2 else 'three'} distinct columns")
    schema = tf._schema()
    if any(schema[name].base_type() in {pl.List, pl.Array, pl.Struct, pl.Object, pl.Binary} for name in names):
        raise TypeError("tabyl() requires scalar numeric, text, factor, boolean, or date columns")
    return names


def _tabyl_counts(tf: Any, names: list[str], show_na: bool, reason: str) -> tuple[Any, Any, pl.DataFrame, str]:
    """The source columns and their combination counts, computed once."""
    from tidy3.frame import note_data_read

    schema = tf._schema()
    levels = {name: tf._category_levels[name] for name in names if name in tf._category_levels}
    source = tf._with_lf(tf._lf.select(names), rowwise=False, category_levels=levels)
    if not show_na:
        missing = pl.any_horizontal([_missing_expr(name, schema[name]) for name in names])
        source = source._with_lf(source._lf.filter(~missing), rowwise=False)
    count_name = "__tabyl_n"
    while count_name in names:
        count_name += "_"
    counts = source._with_lf(source._lf.group_by(names).agg(pl.len().alias(count_name)), rowwise=False)
    note_data_read(reason)
    snapshot = counts._compute()
    return source, snapshot, snapshot.collect(), count_name


def _tabyl_two_way(tf: Any, column: Any, column2: Any, show_na: bool, show_missing_levels: bool) -> Any:
    names = _tabyl_names(tf, [column, column2])
    source, snapshot, data, count_name = _tabyl_counts(
        tf, names, show_na, "tabyl() reads frequency counts to discover table axes",
    )
    row, col = names
    if data.is_empty():
        return _tabyl_wide(source, snapshot, data, names, count_name, None, None)
    rows = _tabyl_axis(source, data, row, show_missing_levels)
    columns = _tabyl_axis(source, data, col, show_missing_levels)
    return _tabyl_wide(source, snapshot, data, names, count_name, rows, columns)


def _tabyl_wide(
    source: Any, snapshot: Any, data: pl.DataFrame, names: list[str], count_name: str,
    rows: pl.Series | None, columns: pl.Series | None,
) -> Any:
    """One column of counts per value of the second variable."""
    row, col = names
    row_levels = {row: source._category_levels[row]} if row in source._category_levels else {}
    if rows is None or columns is None:
        result = data.select(row)
        return snapshot._with_lf(result.lazy(), rowwise=False, category_levels=row_levels,
                                 tabyl=("two_way", result))
    labels = [_tabyl_column_label(value) for value in columns]
    if len(set([row, *labels])) != len(labels) + 1:
        raise ValueError("tabyl() column labels conflict; rename the row column or recode the column values")
    ordered = [(value, label) for value, label in zip(columns, labels) if label not in {"emptystring_", "NA_"}]
    ordered += [(value, label) for value, label in zip(columns, labels) if label == "emptystring_"]
    ordered += [(value, label) for value, label in zip(columns, labels) if label == "NA_"]
    expressions = [
        pl.col(count_name).filter(pl.col(col).eq_missing(pl.lit(value, dtype=columns.dtype))).sum().alias(label)
        for value, label in ordered
    ]
    wide = data.group_by(row).agg(expressions)
    result = rows.to_frame().join(wide, on=row, how="left", nulls_equal=True, maintain_order="left")
    result = result.with_columns(pl.col([label for _, label in ordered]).fill_null(0).cast(pl.Float64))
    return snapshot._with_lf(result.lazy(), rowwise=False, category_levels=row_levels,
                             tabyl=("two_way", result))


def _tabyl_three_way(tf: Any, selections: list[Any], show_na: bool, show_missing_levels: bool) -> dict[str, Any]:
    """Two-way tables split by the third variable, as janitor's list of tabyls.

    As in janitor, ``show_missing_levels`` gives every table all values of
    the first two variables; a missing value appears only in tables that
    have it. Factor tables follow the level order, unlike janitor 2.2.1,
    which reorders some factors with three or more levels.
    """
    names = _tabyl_names(tf, selections)
    source, snapshot, data, count_name = _tabyl_counts(
        tf, names, show_na, "tabyl() reads frequency counts to split tables and discover axes",
    )
    row, col, split = names
    if data.is_empty():
        return {}
    schema = source._schema()
    full = {}
    if show_missing_levels:
        for name in (row, col):
            full[name] = _tabyl_axis(source, data, name, True)
    keys = _tabyl_axis(source, data, split, False)
    result = {}
    for value in keys:
        part = data.filter(pl.col(split).eq_missing(pl.lit(value, dtype=keys.dtype))).drop(split)
        axes = []
        for name in (row, col):
            if name in full:
                # A missing value (null or NaN) appears only in tables that have it.
                present = {_tabyl_column_label(value) for value in
                           part.filter(_missing_expr(name, schema[name]))[name].to_list()}
                axis = full[name]
                keep = pl.Series([not _missing(value) or _tabyl_column_label(value) in present
                                  for value in axis.to_list()], dtype=pl.Boolean)
                axes.append(axis.filter(keep))
            else:
                axes.append(_tabyl_axis(source, part, name, False))
        key = "NA_" if value is None else ("" if value == "" else _tabyl_column_label(value))
        result[key] = _tabyl_wide(source, snapshot, part, [row, col], count_name, *axes)
    return result


class _ReportVerb(Verb):
    """A reporting verb that also runs on each table of a three-variable tabyl.

    ``tabyl(x, y, z)`` returns a dict of tables; ``dict >> verb`` gives a
    dict with the verb applied to each, as janitor maps over its list.
    """

    __slots__ = ()

    def __rrshift__(self, other: Any) -> Any:
        if isinstance(other, dict):
            return {key: table >> self for key, table in other.items()}
        return super().__rrshift__(other)


def adorn_totals(
    where: Any = "row", *, fill: str | None = "-", na_rm: bool = True,
    name: Any = "Total", cols: Any = None,
) -> Verb:
    """Append totals by row, column, or both; keep totals numeric.

    By default numeric columns after the label column are summed. ``cols``
    accepts tidyselect specifications, including the first column explicitly.
    ``fill=None`` keeps unselected columns' types with a missing total cell;
    the row label still becomes text (or an expanded factor).
    """
    from tidy3.frame import _as_levels
    from tidy3.tidyselect import resolve_selection

    _bool("adorn_totals() na_rm", na_rm)
    dimensions = [where] if isinstance(where, str) else list(where)
    if "both" in dimensions:
        dimensions = ["row", "col"]
    if not dimensions or any(value not in {"row", "col"} for value in dimensions):
        raise ValueError('adorn_totals() where must be "row", "col", or both')
    dimensions = list(dict.fromkeys(dimensions))
    labels = [name, name] if isinstance(name, str) else list(name)
    if len(labels) != 2 or any(not isinstance(label, str) or not label for label in labels):
        raise ValueError("adorn_totals() name must be a string or two non-empty strings")
    if fill is not None and not isinstance(fill, str):
        raise TypeError("adorn_totals() fill must be a string or None")

    def _apply(tf):
        if any(dimension in tf._totals for dimension in dimensions):
            raise ValueError("adorn_totals() cannot re-add an existing totals dimension")
        schema = tf._schema()
        names = schema.names()
        selected = names[1:] if cols is None else resolve_selection(tf, [cols])
        selected = [column for column in selected if schema[column].is_numeric() and column not in tf._category_levels]
        if not selected:
            raise ValueError("adorn_totals() requires at least one selected numeric column")
        if "col" in dimensions and labels[1] in names:
            raise ValueError(f"adorn_totals() totals column {labels[1]!r} already exists")
        lf = tf._lf
        levels = dict(tf._category_levels)
        if "row" in dimensions:
            base, totals = [], []
            for column in names:
                dtype = schema[column]
                value = pl.col(column)
                if column in selected:
                    summed = value.fill_nan(None) if na_rm and dtype.is_float() else value
                    total = summed.sum()
                    if not na_rm:
                        total = pl.when(value.is_null().any()).then(None).otherwise(total)
                elif column == names[0]:
                    total_dtype = pl.String
                    recorded = levels.get(column)
                    if recorded is None and isinstance(dtype, pl.Enum):
                        recorded = dtype.categories.to_list()
                    if recorded is not None:
                        category_names = [_header_text(level) for level in recorded]
                        if labels[0] in category_names:
                            raise ValueError("adorn_totals() totals row name is already a factor level")
                        category_names.append(labels[0])
                        total_dtype = pl.Enum(category_names)
                        levels[column] = _as_levels(category_names, ordered=bool(getattr(recorded, "ordered", True)))
                    elif dtype == pl.Categorical:
                        total_dtype = dtype
                    value = _adorn_text(column, dtype).cast(total_dtype)
                    total = pl.lit(labels[0], dtype=total_dtype)
                elif fill is not None:
                    value = _adorn_text(column, dtype)
                    total = pl.lit(fill)
                    levels.pop(column, None)
                else:
                    total = pl.lit(None, dtype=dtype)
                base.append(value.alias(column))
                totals.append(total.alias(column))
            lf = pl.concat([lf.select(base), lf.select(totals)], how="vertical_relaxed")
        if "col" in dimensions:
            values = [pl.col(column).fill_nan(None) if na_rm and schema[column].is_float()
                      else pl.col(column) for column in selected]
            lf = lf.with_columns(
                pl.sum_horizontal(values, ignore_nulls=na_rm).cast(pl.Float64).alias(labels[1])
            )
        markers = {**tf._totals, **{dimension: labels[0 if dimension == "row" else 1] for dimension in dimensions}}
        return tf._with_lf(lf, rowwise=False, category_levels=levels, totals=markers, tabyl=tf._tabyl)

    return _ReportVerb(_apply, "adorn_totals")


def adorn_percentages(denominator: str = "row", *, na_rm: bool = True, cols: Any = None) -> Verb:
    """Divide counts by row, column, or grand totals; results stay numeric.

    By default numeric columns after the label column are divided; ``cols``
    accepts tidyselect specifications, and text or factor columns among them
    are skipped. Totals from :func:`adorn_totals` are not counted twice: a
    totals row is left out of column and grand sums, and a totals column
    becomes 1 (by row) or a share of its own sum, unless ``cols`` leaves it
    out. ``na_rm`` skips missing values in row and column sums; the grand
    total always skips them. A zero sum gives NaN.
    """
    from tidy3.tidyselect import resolve_selection

    _bool("adorn_percentages() na_rm", na_rm)
    if denominator not in {"row", "col", "all"}:
        raise ValueError('adorn_percentages() denominator must be "row", "col", or "all"')

    def _apply(tf):
        schema = tf._schema()
        names = schema.names()
        numeric = {column for column in names
                   if schema[column].is_numeric() and column not in tf._category_levels}
        if not any(column in numeric for column in names[1:]):
            raise ValueError("adorn_percentages() requires a numeric column after the first column")
        totals_column = tf._totals.get("col")
        chosen = names[1:] if cols is None else resolve_selection(tf, [cols])
        # As in janitor, choosing columns without the totals column leaves it alone.
        include_totals = totals_column is not None and (cols is None or totals_column in chosen)
        selected = [column for column in chosen if column in numeric and column != totals_column]
        if not selected:
            return tf._with_lf(tf._lf, rowwise=False, totals=tf._totals, tabyl=tf._tabyl)

        def value(column: str, skip_missing: bool) -> pl.Expr:
            if skip_missing and schema[column].is_float():
                return pl.col(column).fill_nan(None)
            return pl.col(column)

        # A totals row is the last row; it is not part of column or grand sums.
        body = pl.int_range(pl.len()) < pl.len() - 1 if "row" in tf._totals else pl.lit(True)

        def column_sum(column: str, skip_missing: bool) -> pl.Expr:
            values = value(column, skip_missing).filter(body)
            if skip_missing:
                return values.sum()
            return pl.when(values.is_null().any()).then(None).otherwise(values.sum())

        if denominator == "row":
            row_sum = pl.sum_horizontal([value(column, na_rm) for column in selected], ignore_nulls=na_rm)
            divided = [(pl.col(column) / row_sum).alias(column) for column in selected]
            if include_totals:
                divided.append(pl.lit(1.0).alias(totals_column))
        elif denominator == "col":
            divided = [(pl.col(column) / _per_row(column_sum(column, na_rm))).alias(column) for column in selected]
            if include_totals:
                total = _per_row(column_sum(totals_column, False))
                divided.append((pl.col(totals_column) / total).alias(totals_column))
        else:
            grand = _per_row(pl.sum_horizontal([column_sum(column, True) for column in selected]))
            divided = [(pl.col(column) / grand).alias(column) for column in selected]
            if include_totals:
                divided.append((pl.col(totals_column) / grand).alias(totals_column))
        lf = tf._lf.with_columns(expr.cast(pl.Float64) for expr in divided)
        return tf._with_lf(lf, rowwise=False, totals=tf._totals, tabyl=tf._tabyl)

    return _ReportVerb(_apply, "adorn_percentages")


def adorn_pct_formatting(
    digits: int = 1, rounding: str = "half to even", affix_sign: bool = True, *, cols: Any = None,
) -> Verb:
    """Write proportions as percentage text: 0.25 becomes ``"25.0%"``.

    This is the last step of a report: the formatted columns are text and
    can no longer be summed, sorted, or divided. Values are multiplied by
    100, rounded to ``digits`` decimals, and always shown with that many
    decimals. ``rounding="half to even"`` is R's ``round()`` (the nearest
    representable value, ties to even); ``"half up"`` rounds ties away from
    zero. ``affix_sign=False`` leaves off ``"%"``. Missing values and NaN
    become ``"-"``. By default numeric columns after the first are
    formatted; ``cols=`` selects others and skips text and factor columns.
    On a one-variable tabyl the ``n`` column is never formatted, as in
    janitor, which then ignores ``cols`` and turns ``n`` into floats.
    """
    from tidy3.tidyselect import resolve_selection

    if isinstance(digits, bool) or not isinstance(digits, int):
        raise TypeError("adorn_pct_formatting() digits must be an integer")
    if digits < 0:
        raise ValueError("adorn_pct_formatting() digits must be 0 or more")
    if rounding not in _ROUNDING:
        raise ValueError('adorn_pct_formatting() rounding must be "half to even" or "half up"')
    _bool("adorn_pct_formatting() affix_sign", affix_sign)
    sign = "%" if affix_sign else ""

    def text(value: float | None) -> str:
        if value is None or math.isnan(value):
            return "-"
        value *= 100
        if math.isinf(value):
            return ("Inf" if value > 0 else "-Inf") + sign
        rounded = round(value, digits) if rounding == "half to even" else _round_half_up(value, digits)
        # Adding 0.0 turns -0.0 into 0.0, which R shows as "0".
        return f"{rounded + 0.0:.{digits}f}{sign}"

    def batch(series: pl.Series) -> pl.Series:
        return _percent_text(series, digits, rounding, sign, text)

    def _apply(tf):
        schema = tf._schema()
        names = schema.names()
        numeric = [column for column in names
                   if schema[column].is_numeric() and column not in tf._category_levels]
        rounded = []
        if tf._tabyl is not None and tf._tabyl[0] == "one_way":
            chosen = [column for column in numeric if column not in names[:2]]
            # janitor rounds the counts as well, which makes them doubles.
            rounded = [pl.col(column).cast(pl.Float64) for column in numeric if column == names[1]]
        elif cols is None:
            chosen = [column for column in numeric if column != names[0]]
        else:
            chosen = [column for column in resolve_selection(tf, [cols]) if column in numeric]
        if not chosen:
            raise ValueError("adorn_pct_formatting() requires at least one selected numeric column")
        lf = tf._lf.with_columns(
            *rounded,
            *[pl.col(column).map_batches(batch, return_dtype=pl.String, is_elementwise=True) for column in chosen],
        )
        return tf._with_lf(lf, rowwise=False, category_levels=_levels(tf, names),
                           totals=tf._totals, tabyl=tf._tabyl)

    return _ReportVerb(_apply, "adorn_pct_formatting")


def _percent_text(series: pl.Series, digits: int, rounding: str, sign: str, text: Any) -> pl.Series:
    """:func:`adorn_pct_formatting`'s text for a column, rounded in bulk.

    The rounded number is an integer count of the last decimal, so the text
    is that integer with a decimal point put in. Values within a hair of a
    rounding tie, or too large for an exact integer, go through *text*
    one by one, so every value gets exactly the per-value result.
    """
    values = series.cast(pl.Float64).to_numpy() * 100  # missing values become NaN
    scale = 10.0 ** digits
    finite = np.isfinite(values)
    with np.errstate(invalid="ignore", over="ignore"):  # these values go the per-value way
        if rounding == "half to even":
            scaled = values * scale
            fraction = np.abs(scaled - np.floor(scaled))
            exact = finite & (np.abs(fraction - 0.5) > 1e-7) & (np.abs(scaled) < 1e12)
            whole = np.rint(scaled)
        else:
            # janitor's round_half_up, in the same double arithmetic.
            whole = np.trunc(np.abs(values) * scale + 0.5 + math.sqrt(sys.float_info.epsilon)) * np.sign(values)
            exact = finite & (np.abs(whole) < 1e12)
    counts = pl.Series(np.where(exact, whole, 0.0)).cast(pl.Int64)
    magnitude = counts.abs()
    body = (magnitude // int(scale)).cast(pl.String)
    if digits:
        body = body + "." + (magnitude % int(scale)).cast(pl.String).str.zfill(digits)
    out = pl.select(
        pl.when(pl.Series(exact))
        .then(pl.when(counts < 0).then(pl.lit("-")).otherwise(pl.lit("")) + body + pl.lit(sign))
        .when(pl.Series(np.isnan(values))).then(pl.lit("-"))
        .when(pl.Series(np.isposinf(values))).then(pl.lit("Inf" + sign))
        .when(pl.Series(np.isneginf(values))).then(pl.lit("-Inf" + sign))
        .otherwise(pl.lit(None, dtype=pl.String))
    ).to_series()
    rest = np.flatnonzero(finite & ~exact)
    if rest.size:
        originals = series.cast(pl.Float64).gather(rest).to_list()
        out = out.scatter(rest, [text(value) for value in originals])
    return out.alias(series.name)


def _round_half_up(value: float, digits: int) -> float:
    """janitor's ``round_half_up``, step by step in the same double arithmetic."""
    sign = (value > 0) - (value < 0)
    scaled = math.trunc(abs(value) * 10 ** digits + 0.5 + math.sqrt(sys.float_info.epsilon))
    return scaled / 10 ** digits * sign


def adorn_ns(
    position: str = "rear", *, ns: Any = None, format_func: Any = None, cols: Any = None,
) -> Verb:
    """Add counts to a report's cells: ``"25.0% (3)"``, or ``"3 (25.0%)"`` in front.

    Without ``ns``, the counts are the tabyl's own, with the same totals
    added; a verb that changes the data in between clears them, so call it
    straight after the other ``adorn_*`` steps. ``ns=`` supplies a frame
    of the same shape for any table; its values are written as R writes
    them, without ``format_func``. ``format_func`` turns each count (a
    number, or ``None`` when missing) into text; the default writes
    ``1,234``. The part in parentheses is padded on the left to line up
    within each column. Numeric cells are written as janitor writes them:
    15 significant digits in parentheses, 7 (aligned) in front. By
    default the count columns after the first are combined; ``cols=``
    selects others. Combined columns are text. On a one-variable tabyl
    this warns, as janitor does.
    """
    from tidy3.tidyselect import resolve_selection

    if position not in {"rear", "front"}:
        raise ValueError('adorn_ns() position must be "rear" or "front"')
    if format_func is not None and not callable(format_func):
        raise TypeError("adorn_ns() format_func must be a function or None")

    def _apply(tf):
        from tidy3.frame import note_data_read

        note_data_read("adorn_ns() reads the table to line up counts")
        data = tf._lf.collect()
        names = data.columns
        if ns is not None:
            counts = _as_frame(ns).collect()
            if counts.shape != data.shape:
                raise ValueError(f"adorn_ns() ns has shape {counts.shape}; the table has {data.shape}")
            numeric = list(range(1, len(names)))
            counted = None
        else:
            if tf._tabyl is None:
                raise ValueError(
                    "adorn_ns() needs a tabyl's counts, which data-changing steps clear; "
                    "call it right after tabyl() and the other adorn_* steps, or pass ns="
                )
            kind, counts = tf._tabyl
            if kind == "one_way":
                warnings.warn(
                    "adorn_ns() is meant for a two-variable tabyl; for a one-variable tabyl "
                    "consider combining columns with unite()",
                    stacklevel=2,
                )
            if tf._totals:
                from tidy3.frame import TidyFrame

                label = counts.columns[0]
                where = [dimension for dimension in ("row", "col") if dimension in tf._totals]
                totals = adorn_totals(where, name=(tf._totals.get("row", "Total"), tf._totals.get("col", "Total")))
                counts = (TidyFrame(counts.lazy().with_columns(pl.col(label).cast(pl.String))) >> totals).collect()
            if counts.shape != data.shape:
                raise ValueError(
                    "adorn_ns() found counts of a different shape than the table; pass ns="
                )
            numeric = [index for index, column in enumerate(counts.columns)
                       if counts[column].dtype.is_numeric() and index > 0]
            counted = [_count_text(counts[column], format_func) for column in counts.columns]
        if cols is None:
            adorned = numeric
        else:
            chosen = set(resolve_selection(tf, [cols]))
            adorned = [index for index, column in enumerate(names) if column in chosen]
        if counted is None:
            front = _matrix_text(counts) if position == "front" else None
            rear = [_character_text(counts[column]) for column in counts.columns] if position == "rear" else None
        else:
            front = counted if position == "front" else None
            rear = counted if position == "rear" else None
        if position == "rear":
            front = _matrix_text(data, tf._category_levels)
        else:
            rear = [_character_text(data[column]) for column in names]
        columns = []
        for index, column in enumerate(names):
            if index not in adorned:
                columns.append(data[column])
                continue
            wrapped = "(" + rear[index] + ")"
            width = wrapped.str.len_chars().max() or 0
            columns.append((front[index] + " " + wrapped.str.pad_start(width)).alias(column))
        kept = [column for index, column in enumerate(names) if index not in adorned]
        result = pl.DataFrame(columns) if columns else data
        return tf._with_lf(result.lazy(), rowwise=False, category_levels=_levels(tf, kept),
                           totals=tf._totals, tabyl=tf._tabyl)

    return _ReportVerb(_apply, "adorn_ns")


# The text helpers below return String columns. A value's text depends only
# on the value (and, for format(), on the column's set of values), so each
# distinct value is written once in Python and mapped back to the rows;
# integers, text, and logical values are written natively.


def _by_value(series: pl.Series, write: Any) -> pl.Series:
    """*write* applied to each distinct value, mapped back to every row.

    NaN never equals itself, so it is written once and placed by mask.
    """
    nan = series.is_nan() if series.dtype.is_float() else None
    keys = series.fill_nan(None) if nan is not None else series
    texts = {value: write(value) for value in keys.unique().to_list() if value is not None}
    out = keys.replace_strict(texts, default=None, return_dtype=pl.String) if texts else (
        pl.Series(series.name, [None] * series.len(), dtype=pl.String)
    )
    if series.null_count():
        out = pl.select(pl.when(series.is_null()).then(pl.lit(write(None))).otherwise(out)).to_series()
    if nan is not None and nan.any():
        out = pl.select(pl.when(nan).then(pl.lit(write(math.nan))).otherwise(out)).to_series()
    return out.alias(series.name)


def _integer_text(series: pl.Series, big_mark: str = "") -> pl.Series:
    """Integers as decimal text, with *big_mark* between thousands; NA when missing."""
    text = series.cast(pl.String)
    if big_mark:
        digits = series.abs().cast(pl.String)
        # Group from the right: reverse, mark every third digit, reverse back.
        grouped = (digits.str.reverse().str.replace_all(r"([0-9]{3})", "${1}" + big_mark)
                   .str.strip_chars_end(big_mark).str.reverse())
        text = pl.select(pl.when(series < 0).then("-" + grouped).otherwise(grouped)).to_series()
    return text.fill_null("NA").alias(series.name)


def _count_text(series: pl.Series, format_func: Any) -> pl.Series:
    """A count column as janitor's default ``format(x, big.mark = ",")``, trimmed."""
    if not series.dtype.is_numeric():
        return _character_text(series).str.strip_chars()
    if format_func is None:
        return _r_format(series, big_mark=",").str.strip_chars()
    return pl.Series(series.name, [str(format_func(value)).strip() for value in series.to_list()], dtype=pl.String)


def _character_text(series: pl.Series) -> pl.Series:
    """R's ``as.character()`` of each value, then ``paste()``'s ``"NA"``."""
    if series.dtype.is_float():
        return _by_value(series, lambda value: _r_reals([value], 15)[0])
    return _plain_text(series)


def _plain_text(series: pl.Series) -> pl.Series:
    if series.dtype == pl.String:
        return series.fill_null("NA")
    if series.dtype.is_integer():
        return _integer_text(series)
    if series.dtype == pl.Boolean:
        return series.replace_strict({True: "TRUE", False: "FALSE"}, return_dtype=pl.String).fill_null("NA")
    return _by_value(series, lambda value: "NA" if value is None else str(value))


def _matrix_text(frame: pl.DataFrame, levels: Any = ()) -> list[pl.Series]:
    """Cells as R's ``as.matrix()`` writes a data frame.

    A frame of numbers is written as ``as.character()`` writes each value.
    Otherwise numeric columns are written by ``format()``: 7 significant
    digits with the same decimals and width down the column.
    """
    numeric = [frame[column].dtype.is_numeric() and column not in levels for column in frame.columns]
    if all(numeric):
        return [_character_text(frame[column]) for column in frame.columns]
    texts = []
    for column, is_numeric in zip(frame.columns, numeric):
        if not is_numeric:
            texts.append(_plain_text(frame[column]))
            continue
        # Missing values, NaN included, are put back as NA after formatting.
        values = frame[column]
        missing = values.is_null() | (values.is_nan() if values.dtype.is_float() else False)
        formatted = _r_format(values)
        texts.append(pl.select(pl.when(missing).then(pl.lit("NA")).otherwise(formatted)).to_series().alias(column))
    return texts


def _r_format(series: pl.Series, *, big_mark: str = "") -> pl.Series:
    """R's ``format()`` of a numeric column: common decimals, right-aligned."""
    if series.dtype.is_float():
        # The layout depends on the set of values, so the distinct ones decide it.
        distinct = series.unique().to_list()
        written = dict(zip(distinct, _r_reals(distinct, 7, big_mark)))
        texts = _by_value(series, lambda value: "NA" if value is None else "NaN" if math.isnan(value) else written[value])
    else:
        texts = _integer_text(series, big_mark)
    width = texts.str.len_chars().max() or 0
    return texts.str.pad_start(width)


def _r_reals(values: list[Any], digits: int, big_mark: str = "") -> list[str]:
    """Doubles written together as R writes them with ``digits`` significant digits.

    The same number of decimals serves every value; scientific notation is
    used when it is narrower, as R decides with ``scipen = 0``.
    """
    finite = [value for value in values if value is not None and math.isfinite(value)]
    fixed = True
    decimals = 0
    significant = 1
    if finite:
        shapes = []
        for value in finite:
            mantissa, exponent = f"{abs(value):.{digits - 1}e}".split("e")
            shapes.append((len(mantissa.replace(".", "").rstrip("0")) or 1, int(exponent)))
        decimals = max(max(0, sig - 1 - exponent) for sig, exponent in shapes)
        significant = max(sig for sig, _ in shapes)
        negative = any(value < 0 for value in finite)
        left = max(max(1, exponent + 1) for _, exponent in shapes)
        fixed_width = negative + left + decimals + (decimals > 0)
        exponent_width = 4 if max(abs(exponent) for _, exponent in shapes) < 100 else 5
        fixed = fixed_width <= negative + significant + (significant > 1) + exponent_width
    texts = []
    for value in values:
        if value is None:
            texts.append("NA")
        elif math.isnan(value):
            texts.append("NaN")
        elif math.isinf(value):
            texts.append("Inf" if value > 0 else "-Inf")
        elif fixed:
            texts.append(_big_mark(f"{value + 0.0:.{decimals}f}", big_mark))
        else:
            # Adding 0.0 turns -0.0 into 0.0: R never writes a negative zero.
            texts.append(f"{value + 0.0:.{significant - 1}e}")
    return texts


def _big_mark(text: str, mark: str) -> str:
    if not mark:
        return text
    sign = "-" if text.startswith("-") else ""
    whole, dot, fraction = text.lstrip("-").partition(".")
    groups = []
    while len(whole) > 3:
        groups.insert(0, whole[-3:])
        whole = whole[:-3]
    return sign + mark.join([whole, *groups]) + dot + fraction


def _per_row(total: pl.Expr) -> pl.Expr:
    """*total* repeated on every row, so dividing by it is exact.

    Polars divides by a single value as multiplication by its reciprocal
    (3 / 5 gives 0.6000000000000001); R divides each value, giving 0.6.
    """
    return total + pl.int_range(pl.len()).cast(pl.Float64) * 0.0


def _adorn_text(column: str, dtype: Any) -> pl.Expr:
    """Native text conversion for label/fill columns in numeric reports."""
    value = pl.col(column).cast(pl.String)
    if dtype == pl.Boolean:
        return value.replace({"true": "TRUE", "false": "FALSE"})
    if dtype.is_float():
        return value.str.replace(r"\.0$", "")
    return value


def _bool(name: str, value: Any) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")


def _missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    try:
        import pandas as pd

        if value is pd.NA or value is pd.NaT:
            return True
        flag = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(flag) if isinstance(flag, bool) else False


def _which(which: Any) -> tuple[str, ...]:
    if which is None or which == "both" or which == ():
        return ("rows", "cols")
    parts = (which,) if isinstance(which, str) else tuple(which)
    if not parts or any(not isinstance(part, str) or part not in _WHICH for part in parts):
        raise ValueError(
            'remove_empty() which must be "rows", "cols", "both", or both of '
            '"rows" and "cols"'
        )
    ordered: list[str] = []
    for part in parts:
        if part not in ordered:
            ordered.append(part)
    return tuple(ordered)


def _cutoff(cutoff: Any) -> float:
    if isinstance(cutoff, bool) or not isinstance(cutoff, (int, float)):
        raise TypeError("remove_empty() cutoff must be a number")
    value = float(cutoff)
    if not 0 < value <= 1:
        raise ValueError("remove_empty() cutoff must be > 0 and <= 1")
    return value


def _groups(tf: Any, names: list[str]) -> list[str] | None:
    kept = [group for group in (tf._groups or []) if group in names]
    return kept or None


def _levels(tf: Any, names: list[str]) -> dict[str, Any]:
    kept = set(names)
    return {name: levels for name, levels in tf._category_levels.items() if name in kept}


def _renamed(mapping: dict[str, str], names: list[str] | None) -> list[str] | None:
    if not names:
        return None
    return [mapping[name] for name in names]


def _finish_polars(tf: Any, lf: pl.LazyFrame, names: list[str]) -> Any:
    # Dropping every column also drops the rows: Polars cannot store a row
    # count without a column. Pandas keeps that row count.
    selected = lf.select(names) if names else lf.select([])
    return tf._with_lf(selected, _groups(tf, names), category_levels=_levels(tf, names))


def _float_dtype(dtype: Any) -> bool:
    is_float = getattr(dtype, "is_float", None)
    return bool(is_float()) if callable(is_float) else False


def _missing_expr(name: str, dtype: Any) -> pl.Expr:
    missing = pl.col(name).is_null()
    if _float_dtype(dtype):
        missing = missing | pl.col(name).is_nan()
    return missing


def _row_mask_polars(lf: pl.LazyFrame, cutoff: float) -> pl.LazyFrame:
    schema = lf.collect_schema()
    names = schema.names()
    width = len(names)
    if width == 0:
        if cutoff != 1:
            raise ValueError(
                "remove_empty() cutoff needs at least one column when removing rows"
            )
        return lf.filter(pl.lit(False))
    missing = pl.sum_horizontal(
        [_missing_expr(name, dtype).cast(pl.Int64) for name, dtype in schema.items()]
    )
    if cutoff == 1:
        return lf.filter(missing != width)
    return lf.filter(((width - missing) / width) > cutoff)


def _empty_columns(lf: pl.LazyFrame, cutoff: float) -> list[str]:
    """Columns that are empty enough to drop. This collects one count row."""
    from tidy3.frame import note_data_read

    note_data_read("remove_empty() finds empty columns in the data")
    schema = lf.collect_schema()
    names = schema.names()
    if not names:
        return []
    # null_count() reads the stored count of an in-memory column; floats
    # add their NaN values.
    stats = lf.select(
        pl.len().alias("__n"),
        *[
            (
                pl.col(name).null_count().cast(pl.Int64)
                + (pl.col(name).is_nan().sum().cast(pl.Int64) if _float_dtype(dtype) else 0)
            ).alias(f"__c{index}")
            for index, (name, dtype) in enumerate(schema.items())
        ],
    ).collect()
    height = int(stats["__n"][0])
    if cutoff != 1 and height == 0:
        raise ValueError(
            "remove_empty() cutoff needs at least one row when removing columns"
        )
    kept: list[str] = []
    for index, name in enumerate(names):
        missing = int(stats[f"__c{index}"][0])
        if cutoff == 1:
            keep = missing != height
        else:
            keep = ((height - missing) / height) > cutoff
        if keep:
            kept.append(name)
    return kept


def remove_empty(which: Any = "both", *, cutoff: float = 1) -> Verb:
    """Drop empty rows and columns (R **janitor::remove_empty**).

    A row or column is empty when every value is missing. ``""`` is a value,
    so convert blanks to missing first if those should go too. ``cutoff=1``
    (the default) keeps a row or column that has any non-missing value.
    A smaller cutoff keeps it only when the fraction of non-missing values
    is strictly greater than ``cutoff``. Rows are removed before columns.
    ``cutoff`` other than 1 cannot be combined with both rows and columns.

    ``remove_empty("rows")`` on Polars is a lazy filter. Dropping columns
    counts missing values first, so the set of surviving columns depends on
    the data. A zero-row frame has no non-missing values, so ``which="cols"``
    drops every column. Grouping, factor levels, and parsing diagnostics are
    kept for columns that remain. No removal message is printed.
    """
    targets = _which(which)
    limit = _cutoff(cutoff)
    if len(targets) > 1 and limit != 1:
        raise ValueError(
            "remove_empty() cutoff must be used with only one of "
            "which='rows' or which='cols', not both"
        )

    def _apply(tf: Any) -> Any:
        lf = tf._lf
        if "rows" in targets:
            lf = _row_mask_polars(lf, limit)
        names = _empty_columns(lf, limit) if "cols" in targets else list(lf.collect_schema().names())
        return _finish_polars(tf, lf, names)

    return Verb(_apply, "remove_empty")


def _frozen(value: Any) -> Any:
    """A hashable stand-in for a list, array or dict cell, compared by content."""
    if isinstance(value, (list, tuple, np.ndarray)):
        return ("__list", tuple(_frozen(item) for item in value))
    if isinstance(value, dict):
        return ("__dict", tuple((key, _frozen(item)) for key, item in value.items()))
    return value.item() if isinstance(value, np.generic) else value


def _constant_columns(lf: pl.LazyFrame, na_rm: bool) -> list[str]:
    """Columns whose values are not constant. This collects one uniqueness row."""
    from tidy3.frame import note_data_read

    note_data_read("remove_constant() finds constant columns in the data")
    schema = lf.collect_schema()
    names = schema.names()
    if not names:
        return []
    counts = []
    for index, (name, dtype) in enumerate(schema.items()):
        value = pl.col(name)
        if na_rm and _float_dtype(dtype):
            value = pl.when(value.is_nan()).then(None).otherwise(value)
        if na_rm:
            value = value.drop_nulls()
        counts.append(value.n_unique().alias(f"__c{index}"))
    row = lf.select(counts).collect()
    return [name for index, name in enumerate(names) if int(row[f"__c{index}"][0]) > 1]


def remove_constant(*, na_rm: bool = False) -> Verb:
    """Drop columns with fewer than two distinct values (R **janitor::remove_constant**).

    Missing counts as its own value when ``na_rm=False``, so a column that
    mixes one value and missing is kept. ``na_rm=True`` ignores missing
    values first; a column that is entirely missing is still removed.
    On Polars, null and NaN are different values. Pandas stores both as NaN
    in a float column, so that column looks constant. An empty column has
    no values and is removed, including every column of a zero-row frame.
    Unused factor levels do not count. The check reads the current values.
    Grouping and factor levels are kept for columns that remain. No removal
    message is printed.
    """
    _bool("remove_constant() na_rm", na_rm)

    def _apply(tf: Any) -> Any:
        names = _constant_columns(tf._lf, na_rm)
        return _finish_polars(tf, tf._lf, names)

    return Verb(_apply, "remove_constant")


def _quote_levels(levels: Any) -> str:
    return ", ".join(f'"{level}"' for level in levels)


def _factor_class(levels: Any, ordered: bool, strict: bool) -> str:
    if not strict:
        return "factor"
    text = f"factor(levels=c({_quote_levels(levels)}))"
    if ordered:
        return f"ordered, {text}"
    return text


def _plain_polars(dtype: Any) -> str:
    if dtype == pl.Null or getattr(dtype, "base_type", lambda: None)() == pl.Null:
        return "logical"
    if (
        _float_dtype(dtype)
        or bool(getattr(dtype, "is_decimal", lambda: False)())
        or type(dtype).__name__.startswith("Decimal")
    ):
        return "numeric"
    is_integer = getattr(dtype, "is_integer", None)
    if callable(is_integer) and is_integer():
        return "integer"
    # Polars Boolean.is_boolean() is false; compare the dtype itself.
    if dtype == pl.Boolean or getattr(dtype, "base_type", lambda: None)() == pl.Boolean:
        return "logical"
    if dtype == pl.Date or getattr(dtype, "base_type", lambda: None)() == pl.Date:
        return "Date"
    if isinstance(dtype, pl.Datetime):
        return "POSIXct, POSIXt"
    if dtype in {pl.String, pl.Utf8, pl.Categorical} or str(getattr(dtype, "base_type", lambda: dtype)()) in {
        "String",
        "Utf8",
    }:
        return "character"
    return str(dtype)


def _column_class(tf: Any, name: str, strict: bool) -> str:
    recorded = tf._category_levels.get(name)
    levels = list(recorded) if recorded is not None else None
    ordered = bool(getattr(recorded, "ordered", False)) if recorded is not None else False
    dtype = tf._lf.collect_schema()[name]
    if isinstance(dtype, (pl.Enum, pl.Categorical)):
        if levels is None and isinstance(dtype, pl.Enum):
            levels = dtype.categories.to_list()
            ordered = True
        return _factor_class(levels or [], ordered, strict)
    return _plain_polars(dtype)


def describe_class(x: Any, *, strict_description: bool = True) -> str:
    """Describe a column the way janitor's ``describe_class`` does.

    Factors are ``"factor"`` unless ``strict_description=True``, which
    includes the levels and an ``ordered`` prefix. Other common classes use
    R's names: ``integer``, ``numeric``, ``logical``, ``character``,
    ``Date``, and ``POSIXct, POSIXt``. A Python ``int`` is ``integer``;
    pass a float for ``numeric``.
    """
    _bool("describe_class() strict_description", strict_description)
    from tidy3.frame import TidyFrame

    if isinstance(x, TidyFrame):
        raise TypeError(
            "describe_class() describes a column, not a frame; use compare_df_cols()"
        )
    if isinstance(x, pl.Series):
        frame = TidyFrame(pl.DataFrame({x.name or "value": x}).lazy())
        return _column_class(frame, frame.columns[0], strict_description)
    try:
        import pandas as pd

        if isinstance(x, pd.Series):
            frame = TidyFrame(x.to_frame(name=str(x.name or "value")))
            return _column_class(frame, frame.columns[0], strict_description)
    except ImportError:
        pass
    if isinstance(x, bool):
        return "logical"
    if isinstance(x, int):
        return "integer"
    if isinstance(x, float):
        return "numeric"
    if isinstance(x, str):
        return "character"
    if type(x) is date:
        return "Date"
    if isinstance(x, datetime):
        return "POSIXct, POSIXt"
    if x is None:
        return "logical"
    return type(x).__name__


def _as_frame(value: Any) -> Any:
    from tidy3.frame import TidyFrame, tidy

    if isinstance(value, TidyFrame):
        return value
    return tidy(value)


def _expand(name: str, value: Any) -> list[tuple[str, Any]]:
    if isinstance(value, dict):
        if not value:
            raise ValueError(f"compare_df_cols() mapping {name!r} is empty")
        items = []
        for index, (key, item) in enumerate(value.items(), start=1):
            label = str(key) if str(key) else f"{name}_{index}"
            items.append((label, _as_frame(item)))
        return items
    if isinstance(value, (list, tuple)) and not isinstance(value, (str, bytes)):
        if not value:
            raise ValueError(f"compare_df_cols() list {name!r} is empty")
        return [
            (f"{name}_{index}", _as_frame(item))
            for index, item in enumerate(value, start=1)
        ]
    return [(name, _as_frame(value))]


def _input_frames(args: tuple[Any, ...], named: dict[str, Any]) -> list[tuple[str, Any]]:
    items: list[tuple[str, Any]] = []
    for index, value in enumerate(args, start=1):
        items.extend(_expand(f"..{index}", value))
    for name, value in named.items():
        items.extend(_expand(str(name), value))
    if not items:
        raise TypeError("compare_df_cols() requires at least one data frame")
    labels = _unique_names([name for name, _frame in items])
    if any(label == "column_name" for label in labels):
        raise ValueError(
            "compare_df_cols() input names cannot be 'column_name'"
        )
    return list(zip(labels, [frame for _name, frame in items]))


def _unique_names(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    result: list[str] = []
    for name in names:
        if name not in seen:
            seen[name] = 1
            result.append(name)
            continue
        seen[name] += 1
        candidate = f"{name}_{seen[name]}"
        while candidate in seen:
            seen[name] += 1
            candidate = f"{name}_{seen[name]}"
        seen[candidate] = 1
        result.append(candidate)
    return result


def _bind_match(classes: list[str | None], method: str) -> bool:
    if method == "rbind":
        first = classes[0]
        return all(value == first for value in classes[1:])
    present = [value for value in classes if value is not None]
    anchor = present[0] if present else None
    return all(value is None or value == anchor for value in classes[1:])


def _class_frame(labels: list[str], rows: list[tuple[str, list[str | None]]]) -> Any:
    from tidy3.frame import tidy

    columns: dict[str, list[Any]] = {"column_name": [name for name, _classes in rows]}
    for index, label in enumerate(labels):
        columns[label] = [classes[index] for _name, classes in rows]
    frame = pl.DataFrame(
        {name: pl.Series(name, values, dtype=pl.String) for name, values in columns.items()}
    )
    return tidy(frame)


def compare_df_cols(
    *frames: Any,
    return_: str = "all",
    bind_method: str = "bind_rows",
    strict_description: bool = False,
    **named: Any,
) -> Any:
    """Compare column names and classes before binding rows.

    Pass frames by keyword (``compare_df_cols(old=old, new=new)``) or in a
    mapping. Positional frames are named ``..1``, ``..2``, and so on. A list
    uses ``name_1``, ``name_2``. The result has ``column_name`` plus one
    column of class descriptions per frame. A column missing from a frame
    is missing in that description.

    ``return_`` is ``"all"``, ``"match"``, or ``"mismatch"``. With
    ``bind_method="bind_rows"`` (the default), a column missing from some
    frames still matches when every present class agrees. ``"rbind"``
    requires every frame to have the column and the same class. More than
    one frame sorts rows by column name, as R's ``merge`` does; one frame
    keeps its column order. ``strict_description=True`` treats different
    factor levels as different classes.

    Descriptions come from dtypes and recorded factor levels, not from a
    row scan, except that a pandas object column may be read far enough to
    tell a date from text. A frame with no columns is omitted. No input may
    be named ``column_name``.
    """
    if return_ not in _RETURNS:
        raise ValueError('compare_df_cols() return_ must be "all", "match", or "mismatch"')
    if bind_method not in _BIND:
        raise ValueError('compare_df_cols() bind_method must be "bind_rows" or "rbind"')
    _bool("compare_df_cols() strict_description", strict_description)
    inputs = _input_frames(frames, named)
    present: list[tuple[str, Any]] = []
    for label, frame in inputs:
        if frame.width == 0:
            warnings.warn(
                f"{label} has zero columns and will not appear in output.",
                stacklevel=2,
            )
            continue
        present.append((label, frame))
    if not present:
        return _class_frame([], [])
    labels = [label for label, _frame in present]
    order: list[str] = []
    seen: set[str] = set()
    described: list[dict[str, str]] = []
    for _label, frame in present:
        classes = {
            name: _column_class(frame, name, strict_description) for name in frame.columns
        }
        described.append(classes)
        for name in frame.columns:
            if name not in seen:
                seen.add(name)
                order.append(name)
    if len(present) > 1:
        order = sorted(order)
    rows = [
        (
            name,
            [classes.get(name) for classes in described],
        )
        for name in order
    ]
    if return_ != "all" and len(present) == 1:
        warnings.warn(
            "Only one data frame was provided, so all its classes are returned.",
            stacklevel=2,
        )
    elif return_ != "all":
        want = return_ == "match"
        rows = [row for row in rows if _bind_match(row[1], bind_method) is want]
    return _class_frame(labels, rows)


def compare_df_cols_same(
    *frames: Any,
    bind_method: str = "bind_rows",
    verbose: bool = True,
    **named: Any,
) -> bool:
    """Whether the frames can be row-bound under ``bind_method``.

    This is true when :func:`compare_df_cols` reports no mismatches. One
    frame is not the same as another frame, so a single frame with columns
    returns ``False``, as janitor does. ``verbose`` is accepted and ignored:
    the mismatch table is not printed. Use ``compare_df_cols(return_="mismatch")``.
    """
    _bool("compare_df_cols_same() verbose", verbose)
    mismatches = compare_df_cols(
        *frames,
        return_="mismatch",
        bind_method=bind_method,
        **named,
    )
    return mismatches.collect().height == 0


def _column_position(columns: list[str], column: Any) -> int:
    if isinstance(column, str):
        if column not in columns:
            raise KeyError(f"find_header() column not found: {column!r}")
        return columns.index(column)
    if isinstance(column, bool) or not isinstance(column, int) or column < 1:
        raise TypeError("find_header() column must be a name or a positive integer")
    if column > len(columns):
        raise IndexError(f"find_header() column position out of range: {column}")
    return column - 1


def _materialized(tf: Any) -> Any:
    from tidy3.frame import note_data_read

    note_data_read("row_to_names() takes column names from a row of the data")
    return tf._lf.collect()


def _cell(data: Any, row: int, column: int) -> Any:
    if isinstance(data, pl.DataFrame):
        return data.row(row)[column]
    return data.iat[row, column]


def _height(data: Any) -> int:
    if isinstance(data, pl.DataFrame):
        return data.height
    return len(data)


def _header_text(value: Any) -> str:
    if _missing(value):
        return "NA"
    # numpy.bool is not a Python bool, and str() of it is "True".
    if isinstance(value, bool) or type(value).__name__ in {"bool", "bool_"}:
        return "TRUE" if bool(value) else "FALSE"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, datetime):
        return value.isoformat(sep=" ")
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def find_header(
    data: Any = None,
    *,
    value: Any = None,
    column: Any = 1,
) -> Any:
    """Return the 1-based header row (R **janitor::find_header**).

    With no ``value``, this is the first row that has no missing values.
    With ``value``, this is the first row where ``column`` (a name or a
    1-based position, defaulting to the first column) equals that value.
    Several matches warn and use the first row. The search reads the
    current rows. ``find_header()`` without ``data`` is the pipe form.
    """
    if data is None:
        def _apply(tf: Any) -> int:
            return _find_header(tf, value=value, column=column)

        return Verb(_apply, "find_header")
    from tidy3.frame import TidyFrame, tidy

    frame = data if isinstance(data, TidyFrame) else tidy(data)
    return _find_header(frame, value=value, column=column)


def _find_header(tf: Any, *, value: Any, column: Any, data: Any = None) -> int:
    if data is None:
        data = _materialized(tf)
    columns = [str(name) for name in (data.columns if isinstance(data, pl.DataFrame) else data.columns)]
    height = _height(data)
    if value is None:
        if column != 1:
            raise TypeError(
                "find_header() column is only used together with value"
            )
        for row in range(height):
            if all(not _missing(_cell(data, row, index)) for index in range(len(columns))):
                return row + 1
        raise ValueError("No complete rows (rows with zero missing values) were found.")
    if not columns:
        raise ValueError(f"The value {value!r} was not found in column {column}")
    index = _column_position(columns, column)
    matches = [
        row + 1
        for row in range(height)
        if not _missing(_cell(data, row, index)) and _cell(data, row, index) == value
    ]
    if not matches:
        raise ValueError(f"The value {value!r} was not found in column {column}")
    if len(matches) > 1:
        warnings.warn(
            f"The value {value!r} was found {len(matches)} times in column {column}; "
            "using the first row.",
            stacklevel=2,
        )
    return matches[0]


def row_to_names(
    row_number: Any,
    *,
    remove_row: bool = True,
    remove_rows_above: bool = True,
    value: Any = None,
    column: Any = None,
) -> Verb:
    """Use one row as the column names (R **janitor::row_to_names**).

    ``row_number`` is 1-based, or ``"find_header"`` to call
    :func:`find_header`. ``value`` and ``column`` are only allowed with
    ``"find_header"``. The name row and the rows above it are removed by
    default. A missing name becomes ``"NA"``. Duplicate names are made
    unique (``id``, ``id_2``) because a frame cannot repeat column names;
    janitor 2.2.1 only warns. Reading the name row runs the plan up to
    this verb. Grouping and factor levels follow the new names.
    """
    _bool("row_to_names() remove_row", remove_row)
    _bool("row_to_names() remove_rows_above", remove_rows_above)
    if row_number != "find_header" and (value is not None or column is not None):
        raise TypeError(
            "row_to_names() value and column are only allowed when "
            "row_number='find_header'"
        )
    if row_number != "find_header" and (
        isinstance(row_number, bool)
        or not isinstance(row_number, int)
    ):
        raise TypeError("row_to_names() row_number must be a positive integer or 'find_header'")

    def _apply(tf: Any) -> Any:
        data = _materialized(tf)
        columns = [str(name) for name in data.columns]
        height = _height(data)
        if row_number == "find_header":
            chosen = _find_header(
                tf,
                value=value,
                column=1 if column is None else column,
                data=data,
            )
        else:
            chosen = row_number
        if isinstance(chosen, bool) or not isinstance(chosen, int) or chosen < 1 or chosen > height:
            raise ValueError(
                f"row_to_names() row_number {chosen} is outside 1:{height}"
            )
        raw = [_header_text(_cell(data, chosen - 1, index)) for index in range(len(columns))]
        names = _unique_names(raw)
        if names != raw:
            warnings.warn(
                f"Row {chosen} does not provide unique names. "
                "They were made unique; janitor only warns.",
                stacklevel=2,
            )
        removed: set[int] = set()
        if remove_row:
            removed.add(chosen)
        if remove_rows_above:
            removed.update(range(1, chosen))
        keep = [index for index in range(height) if (index + 1) not in removed]
        mapping = dict(zip(columns, names))
        groups = _renamed(mapping, tf._groups)
        levels = {
            mapping[name]: levels
            for name, levels in tf._category_levels.items()
            if name in mapping
        }
        sliced = data[keep]
        sliced.columns = names
        return tf._with_lf(sliced.lazy(), groups, category_levels=levels)

    return Verb(_apply, "row_to_names")
