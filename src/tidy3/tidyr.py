"""Tidy reshaping and missing-data verbs for both tidy3 backends."""

from __future__ import annotations

import inspect
import re
import warnings
from dataclasses import dataclass
from typing import Any, Iterable

import polars as pl

from tidy3.tidyselect import resolve_selection
from tidy3.verbs import Verb, _operation_groups


@dataclass(frozen=True)
class Nesting:
    columns: tuple[Any, ...]


def nesting(*cols: Any) -> Nesting:
    """Keep only combinations of columns that already occur together."""
    if not cols:
        raise TypeError("nesting() requires at least one column")
    return Nesting(tuple(cols))


@dataclass(frozen=True)
class FullSeq:
    """``full_seq("year", 1)``: filled in against a frame's column."""

    column: str
    period: float
    tol: float


def full_seq(x: Any, period: float, tol: float = 1e-6) -> Any:
    """Every value from ``min(x)`` to ``max(x)`` in steps of ``period``.

    Give a column name inside ``expand()``/``complete()``
    (``complete("country", year=full_seq("year", 1))``), or values for a
    plain list (``full_seq([1, 2, 5], 1)`` is ``[1, 2, 3, 4, 5]``). Raises
    if ``x`` has missing values or is not on a regular grid, like tidyr.
    """
    if not period or period <= 0:
        raise ValueError("full_seq() period must be a positive number")
    if isinstance(x, str):
        return FullSeq(x, float(period), float(tol))
    return _full_seq_values(list(x), period, tol)


def _full_seq_values(values: list[Any], period: float, tol: float) -> list[Any]:
    import math

    if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in values):
        raise ValueError("full_seq() x must not contain missing values")
    if not values:
        return []
    numbers = [float(v) for v in values]
    low, high = min(numbers), max(numbers)
    for value in numbers:
        offset = (value - low) % period
        if offset > tol and period - offset > tol:
            raise ValueError("full_seq() x is not a regular sequence")
    count = int(round((high - low) / period))
    # Decimal numbers, like R's full_seq(); expand()/complete() then cast to
    # the column's own type, so a whole-number year column stays whole.
    return [low + index * period for index in range(count + 1)]


def _value_list(tf: Any, name: str, spec: Any) -> list[Any]:
    """Unique sorted values for a named ``expand()``/``complete()`` input."""
    if isinstance(spec, FullSeq):
        column = tf.collect(as_="pandas")[spec.column]
        return _full_seq_values(column.dropna().tolist(), spec.period, spec.tol)
    values = list(spec.tolist() if hasattr(spec, "tolist") else spec)
    return _unique_sorted(values)


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and value != value)


def _unique_sorted(values: list[Any]) -> list[Any]:
    """tidyr's crossing order: distinct values sorted, missing last."""
    present = sorted({v for v in values if not _is_missing(v)})
    return [*present, None] if any(_is_missing(v) for v in values) else present


def _named_values_frame(tf: Any, name: str, values: list[Any], backend: str) -> Any:
    """One-column frame of *values*, cast to the column's type when present."""
    if backend == "pandas":
        import pandas as pd

        frame = pd.DataFrame({name: values})
        if name in tf._pdf.columns:
            try:
                frame[name] = frame[name].astype(tf._pdf[name].dtype)
            except (TypeError, ValueError):
                pass
        return frame
    frame = pl.DataFrame({name: values}, strict=False).lazy()
    schema = tf._lf.collect_schema()
    if name in schema:
        frame = frame.with_columns(pl.col(name).cast(schema[name], strict=False))
    return frame


def _columns(tf: Any) -> list[str]:
    if tf._backend == "pandas":
        return [str(name) for name in tf._pdf.columns]
    return tf._lf.collect_schema().names()


def _temp_name(columns: Iterable[str], base: str) -> str:
    occupied = set(columns)
    name = base
    index = 1
    while name in occupied:
        name = f"{base}_{index}"
        index += 1
    return name


def _result_groups(tf: Any, columns: Iterable[str]) -> list[str] | None:
    available = set(columns)
    groups = [name for name in (tf._groups or []) if name in available]
    return groups or None


def _pandas_nested_frames(
    pdf: Any,
    identifiers: list[str],
    nested: list[str],
    *,
    column: str,
) -> Any:
    """Build one nested DataFrame per group without ``to_dict(records)``.

    Matches ``group_nest``'s pandas representation: nested cells are
    DataFrames (not lists of dicts), which is far cheaper on wide frames
    and still unnests via ``unnest()``.
    """
    import pandas as pd

    if not identifiers:
        return pd.DataFrame({column: [pdf.loc[:, nested].reset_index(drop=True)]})

    body = pdf.loc[:, nested]
    grouping_key = identifiers[0] if len(identifiers) == 1 else identifiers
    rows: list[dict[str, Any]] = []
    for key, positions in pdf.groupby(
        grouping_key, sort=False, dropna=False, observed=True
    ).groups.items():
        key_tuple = key if isinstance(key, tuple) else (key,)
        row = dict(zip(identifiers, key_tuple))
        row[column] = body.loc[positions].reset_index(drop=True)
        rows.append(row)
    return pd.DataFrame(rows, columns=[*identifiers, column])


def _is_float_dtype(dtype: Any) -> bool:
    try:
        return dtype.base_type() in {pl.Float32, pl.Float64}
    except AttributeError:
        return False


def drop_na(*cols: Any) -> Verb:
    """Drop rows missing a value in any selected column."""

    def _apply(tf):
        selected = resolve_selection(tf, cols) if cols else _columns(tf)
        if not selected:
            return tf
        if tf._backend == "pandas":
            return tf._with_pdf(
                tf._pdf.dropna(subset=selected), groups=tf._groups
            )
        schema = tf._lf.collect_schema()
        predicates = []
        for name in selected:
            valid = pl.col(name).is_not_null()
            if _is_float_dtype(schema[name]):
                valid = valid & pl.col(name).is_not_nan()
            predicates.append(valid)
        return tf._with_lf(
            tf._lf.filter(pl.all_horizontal(predicates)), groups=tf._groups
        )

    return Verb(_apply, "drop_na")


def replace_na(replace: dict[str, Any]) -> Verb:
    """Replace missing values using a ``{column: value}`` mapping."""
    if not isinstance(replace, dict) or not replace:
        raise TypeError("replace_na() requires a non-empty replacement mapping")

    def _apply(tf):
        columns = _columns(tf)
        missing = [name for name in replace if name not in columns]
        if missing:
            raise KeyError(f"replace_na() columns not found: {missing}")
        if tf._backend == "pandas":
            return tf._with_pdf(
                tf._pdf.fillna(value=replace), groups=tf._groups
            )
        schema = tf._lf.collect_schema()
        expressions = []
        for name, value in replace.items():
            expression = pl.col(name)
            if _is_float_dtype(schema[name]):
                expression = expression.fill_nan(value)
            expressions.append(expression.fill_null(value).alias(name))
        return tf._with_lf(
            tf._lf.with_columns(expressions), groups=tf._groups
        )

    return Verb(_apply, "replace_na")


def fill(
    *cols: Any,
    direction: str = "down",
    by: Any = None,
) -> Verb:
    """Fill missing values down/up, respecting persistent or transient groups."""
    if not cols:
        raise TypeError("fill() requires at least one column selection")
    if direction not in {"down", "up", "downup", "updown"}:
        raise ValueError("direction must be down, up, downup, or updown")

    def _apply(tf):
        selected = resolve_selection(tf, cols)
        groups, transient = _operation_groups(tf, by, "fill")
        if tf._backend == "pandas":
            pdf = tf._pdf.copy(deep=False)

            def apply_direction(frame, method):
                if groups:
                    return frame.groupby(
                        list(groups), sort=False, dropna=False
                    )[selected].transform(method)
                return getattr(frame[selected], method)()

            # Replace whole columns: writing through .loc into a shallow
            # copy changes the caller's frame on pandas 2 (no copy-on-write).
            if direction in {"down", "downup"}:
                pdf[selected] = apply_direction(pdf, "ffill")
            if direction in {"up", "updown"}:
                pdf[selected] = apply_direction(pdf, "bfill")
            if direction == "downup":
                pdf[selected] = apply_direction(pdf, "bfill")
            elif direction == "updown":
                pdf[selected] = apply_direction(pdf, "ffill")
            return tf._with_pdf(
                pdf, groups=None if transient else tf._groups
            )

        schema = tf._lf.collect_schema()
        strategies = {
            "down": ["forward"],
            "up": ["backward"],
            "downup": ["forward", "backward"],
            "updown": ["backward", "forward"],
        }[direction]
        expressions = []
        for name in selected:
            expression = pl.col(name)
            if _is_float_dtype(schema[name]):
                expression = expression.fill_nan(None)
            for strategy in strategies:
                expression = expression.fill_null(strategy=strategy)
            if groups:
                expression = expression.over(groups)
            expressions.append(expression.alias(name))
        return tf._with_lf(
            tf._lf.with_columns(expressions),
            groups=None if transient else tf._groups,
        )

    return Verb(_apply, "fill")


def _expansion_units(tf: Any, specs: tuple[Any, ...]) -> list[list[str]]:
    units: list[list[str]] = []
    for spec in specs:
        if isinstance(spec, Nesting):
            selected = resolve_selection(tf, spec.columns)
            if selected:
                units.append(selected)
        else:
            units.extend([[name] for name in resolve_selection(tf, [spec])])
    flattened = [name for unit in units for name in unit]
    if len(flattened) != len(set(flattened)):
        raise ValueError("expand() columns must be selected only once")
    return units


def expand(*cols: Any, **values: Any) -> Verb:
    """Generate observed-value combinations, within groups when grouped.

    Named inputs give the values to use instead of the observed ones, as in
    tidyr: ``expand("country", year=full_seq("year", 1))``.
    """
    if not cols and not values:
        raise TypeError("expand() requires at least one column")
    named_specs = dict(values)

    def _apply(tf):
        units = _expansion_units(tf, cols)
        groups = list(tf._groups or [])
        named = {name: _value_list(tf, name, spec) for name, spec in named_specs.items()}
        selected = [name for unit in units for name in unit] + list(named)
        overlap = [name for name in selected if name in groups]
        if overlap:
            raise ValueError(
                f"expand() cannot expand grouping columns: {overlap}"
            )
        if tf._backend == "pandas":
            pdf = tf._pdf
            if groups:
                result = pdf.loc[:, groups].drop_duplicates()
                for unit in units:
                    values = pdf.loc[:, [*groups, *unit]].drop_duplicates()
                    result = result.merge(
                        values, on=groups, how="inner", sort=False
                    )
            else:
                result = None
                for unit in units:
                    values = pdf.loc[:, unit].drop_duplicates()
                    if result is None:
                        result = values
                    else:
                        result = result.merge(values, how="cross")
            for name, listed in named.items():
                frame = _named_values_frame(tf, name, listed, "pandas")
                result = frame if result is None else result.merge(frame, how="cross")
            if result is None:
                result = pdf.iloc[:0, :0]
            ordered = [*groups, *selected]
            result = result.loc[:, ordered].sort_values(
                ordered, kind="stable", na_position="last"
            ).reset_index(drop=True)
            return tf._with_pdf(result, groups=groups or None, rowwise=False)

        if groups:
            result = tf._lf.select(groups).unique(maintain_order=True)
            for unit in units:
                values = tf._lf.select([*groups, *unit]).unique(
                    maintain_order=True
                )
                result = result.join(
                    values, on=groups, how="inner", maintain_order="left_right"
                )
        else:
            result = None
            for unit in units:
                values = tf._lf.select(unit).unique(maintain_order=True)
                result = (
                    values
                    if result is None
                    else result.join(values, how="cross")
                )
        for name, listed in named.items():
            frame = _named_values_frame(tf, name, listed, "polars")
            result = frame if result is None else result.join(frame, how="cross")
        if result is None:
            result = tf._lf.select([]).head(0)
        ordered = [*groups, *selected]
        return tf._with_lf(
            result.select(ordered).sort(ordered),
            groups=groups or None,
            rowwise=False,
        )

    return Verb(_apply, "expand")


def _grid_units(args: tuple[Any, ...], kwargs: dict[str, Any], *, crossing: bool) -> tuple[list[Any], str]:
    """pandas frames for each expand_grid()/crossing() input, and the backend."""
    import pandas as pd

    from tidy3.frame import TidyFrame

    backend = "polars"
    units = []
    for arg in args:
        if isinstance(arg, TidyFrame):
            backend = arg._backend if not units else backend
            frame = arg.collect(as_="pandas")
        elif isinstance(arg, pl.DataFrame):
            frame = arg.to_pandas()
        else:
            frame = pd.DataFrame(arg)
        if crossing:
            frame = frame.drop_duplicates().sort_values(
                list(frame.columns), kind="stable", na_position="last"
            )
        units.append(frame.reset_index(drop=True))
    for name, values in kwargs.items():
        listed = list(values.tolist() if hasattr(values, "tolist") else values)
        if crossing:
            listed = _unique_sorted(listed)
        units.append(pd.DataFrame({name: listed}))
    return units, backend


def _cross(units: list[Any], backend: str) -> Any:
    import pandas as pd

    from tidy3.frame import tidy

    if not units:
        return tidy(pd.DataFrame(), backend=backend)
    result = units[0]
    for unit in units[1:]:
        result = result.merge(unit, how="cross")
    return tidy(result.reset_index(drop=True), backend=backend)


def expand_grid(*frames: Any, **values: Any) -> Any:
    """Every combination of the inputs, as given (tidyr ``expand_grid()``).

    Tables keep their rows together; named inputs are value lists. Nothing
    is de-duplicated or sorted, and the first input varies slowest::

        expand_grid(x=[1, 2], y=["a", "b"])
        expand_grid(stores, week=range(1, 53))
    """
    units, backend = _grid_units(frames, values, crossing=False)
    return _cross(units, backend)


def crossing(*frames: Any, **values: Any) -> Any:
    """Like :func:`expand_grid`, on each input's distinct values, sorted.

    Missing values sort last, as in tidyr ``crossing()``.
    """
    units, backend = _grid_units(frames, values, crossing=True)
    return _cross(units, backend)


def uncount(weights: Any, *, remove: bool = True, id: str | None = None) -> Verb:
    """Repeat each row ``weights`` times (tidyr ``uncount()``).

    ``weights`` is a column name or a number. The weights column is dropped
    unless ``remove=False``; ``id="copy"`` adds each copy's 1-based number.
    """

    def _apply(tf):
        columns = _columns(tf)
        name = weights if isinstance(weights, str) else None
        if name is not None and name not in columns:
            raise KeyError(f"uncount() weights column not found: {name!r}")
        if tf._backend == "pandas":
            pdf = tf._pdf
            counts = pdf[name] if name else _constant_series(len(pdf), weights)
            _check_weights(counts.isna().any(), (counts < 0).any(), (counts % 1 != 0).any())
            counts = counts.astype("int64")
            out = pdf.loc[pdf.index.repeat(counts)]
            if id is not None:
                out = out.assign(**{id: out.groupby(level=0).cumcount() + 1})
            out = out.reset_index(drop=True)
            if remove and name is not None:
                out = out.drop(columns=name)
            ordered = list(out.columns)
            return tf._with_pdf(out, groups=_result_groups(tf, ordered))
        count = pl.col(name) if name else pl.lit(weights)
        missing, negative, fractional = (
            tf._lf.select(
                count.is_null().any().alias("missing"),
                (count < 0).any().alias("negative"),
                (count.cast(pl.Float64) % 1 != 0).any().alias("fractional"),
            )
            .collect()
            .row(0)
        )
        _check_weights(missing, negative, fractional)
        copy = _temp_name(columns, "__tidy3_copy")
        # Drop zero weights first: explode() never sees an empty list, whose
        # handling differs between Polars 1 and 2.
        lf = (
            tf._lf.filter(count > 0)
            .with_columns(pl.int_ranges(0, count.cast(pl.Int64)).alias(copy))
            .explode(copy, **_EXPLODE_OPTIONS)
        )
        if id is not None:
            lf = lf.with_columns((pl.col(copy) + 1).alias(id))
        drop = [copy] + ([name] if remove and name is not None else [])
        lf = lf.drop(drop)
        ordered = lf.collect_schema().names()
        return tf._with_lf(lf, groups=_result_groups(tf, ordered))

    return Verb(_apply, "uncount")


# Polars 1.4x warns unless empty_as_null is explicit; older 1.x lacks it.
_EXPLODE_OPTIONS = (
    {"empty_as_null": True}
    if "empty_as_null" in inspect.signature(pl.LazyFrame.explode).parameters
    else {}
)


def _constant_series(length: int, value: Any) -> Any:
    import pandas as pd

    return pd.Series([value] * length)


def _check_weights(missing: bool, negative: bool, fractional: bool) -> None:
    if missing:
        raise ValueError("uncount() weights must not be missing")
    if negative:
        raise ValueError("uncount() weights must be zero or more")
    if fractional:
        raise ValueError("uncount() weights must be whole numbers")


def complete(
    *cols: Any,
    fill: dict[str, Any] | None = None,
    explicit: bool = True,
    **values: Any,
) -> Verb:
    """Make implicit missing combinations explicit, optionally filling them.

    Named inputs give the values to complete over, as in tidyr:
    ``complete("country", year=full_seq("year", 1))``.
    """
    if not cols and not values:
        raise TypeError("complete() requires at least one column")
    if fill is not None and not isinstance(fill, dict):
        raise TypeError("complete() fill must be a mapping or None")

    def _apply(tf):
        expanded = expand(*cols, **values)._fn(tf)
        groups = list(tf._groups or [])
        units = _expansion_units(tf, cols)
        keys = [*groups, *(name for unit in units for name in unit), *values]
        marker = _temp_name(_columns(tf), "__tidy3_complete")
        replacements = fill or {}
        if tf._backend == "pandas":
            original = tf._pdf.assign(**{marker: True})
            output = expanded._pdf.merge(
                original, on=keys, how="left", sort=False
            )
            for name, value in replacements.items():
                if name not in output.columns:
                    raise KeyError(f"complete() fill column not found: {name!r}")
                mask = output[name].isna()
                if not explicit:
                    mask &= output[marker].isna()
                output.loc[mask, name] = value
            output = output.drop(columns=marker)
            return tf._with_pdf(
                output, groups=groups or None, rowwise=False
            )

        original = tf._lf.with_columns(pl.lit(True).alias(marker))
        output = expanded._lf.join(
            original,
            on=keys,
            how="left",
            maintain_order="left_right",
        )
        schema = output.collect_schema()
        expressions = []
        for name, value in replacements.items():
            if name not in schema:
                raise KeyError(f"complete() fill column not found: {name!r}")
            missing = pl.col(name).is_null()
            if _is_float_dtype(schema[name]):
                missing = missing | pl.col(name).is_nan()
            if not explicit:
                missing = missing & pl.col(marker).is_null()
            expressions.append(
                pl.when(missing)
                .then(pl.lit(value))
                .otherwise(pl.col(name))
                .alias(name)
            )
        if expressions:
            output = output.with_columns(expressions)
        return tf._with_lf(
            output.drop(marker), groups=groups or None, rowwise=False
        )

    return Verb(_apply, "complete")


def pivot_longer(
    cols: Any,
    *,
    names_to: str | list[str] | tuple[str, ...] | None = "name",
    values_to: str = "value",
    names_prefix: str | None = None,
    names_sep: str | None = None,
    names_pattern: str | None = None,
    values_drop_na: bool = False,
    cols_vary: str = "fastest",
) -> Verb:
    """Lengthen selected columns, with optional name splitting."""
    if cols_vary not in {"fastest", "slowest"}:
        raise ValueError("cols_vary must be 'fastest' or 'slowest'")
    if not isinstance(values_to, str) or not values_to:
        raise TypeError("values_to must be a non-empty string")
    if names_sep is not None and names_pattern is not None:
        raise ValueError("supply only one of names_sep or names_pattern")
    name_columns = (
        []
        if names_to is None
        else [names_to]
        if isinstance(names_to, str)
        else list(names_to)
    )
    value_sentinel = ".value" in name_columns
    if len(name_columns) > 1 and names_sep is None and names_pattern is None:
        raise ValueError(
            "multiple names_to columns require names_sep or names_pattern"
        )

    def _apply(tf):
        pivoted = resolve_selection(tf, [cols])
        if not pivoted:
            raise ValueError("pivot_longer() selected no columns")
        all_columns = _columns(tf)
        identifiers = [name for name in all_columns if name not in pivoted]
        output_name_columns = [
            name for name in name_columns if name != ".value"
        ]
        collisions = set(identifiers) & (
            ({values_to} if not value_sentinel else set())
            | set(output_name_columns)
        )
        if collisions:
            raise ValueError(
                f"pivot_longer() output names collide with existing columns: {sorted(collisions)}"
            )
        variable = _temp_name(all_columns, "__tidy3_name")
        row = _temp_name([*all_columns, variable], "__tidy3_row")

        if tf._backend == "pandas":
            import pandas as pd

            pdf = tf._pdf.copy(deep=False)
            pdf[row] = range(len(pdf))
            out = pdf.melt(
                id_vars=[*identifiers, row],
                value_vars=pivoted,
                var_name=variable,
                value_name=values_to,
            )
            if cols_vary == "fastest":
                out = out.sort_values(row, kind="stable")
            names = out[variable].astype("string")
            if names_prefix:
                names = names.str.replace(
                    rf"^(?:{names_prefix})", "", regex=True
                )
            if len(name_columns) == 1:
                out[name_columns[0]] = names
            elif len(name_columns) > 1:
                if names_pattern:
                    pieces = names.str.extract(names_pattern)
                else:
                    pieces = names.str.split(
                        names_sep, n=len(name_columns) - 1, expand=True
                    )
                if pieces.shape[1] != len(name_columns):
                    raise ValueError(
                        "name split did not produce the requested number of columns"
                    )
                for index, name in enumerate(name_columns):
                    out[name] = pieces.iloc[:, index]
            out = out.drop(columns=variable)
            if value_sentinel:
                value_names = out[".value"].drop_duplicates().tolist()
                duplicate_outputs = set(identifiers) & set(value_names)
                if duplicate_outputs:
                    raise ValueError(
                        "pivot_longer() .value names collide with identifiers: "
                        f"{sorted(duplicate_outputs)}"
                    )
                other_names = [
                    name for name in name_columns if name != ".value"
                ]
                index = [*identifiers, row, *other_names]
                out = out.pivot(
                    index=index,
                    columns=".value",
                    values=values_to,
                ).reset_index()
                out.columns.name = None
                if values_drop_na:
                    out = out.dropna(subset=value_names, how="all")
                ordered = [*identifiers, *other_names, *value_names]
            else:
                if values_drop_na:
                    out = out.dropna(subset=[values_to])
                ordered = [*identifiers, *name_columns, values_to]
            out = out.drop(columns=row).reset_index(drop=True)
            out = out.loc[:, ordered]
            return tf._with_pdf(
                out,
                groups=_result_groups(tf, ordered),
                rowwise=False,
            )

        lf = tf._lf.with_row_index(row).unpivot(
            on=pivoted,
            index=[*identifiers, row],
            variable_name=variable,
            value_name=values_to,
        )
        # Sort explicitly: unpivot()'s row order changed in Polars 2.
        if cols_vary == "fastest":
            lf = lf.sort(row, maintain_order=True)
        else:
            position = pl.col(variable).replace_strict(
                {name: index for index, name in enumerate(pivoted)},
                return_dtype=pl.UInt32,
            )
            lf = lf.sort(position, row, maintain_order=True)
        name_expression = pl.col(variable)
        if names_prefix:
            name_expression = name_expression.str.replace(
                rf"^(?:{names_prefix})", ""
            )
        if len(name_columns) == 1:
            lf = lf.with_columns(name_expression.alias(name_columns[0]))
        elif len(name_columns) > 1:
            if names_pattern:
                pieces = [
                    name_expression.str.extract(names_pattern, index + 1).alias(name)
                    for index, name in enumerate(name_columns)
                ]
            else:
                split = name_expression.str.split_exact(
                    names_sep, len(name_columns) - 1
                )
                pieces = [
                    split.struct.field(f"field_{index}").alias(name)
                    for index, name in enumerate(name_columns)
                ]
            lf = lf.with_columns(pieces)
        lf = lf.drop(variable)
        if value_sentinel:
            value_names = (
                lf.select(".value")
                .unique(maintain_order=True)
                .collect()
                .get_column(".value")
                .to_list()
            )
            duplicate_outputs = set(identifiers) & set(value_names)
            if duplicate_outputs:
                raise ValueError(
                    "pivot_longer() .value names collide with identifiers: "
                    f"{sorted(duplicate_outputs)}"
                )
            other_names = [
                name for name in name_columns if name != ".value"
            ]
            index = [*identifiers, row, *other_names]
            lf = lf.pivot(
                on=".value",
                on_columns=value_names,
                index=index,
                values=values_to,
                aggregate_function="first",
                maintain_order=True,
            )
            if values_drop_na:
                lf = lf.filter(
                    pl.any_horizontal(
                        pl.col(name).is_not_null() for name in value_names
                    )
                )
            ordered = [*identifiers, *other_names, *value_names]
        else:
            if values_drop_na:
                lf = lf.filter(pl.col(values_to).is_not_null())
                schema = lf.collect_schema()
                if _is_float_dtype(schema[values_to]):
                    lf = lf.filter(pl.col(values_to).is_not_nan())
            ordered = [*identifiers, *name_columns, values_to]
        lf = lf.drop(row).select(ordered)
        return tf._with_lf(
            lf,
            groups=_result_groups(tf, ordered),
            rowwise=False,
        )

    return Verb(_apply, "pivot_longer")


def _warn_not_unique(values: list[str], keys: list[str], temp_id: str | None) -> None:
    """tidyr's warning for values that are not uniquely identified."""
    shown = ", ".join(f"`{name}`" for name in values)
    by = [key for key in keys if key != temp_id]
    warnings.warn(
        f"Values from {shown} are not uniquely identified; output will contain "
        "list-cols.\n"
        '* Use values_fn="list" to suppress this warning.\n'
        '* Use values_fn="mean" (or "sum", "first", ...) to summarise duplicates.\n'
        "* Use the following tidy3 code to identify duplicates:\n"
        f"  data >> summarise(n=n(), by={by!r}) >> filter(col(\"n\") > 1)",
        UserWarning,
        stacklevel=4,
    )


def pivot_wider(
    *,
    names_from: Any = "name",
    values_from: Any = "value",
    id_cols: Any = None,
    names_prefix: str = "",
    names_sort: bool = False,
    values_fill: Any = None,
    values_fn: str | None = None,
    names: Iterable[Any] | None = None,
) -> Verb:
    """Widen a name/value pair; discovers output names when not supplied.

    Like tidyr, values that are not uniquely identified by the id and name
    columns become list columns with a warning; ``values_fn="list"`` asks for
    list columns without the warning, and ``values_fn="mean"`` (or ``sum``,
    ``first``, …) summarises duplicates instead.
    """
    if values_fn is not None and values_fn not in {
        "first", "last", "sum", "min", "max", "mean", "median", "len", "list"
    }:
        raise ValueError(
            "values_fn must be first, last, sum, min, max, mean, median, len, "
            "list, or None"
        )

    def _apply(tf):
        name_columns = resolve_selection(tf, [names_from])
        value_columns = resolve_selection(tf, [values_from])
        if not name_columns or not value_columns:
            raise ValueError(
                "pivot_wider() must select names_from and values_from columns"
            )
        name_argument: Any = (
            name_columns[0] if len(name_columns) == 1 else name_columns
        )
        value_argument: Any = (
            value_columns[0] if len(value_columns) == 1 else value_columns
        )
        all_columns = _columns(tf)
        if id_cols is None:
            identifiers = [
                name for name in all_columns
                if name not in {*name_columns, *value_columns}
            ]
        else:
            identifiers = resolve_selection(tf, [id_cols])
        temp_id = None
        if not identifiers:
            temp_id = _temp_name(all_columns, "__tidy3_id")
            identifiers = [temp_id]

        if tf._backend == "pandas":
            pdf = tf._pdf.copy(deep=False)
            if temp_id:
                pdf[temp_id] = 0
            if names is None:
                name_values = pdf.loc[:, name_columns].drop_duplicates()
                if names_sort:
                    name_values = name_values.sort_values(
                        name_columns, kind="stable"
                    )
                combinations = list(
                    name_values.itertuples(index=False, name=None)
                )
            elif len(name_columns) == 1:
                combinations = [(value,) for value in names]
            else:
                combinations = [tuple(value) for value in names]
            as_lists = values_fn == "list" or (
                values_fn is None
                and bool(pdf.duplicated(subset=[*identifiers, *name_columns]).any())
            )
            if as_lists and values_fn is None:
                _warn_not_unique(value_columns, [*identifiers, *name_columns], temp_id)
            if as_lists:
                out = pdf.pivot_table(
                    index=identifiers,
                    columns=name_argument,
                    values=value_argument,
                    aggfunc=list,
                    sort=names_sort,
                )
                # Combinations with no rows are missing (tidyr's NULL).
                out = out.astype(object).where(out.notna(), None)
            elif values_fn is None:
                out = pdf.pivot(
                    index=identifiers,
                    columns=name_argument,
                    values=value_argument,
                )
            else:
                aggfunc = "size" if values_fn == "len" else values_fn
                out = pdf.pivot_table(
                    index=identifiers,
                    columns=name_argument,
                    values=value_argument,
                    aggfunc=aggfunc,
                    sort=names_sort,
                )
            if len(value_columns) > 1:
                desired = [
                    (value, *combination)
                    for value in value_columns
                    for combination in combinations
                ]
            elif len(name_columns) > 1:
                desired = combinations
            else:
                desired = [combination[0] for combination in combinations]
            out = out.reindex(columns=desired)
            out = out.reset_index()
            out.columns.name = None
            generated = [
                (
                    f"{value}_{'_'.join(map(str, combination))}"
                    if len(value_columns) > 1
                    else "_".join(map(str, combination))
                )
                for value in value_columns
                for combination in combinations
            ] if len(value_columns) > 1 else [
                "_".join(map(str, combination))
                for combination in combinations
            ]
            out.columns = [*identifiers, *generated]
            if temp_id:
                out = out.drop(columns=temp_id)
                identifiers = []
            value_names = generated
            out = out.rename(
                columns={name: f"{names_prefix}{name}" for name in value_names}
            )
            if values_fill is not None:
                out = out.fillna(values_fill)
            ordered = list(out.columns)
            return tf._with_pdf(
                out,
                groups=_result_groups(tf, ordered),
                rowwise=False,
            )

        lf = tf._lf
        if temp_id:
            lf = lf.with_columns(pl.lit(0).alias(temp_id))
        # tidyr turns values that are not uniquely identified into lists, so
        # check for duplicates (with the name discovery when there is one).
        duplicates = None
        if values_fn is None:
            count = _temp_name(all_columns, "__tidy3_n")
            duplicates = (
                lf.group_by([*identifiers, *name_columns])
                .agg(pl.len().alias(count))
                .select((pl.col(count).max() > 1).fill_null(False))
            )
        if names is None:
            discovery = lf.select(name_columns).unique(
                maintain_order=not names_sort
            )
            if duplicates is not None:
                discovered, duplicated = pl.collect_all([discovery, duplicates])
                duplicates = duplicated
            else:
                discovered = discovery.collect()
            if names_sort:
                discovered = discovered.sort(name_columns)
            combinations = discovered.rows()
            on_columns: Any = (
                discovered.get_column(name_columns[0])
                if len(name_columns) == 1
                else discovered
            )
        else:
            if len(name_columns) == 1:
                on_columns = list(names)
                combinations = [(value,) for value in names]
            else:
                combinations = [tuple(value) for value in names]
                on_columns = pl.DataFrame(
                    {
                        column: [row[index] for row in combinations]
                        for index, column in enumerate(name_columns)
                    }
                )
        if isinstance(duplicates, pl.LazyFrame):
            duplicates = duplicates.collect()
        as_lists = values_fn == "list" or (
            duplicates is not None and bool(duplicates.item())
        )
        if as_lists and values_fn is None:
            _warn_not_unique(value_columns, [*identifiers, *name_columns], temp_id)
        out = lf.pivot(
            on=name_argument,
            on_columns=on_columns,
            index=identifiers,
            values=value_argument,
            aggregate_function=pl.element() if as_lists else values_fn,
            maintain_order=True,
        )
        if temp_id:
            out = out.drop(temp_id)
            identifiers = []
        output_columns = out.collect_schema().names()
        actual_values = [
            name for name in output_columns if name not in identifiers
        ]
        generated = (
            [
                f"{value}_{'_'.join(map(str, combination))}"
                for value in value_columns
                for combination in combinations
            ]
            if len(value_columns) > 1
            else [
                "_".join(map(str, combination))
                for combination in combinations
            ]
        )
        rename = {
            old: f"{names_prefix}{new}"
            for old, new in zip(actual_values, generated)
        }
        if rename:
            out = out.rename(rename)
        if as_lists:
            # Combinations with no rows come back as [] — make them missing.
            out = out.with_columns(
                pl.when(pl.col(name).list.len() > 0).then(pl.col(name)).alias(name)
                for name in (
                    rename.get(column, column)
                    for column in output_columns
                    if column not in identifiers
                )
            )
        if values_fill is not None:
            value_outputs = [rename.get(name, name) for name in output_columns if name not in identifiers]
            if isinstance(values_fill, dict):
                expressions = [
                    pl.col(name).fill_null(values_fill.get(name)).alias(name)
                    for name in value_outputs
                    if name in values_fill
                ]
            else:
                expressions = [
                    pl.col(name).fill_null(values_fill).alias(name)
                    for name in value_outputs
                ]
            if expressions:
                out = out.with_columns(expressions)
        ordered = out.collect_schema().names()
        return tf._with_lf(
            out,
            groups=_result_groups(tf, ordered),
            rowwise=False,
        )

    return Verb(_apply, "pivot_wider")


def separate(
    column: str,
    into: Iterable[str | None],
    *,
    sep: str = r"[^A-Za-z0-9]+",
    remove: bool = True,
    convert: bool = False,
    extra: str = "warn",
    fill: str = "warn",
) -> Verb:
    """Split one string column into several columns using a regex."""
    names = list(into)
    if not names:
        raise TypeError("separate() requires at least one output name")
    if extra not in {"warn", "drop", "merge"}:
        raise ValueError("extra must be warn, drop, or merge")
    if fill not in {"warn", "right", "left"}:
        raise ValueError("fill must be warn, right, or left")

    def _apply(tf):
        columns = _columns(tf)
        if column not in columns:
            raise KeyError(f"column not found: {column!r}")
        kept_names = [name for name in names if name is not None]
        collisions = [
            name for name in kept_names if name in columns and name != column
        ]
        if collisions:
            raise ValueError(f"separate() output columns already exist: {collisions}")
        if tf._backend == "pandas":
            import pandas as pd

            maxsplit = len(names) - 1 if extra == "merge" else -1
            pieces = tf._pdf[column].astype("string").str.split(
                sep, n=maxsplit, expand=True, regex=True
            )
            if pieces.shape[1] > len(names):
                pieces = pieces.iloc[:, : len(names)]
            if pieces.shape[1] < len(names):
                padded = pd.DataFrame(
                    pd.NA, index=pieces.index, columns=range(len(names))
                )
                offset = len(names) - pieces.shape[1] if fill == "left" else 0
                for index in range(pieces.shape[1]):
                    padded.iloc[:, offset + index] = pieces.iloc[:, index]
                pieces = padded
            if fill == "left":
                pieces = pieces.apply(
                    lambda row: pd.Series(
                        [pd.NA] * (len(names) - row.notna().sum())
                        + row.dropna().tolist(),
                        index=pieces.columns,
                    ),
                    axis=1,
                )
            output = tf._pdf.copy(deep=False)
            position = columns.index(column)
            if remove:
                output = output.drop(columns=column)
            for index, name in enumerate(names):
                if name is None:
                    continue
                values = pieces.iloc[:, index]
                if convert:
                    values = values.replace("NA", pd.NA)
                    observed = values.dropna().astype("string").str.upper()
                    if len(observed) and observed.isin(
                        ["TRUE", "FALSE", "T", "F"]
                    ).all():
                        upper = values.astype("string").str.upper()
                        values = upper.map(
                            {
                                "TRUE": True,
                                "T": True,
                                "FALSE": False,
                                "F": False,
                            }
                        ).astype("boolean")
                    else:
                        try:
                            values = pd.to_numeric(values, errors="raise")
                        except (TypeError, ValueError):
                            pass
                output.insert(position, name, values)
                position += 1
            return tf._with_pdf(
                output,
                groups=_result_groups(tf, output.columns),
                rowwise=False,
            )

        sentinel = "\x1f"
        source = pl.col(column).cast(pl.String).str.replace_all(sep, sentinel)
        if extra == "merge":
            split = source.str.splitn(sentinel, len(names))
        else:
            split = source.str.split(sentinel)
        schema = tf._lf.collect_schema()
        expressions: dict[str, pl.Expr] = {}
        for index, name in enumerate(names):
            if name is None:
                continue
            if extra == "merge":
                value = split.struct.field(f"field_{index}").str.replace_all(
                    sentinel, sep, literal=True
                )
            else:
                if fill == "left":
                    value = (
                        pl.when(split.list.len() < len(names))
                        .then(
                            split.list.get(
                                index - len(names), null_on_oob=True
                            )
                        )
                        .otherwise(split.list.get(index, null_on_oob=True))
                    )
                else:
                    value = split.list.get(index, null_on_oob=True)
            expressions[name] = value.alias(name)
        output = []
        for name in schema.names():
            if name == column:
                if not remove:
                    output.append(pl.col(column))
                output.extend(expressions.values())
            else:
                output.append(pl.col(name))
        lf = tf._lf.select(output)
        if convert and kept_names:
            lf = lf.with_columns(
                pl.when(pl.col(name) == "NA")
                .then(None)
                .otherwise(pl.col(name))
                .alias(name)
                for name in kept_names
            )
            stats = []
            for index, name in enumerate(kept_names):
                value = pl.col(name)
                valid_int = value.is_null() | value.cast(
                    pl.Int64, strict=False
                ).is_not_null()
                valid_float = value.is_null() | value.cast(
                    pl.Float64, strict=False
                ).is_not_null()
                upper = value.str.to_uppercase()
                valid_bool = value.is_null() | upper.is_in(
                    ["TRUE", "FALSE", "T", "F"]
                )
                stats.extend(
                    [
                        value.is_not_null().any().alias(f"n_{index}"),
                        valid_int.all().alias(f"i_{index}"),
                        valid_float.all().alias(f"f_{index}"),
                        valid_bool.all().alias(f"b_{index}"),
                    ]
                )
            inferred = lf.select(stats).collect().row(0, named=True)
            conversions = []
            for index, name in enumerate(kept_names):
                if not inferred[f"n_{index}"]:
                    continue
                if inferred[f"i_{index}"]:
                    expression = pl.col(name).cast(pl.Int64)
                elif inferred[f"f_{index}"]:
                    expression = pl.col(name).cast(pl.Float64)
                elif inferred[f"b_{index}"]:
                    upper = pl.col(name).str.to_uppercase()
                    expression = (
                        pl.when(upper.is_in(["TRUE", "T"]))
                        .then(True)
                        .when(upper.is_in(["FALSE", "F"]))
                        .then(False)
                        .otherwise(None)
                    )
                else:
                    continue
                conversions.append(expression.alias(name))
            if conversions:
                lf = lf.with_columns(conversions)
        ordered = lf.collect_schema().names()
        return tf._with_lf(
            lf,
            groups=_result_groups(tf, ordered),
            rowwise=False,
        )

    return Verb(_apply, "separate")


def unite(
    column: str,
    *cols: Any,
    sep: str = "_",
    remove: bool = True,
    na_rm: bool = False,
) -> Verb:
    """Combine selected columns into one string column."""
    if not isinstance(column, str) or not column:
        raise TypeError("unite() output column must be a non-empty string")

    def _apply(tf):
        selected = resolve_selection(tf, cols) if cols else _columns(tf)
        if not selected:
            raise ValueError("unite() selected no columns")
        columns = _columns(tf)
        if column in columns and column not in selected:
            raise ValueError(f"unite() output column already exists: {column!r}")
        position = min(columns.index(name) for name in selected)
        if tf._backend == "pandas":
            import pandas as pd

            values = tf._pdf[selected].astype("string")
            if na_rm:
                combined = values.fillna("").agg(
                    lambda row: sep.join(value for value in row if value != ""),
                    axis=1,
                )
            else:
                combined = values.fillna("NA").agg(sep.join, axis=1)
            output = tf._pdf.copy(deep=False)
            if remove:
                output = output.drop(columns=selected)
            if column in output:
                output[column] = combined
            else:
                output.insert(position, column, combined)
            return tf._with_pdf(
                output,
                groups=_result_groups(tf, output.columns),
                rowwise=False,
            )

        values = [pl.col(name).cast(pl.String) for name in selected]
        if not na_rm:
            values = [value.fill_null("NA") for value in values]
        combined = pl.concat_str(
            values, separator=sep, ignore_nulls=na_rm
        ).alias(column)
        output = []
        inserted = False
        for index, name in enumerate(columns):
            if index == position:
                output.append(combined)
                inserted = True
            if name in selected and remove:
                continue
            if name == column and inserted:
                continue
            output.append(pl.col(name))
        lf = tf._lf.select(output)
        ordered = lf.collect_schema().names()
        return tf._with_lf(
            lf,
            groups=_result_groups(tf, ordered),
            rowwise=False,
        )

    return Verb(_apply, "unite")


def nest(
    column: str = "data",
    *,
    cols: Any = None,
    by: Any = None,
) -> Verb:
    """Collapse selected columns into a nested list-column.

    Polars stores list-of-struct; pandas stores a DataFrame per group
    (same representation as ``group_nest``).
    """
    if not isinstance(column, str) or not column:
        raise TypeError("nest() column must be a non-empty string")
    if cols is not None and by is not None:
        raise ValueError("nest() accepts only one of cols= or by=")

    def _apply(tf):
        columns = _columns(tf)
        if column in columns:
            raise ValueError(f"nest() output column already exists: {column!r}")
        if by is not None:
            identifiers = resolve_selection(tf, [by])
            nested = [name for name in columns if name not in identifiers]
        elif cols is not None:
            nested = resolve_selection(tf, [cols])
            identifiers = [name for name in columns if name not in nested]
        elif tf._groups:
            identifiers = list(tf._groups)
            nested = [name for name in columns if name not in identifiers]
        else:
            raise TypeError("nest() requires cols=, by=, or a grouped frame")
        if not nested:
            raise ValueError("nest() selected no columns to nest")

        if tf._backend == "pandas":
            output = _pandas_nested_frames(
                tf._pdf, identifiers, nested, column=column
            )
            groups = _result_groups(tf, identifiers)
            return tf._with_pdf(output, groups=groups, rowwise=False)

        # Prefer exclude() when nesting "everything except keys" so the plan
        # tracks the schema without rebuilding a Python name list at collect.
        if identifiers and set(nested) == set(columns) - set(identifiers):
            if len(identifiers) == 1:
                records = pl.struct(pl.exclude(identifiers[0]))
            else:
                records = pl.struct(pl.exclude(identifiers))
        else:
            records = pl.struct(nested)
        if identifiers:
            output = tf._lf.group_by(
                identifiers, maintain_order=True
            ).agg(records.alias(column))
        else:
            output = tf._lf.select(records.implode().alias(column))
        groups = _result_groups(tf, identifiers)
        return tf._with_lf(output, groups=groups, rowwise=False)

    return Verb(_apply, "nest")


def unnest_longer(
    column: str,
    *,
    values_to: str | None = None,
    indices_to: str | None = None,
    keep_empty: bool = False,
) -> Verb:
    """Expand each element of a list-column into its own row."""

    def _apply(tf):
        columns = _columns(tf)
        if column not in columns:
            raise KeyError(f"column not found: {column!r}")
        value_name = values_to or column
        if value_name != column and value_name in columns:
            raise ValueError(f"unnest_longer() output column exists: {value_name!r}")
        if indices_to and indices_to in columns:
            raise ValueError(f"unnest_longer() index column exists: {indices_to!r}")
        if tf._backend == "pandas":
            import pandas as pd

            pdf = tf._pdf.copy(deep=False)
            if not keep_empty:
                keep = pdf[column].map(
                    lambda value: isinstance(value, (list, tuple))
                    and len(value) > 0
                )
                pdf = pdf.loc[keep]
            if indices_to:
                pdf[indices_to] = pdf[column].map(
                    lambda value: list(range(1, len(value) + 1))
                    if isinstance(value, (list, tuple))
                    else [pd.NA]
                    if keep_empty
                    else []
                )
                pdf = pdf.explode([column, indices_to], ignore_index=True)
            else:
                pdf = pdf.explode(column, ignore_index=True)
            if value_name != column:
                pdf = pdf.rename(columns={column: value_name})
            if indices_to:
                ordered = list(pdf.columns)
                ordered.remove(indices_to)
                ordered.insert(ordered.index(value_name), indices_to)
                pdf = pdf.loc[:, ordered]
            return tf._with_pdf(
                pdf,
                groups=_result_groups(tf, pdf.columns),
                rowwise=False,
            )

        lf = tf._lf
        explode_columns = [column]
        if indices_to:
            lf = lf.with_columns(
                pl.int_ranges(1, pl.col(column).list.len() + 1).alias(indices_to)
            )
            explode_columns.append(indices_to)
        lf = lf.explode(
            explode_columns,
            empty_as_null=keep_empty,
            keep_nulls=keep_empty,
        )
        if value_name != column:
            lf = lf.rename({column: value_name})
        ordered = lf.collect_schema().names()
        if indices_to:
            ordered.remove(indices_to)
            ordered.insert(ordered.index(value_name), indices_to)
            lf = lf.select(ordered)
        return tf._with_lf(
            lf,
            groups=_result_groups(tf, ordered),
            rowwise=False,
        )

    return Verb(_apply, "unnest_longer")


def unnest(
    column: str,
    *,
    keep_empty: bool = False,
    names_sep: str | None = None,
) -> Verb:
    """Expand a list-column, widening nested row records when present.

    On the pandas backend, nested cells may be list/tuple of dicts (legacy /
    polars-handoff style) or DataFrames (``nest`` / ``group_nest`` /
    ``nest_join``).
    """

    def _apply(tf):
        columns = _columns(tf)
        if column not in columns:
            raise KeyError(f"column not found: {column!r}")
        if tf._backend == "pandas":
            import pandas as pd

            pdf = tf._pdf.copy(deep=False)
            sample = next(
                (
                    value
                    for value in pdf[column]
                    if value is not None and not (isinstance(value, float) and pd.isna(value))
                ),
                None,
            )

            # Nested DataFrames from nest() / group_nest() / nest_join().
            if isinstance(sample, pd.DataFrame):
                pieces: list[pd.DataFrame] = []
                parent_columns = [name for name in pdf.columns if name != column]
                for _, row in pdf.iterrows():
                    nested = row[column]
                    parent = {name: row[name] for name in parent_columns}
                    if not isinstance(nested, pd.DataFrame) or nested.empty:
                        if keep_empty:
                            pieces.append(pd.DataFrame([parent]))
                        continue
                    body = nested.reset_index(drop=True)
                    if names_sep is not None:
                        body = body.add_prefix(f"{column}{names_sep}")
                    collisions = set(body.columns) & set(parent_columns)
                    if collisions:
                        raise ValueError(
                            f"unnest() output columns collide: {sorted(collisions)}; "
                            "supply names_sep"
                        )
                    parent_frame = pd.DataFrame(
                        [parent] * len(body), columns=parent_columns
                    )
                    pieces.append(
                        pd.concat(
                            [parent_frame.reset_index(drop=True), body],
                            axis=1,
                        )
                    )
                if not pieces:
                    return tf._with_pdf(
                        pdf.iloc[0:0].drop(columns=column),
                        groups=_result_groups(
                            tf, [name for name in columns if name != column]
                        ),
                        rowwise=False,
                    )
                output = pd.concat(pieces, ignore_index=True)
                return tf._with_pdf(
                    output,
                    groups=_result_groups(tf, output.columns),
                    rowwise=False,
                )

            if not keep_empty:
                keep = pdf[column].map(
                    lambda value: isinstance(value, (list, tuple))
                    and len(value) > 0
                )
                pdf = pdf.loc[keep]
            pdf = pdf.explode(column, ignore_index=True)
            first = next(
                (value for value in pdf[column] if isinstance(value, dict)),
                None,
            )
            if first is None:
                return tf._with_pdf(
                    pdf,
                    groups=_result_groups(tf, pdf.columns),
                    rowwise=False,
                )
            expanded = pd.DataFrame(
                pdf[column]
                .map(lambda value: value if isinstance(value, dict) else {})
                .tolist(),
                index=pdf.index,
            )
            if names_sep is not None:
                expanded = expanded.add_prefix(f"{column}{names_sep}")
            collisions = set(expanded.columns) & (set(columns) - {column})
            if collisions:
                raise ValueError(
                    f"unnest() output columns collide: {sorted(collisions)}; supply names_sep"
                )
            position = columns.index(column)
            output = pdf.drop(columns=column)
            for offset, name in enumerate(expanded.columns):
                output.insert(position + offset, name, expanded[name])
            return tf._with_pdf(
                output,
                groups=_result_groups(tf, output.columns),
                rowwise=False,
            )

        dtype = tf._lf.collect_schema()[column]
        if dtype.base_type() != pl.List:
            raise TypeError("unnest() requires a list column")
        inner = dtype.inner
        output = tf._lf.explode(
            column,
            empty_as_null=keep_empty,
            keep_nulls=keep_empty,
        )
        if inner.base_type() == pl.Struct:
            output = output.unnest(column, separator=names_sep)
        ordered = output.collect_schema().names()
        return tf._with_lf(
            output,
            groups=_result_groups(tf, ordered),
            rowwise=False,
        )

    return Verb(_apply, "unnest")


def unnest_wider(column: str, *, names_sep: str | None = None) -> Verb:
    """Expand a struct/dict or fixed-width list column into columns."""

    def _apply(tf):
        columns = _columns(tf)
        if column not in columns:
            raise KeyError(f"column not found: {column!r}")
        if tf._backend == "pandas":
            import pandas as pd

            values = tf._pdf[column]
            first = next((value for value in values if value is not None), None)
            if isinstance(first, dict):
                expanded = pd.DataFrame(values.map(lambda value: value or {}).tolist(), index=tf._pdf.index)
            elif isinstance(first, (list, tuple)):
                if names_sep is None:
                    raise ValueError("names_sep is required for unnamed list values")
                expanded = pd.DataFrame(values.tolist(), index=tf._pdf.index)
                expanded.columns = [
                    f"{column}{names_sep}{index + 1}"
                    for index in range(expanded.shape[1])
                ]
            else:
                raise TypeError("unnest_wider() requires dict/struct or list values")
            collisions = set(expanded.columns) & (set(columns) - {column})
            if collisions:
                if names_sep is None:
                    raise ValueError(
                        f"unnest_wider() output columns collide: {sorted(collisions)}; supply names_sep"
                    )
                expanded = expanded.add_prefix(f"{column}{names_sep}")
            position = columns.index(column)
            left = tf._pdf.drop(columns=column)
            for offset, name in enumerate(expanded.columns):
                left.insert(position + offset, name, expanded[name])
            return tf._with_pdf(
                left,
                groups=_result_groups(tf, left.columns),
                rowwise=False,
            )

        dtype = tf._lf.collect_schema()[column]
        if dtype.base_type() == pl.Struct:
            lf = tf._lf.unnest(column, separator=names_sep)
        elif dtype.base_type() == pl.List:
            if names_sep is None:
                raise ValueError("names_sep is required for unnamed list values")
            width = (
                tf._lf.select(pl.col(column).list.len().max())
                .collect()
                .item()
                or 0
            )
            expressions = [
                pl.col(column)
                .list.get(index, null_on_oob=True)
                .alias(f"{column}{names_sep}{index + 1}")
                for index in range(width)
            ]
            output = []
            for name in columns:
                if name == column:
                    output.extend(expressions)
                else:
                    output.append(pl.col(name))
            lf = tf._lf.select(output)
        else:
            raise TypeError("unnest_wider() requires a struct or list column")
        ordered = lf.collect_schema().names()
        return tf._with_lf(
            lf,
            groups=_result_groups(tf, ordered),
            rowwise=False,
        )

    return Verb(_apply, "unnest_wider")


def separate_longer_delim(cols: Any, delim: str) -> Verb:
    """Split delimited values into multiple rows."""
    if not isinstance(delim, str) or not delim:
        raise ValueError("separate_longer_delim() delim must be non-empty")

    def _apply(tf):
        import pandas as pd

        selected = resolve_selection(tf, [cols])
        if len(selected) != 1:
            raise ValueError("separate_longer_delim() currently accepts one column")
        column = selected[0]
        pdf = tf.collect(as_="pandas").copy()
        pdf[column] = pdf[column].map(
            lambda value: value.split(delim) if isinstance(value, str) else [value]
        )
        pdf = pdf.explode(column, ignore_index=True)
        from tidy3.frame import tidy

        return tidy(pdf, backend=tf._backend)

    return Verb(_apply, "separate_longer_delim")


def separate_wider_delim(
    column: str,
    names: list[str] | tuple[str, ...],
    delim: str,
    *,
    too_few: str = "error",
    too_many: str = "error",
) -> Verb:
    """Split one delimited column into named columns, like tidyr.

    ``too_few`` is ``error`` (default), ``align_start``, or ``align_end``;
    ``too_many`` is ``error`` (default), ``drop``, or ``merge`` (the extra
    pieces stay joined in the last column). Missing values stay missing.
    """
    if not names:
        raise TypeError("separate_wider_delim() requires output names")
    if too_few not in {"align_start", "align_end", "error"}:
        raise ValueError("too_few must be align_start, align_end, or error")
    if too_many not in {"error", "drop", "merge"}:
        raise ValueError("too_many must be error, drop, or merge")
    width = len(names)

    def _apply(tf):
        import pandas as pd
        from tidy3.frame import tidy

        selected = resolve_selection(tf, [column])
        if len(selected) != 1:
            raise ValueError("separate_wider_delim() requires one column")
        source = selected[0]
        pdf = tf.collect(as_="pandas").copy()
        present = pdf[source].notna()
        pieces = [
            str(value).split(delim) if keep else None
            for value, keep in zip(pdf[source], present)
        ]
        counts = [len(values) for values in pieces if values is not None]
        too_short = sum(count < width for count in counts)
        too_long = sum(count > width for count in counts)
        if too_few == "error" and too_short:
            raise ValueError(
                f"separate_wider_delim() expected {width} pieces in each value "
                f"of {source!r}; {too_short} value(s) had fewer. "
                'Use too_few="align_start" or "align_end" to fill with missing.'
            )
        if too_many == "error" and too_long:
            raise ValueError(
                f"separate_wider_delim() expected {width} pieces in each value "
                f"of {source!r}; {too_long} value(s) had more. "
                'Use too_many="drop" or "merge".'
            )

        def align(values):
            if values is None:
                return [None] * width
            if len(values) > width:
                if too_many == "merge":
                    return [*values[: width - 1], delim.join(values[width - 1 :])]
                return values[:width]
            pad = [None] * (width - len(values))
            return pad + values if too_few == "align_end" else values + pad

        expanded = pd.DataFrame(
            [align(values) for values in pieces],
            columns=list(names),
            index=pdf.index,
            dtype=object,
        )
        # The new columns take the place of the split one, as in tidyr.
        at = pdf.columns.get_loc(source)
        pdf = pd.concat(
            [pdf.iloc[:, :at], expanded, pdf.iloc[:, at + 1 :]], axis=1
        )
        return tidy(pdf, backend=tf._backend)

    return Verb(_apply, "separate_wider_delim")


def hoist(column: str, *paths: Any, **named_paths: Any) -> Verb:
    """Extract named fields from dictionary/list columns."""
    def _apply(tf):
        import pandas as pd
        from tidy3.frame import tidy

        pdf = tf.collect(as_="pandas").copy()
        specs = list(named_paths.items()) or [(str(path), path) for path in paths]
        for name, path in specs:
            keys = path if isinstance(path, (list, tuple)) else [path]
            pdf[name] = pdf[column].map(
                lambda value: _pluck(value, keys)
            )
        return tidy(pdf, backend=tf._backend)
    return Verb(_apply, "hoist")


def _pluck(value: Any, keys: list[Any]) -> Any:
    for key in keys:
        if value is None:
            return None
        if isinstance(value, dict):
            value = value.get(key)
        elif isinstance(value, (list, tuple)) and isinstance(key, int):
            value = value[key] if -len(value) <= key < len(value) else None
        else:
            return None
    return value


def pack(name: str, *cols: Any) -> Verb:
    """Pack selected columns into a structured column."""
    def _apply(tf):
        import pandas as pd
        from tidy3.frame import tidy

        selected = resolve_selection(tf, cols or [c for c in _columns(tf) if c not in (tf._groups or [])])
        if tf._backend == "polars":
            lf = tf._lf.with_columns(pl.struct(selected).alias(name)).drop(selected)
            return tf._with_lf(lf, groups=_result_groups(tf, lf.collect_schema().names()))
        pdf = tf.collect(as_="pandas").copy()
        pdf[name] = pdf[selected].to_dict(orient="records")
        pdf = pdf.drop(columns=selected)
        return tidy(pdf, backend=tf._backend)
    return Verb(_apply, "pack")


def unpack(column: str) -> Verb:
    """Expand a structured dictionary column into ordinary columns."""
    def _apply(tf):
        import pandas as pd
        from tidy3.frame import tidy

        if tf._backend == "polars":
            return tf >> unnest_wider(column)
        pdf = tf.collect(as_="pandas").copy()
        expanded = pd.json_normalize(pdf[column]).set_index(pdf.index)
        pdf = pd.concat([pdf.drop(columns=column), expanded], axis=1)
        return tidy(pdf, backend=tf._backend)
    return Verb(_apply, "unpack")


__all__ = [
    "crossing",
    "expand_grid",
    "full_seq",
    "uncount",
    "complete",
    "drop_na",
    "expand",
    "fill",
    "nest",
    "nesting",
    "pivot_longer",
    "pivot_wider",
    "replace_na",
    "separate",
    "unite",
    "unnest",
    "unnest_longer",
    "unnest_wider",
    "separate_longer_delim",
    "separate_wider_delim",
    "hoist",
    "pack",
    "unpack",
]
